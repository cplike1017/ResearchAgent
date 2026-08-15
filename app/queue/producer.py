"""
Redis Job 队列（Producer 侧 + 共享操作）。

自行用 Redis List 实现队列（BLPOP/RPUSH），不引入 Celery / RQ，以看清机制：

    - 队列本体：Redis List `agent:jobs:queue`
    - Job 数据：Redis Hash `agent:jobs:{job_id}`（字段 = Job 模型）
    - 幂等：Redis String `agent:requests:{request_id}`（SET NX + TTL）

幂等（Idempotency）：
    客户端重复提交相同 request_id 时，第二次入队直接返回第一次的 job，
    绝不重复执行 —— 防止网络重试导致 Agent 重复运行。
"""
import json
from datetime import datetime, timezone

from redis.asyncio import Redis

from app.config import Settings
from app.errors import QueueError
from app.queue.models import Job, JobStatus
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RedisJobQueue:
    """基于 Redis List + Hash 的最小任务队列。"""

    def __init__(
        self,
        redis: Redis,
        settings: Settings,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self._redis = redis
        self.queue_name = settings.queue_name
        self.job_key_prefix = settings.job_key_prefix
        self.request_key_prefix = settings.request_key_prefix
        self.max_attempts = settings.max_attempts
        self.job_ttl = settings.job_ttl_seconds
        self.recorder = recorder  # None = 不追踪

    # ------------------------------------------------------------------
    # 键名
    # ------------------------------------------------------------------
    def _job_key(self, job_id: str) -> str:
        return f"{self.job_key_prefix}{job_id}"

    def _request_key(self, request_id: str) -> str:
        return f"{self.request_key_prefix}{request_id}"

    # ------------------------------------------------------------------
    # 序列化：Hash 字段是字符串，dict 字段需要 JSON 编码
    # ------------------------------------------------------------------
    @staticmethod
    def _job_to_mapping(job: Job) -> dict:
        d = job.model_dump(mode="json")
        for key in ("input", "user", "result", "error", "trace_context"):
            value = d.get(key)
            d[key] = json.dumps(value, ensure_ascii=False) if value is not None else ""
        return d

    @staticmethod
    def _job_from_mapping(raw: dict) -> Job:
        d = dict(raw)
        for key in ("input", "user", "result", "error", "trace_context"):
            value = d.get(key)
            if value in (None, ""):
                d[key] = {} if key != "result" else None
            else:
                try:
                    d[key] = json.loads(value)
                except (TypeError, json.JSONDecodeError):
                    d[key] = {} if key != "result" else None
        return Job(**d)

    # ------------------------------------------------------------------
    # 写入 / 读取
    # ------------------------------------------------------------------
    async def save_job(self, job: Job) -> Job:
        """把 Job 完整写入 Hash 并设置 TTL。"""
        await self._redis.hset(self._job_key(job.job_id), mapping=self._job_to_mapping(job))
        await self._redis.expire(self._job_key(job.job_id), self.job_ttl)
        return job

    async def get_job(self, job_id: str) -> Job | None:
        raw = await self._redis.hgetall(self._job_key(job_id))
        if not raw:
            return None
        return self._job_from_mapping(raw)

    # ------------------------------------------------------------------
    # Producer：入队（含幂等）
    # ------------------------------------------------------------------
    async def enqueue(self, job: Job) -> Job:
        """
        入队一个 Job。

        幂等流程：
            1. request_id 已有记录 -> 直接返回已有 Job（不重复入队）；
            2. SET NX 原子占位（防并发重复提交）；
            3. 写入 Job Hash + RPUSH 到队列 List。
        """
        if self.recorder is None or not self.recorder.enabled:
            return await self._enqueue_impl(job)

        async with trace_span(
            "redis.enqueue",
            "queue",
            input={"job_id": job.job_id, "request_id": job.request_id},
            attributes={"job_id": job.job_id, "request_id": job.request_id},
            recorder=self.recorder,
        ) as span:
            enqueued = await self._enqueue_impl(job)
            span.output = {"job_id": enqueued.job_id, "status": enqueued.status.value}
            return enqueued

    async def _enqueue_impl(self, job: Job) -> Job:
        existing = await self._redis.get(self._request_key(job.request_id))
        if existing is not None:
            return await self.get_job(existing)

        # SET NX：仅当键不存在时写入，返回 True 表示本进程抢占成功
        acquired = await self._redis.set(
            self._request_key(job.request_id), job.job_id, nx=True, ex=self.job_ttl
        )
        if not acquired:
            # 并发下被其他进程抢占了：返回已有的 Job
            winner = await self._redis.get(self._request_key(job.request_id))
            return await self.get_job(winner)

        job.created_at = job.created_at or utc_now()
        await self.save_job(job)
        await self._redis.rpush(self.queue_name, job.job_id)
        return job

    # ------------------------------------------------------------------
    # 状态更新
    # ------------------------------------------------------------------
    async def update_status(
        self,
        job_id: str,
        status: JobStatus,
        *,
        result: dict | None = None,
        error: dict | None = None,
    ) -> Job:
        job = await self.get_job(job_id)
        if job is None:
            raise QueueError(f"Job 不存在: {job_id}")
        job.status = status
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        await self.save_job(job)
        return job

    # ------------------------------------------------------------------
    # Consumer：出队 / 重试
    # ------------------------------------------------------------------
    async def pop(self, timeout: float = 0) -> Job | None:
        """BLPOP 阻塞弹出队头 Job；超时返回 None。"""
        item = await self._redis.blpop(self.queue_name, timeout=timeout)
        if item is None:
            return None
        job_id = item[1]
        return await self.get_job(job_id)

    async def requeue(self, job_id: str) -> None:
        """重新入队（重试）。"""
        await self._redis.rpush(self.queue_name, job_id)

    async def queue_length(self) -> int:
        return await self._redis.llen(self.queue_name)
