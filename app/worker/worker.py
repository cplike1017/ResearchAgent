"""
Worker：独立进程，从 Redis 队列消费 Job 并执行 Agent。

执行模型：
    HTTP Gateway（FastAPI）
        ↓ RPUSH
    Redis List（agent:jobs:queue）
        ↓ BLPOP
    Worker（本模块）
        ↓
    Agent Runtime（Session / Checkpoint / Context Builder / Tools）

启动多个 Worker（docker compose up --scale worker=3）时，
Redis BLPOP 天然把请求分发给不同 Worker —— 并发能力由此而来。

第六阶段会在取到 Job 后把 trace_context 注入 contextvars，继续同一 Trace。
"""
import asyncio
import os
import signal

import redis.asyncio as aioredis

from app.agent.runtime import AgentRuntime
from app.checkpoint.repository import SQLiteCheckpointRepository
from app.config import Settings, get_settings
from app.queue.consumer import process_job
from app.queue.producer import RedisJobQueue
from app.session.repository import SQLiteSessionRepository
from app.tracing.recorder import TraceRecorder


def build_runtime_factory(settings: Settings, recorder: TraceRecorder | None = None):
    """返回 () -> AgentRuntime 的工厂：每个 Job 共享同一 SQLite。"""
    session_repo = SQLiteSessionRepository(settings.database_url)
    checkpoint_repo = SQLiteCheckpointRepository(settings.database_url)

    def factory() -> AgentRuntime:
        return AgentRuntime(
            settings=settings,
            session_repo=session_repo,
            checkpoint_repo=checkpoint_repo,
            recorder=recorder,
        )

    return factory


async def run_worker(
    settings: Settings | None = None,
    *,
    shutdown_event: asyncio.Event | None = None,
    worker_id: str = "worker",
    queue: RedisJobQueue | None = None,
) -> None:
    """Worker 主循环：BLPOP -> 处理 -> 循环。

    :param queue: 可注入的队列（测试用 fakeredis；缺省按 settings.redis_url 自建）
    """
    settings = settings or get_settings()
    owns_redis = queue is None
    recorder = TraceRecorder(
        settings.trace_file,
        enabled=settings.trace_enabled,
        capture_content=settings.trace_capture_content,
    )
    if queue is None:
        redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        queue = RedisJobQueue(redis, settings, recorder=recorder)
    runtime_factory = build_runtime_factory(settings, recorder=recorder)

    print(f"[{worker_id}] 启动，监听队列 {settings.queue_name}（max_attempts={settings.max_attempts}）", flush=True)
    try:
        while shutdown_event is None or not shutdown_event.is_set():
            # BLPOP 阻塞 1 秒：既能及时取任务，又能定期检查退出信号
            job = await queue.pop(timeout=1.0)
            if job is None:
                continue
            print(f"[{worker_id}] 消费 job={job.job_id} status={job.status.value} attempt={job.attempt}", flush=True)
            try:
                await process_job(queue, runtime_factory, job, recorder=recorder)
            except Exception as exc:  # process_job 已兜底，这里防御最后一层
                print(f"[{worker_id}] job={job.job_id} 处理异常: {exc}", flush=True)
            done = await queue.get_job(job.job_id)
            if done is not None:
                print(
                    f"[{worker_id}] job={job.job_id} -> {done.status.value}"
                    + (f"（重试至 attempt={done.attempt}）" if done.attempt else ""),
                    flush=True,
                )
    finally:
        if owns_redis:
            await redis.aclose()
        print(f"[{worker_id}] 已退出", flush=True)


async def main() -> None:
    settings = get_settings()
    worker_id = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, shutdown_event.set)
        except NotImplementedError:
            pass  # Windows 上部分信号不支持 add_signal_handler
    try:
        await run_worker(settings, shutdown_event=shutdown_event, worker_id=worker_id)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    asyncio.run(main())
