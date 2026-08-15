"""
Stage 4 Demo：Redis Queue + Worker。

运行前提：需要可用的 Redis。
    方式一（推荐）：docker compose up -d redis
    方式二：本地已有 redis-server（修改 .env 的 REDIS_URL）

运行：python -m demos.stage4_demo

展示：
    A. 同时提交 10 个请求 -> 3 个并发 Worker 消费 -> 全部完成
    B. 失败重试：Worker 崩溃 -> attempt 递增重入队 -> 达到上限 FAILED（不无限重试）

说明：本 Demo 在同一进程内用 asyncio 任务模拟 3 个 Worker（对应生产环境的
3 个容器/进程）；消息分发机制与多进程完全一致（Redis BLPOP）。
"""
import asyncio
import random
import time

from app.config import get_settings
from app.queue.models import Job, JobStatus
from app.queue.producer import RedisJobQueue, utc_now

SEPARATOR = "=" * 64

# 10 个并发请求（混用工具 / 非工具场景）
REQUESTS = [
    ("查询北京天气", "北京"),
    ("计算 123 * 456", "56088"),
    ("查询上海天气", "上海"),
    ("你好", None),
    ("计算 2 ** 10", "1024"),
    ("查询广州天气", "广州"),
    ("你好", None),
    ("计算 100 / 4", "25"),
    ("查询成都天气", "成都"),
    ("你好", None),
]


class FailingRuntime:
    """模拟 Worker 内部崩溃的运行时（用于重试演示）。"""

    async def run(self, *args, **kwargs):
        raise RuntimeError("模拟 Worker 崩溃")


async def main() -> None:
    settings = get_settings()
    import redis.asyncio as aioredis

    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    queue = RedisJobQueue(redis, settings)

    # 0) 检查 Redis 可用性
    try:
        await redis.ping()
    except Exception as exc:  # noqa: BLE001
        print(f"无法连接 Redis ({settings.redis_url}): {exc}")
        print("请先启动 Redis：docker compose up -d redis")
        await redis.aclose()
        return

    print(f"Redis 已连接: {settings.redis_url}")
    print(f"队列名: {settings.queue_name} | max_attempts: {settings.max_attempts}")

    # =================================================================
    print("\n" + SEPARATOR)
    print("阶段 A：同时提交 10 个请求，3 个并发 Worker 消费")
    print(SEPARATOR)

    # 1) 入队 10 个 Job（run 后缀保证每次 Demo 可重复执行，不被幂等拦截）
    run_suffix = str(int(time.time()))
    job_ids = []
    for i, (message, _) in enumerate(REQUESTS):
        job = Job(
            job_id=f"job_demo_{run_suffix}_{i}",
            request_id=f"req_demo_{run_suffix}_{i}",
            session_id=f"session_demo_{run_suffix}_{i}",
            input={"message": message},
            created_at=utc_now(),
        )
        await queue.enqueue(job)
        job_ids.append(job.job_id)
    print(f"已入队 {len(job_ids)} 个 Job，队列长度 = {await queue.queue_length()}")

    # 2) 启动 3 个并发消费者（模拟 3 个 Worker 进程）
    # 注意：每个 Worker 使用独立的 Redis 客户端 —— 与生产环境一致
    # （同一客户端实例上的并发 BLPOP 可能被连接池串行化，导致任务全被一个消费者拿走）
    consumed_by: dict[str, str] = {}

    async def consumer(name: str):
        import redis.asyncio as aioredis

        worker_redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        worker_queue = RedisJobQueue(worker_redis, settings)
        try:
            while True:
                job = await worker_queue.pop(timeout=0.2)
                if job is None:
                    return
                consumed_by[job.job_id] = name
                # 随机抖动模拟各 Worker 处理速度差异（让任务分发更真实可见）
                await asyncio.sleep(random.uniform(0.01, 0.05))
                # 用真实 AgentRuntime 处理
                from app.agent.runtime import AgentRuntime

                from app.checkpoint.repository import SQLiteCheckpointRepository
                from app.session.repository import SQLiteSessionRepository

                rt = AgentRuntime(
                    settings=settings,
                    session_repo=SQLiteSessionRepository(settings.database_url),
                    checkpoint_repo=SQLiteCheckpointRepository(settings.database_url),
                )
                from app.queue.consumer import process_job

                done = await process_job(worker_queue, lambda: rt, job)
                print(f"  [{name}] {job.job_id} -> {done.status.value}", flush=True)
        finally:
            await worker_redis.aclose()

    workers = [asyncio.create_task(consumer(f"worker-{i}")) for i in range(3)]
    await asyncio.gather(*workers)

    # 3) 汇总
    print("\n结果汇总：")
    ok = 0
    for i, (message, expect) in enumerate(REQUESTS):
        job = await queue.get_job(f"job_demo_{run_suffix}_{i}")
        worker = consumed_by.get(job.job_id, "-")
        answer = (job.result or {}).get("answer", "") if job.status == JobStatus.SUCCEEDED else ""
        matched = expect is None or expect in answer
        ok += 1 if matched else 0
        print(f"  job={job.job_id:<12} {job.status.value:<10} worker={worker:<9} "
              f"输入={message:<14} 答案={answer[:30]}")

    print(f"\n分发到不同 Worker 的请求: {len(set(consumed_by.values()))} 个 Worker 参与消费")
    print(f"正确率: {ok}/{len(REQUESTS)}")
    print(f"队列剩余: {await queue.queue_length()}")

    # =================================================================
    print("\n" + SEPARATOR)
    print("阶段 B：失败重试（Worker 崩溃 -> 重入队 -> 达到上限 FAILED）")
    print(SEPARATOR)

    bad_job = Job(
        job_id=f"job_demo_bad_{run_suffix}",
        request_id=f"req_demo_bad_{run_suffix}",
        session_id=f"session_demo_bad_{run_suffix}",
        input={"message": "触发崩溃"},
        created_at=utc_now(),
    )
    await queue.enqueue(bad_job)
    from app.queue.consumer import process_job

    for i in range(settings.max_attempts):
        popped = await queue.pop(timeout=0.2)
        if popped is None:
            break
        await process_job(queue, lambda: FailingRuntime(), popped)
        state = await queue.get_job(bad_job.job_id)
        print(f"  第 {i + 1} 次尝试 -> attempt={state.attempt} status={state.status.value}")
        if state.status == JobStatus.FAILED:
            print(f"  最终错误: {state.error}")
            break
    print(f"队列剩余: {await queue.queue_length()}（无无限重试）")

    await redis.aclose()
    print("\nDemo 完成。")


if __name__ == "__main__":
    asyncio.run(main())
