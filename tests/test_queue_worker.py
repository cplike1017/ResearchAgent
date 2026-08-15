"""
Stage 4 测试：Redis Queue + Worker + 幂等 + 重试 + API。

使用 fakeredis（内存假实现），完全离线。
"""
import asyncio

import pytest
from fastapi.testclient import TestClient

from app.api.routes import router
from app.config import Settings
from app.main import create_app
from app.queue.consumer import process_job
from app.queue.models import Job, JobStatus
from app.queue.producer import RedisJobQueue, utc_now
from app.worker.worker import run_worker


@pytest.fixture
async def queue(settings):
    """fakeredis 支撑的队列。"""
    import fakeredis.aioredis

    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    q = RedisJobQueue(redis, settings)
    yield q
    await redis.aclose()


def _make_job(request_id: str, message: str, session_id: str = "session_q") -> Job:
    return Job(
        job_id=f"job_{request_id}",
        request_id=request_id,
        session_id=session_id,
        input={"message": message},
        created_at=utc_now(),
    )


# ---------------------------------------------------------------------------
# 入队 / 读取
# ---------------------------------------------------------------------------
async def test_enqueue_and_roundtrip(queue):
    job = _make_job("req_1", "查询北京天气")
    await queue.enqueue(job)

    got = await queue.get_job(job.job_id)
    assert got is not None
    assert got.status == JobStatus.QUEUED
    assert got.input == {"message": "查询北京天气"}
    assert await queue.queue_length() == 1


async def test_pop_consumes_job(queue):
    await queue.enqueue(_make_job("req_2", "你好"))
    popped = await queue.pop(timeout=0.1)
    assert popped is not None
    assert popped.request_id == "req_2"
    assert await queue.queue_length() == 0


async def test_pop_timeout_returns_none(queue):
    assert await queue.pop(timeout=0.01) is None


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------
async def test_idempotency_same_request_id(queue):
    """相同 request_id 重复提交 -> 返回同一个 Job，队列不重复。"""
    job = _make_job("req_same", "查询北京天气")
    first = await queue.enqueue(job)
    second = await queue.enqueue(_make_job("req_same", "查询北京天气"))
    assert first.job_id == second.job_id
    assert await queue.queue_length() == 1


async def test_idempotency_different_request_ids(queue):
    a = await queue.enqueue(_make_job("req_a", "你好"))
    b = await queue.enqueue(_make_job("req_b", "你好"))
    assert a.job_id != b.job_id
    assert await queue.queue_length() == 2


# ---------------------------------------------------------------------------
# Worker 处理
# ---------------------------------------------------------------------------
async def test_process_job_success(queue, runtime):
    job = await queue.enqueue(_make_job("req_ok", "查询北京天气", session_id="session_ok"))
    done = await process_job(queue, lambda: runtime, job)
    assert done.status == JobStatus.SUCCEEDED
    assert "北京" in done.result["answer"]
    assert done.result["session_id"] == "session_ok"


async def test_process_job_status_transitions(queue, runtime):
    job = await queue.enqueue(_make_job("req_trans", "查询北京天气"))
    assert job.status == JobStatus.QUEUED
    await process_job(queue, lambda: runtime, job)
    # 中间态 RUNNING 已被覆盖为 SUCCEEDED，但可通过日志/钩子观察；这里验证终态
    final = await queue.get_job(job.job_id)
    assert final.status == JobStatus.SUCCEEDED


async def test_retry_until_max_attempts(queue):
    """连续失败 -> attempt 递增重入队 -> 达到 max_attempts 后 FAILED。"""

    class FailingRuntime:
        """模拟 Worker 内部崩溃的运行时。"""

        async def run(self, *args, **kwargs):
            raise RuntimeError("worker 内部错误")

    job = await queue.enqueue(_make_job("req_fail", "你好"))

    for expected_attempt in (0, 1, 2):
        popped = await queue.pop(timeout=0.1)
        assert popped is not None
        await process_job(queue, lambda: FailingRuntime(), popped)
        state = await queue.get_job(job.job_id)
        if expected_attempt < 2:
            assert state.status == JobStatus.QUEUED  # 已重新入队
            assert state.attempt == expected_attempt + 1
        else:
            assert state.status == JobStatus.FAILED  # 用尽重试
            assert state.error["type"] == "RuntimeError"
    assert await queue.queue_length() == 0  # 不再无限重试


async def test_worker_loop_processes_jobs(queue, runtime, settings):
    """run_worker 主循环：入队 3 个 -> Worker 消费 -> 全部 SUCCEEDED。"""
    for i in range(3):
        await queue.enqueue(_make_job(f"req_w{i}", "你好", session_id=f"session_w{i}"))

    shutdown = asyncio.Event()
    worker_task = asyncio.create_task(
        run_worker(settings=settings, shutdown_event=shutdown, worker_id="test-worker", queue=queue)
    )
    try:
        # 轮询等待全部完成（最多 5 秒）
        for _ in range(50):
            jobs = [await queue.get_job(f"job_req_w{i}") for i in range(3)]
            if all(j is not None and j.status == JobStatus.SUCCEEDED for j in jobs):
                break
            await asyncio.sleep(0.1)
        for i in range(3):
            final = await queue.get_job(f"job_req_w{i}")
            assert final is not None and final.status == JobStatus.SUCCEEDED
    finally:
        shutdown.set()
        await worker_task


async def test_concurrent_consumers(queue, runtime):
    """3 个并发消费者处理 10 个 Job：全部成功且无重复消费。"""
    for i in range(10):
        await queue.enqueue(_make_job(f"req_c{i}", "计算 1 + 1", session_id=f"session_c{i}"))

    async def consumer(name: str):
        processed = []
        while True:
            job = await queue.pop(timeout=0.1)
            if job is None:
                break
            await process_job(queue, lambda: runtime, job)
            processed.append(job.job_id)
        return processed

    results = await asyncio.gather(*(consumer(f"w{i}") for i in range(3)))
    all_ids = [jid for group in results for jid in group]
    assert len(all_ids) == 10
    assert len(set(all_ids)) == 10  # 每个 job 恰好被消费一次
    for i in range(10):
        final = await queue.get_job(f"job_req_c{i}")
        assert final.status == JobStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# HTTP API（TestClient + fakeredis）
# ---------------------------------------------------------------------------
@pytest.fixture
def api_client(settings):
    import fakeredis.aioredis

    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app = create_app(settings, redis=fake_redis)
    with TestClient(app) as client:  # 触发 lifespan
        yield client


def test_api_chat_enqueue(api_client):
    resp = api_client.post("/api/chat", json={"message": "查询北京天气"})
    assert resp.status_code == 202
    body = resp.json()
    assert body["request_id"].startswith("req_")
    assert body["job_id"].startswith("job_")
    assert body["status"] == "QUEUED"

    job = api_client.get(f"/api/jobs/{body['job_id']}")
    assert job.status_code == 200
    assert job.json()["input"]["message"] == "查询北京天气"


def test_api_idempotency(api_client):
    payload = {"message": "你好", "idempotency_key": "my-key-1"}
    r1 = api_client.post("/api/chat", json=payload).json()
    r2 = api_client.post("/api/chat", json=payload).json()
    assert r1["job_id"] == r2["job_id"]


def test_api_job_not_found(api_client):
    assert api_client.get("/api/jobs/nope").status_code == 404


def test_api_health(api_client):
    assert api_client.get("/health").json() == {"status": "ok"}
