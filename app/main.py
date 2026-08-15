"""
FastAPI 应用入口。

启动：
    本地:  uvicorn app.main:app --port 8000
    Docker: docker compose up --build

lifespan 中创建 Redis 连接并挂到 app.state.queue；
Worker 是独立进程（app.worker.worker），由 docker compose 或手动启动。
"""
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI

from app.api.routes import router
from app.config import Settings, get_settings
from app.queue.producer import RedisJobQueue
from app.tracing.recorder import TraceRecorder


def create_app(settings: Settings | None = None, redis=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 每个进程一个 Redis 异步连接池（测试可注入 fakeredis）
        client = redis or aioredis.from_url(settings.redis_url, decode_responses=True)
        # 每个进程一个 Trace Recorder（与 Worker 共享同一个 JSONL 文件）
        recorder = TraceRecorder(
            settings.trace_file,
            enabled=settings.trace_enabled,
            capture_content=settings.trace_capture_content,
        )
        app.state.queue = RedisJobQueue(client, settings, recorder=recorder)
        app.state.recorder = recorder
        app.state.settings = settings
        print(f"[api] Redis 队列已连接: {settings.redis_url}", flush=True)
        yield
        await client.aclose()

    app = FastAPI(title="agent-runtime", version=settings.agent_version, lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
