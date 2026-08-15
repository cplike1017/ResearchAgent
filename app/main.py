"""
FastAPI 应用入口（Web UI 版）。

启动：
    本地:  uvicorn app.main:app --port 8000
    Docker: docker compose up --build

lifespan 中：
    1. 创建 Redis 连接并挂到 app.state.queue（异步队列模式）；
    2. 构建 Web 运行时（app.state.runtime）：AgentRuntime + 记忆 + MCP + 技能，
       供 /api/web/* 进程内直连使用（不需要 Worker）。

Web UI 页面：http://localhost:8000/
"""
import asyncio
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.agent.runtime import AgentRuntime
from app.api.routes import router
from app.api.web import router as web_router
from app.checkpoint.repository import SQLiteCheckpointRepository
from app.config import Settings, get_settings
from app.llm.client import create_llm_client
from app.mcp.client import MCPClientManager
from app.memory.store import MemoryStore
from app.queue.producer import RedisJobQueue
from app.session.repository import SQLiteSessionRepository
from app.skills.manager import SkillManager
from app.tools.builtin import build_default_registry
from app.tracing.recorder import TraceRecorder


def build_web_runtime(settings: Settings, recorder: TraceRecorder) -> AgentRuntime:
    """构建 Web 进程内运行时：AgentRuntime + 记忆 + MCP + 技能。"""
    llm = create_llm_client(settings)
    session_repo = SQLiteSessionRepository(settings.database_url)
    checkpoint_repo = SQLiteCheckpointRepository(settings.database_url)

    # 记忆层（配置开启才启用）
    memory = None
    if settings.memory_enabled:
        memory = MemoryStore(settings=settings, recorder=recorder, llm=llm)

    # MCP（配置了 server 且非 test 环境才连接；连接失败不阻塞 Web 启动）
    mcp_client = None
    if settings.environment != "test":
        try:
            servers = settings.mcp_servers or "[]"
            if servers.strip("[] "):
                mcp_client = MCPClientManager(settings)
        except Exception:
            mcp_client = None

    # 技能（目录存在才启用）
    skill_manager = None
    if settings.skills_enabled:
        skill_manager = SkillManager(settings=settings, llm=llm)

    return AgentRuntime(
        llm=llm,
        registry=build_default_registry(),
        session_repo=session_repo,
        checkpoint_repo=checkpoint_repo,
        recorder=recorder,
        memory=memory,
        mcp_client=mcp_client,
        skill_manager=skill_manager,
        settings=settings,
    )


def create_app(settings: Settings | None = None, redis=None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 1) Redis 异步连接池（异步队列模式；测试可注入 fakeredis）
        client = redis or aioredis.from_url(settings.redis_url, decode_responses=True)
        # 2) Trace Recorder（与 Worker 共享同一个 JSONL 文件）
        recorder = TraceRecorder(
            settings.trace_file,
            enabled=settings.trace_enabled,
            capture_content=settings.trace_capture_content,
        )
        app.state.queue = RedisJobQueue(client, settings, recorder=recorder)
        app.state.recorder = recorder
        app.state.settings = settings
        # 3) Web 进程内运行时（记忆 + MCP + 技能）
        app.state.runtime = build_web_runtime(settings, recorder)

        # 4) 后台预初始化 MCP（不阻塞启动；首次请求时已就绪）
        if app.state.runtime.mcp_client is not None:
            try:
                await app.state.runtime.mcp_client.connect_all()
                from app.mcp.bridge import MCPBridge

                count = MCPBridge(app.state.runtime.mcp_client).register_all(app.state.runtime.registry)
                print(f"[api] MCP 预初始化完成：注册 {count} 个工具", flush=True)
            except Exception as exc:
                print(f"[api] MCP 预初始化失败（不影响内置工具）: {exc}", flush=True)
            app.state.runtime._mcp_initialized = True

        print(f"[api] Redis 队列已连接: {settings.redis_url}", flush=True)
        print(f"[api] Web 运行时就绪（mode={settings.agent_mode}, memory={settings.memory_enabled}）", flush=True)
        print(f"[api] Redis 队列已连接: {settings.redis_url}", flush=True)
        print(f"[api] Web 运行时就绪（mode={settings.agent_mode}, memory={settings.memory_enabled}）", flush=True)
        yield
        if app.state.runtime.mcp_client is not None:
            # 关闭 MCP：尽力清理并抑制 SDK 跨 task 噪音（捕获 BaseExceptionGroup）
            await _quiet_mcp_close(app.state.runtime.mcp_client)
        if app.state.runtime.memory is not None:
            app.state.runtime.memory.close()
        await client.aclose()

    app = FastAPI(title="agent-runtime", version=settings.agent_version, lifespan=lifespan)
    app.include_router(router)
    app.include_router(web_router)
    # Web UI 静态资源
    app.mount("/", StaticFiles(directory="app/static", html=True), name="static")
    return app


app = create_app()


async def _quiet_mcp_close(mcp_client) -> None:
    """静默关闭 MCP 连接。

    注意：uvicorn lifespan 的 startup/shutdown 在同一 task，MCP SDK 的
    stdio_client.__aexit__ 可以直接 await 干净关闭。这里仅兜底捕获
    BaseExceptionGroup（SDK 在异常路径下可能抛出，且不被 except Exception
    捕获），防止关闭阶段抛错。
    """
    try:
        await mcp_client.close()
    except BaseExceptionGroup:
        pass
    except Exception:
        pass
