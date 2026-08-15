"""
HTTP Gateway 路由（第四阶段起）。

职责（Gateway 只做"接单"，不做"执行"）：
    POST /api/chat       校验参数 -> 生成 request_id/job_id -> 写 Redis 队列 -> 返回
    GET  /api/jobs/{id}  查询 Job 状态
    GET  /api/traces/{id} 查询 Trace 调用树（第六阶段）
    GET  /health         健康检查

设计要点：Gateway 进程内绝不直接执行 Agent —— 执行交给 Worker，
这样 HTTP 层可以快速返回、Worker 可以横向扩展。

Trace 起点：gateway.request Span 在此创建，trace_context 随 Job 写入 Redis，
Worker 消费后恢复同一 trace_id —— 这是跨进程链路不断的关键。
"""
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from app.queue.models import Job
from app.queue.producer import RedisJobQueue, utc_now
from app.tools.schemas import UserContext
from app.tracing.context import get_trace_context
from app.tracing.span import trace_span

router = APIRouter()


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """提交一个对话任务的请求体。"""

    message: str = Field(description="用户消息")
    session_id: str | None = Field(default=None, description="会话 ID；缺省时自动创建")
    idempotency_key: str | None = Field(
        default=None, description="幂等键：相同键的重复提交只执行一次"
    )
    # 权限上下文（第五阶段 Tool Permission / Policy 使用）
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# POST /api/chat：入队（创建根 Trace 并传播）
# ---------------------------------------------------------------------------
@router.post("/api/chat", status_code=202)
async def chat(req: ChatRequest, request: Request) -> dict:
    queue: RedisJobQueue = request.app.state.queue
    recorder = request.app.state.recorder

    request_id = req.idempotency_key or f"req_{uuid4().hex[:12]}"
    session_id = req.session_id or f"session_{uuid4().hex[:12]}"

    async def _do_chat() -> dict:
        job = Job(
            job_id=f"job_{uuid4().hex[:12]}",
            request_id=request_id,
            session_id=session_id,
            input={"message": req.message},
            # 权限上下文随 Job 流转到 Worker（Stage 5）
            user=UserContext(
                user_id=req.user_id or "",
                roles=req.roles,
                permissions=req.permissions,
            ).model_dump(),
            # 幂等入队前写入 Trace 传播上下文（Stage 6 关键）
            trace_context=get_trace_context(),
            created_at=utc_now(),
        )
        enqueued = await queue.enqueue(job)
        return {
            "request_id": enqueued.request_id,
            "job_id": enqueued.job_id,
            "session_id": enqueued.session_id,
            "status": enqueued.status.value,
        }

    if recorder is None or not recorder.enabled:
        return await _do_chat()

    async with trace_span(
        "gateway.request",
        "gateway",
        input={"message": req.message, "session_id": session_id, "user_id": req.user_id},
        attributes={"session_id": session_id, "request_id": request_id},
        recorder=recorder,
    ) as span:
        body = await _do_chat()
        span.output = {"job_id": body["job_id"], "status": body["status"]}
        return body


# ---------------------------------------------------------------------------
# GET /api/jobs/{job_id}：查询状态
# ---------------------------------------------------------------------------
@router.get("/api/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict:
    queue: RedisJobQueue = request.app.state.queue
    job = await queue.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job 不存在: {job_id}")
    return job.model_dump(mode="json")


# ---------------------------------------------------------------------------
# GET /api/traces/{trace_id}：查询 Trace 调用树（第六阶段）
# ---------------------------------------------------------------------------
@router.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str, request: Request) -> dict:
    recorder = request.app.state.recorder
    if recorder is None:
        raise HTTPException(status_code=404, detail="Tracing 未启用")
    tree = recorder.build_tree(trace_id)
    if not tree["spans"]:
        raise HTTPException(status_code=404, detail=f"Trace 不存在: {trace_id}")
    return tree


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}
