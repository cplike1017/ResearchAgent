"""
Web UI 路由（Stage 12）。

提供进程内直连模式（不依赖 Redis 队列 / Worker，单进程演示友好）：
    POST /api/web/chat        同步聊天（完整结果：answer + plan + tool_calls + trace）
    POST /api/web/chat/stream SSE 流式（每步决策 / 工具调用实时推送）
    GET  /api/web/tools       工具列表（内置 + MCP）
    GET  /api/web/skills      技能列表
    GET  /api/web/mcp         MCP server 状态
    GET  /api/web/sessions    会话历史

运行时从 request.app.state.runtime 取（main.py lifespan 构建的 AppRuntime 容器）。
"""
import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.agent.react_loop import LoopHooks
from app.errors import AgentError
from app.tools.schemas import ToolResult

router = APIRouter(prefix="/api/web", tags=["web"])


class WebChatRequest(BaseModel):
    """Web 聊天请求。"""

    message: str = Field(description="用户消息")
    session_id: str | None = Field(default=None, description="会话 ID；缺省自动创建")
    agent_mode: str | None = Field(default=None, description="react | plan；缺省用配置")


def _get_runtime(request: Request):
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="运行时未初始化")
    return runtime


# ---------------------------------------------------------------------------
# 同步聊天
# ---------------------------------------------------------------------------
@router.post("/chat")
async def web_chat(req: WebChatRequest, request: Request) -> dict:
    app = request.app.state
    runtime = _get_runtime(request)
    settings = app.settings

    # 会话级 agent_mode 覆盖
    agent_mode = req.agent_mode or settings.agent_mode
    old_mode = settings.agent_mode
    settings.agent_mode = agent_mode
    try:
        result = await runtime.run(req.message, session_id=req.session_id)
    finally:
        settings.agent_mode = old_mode

    return {
        "session_id": result.session_id,
        "answer": result.answer,
        "steps": result.steps,
        "plan": [s.model_dump() for s in result.plan],
        "plan_revisions": result.plan_revisions,
        "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in result.tool_calls],
        "trace_id": result.trace_id,
        "checkpoint_id": result.checkpoint_id,
        "mode": agent_mode,
        # Trace 树（Agent 工作流可视化；tracing 关闭时为 None）
        "trace": _build_trace_tree(request, result.trace_id),
    }


def _build_trace_tree(request: Request, trace_id: str | None) -> dict | None:
    """构建 Trace 调用树（供前端渲染 Agent 工作流）。"""
    if not trace_id:
        return None
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is None or not recorder.enabled:
        return None
    try:
        tree = recorder.build_tree(trace_id)
        return tree if tree.get("spans") else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# SSE 流式聊天（复用 LoopHooks 推送事件）
# ---------------------------------------------------------------------------
@router.post("/chat/stream")
async def web_chat_stream(req: WebChatRequest, request: Request) -> StreamingResponse:
    runtime = _get_runtime(request)
    settings = request.app.state.settings
    agent_mode = req.agent_mode or settings.agent_mode

    async def event_gen() -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue()
        session_id = req.session_id

        async def _emit(event: str, data: dict) -> None:
            await queue.put(f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n")

        # 流式事件钩子：每个关键节点推送
        async def _hook_after_decision(response, step: int) -> None:
            await _emit("step", {
                "step": step,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
                "content_preview": (response.content or "")[:200],
                "is_final": response.is_final_answer,
            })

        async def _hook_after_tool(tc, envelope: ToolResult, step: int) -> None:
            await _emit("tool_result", {
                "step": step,
                "tool": tc.name,
                "arguments": tc.arguments,
                "success": envelope.success,
                "data": str(envelope.data)[:300] if envelope.data else None,
                "error": envelope.error.model_dump() if envelope.error else None,
            })

        async def _hook_before_final(response, step: int) -> None:
            await _emit("final", {"step": step, "content": response.content or ""})

        hooks = LoopHooks(
            after_decision=_hook_after_decision,
            after_tool=_hook_after_tool,
            before_final=_hook_before_final,
        )

        # 生成器：先跑 Agent，同时消费队列推送
        async def _run() -> None:
            old_mode = settings.agent_mode
            settings.agent_mode = agent_mode
            try:
                result = await runtime.run(req.message, session_id=session_id, extra_hooks=hooks)
                # 附带 Trace 树（Agent 工作流可视化）
                trace_tree = None
                if result.trace_id:
                    try:
                        trace_tree = request.app.state.recorder.build_tree(result.trace_id)
                    except Exception:
                        trace_tree = None
                await _emit("done", {
                    "session_id": result.session_id,
                    "answer": result.answer,
                    "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in result.tool_calls],
                    "plan": [s.model_dump() for s in result.plan],
                    "plan_revisions": result.plan_revisions,
                    "trace_id": result.trace_id,
                    "trace": trace_tree,
                })
            except AgentError as exc:
                await _emit("error", {"type": type(exc).__name__, "message": str(exc)})
            except Exception as exc:
                await _emit("error", {"type": type(exc).__name__, "message": str(exc)})
            finally:
                settings.agent_mode = old_mode
                await queue.put(None)  # 结束信号

        task = asyncio.create_task(_run())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if not task.done():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# 能力列表
# ---------------------------------------------------------------------------
@router.get("/tools")
async def web_tools(request: Request) -> dict:
    runtime = _get_runtime(request)
    tools = []
    for t in runtime.registry.all():
        tools.append({
            "name": t.name,
            "description": t.description,
            "risk_level": t.risk_level,
            "required_permission": t.required_permission,
        })
    return {"tools": tools, "count": len(tools)}


@router.get("/skills")
async def web_skills(request: Request) -> dict:
    runtime = _get_runtime(request)
    skills = []
    if runtime.skill_manager is not None:
        skills = [
            {"name": s.name, "description": s.description, "triggers": s.triggers, "version": s.version}
            for s in runtime.skill_manager.all_skills()
        ]
    return {"skills": skills, "count": len(skills)}


@router.get("/mcp")
async def web_mcp(request: Request) -> dict:
    runtime = _get_runtime(request)
    servers = []
    if runtime.mcp_client is not None:
        servers = [
            {"name": c.name, "transport": c.transport, "tool_count": len(c.tools)}
            for c in runtime.mcp_client.connections
        ]
    return {"servers": servers, "count": len(servers)}


# ---------------------------------------------------------------------------
# Trace 树（Agent 工作流可视化）
# ---------------------------------------------------------------------------
@router.get("/traces/{trace_id}")
async def web_trace(trace_id: str, request: Request) -> dict:
    recorder = getattr(request.app.state, "recorder", None)
    if recorder is None or not recorder.enabled:
        raise HTTPException(status_code=404, detail="Tracing 未启用")
    tree = recorder.build_tree(trace_id)
    if not tree["spans"]:
        raise HTTPException(status_code=404, detail=f"Trace 不存在: {trace_id}")
    return tree


# ---------------------------------------------------------------------------
# 会话历史
# ---------------------------------------------------------------------------
@router.get("/sessions")
async def web_sessions(request: Request) -> dict:
    runtime = _get_runtime(request)
    if runtime.session_repo is None:
        return {"sessions": []}
    # 复用 SQLite 查询：列出会话（按 updated_at 排序）
    try:
        conn = runtime.session_repo._conn
        rows = conn.execute(
            "SELECT session_id, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()
        sessions = [dict(r) for r in rows]
    except Exception:
        sessions = []
    return {"sessions": sessions}


@router.get("/sessions/{session_id}/messages")
async def web_session_messages(session_id: str, request: Request) -> dict:
    runtime = _get_runtime(request)
    if runtime.session_repo is None:
        raise HTTPException(status_code=404, detail="会话存储未启用")
    messages = runtime.session_repo.list_messages(session_id)
    return {"session_id": session_id, "messages": messages}


# ---------------------------------------------------------------------------
# 会话删除（含关联消息与检查点）
# ---------------------------------------------------------------------------
@router.delete("/sessions/{session_id}")
async def web_session_delete(session_id: str, request: Request) -> dict:
    runtime = _get_runtime(request)
    if runtime.session_repo is None:
        raise HTTPException(status_code=404, detail="会话存储未启用")
    # session 与 checkpoint 同库：统一用 session_repo 连接删除（避免多连接 WAL 锁）
    try:
        conn = runtime.session_repo._conn
        conn.execute("DELETE FROM checkpoints WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除失败: {exc}")
    return {"deleted": session_id}


# ---------------------------------------------------------------------------
# 文件上传（写入沙箱目录，供 file_read 工具使用）
# ---------------------------------------------------------------------------
@router.post("/upload")
async def web_upload(request: Request) -> dict:
    from pathlib import Path

    runtime = _get_runtime(request)
    settings = request.app.state.settings
    sandbox = Path(settings.sandbox_dir).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    form = await request.form()
    file = form.get("file")
    if file is None or not getattr(file, "filename", None):
        raise HTTPException(status_code=400, detail="缺少文件")

    filename = getattr(file, "filename", "upload.txt")
    # 安全：只保留文件名（防路径穿越）
    safe_name = Path(filename).name
    target = (sandbox / safe_name).resolve()
    if not str(target).startswith(str(sandbox)):
        raise HTTPException(status_code=400, detail="非法文件名")

    try:
        content = await file.read()
        # 限 1MB
        if len(content) > 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件超过 1MB 限制")
        target.write_bytes(content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存失败: {exc}")

    return {
        "filename": safe_name,
        "path": str(target),
        "size": len(content),
        "hint": f"已上传到沙箱，可用 file_read 读取 {safe_name}",
    }


# ---------------------------------------------------------------------------
# 沙箱文件列表（前端"已上传文件"面板）
# ---------------------------------------------------------------------------
@router.get("/files")
async def web_files(request: Request) -> dict:
    from pathlib import Path

    settings = request.app.state.settings
    sandbox = Path(settings.sandbox_dir).resolve()
    files = []
    if sandbox.exists():
        for p in sorted(sandbox.iterdir()):
            if p.is_file():
                files.append({"name": p.name, "size": p.stat().st_size})
    return {"files": files, "sandbox": str(sandbox)}
