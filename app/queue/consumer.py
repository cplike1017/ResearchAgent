"""
队列消费者（Consumer 侧）：取出 Job -> 执行 Agent -> 结果 / 重试。

重试（Retry）语义：
    max_attempts 默认 3。每次失败 attempt += 1 并重新入队；
    达到上限后标记 FAILED —— 避免无限重试。

注意：重试会导致工具可能重复执行（例如工具执行成功后、回写结果前进程崩溃），
这是分布式系统的经典问题，需要结合幂等工具或去重机制解决（教学点）。
"""
from typing import Awaitable, Callable

from app.errors import error_to_dict
from app.queue.models import Job, JobStatus
from app.queue.producer import RedisJobQueue
from app.tools.schemas import UserContext
from app.tracing.context import current_span_id, current_trace_id, set_trace_context
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span

# 运行时工厂：() -> AgentRuntime（由 Worker 提供，负责组装 Session/Checkpoint 等依赖）
RuntimeFactory = Callable[[], Awaitable]


async def process_job(
    queue: RedisJobQueue,
    runtime_factory: Callable[[], "object"],
    job: Job,
    recorder: TraceRecorder | None = None,
) -> Job:
    """
    消费一个 Job 的完整流程：
        QUEUED -> RUNNING -> SUCCEEDED
                        \\-> (重试) QUEUED (attempt+1)
                        \\-> FAILED (达到 max_attempts)

    Trace 传播：从 job.trace_context 恢复 trace_id / parent_span_id，
    让 worker.process 及其子 Span 与 Gateway 属于同一条 Trace。
    """
    await queue.update_status(job.job_id, JobStatus.RUNNING)
    recorder = recorder or getattr(queue, "recorder", None)

    # ---- 跨进程 Trace 传播：保存旧上下文，处理完恢复 ----
    old_trace, old_span = current_trace_id.get(), current_span_id.get()
    ctx = job.trace_context or {}
    set_trace_context(ctx.get("trace_id"), ctx.get("parent_span_id"))
    try:
        message = (job.input or {}).get("message", "")
        # 从 Job 恢复调用方身份（Stage 5 权限校验用）
        user = UserContext(**job.user) if job.user else None
        runtime = runtime_factory()

        traced = recorder is not None and recorder.enabled
        if traced:
            async with trace_span(
                "worker.process",
                "worker",
                input={"job_id": job.job_id, "session_id": job.session_id},
                attributes={"job_id": job.job_id, "attempt": job.attempt},
                recorder=recorder,
            ) as span:
                result = await runtime.run(message, session_id=job.session_id, user=user)
                span.output = {
                    "session_id": result.session_id,
                    "steps": result.steps,
                    "tool_calls": len(result.tool_calls),
                }
        else:
            result = await runtime.run(message, session_id=job.session_id, user=user)

        payload = {
            "answer": result.answer,
            "session_id": result.session_id,
            "trace_id": result.trace_id,
        }
        return await queue.update_status(job.job_id, JobStatus.SUCCEEDED, result=payload)
    except Exception as exc:  # 任何异常都走重试 / 失败路径
        error = error_to_dict(exc)
        fresh = await queue.get_job(job.job_id)
        if fresh is None:
            raise
        if fresh.attempt + 1 < queue.max_attempts:
            # 还有重试额度：attempt + 1，重新入队
            fresh.attempt += 1
            fresh.status = JobStatus.QUEUED
            fresh.error = error
            await queue.save_job(fresh)
            await queue.requeue(fresh.job_id)
            return fresh
        # 用尽重试次数：最终失败
        return await queue.update_status(fresh.job_id, JobStatus.FAILED, error=error)
    finally:
        # 恢复调用前的 Trace 上下文，避免污染下一个 Job
        current_trace_id.set(old_trace)
        current_span_id.set(old_span)
