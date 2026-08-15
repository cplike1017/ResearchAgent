"""
trace_span：以 with 语法创建 Span 的上下文管理器。

用法：
    with trace_span("llm_call", "llm") as span:
        result = await llm.chat(...)
        span.output = result

必须正确处理四件事（面试点）：
    1. 正常完成  —— 退出时 end_span(OK)
    2. 异常      —— 捕获后 end_span(ERROR) 并重新抛出
    3. finally   —— 无论成功失败都恢复 ContextVar（token/reset）
    4. Context 恢复 —— 用 token 把 current_trace_id / current_span_id 还原

@contextmanager 原理（文档 docs/stage6.md 有详细解释）：
    with 语句会调用 __enter__ / __exit__；
    @contextmanager 把"yield 之前的代码"变成 __enter__，
    把"yield 之后的代码"变成 __exit__ 的组成部分 —— 所以
    try/yield/finally 的顺序决定了异常怎么被捕获、资源怎么被清理。
"""
import contextlib
import time
from typing import Any, AsyncIterator, Iterator

from app.tracing.context import current_span_id, current_trace_id
from app.tracing.recorder import TraceRecorder, get_default_recorder
from app.tracing.models import Span


@contextlib.asynccontextmanager
async def trace_span(
    name: str,
    span_type: str,
    *,
    input: Any = None,
    attributes: dict | None = None,
    recorder: TraceRecorder | None = None,
) -> AsyncIterator[Span]:
    """异步版本的 Span 上下文管理器（Agent / Gateway / Worker 等异步代码使用）。"""
    recorder = recorder or get_default_recorder()
    span = recorder.start_span(name, span_type, input=input, attributes=attributes)
    # 用 token 记住旧值，finally 中 reset 恢复 —— 保证嵌套与并发安全
    token_trace = current_trace_id.set(span.trace_id)
    token_span = current_span_id.set(span.span_id)
    try:
        yield span
        recorder.end_span(span, output=getattr(span, "output", None))
    except Exception as exc:
        recorder.end_span(span, error=exc)
        raise
    finally:
        current_trace_id.reset(token_trace)
        current_span_id.reset(token_span)


@contextlib.contextmanager
def trace_span_sync(
    name: str,
    span_type: str,
    *,
    input: Any = None,
    attributes: dict | None = None,
    recorder: TraceRecorder | None = None,
) -> Iterator[Span]:
    """同步版本的 Span 上下文管理器（同步代码使用；机制与异步版一致）。"""
    recorder = recorder or get_default_recorder()
    span = recorder.start_span(name, span_type, input=input, attributes=attributes)
    token_trace = current_trace_id.set(span.trace_id)
    token_span = current_span_id.set(span.span_id)
    try:
        yield span
        recorder.end_span(span, output=getattr(span, "output", None))
    except Exception as exc:
        recorder.end_span(span, error=exc)
        raise
    finally:
        current_trace_id.reset(token_trace)
        current_span_id.reset(token_span)


def time_ms() -> float:
    """当前毫秒时间戳（span 计时的辅助函数）。"""
    return time.perf_counter() * 1000
