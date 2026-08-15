"""
Trace 上下文传播（contextvars）。

为什么用 ContextVar 而不是手动传参？
    - 全链路（Gateway -> Redis -> Worker -> Agent -> Tool）都要携带 trace_id，
      手动传参要改动每一个函数签名，侵入性极强；
    - ContextVar 让"当前 Trace 上下文"成为隐式全局状态：
      任何深层的函数只要读取 current_trace_id / current_span_id 就能加入同一链路；
    - asyncio 会自动把 ContextVar 随 await / 任务复制传播，天然适配异步代码。

跨进程传播（Redis）：
    进程内靠 ContextVar，进程间靠显式字段：
        Job.trace_context = {"trace_id": ..., "parent_span_id": ...}
    Worker 消费时调用 set_trace_context() 恢复，链路就接上了。
"""
from contextvars import ContextVar

# 当前 Trace ID（None = 尚未进入任何 Trace）
current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)
# 当前 Span ID（子 Span 的 parent_span_id 取自它）
current_span_id: ContextVar[str | None] = ContextVar("current_span_id", default=None)


def get_trace_context() -> dict:
    """读取当前 Trace 上下文（用于写入 Job 进行跨进程传播）。"""
    return {
        "trace_id": current_trace_id.get(),
        "parent_span_id": current_span_id.get(),
    }


def set_trace_context(trace_id: str | None, parent_span_id: str | None = None) -> None:
    """恢复 Trace 上下文（Worker 消费 Job 后调用）。"""
    if trace_id is None:
        return
    current_trace_id.set(trace_id)
    if parent_span_id:
        current_span_id.set(parent_span_id)
