"""编排深度上下文（多级编排 / 递归防御）。

为什么需要深度追踪？
    多级编排 = 子 agent 也能调用 delegate 再往下委派。若不加限制，理论上可以
    无限嵌套（agent 委派 agent 委派 agent …）。深度 ContextVar 解决两件事：

    1. 可见性控制：只有 depth < orchestrator_max_depth 的子 agent 才能看到
       delegate 工具（叶子层物理移除，模型根本不会产生嵌套调用）；
    2. 兜底防御：delegate handler 与 runner.run 入口都检查深度上限，
       即使绕过工具可见性直接调用也会被拒绝。

深度语义（depth 表示"当前执行处于编排的第几层"）：
    - 主 agent 调用 delegate 工具      → runner.run 进入 depth=1，子 agent 执行时读到 1
    - 子 agent 再调用 delegate         → 嵌套 runner.run 进入 depth=2，孙 agent 读到 2
    - max_depth=2 时：depth=1 的子 agent 可再委派；depth=2 的孙 agent 是叶子

ContextVar 是任务局部（task-local）的：并行子 agent 各自嵌套委派时深度互不干扰，
asyncio 任务创建时会复制当前上下文，与 trace_span 的 ContextVar 机制一致。
"""
from contextvars import ContextVar

# 当前编排深度（0 = 尚未进入任何编排，即主 agent 侧）
orchestration_depth: ContextVar[int] = ContextVar("orchestration_depth", default=0)
