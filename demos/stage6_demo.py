"""
Stage 6 Demo：Tracing + Evaluation。

运行：python -m demos.stage6_demo

展示：
    1. 一次 Agent 请求的完整 Trace（gateway -> redis -> worker -> agent -> tool）
    2. Trace 树（按 parent_span_id 重建），含 duration / status
    3. 故意制造 Tool Timeout -> Trace 中出现 ERROR + ToolTimeoutError
    4. 演示 Trace -> Checkpoint 的失败复现链路

完整的评测与回归报告请运行：
    python -m evals.runner
    python -m evals.runner --compare evals/runs/<baseline>.json
"""
import asyncio
import time

from app.agent.react_loop import LoopHooks
from app.agent.runtime import AgentRuntime
from app.checkpoint.repository import SQLiteCheckpointRepository
from app.config import get_settings
from app.llm.client import create_llm_client
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolDefinition
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span
from pydantic import BaseModel, Field

SEPARATOR = "=" * 64


def print_tree(nodes: list[dict], prefix: str = "") -> None:
    """把 Trace 树打印为缩进文本。"""
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        connector = "└── " if is_last else "├── "
        status_mark = "ERROR" if node["status"] == "ERROR" else f"{node['duration_ms']}ms"
        print(f"{prefix}{connector}{node['name']}  [{node['span_type']}] {status_mark}")
        if node.get("error"):
            print(f"{prefix}{'    ' if is_last else '│   '}    └── error: {node['error']['type']}: {node['error']['message'][:60]}")
        child_prefix = prefix + ("    " if is_last else "│   ")
        print_tree(node["children"], child_prefix)


class SlowArgs(BaseModel):
    delay: float = Field(description="延迟秒数")


async def main() -> None:
    settings = get_settings()
    recorder = TraceRecorder(
        settings.trace_file, enabled=True, capture_content=settings.trace_capture_content
    )
    registry = build_default_registry()
    gateway = ToolGateway(registry, settings=settings, recorder=recorder)

    # 附加一个慢工具用于超时演示
    registry.register(
        ToolDefinition(
            name="slow_tool",
            description="慢工具",
            input_model=SlowArgs,
            handler=lambda delay: (time.sleep(delay) or "slow done"),
            timeout_seconds=0.2,
        )
    )

    llm = create_llm_client(settings)
    runtime = AgentRuntime(
        llm=llm,
        registry=registry,
        tool_gateway=gateway,
        session_repo=SQLiteSessionRepository(settings.database_url),
        checkpoint_repo=SQLiteCheckpointRepository(settings.database_url),
        recorder=recorder,
        settings=settings,
    )

    # =================================================================
    print(SEPARATOR)
    print("1) 一次 Agent 请求的完整 Trace")
    print(SEPARATOR)
    async with trace_span("gateway.request", "gateway", input={"message": "查询北京天气"}, recorder=recorder) as root:
        result = await runtime.run("查询北京天气", session_id="stage6_demo_session")
        trace_id = root.trace_id
    print(f"trace_id: {trace_id}")
    print(f"answer  : {result.answer}")
    print(f"steps   : {result.steps} | checkpoint_id: {result.checkpoint_id}")

    tree = recorder.build_tree(trace_id)
    print("\nTrace 调用树：")
    print_tree(tree["spans"])

    # =================================================================
    print("\n" + SEPARATOR)
    print("2) 故意制造 Tool Timeout -> Trace 中看到 ERROR + ToolTimeoutError")
    print(SEPARATOR)
    async with trace_span("gateway.request", "gateway", input={"message": "slow"}, recorder=recorder) as root2:
        env = await gateway.execute("slow_tool", {"delay": 5.0})
        trace2 = root2.trace_id
    print(f"trace_id: {trace2}")
    print(f"tool 结果: success={env.success} error={env.error}")
    tree2 = recorder.build_tree(trace2)
    print("\nTrace 调用树（注意 tool.execute 的 ERROR 状态）：")
    print_tree(tree2["spans"])

    # =================================================================
    print("\n" + SEPARATOR)
    print("3) Eval Failure -> Trace -> Checkpoint 复现链路")
    print(SEPARATOR)
    print("每个 Eval CaseResult 保存 case_id / trace_id / checkpoint_id，")
    print("失败时可从 trace 定位、从 checkpoint 恢复状态复现。运行：")
    print("    python -m evals.runner")
    print("    python -m evals.runner --compare evals/runs/<baseline>.json")
    print("\nDemo 完成。")


if __name__ == "__main__":
    asyncio.run(main())
