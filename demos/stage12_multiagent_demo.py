"""
Stage 12 Demo：多 Agent 编排（Manager/Worker 模式）。

运行：python -m demos.stage12_multiagent_demo

展示：
    1. OrchestratorPlanner 自动分工（LLM 拆解任务 → researcher/analyst/writer）
    2. SubAgentExecutor 独立执行（各自人设 + 工具白名单隔离）
    3. 依赖图推进 + 并行执行
    4. 主管 LLM 合成最终回答
    5. delegate 工具：主 agent 在对话中直接委派

配置真实 LLM_BASE_URL / LLM_API_KEY 后效果最佳；无 Key 时降级 stub 单步。
"""
import asyncio
import os

from app.config import get_settings
from app.llm.client import create_llm_client
from app.orchestrator.runner import OrchestratorRunner
from app.tools.builtin import build_default_registry
from app.tracing.recorder import TraceRecorder

SEPARATOR = "=" * 64


def _print_trace_tree(tree: dict, indent: int = 0) -> None:
    """打印 Trace 树（orchestrator.run → agent.run → llm_call/tool）。"""
    for node in tree.get("spans", []):
        status = node.get("status", "?")
        mark = "✓" if status == "OK" else ("✗" if status == "ERROR" else "?")
        print(
            f"{'  ' * indent}{mark} {node['name']} "
            f"({node.get('duration_ms', 0):.0f}ms)"
        )
        _print_trace_tree({"spans": node.get("children", [])}, indent + 1)


async def main() -> None:
    settings = get_settings()
    llm = create_llm_client(settings)
    registry = build_default_registry()
    recorder = TraceRecorder(
        settings.trace_file,
        enabled=settings.trace_enabled,
        capture_content=settings.trace_capture_content,
    )

    print(SEPARATOR)
    print("Stage 12 多 Agent 编排 Demo")
    print(SEPARATOR)

    runner = OrchestratorRunner(
        llm=llm,
        registry=registry,
        settings=settings,
        recorder=recorder,
    )

    # ------------------------------------------------------------------
    # 1) 自动分工：研究课题 → researcher 检索 / analyst 分析 / writer 成稿
    # ------------------------------------------------------------------
    task = (
        "调研 2024 年大语言模型的技术进展，并给出对 RAG 应用开发的 3 条建议。"
        "先联网检索最新进展，再分析趋势，最后整理成报告。"
    )
    print("\n1) 自动分工编排")
    print(SEPARATOR)
    print(f"任务: {task}\n")

    result = await runner.run(task)

    print(f"分工理由: {result.plan.rationale}\n")
    for i, step in enumerate(result.plan.steps):
        print(f"  步骤{i}: [{step.agent}] {step.task}")
        if step.depends_on:
            print(f"         依赖步骤: {step.depends_on}")

    print("\n各子 Agent 结果:")
    for r in result.agent_results:
        print(f"\n  [{r.agent}] status={r.status} steps={r.steps} ({r.duration_ms:.0f}ms)")
        print(f"  tools={[t['name'] for t in r.tool_calls]}")
        print(f"  摘要: {(r.answer or '')[:200]}")

    print(f"\n最终合成回答（前 600 字）:\n{result.final_answer[:600]}")
    print(f"\n整体状态: {result.status}  耗时: {result.duration_ms:.0f}ms")

    # ------------------------------------------------------------------
    # 2) delegate 工具：主 agent 直接委派（Trace 树可视化）
    # ------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("2) delegate 工具：主 agent 直接委派")
    print(SEPARATOR)

    if result.trace_id:
        tree = recorder.build_tree(result.trace_id)
        print(f"\nTrace 树 (trace_id={result.trace_id}):")
        _print_trace_tree(tree)

    # 显式指定子 agent（跳过自动分工）
    print("\n3) 显式指定子 Agent 名单")
    print(SEPARATOR)
    explicit = await runner.run("计算 15*4 和 23+17 的结果，然后整理成一行结论。", agents=["analyst", "writer"])
    print(f"最终回答: {explicit.final_answer[:300]}")

    # ------------------------------------------------------------------
    # 4) Web 运行时集成（delegate 工具注册）
    # ------------------------------------------------------------------
    print("\n" + SEPARATOR)
    print("4) delegate 工具已注入 AgentRuntime（Web UI 可用）")
    print(SEPARATOR)
    from app.agent.runtime import AgentRuntime

    web_runtime = AgentRuntime(
        llm=llm,
        registry=registry,
        settings=settings,
        orchestrator=runner,
    )
    tool = web_runtime.registry.get("delegate")
    print(f"已注册工具: {tool.name} — {tool.description[:80]}...")
    print(f"registry 工具总数: {len(web_runtime.registry.all())}")

    print("\nDemo 完成。可在 Web UI (http://localhost:8000) 的聊天中直接让主 agent 调用 delegate 工具。")


if __name__ == "__main__":
    asyncio.run(main())
