"""
Stage 9 Demo：规划层（Plan-and-Execute + 反思）+ 真实工具。

运行：python -m demos.stage9_demo

展示：
    1. Planner 把多城市天气任务分解为步骤
    2. PlanExecutor 逐步执行（每步一个小 ReAct 循环）
    3. Reflector 检查结果，失败时重规划（最多 N 次）
    4. 真实工具：web_search（Tavily）/ http_get / get_time
    5. 对比：react 模式 vs plan 模式

依赖：
    - LLM：.env 配好 LLM_BASE_URL / LLM_API_KEY（真实模型体验最佳；无则用 Stub）
    - 真实工具：web_search 需 TAVILY_API_KEY（可选）
"""
import asyncio

from app.agent.runtime import AgentRuntime
from app.agent.state import AgentState
from app.config import get_settings
from app.llm.client import create_llm_client
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry
from app.tracing.recorder import TraceRecorder

SEPARATOR = "=" * 64


def print_plan(plan) -> None:
    for s in plan:
        hint = f" [tools: {', '.join(s.tools_hint)}]" if s.tools_hint else ""
        print(f"  [{s.status}] {s.description}{hint}")
        if s.result:
            print(f"      -> {s.result[:80]}")


async def main() -> None:
    settings = get_settings()
    recorder = TraceRecorder(
        settings.trace_file, enabled=True, capture_content=settings.trace_capture_content
    )

    print(SEPARATOR)
    print("Stage 9 规划层 Demo")
    print(SEPARATOR)
    print(f"LLM      : {settings.llm_provider_resolved} / {settings.llm_model}")
    print(f"模式     : {settings.agent_mode} | 规划策略: {settings.planner_strategy}")
    print(f"工具     : {[t.name for t in build_default_registry().all()]}")
    print()

    llm = create_llm_client(settings)
    session_repo = SQLiteSessionRepository(settings.database_url)
    runtime = AgentRuntime(
        llm=llm,
        registry=build_default_registry(),
        session_repo=session_repo,
        recorder=recorder,
        settings=settings,
    )

    # ---- 1) 多城市天气（规划拆步）----
    print(SEPARATOR)
    print("1) 多城市天气任务（planner 拆成多步）")
    print(SEPARATOR)
    r1 = await runtime.run("查询北京和上海天气", session_id="stage9_demo")
    print(f"回答: {r1.answer}")
    print(f"计划（{len(r1.plan)} 步，重规划 {r1.plan_revisions} 次）：")
    print_plan(r1.plan)

    # ---- 2) 真实工具演示（web_search / get_time）----
    print("\n" + SEPARATOR)
    print("2) 真实工具：get_time + web_search（Tavily）")
    print(SEPARATOR)

    from app.tools.builtin.web import get_time_handler, web_search_handler

    print(f"get_time -> {get_time_handler()}")
    if settings.tavily_api_key:
        try:
            result = web_search_handler("2024 巴黎奥运会 金牌榜", max_results=2)
            print("web_search ->")
            print(result)
        except Exception as exc:
            print(f"web_search 失败: {exc}")
    else:
        print("web_search 跳过（未配置 TAVILY_API_KEY）")

    print("\nDemo 完成。")
    print("\n提示：将 .env 的 AGENT_MODE=react 切回 plan 可对比两种模式行为差异。")


if __name__ == "__main__":
    asyncio.run(main())
