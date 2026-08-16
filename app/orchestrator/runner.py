"""编排运行器（OrchestratorRunner，Stage 12 核心）：主管（Manager）驱动多个子 agent 协作。

执行模型：
    1. 规划：OrchestratorPlanner 把任务拆成 SubTask 列表（agent + task + depends_on）
    2. 执行：按依赖图推进 —— 每一轮取"依赖已全部完成"的步骤，
       在同轮内并行执行（信号量限流），完成后把结果写入上下文
    3. 合成：所有步骤完成后，由"主管 LLM"把各子 agent 结果整合为最终回答
       （单步任务直接返回该子 agent 的答案，省一次 LLM 调用）

失败语义（编排不因单个子 agent 失败而崩溃）：
    - 子 agent 内部异常 → AgentRunResult.status = FAILED，其余步骤照常
    - 某步骤失败时，依赖它的步骤照常执行（context 中标注来源失败）
    - 全部失败 → status = FAILED；部分失败 → status = PARTIAL

追踪：整个编排包在 orchestrator.run span 下，子 agent 的 agent.run span 自动嵌套，
Web UI 的 Trace 树可完整展示"主管 → 员工 → 工具调用"。
"""
import asyncio
import time

from app.config import Settings, get_settings
from app.llm.client import BaseLLMClient
from app.orchestrator.context import orchestration_depth
from app.orchestrator.executor import SubAgentExecutor
from app.orchestrator.models import AgentRunResult, OrchestrationPlan, OrchestrationResult
from app.orchestrator.planner import OrchestratorPlanner
from app.orchestrator.profiles import AgentProfile, BUILTIN_PROFILES, get_profile
from app.tools.registry import ToolRegistry
from app.tracing.context import current_trace_id
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span

_SYNTHESIS_PROMPT = """你是多 Agent 编排的主管。下面是各子 Agent 针对同一任务的执行结果，请整合为一份最终回答。

用户任务：
{task}

各子 Agent 结果：
{results}

要求：
1. 直接给出最终回答，不要复述过程；
2. 融合各子 Agent 的结论，矛盾处给出你的判断；
3. 失败的子 Agent 结果跳过，不要提及内部错误细节。"""


class OrchestratorRunner:
    """多 Agent 编排运行器。"""

    def __init__(
        self,
        *,
        llm: BaseLLMClient,
        registry: ToolRegistry,
        settings: Settings | None = None,
        recorder: TraceRecorder | None = None,
        profiles: list[AgentProfile] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.registry = registry
        self.recorder = recorder
        self.profiles = profiles or BUILTIN_PROFILES
        self.planner = OrchestratorPlanner(self.settings, llm=llm)
        self.executor = SubAgentExecutor(
            llm=llm,
            master_registry=registry,
            settings=self.settings,
            recorder=recorder,
        )

    # ------------------------------------------------------------------
    async def run(
        self,
        task: str,
        agents: list[str] | None = None,
        context: str = "",
        max_parallel: int | None = None,
    ) -> OrchestrationResult:
        """执行一次多 Agent 编排（支持嵌套：子 agent 也可再委派）。

        :param task:   用户任务
        :param agents: 显式指定子 agent 名单（None = 让 planner 自动分工）
        :param context: 主 agent 提供的背景信息（拼进子 agent 的任务）
        :param max_parallel: 并行上限（默认取配置 orchestrator_max_parallel）

        多级编排：进入时 depth+1、退出时恢复。深度由 ContextVar 追踪，
        并行子 agent 的嵌套委派互不干扰；超限由 delegate 工具可见性
        （叶子层无 delegate）与 handler 兜底双重防御。
        """
        token = orchestration_depth.set(orchestration_depth.get() + 1)
        try:
            return await self._run(task, agents, context, max_parallel)
        finally:
            orchestration_depth.reset(token)

    async def _run(
        self,
        task: str,
        agents: list[str] | None,
        context: str,
        max_parallel: int | None,
    ) -> OrchestrationResult:
        start = time.perf_counter()
        plan = await self.planner.plan(task, agents=agents)

        if self.recorder is None or not self.recorder.enabled:
            return await self._run_impl(task, plan, context, max_parallel, start)

        async with trace_span(
            "orchestrator.run",
            "orchestrator",
            input={"task": task, "agents": agents, "steps": len(plan.steps), "depth": orchestration_depth.get()},
            attributes={"agents": agents, "steps": len(plan.steps), "depth": orchestration_depth.get()},
            recorder=self.recorder,
        ) as span:
            result = await self._run_impl(task, plan, context, max_parallel, start)
            span.output = {
                "final_answer": result.final_answer,
                "status": result.status,
                "steps": len(result.agent_results),
                "succeeded": sum(1 for r in result.agent_results if r.status == "SUCCEEDED"),
            }
            result.trace_id = current_trace_id.get()
            return result

    # ------------------------------------------------------------------
    async def _run_impl(
        self,
        task: str,
        plan: OrchestrationPlan,
        context: str,
        max_parallel: int | None,
        start: float,
    ) -> OrchestrationResult:
        result = OrchestrationResult(task=task, plan=plan)
        if not plan.steps:
            result.status = "FAILED"
            result.final_answer = "编排计划为空，无法执行。"
            result.duration_ms = round((time.perf_counter() - start) * 1000, 3)
            return result

        limit = max_parallel or self.settings.orchestrator_max_parallel
        semaphore = asyncio.Semaphore(max(1, limit))

        results: dict[int, AgentRunResult] = {}
        done: set[int] = set()

        # 按依赖图逐轮推进：每轮取依赖全部完成的步骤，并行执行
        while len(done) < len(plan.steps):
            ready = [
                i for i, step in enumerate(plan.steps)
                if i not in done and all(d in done for d in step.depends_on)
            ]
            if not ready:
                # 依赖环 / 死锁防御：剩余步骤标记 SKIPPED
                for i in range(len(plan.steps)):
                    if i not in done:
                        results[i] = AgentRunResult(
                            agent=plan.steps[i].agent,
                            task=plan.steps[i].task,
                            status="SKIPPED",
                            error="依赖未完成（依赖环或前置失败）",
                        )
                break

            async def _run_step(i: int) -> None:
                async with semaphore:
                    step = plan.steps[i]
                    profile = get_profile(step.agent)
                    dep_context = _build_context(results, step.depends_on)
                    full_context = "\n".join(filter(None, [context, dep_context]))
                    results[i] = await self.executor.execute(profile, step.task, full_context)
                    done.add(i)

            await asyncio.gather(*(_run_step(i) for i in ready))

        # 组装结果列表（保持计划顺序）
        result.agent_results = [results[i] for i in range(len(plan.steps))]

        # 合成最终回答
        result.final_answer = await self._synthesize(task, result.agent_results)
        succeeded = sum(1 for r in result.agent_results if r.status == "SUCCEEDED")
        if succeeded == 0:
            result.status = "FAILED"
        elif succeeded < len(result.agent_results):
            result.status = "PARTIAL"
        else:
            result.status = "SUCCEEDED"
        result.duration_ms = round((time.perf_counter() - start) * 1000, 3)
        return result

    # ------------------------------------------------------------------
    async def _synthesize(self, task: str, results: list[AgentRunResult]) -> str:
        """合成最终回答：单步直接返回；多步由主管 LLM 整合；LLM 不可用则拼接。"""
        if len(results) == 1:
            r = results[0]
            if r.status == "SUCCEEDED":
                return r.answer
            return f"（子 Agent {r.agent} 执行失败：{r.error}）"

        ok = [r for r in results if r.status == "SUCCEEDED"]
        if not ok:
            return "；".join(f"{r.agent} 失败：{r.error}" for r in results)

        if self.llm is None:
            return _join_answers(results)

        results_block = "\n\n".join(
            f"【{r.agent}】\n{r.answer or '(无输出)'}" + (f"\n（失败：{r.error}）" if r.status != "SUCCEEDED" else "")
            for r in results
        )
        try:
            resp = await self.llm.chat(
                [
                    {"role": "system", "content": "你只输出最终回答本身。"},
                    {"role": "user", "content": _SYNTHESIS_PROMPT.format(task=task, results=results_block)},
                ]
            )
            if resp.content and resp.content.strip():
                return resp.content.strip()
        except Exception:
            pass
        return _join_answers(results)


def _build_context(results: dict[int, AgentRunResult], deps: list[int]) -> str:
    """把依赖步骤的结果拼成上下文（失败/跳过步骤标注来源状态）。"""
    blocks: list[str] = []
    for d in deps:
        r = results.get(d)
        if r is None:
            continue
        if r.status == "SUCCEEDED":
            blocks.append(f"步骤{d}（{r.agent}）：{r.answer}")
        else:
            blocks.append(f"步骤{d}（{r.agent}）：执行{r.status}（{r.error}）")
    return "\n".join(blocks)


def _join_answers(results: list[AgentRunResult]) -> str:
    """无 LLM 时的兜底拼接。"""
    parts = []
    for r in results:
        if r.status == "SUCCEEDED":
            parts.append(f"【{r.agent}】\n{r.answer}")
        else:
            parts.append(f"【{r.agent}】执行{r.status}：{r.error}")
    return "\n\n".join(parts)
