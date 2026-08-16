"""编排规划器（OrchestratorPlanner）：把用户任务拆解为"哪个子 agent 干什么"。

与 Stage 9 Planner 的区别：
    Stage 9 Planner  拆的是"工具步骤"（单 agent 内，steps 有 tools_hint）
    本 Planner       拆的是"角色步骤"（多 agent 间，steps 有 agent + depends_on）

策略（与项目三策略同构）：
    llm  —— 让 LLM 决定分工：给定任务 + 可用档案 → JSON 计划
            {"rationale": "...", "steps": [{"agent": "researcher", "task": "...", "depends_on": []}]}
    stub —— 确定性兜底：单步全交给 generalist（离线 / LLM 不可用时）

LLM 输出防御：
    1. 从回复中提取 JSON（容忍 ```json 代码块包裹 / 前后废话）；
    2. 校验字段，未知 agent 名回退 generalist；
    3. 步骤数上限 max_steps，依赖下标越界自动剔除；
    4. 任何解析失败 → 降级 stub 单步，绝不抛异常让编排中断。
"""
import json
import re
from typing import Any

from app.config import Settings, get_settings
from app.llm.client import BaseLLMClient
from app.orchestrator.models import OrchestrationPlan, SubTask

_PLAN_PROMPT = """你是一名多 Agent 编排主管（Orchestrator）。请把用户任务拆解为多个子 Agent 的协作计划。

可用子 Agent 档案：
{profiles}

要求：
1. 输出严格 JSON（不要 Markdown 代码块、不要多余文字），格式：
   {{"rationale": "分工理由", "steps": [{{"agent": "档案名", "task": "该 agent 的具体任务", "depends_on": []}}]}}
2. depends_on 填写依赖步骤的下标（0 起）；无依赖填 []；依赖关系要正确（如分析师依赖研究员的资料）。
3. 最多 {max_steps} 步；能用并行解决的不要串行。
4. task 要自包含：子 agent 看不到其他步骤的完整对话，只能看到依赖步骤的结果。

用户任务：
{task}"""


class OrchestratorPlanner:
    """多 Agent 编排规划器。"""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLMClient | None = None,
        profile_registry: "ProfileRegistry | None" = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.strategy = self.settings.orchestrator_planner_strategy
        # 档案注册表：planner 用它的 names 做校验、用它的描述喂给 LLM
        # （动态注册的档案会立即参与分工）
        from app.orchestrator.registry import ProfileRegistry

        self.profile_registry = profile_registry or ProfileRegistry(self.settings)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    async def plan(
        self,
        task: str,
        agents: list[str] | None = None,
    ) -> OrchestrationPlan:
        """生成编排计划。

        :param agents: 调用方显式指定的子 agent 名单；非空时跳过 LLM 分工，
            直接按名单生成单步计划（每个指定 agent 一步，任务原样交给它）。
        """
        names = self.profile_registry.names()
        if agents:
            return OrchestrationPlan(
                rationale="调用方显式指定子 Agent",
                steps=[
                    SubTask(
                        agent=(name if name in names else "generalist"),
                        task=task,
                        depends_on=[],
                    )
                    for name in agents
                ],
            )
        if self.strategy == "llm" and self.llm is not None:
            plan = await self._llm_plan(task)
            if plan is not None:
                return plan
        return self._stub_plan(task)

    # ------------------------------------------------------------------
    # llm：模型分工
    # ------------------------------------------------------------------
    async def _llm_plan(self, task: str) -> OrchestrationPlan | None:
        profiles_block = "\n".join(
            f"- {p.name}: {p.description}" for p in self.profile_registry.all()
        )
        prompt = _PLAN_PROMPT.format(
            profiles=profiles_block,
            max_steps=self.settings.orchestrator_max_steps,
            task=task,
        )
        try:
            resp = await self.llm.chat(
                [
                    {"role": "system", "content": "你只输出 JSON。"},
                    {"role": "user", "content": prompt},
                ]
            )
        except Exception:
            return None
        return self._parse_plan(resp.content or "")

    def _parse_plan(self, raw: str) -> OrchestrationPlan | None:
        """解析 LLM 返回的计划 JSON（容忍代码块 / 前后文字）。"""
        data = _extract_json(raw)
        if not data:
            return None
        steps = data.get("steps")
        if not isinstance(steps, list) or not steps:
            return None
        parsed: list[SubTask] = []
        for i, s in enumerate(steps[: self.settings.orchestrator_max_steps]):
            if not isinstance(s, dict):
                continue
            agent = str(s.get("agent", "generalist"))
            if agent not in self.profile_registry.names():
                agent = "generalist"
            task_text = str(s.get("task", "")).strip()
            if not task_text:
                continue
            deps = s.get("depends_on") or []
            if not isinstance(deps, list):
                deps = []
            deps = [d for d in deps if isinstance(d, int) and 0 <= d < i]  # 只能依赖已存在的步
            parsed.append(SubTask(agent=agent, task=task_text, depends_on=deps))
        if not parsed:
            return None
        return OrchestrationPlan(
            rationale=str(data.get("rationale", "")),
            steps=parsed,
        )

    # ------------------------------------------------------------------
    # stub：确定性兜底
    # ------------------------------------------------------------------
    def _stub_plan(self, task: str) -> OrchestrationPlan:
        return OrchestrationPlan(
            rationale="stub 规划器：单步全交给 generalist",
            steps=[SubTask(agent="generalist", task=task, depends_on=[])],
        )


def _extract_json(raw: str) -> dict | None:
    """从模型回复中提取 JSON 对象（容忍 ```json 包裹 / 前后解释文字）。"""
    if not raw:
        return None
    text = raw.strip()
    # 1) 去掉 ```json ... ``` 代码块
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    # 2) 找第一个 { 到最后一个 } 之间的内容
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        # 3) 容忍尾部逗号等小瑕疵：逐行修复后重试一次
        try:
            fixed = _fix_json(text[start : end + 1])
            obj = json.loads(fixed)
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _fix_json(text: str) -> str:
    """粗暴修复常见 JSON 错误（尾逗号 / 单引号）。仅供解析失败时兜底。"""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text
