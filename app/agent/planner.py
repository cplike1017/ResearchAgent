"""
Planner（规划器，Stage 9 核心）：把用户任务分解为有序步骤。

为什么需要规划？—— 复杂任务（"查北京和上海的天气并对比"）如果直接丢给
单层 ReAct，模型可能来回绕圈、漏步骤。先把任务拆成 PlanStep 列表，
再逐步执行，每步目标明确、可验证、可追踪。

策略（与项目三策略同构）：
    stub —— 规则分解：基于关键词启发式拆步（"和/与/并/同时"→多任务；
            含"天气/计算/搜索"等信号→对应工具步骤）。确定性，离线可用。
    llm  —— LLM 计划：让模型把任务拆成步骤（真实规划，需真实 LLM）。
    off  —— 不规划（等同 react 模式）。

输出 PlanStep 列表：每步含 description / tools_hint / order。
"""
import json
import re
from uuid import uuid4

from app.agent.models import PlanStep
from app.config import Settings
from app.llm.client import BaseLLMClient


class Planner:
    """任务规划器。"""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLMClient | None = None,
        tool_names: list[str] | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.llm = llm
        self.strategy = self.settings.planner_strategy
        # 可用工具名（LLM 计划时告知模型可用的工具）
        self.tool_names = tool_names or []

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    async def plan(self, task: str, memory_context: list[str] | None = None) -> list[PlanStep]:
        """
        把任务分解为计划步骤。

        :param memory_context: Stage 8 记忆层检索到的相关记忆（可空），
            供 LLM 规划参考（如"用户上次查询过北京天气"）。
        """
        if self.strategy == "off":
            return []
        if self.strategy == "llm" and self.llm is not None:
            return await self._llm_plan(task, memory_context)
        return self._stub_plan(task, memory_context)

    # ------------------------------------------------------------------
    # stub：确定性规则分解（默认，离线可用）
    # ------------------------------------------------------------------
    def _stub_plan(self, task: str, memory_context: list[str] | None = None) -> list[PlanStep]:
        """规则启发式拆步。

        触发拆分的信号：
            - 并列词（和/与/并/同时/分别）且出现多个实体 → 每实体一步；
            - 明确动词 + 实体（"查询X"、"计算X"）→ 每 (动词,实体) 一步；
            - 否则单步（直接交给 ReAct 处理）。

        记忆影响（Stage 8 联动）：stub 规则不依赖记忆（规则已足够），
        记忆主要服务于 LLM 规划（_llm_plan）。
        """
        steps: list[PlanStep] = []

        # 1) 天气多城市："查询北京和上海天气" -> 每城市一步
        weather_cities = _parse_weather_cities(task)
        if len(weather_cities) >= 2:
            for city in weather_cities:
                steps.append(
                    _make_step(f"查询 {city} 的天气", tools_hint=["get_weather"], order=len(steps))
                )
            return steps

        # 2) 单城市天气 / 单计算：单步但带工具提示
        if _contains_weather(task) and weather_cities:
            steps.append(
                _make_step(f"查询 {weather_cities[0]} 的天气", tools_hint=["get_weather"], order=0)
            )
            return steps

        # 3) 计算：支持"计算 X 和 Y"拆多步
        calc_exprs = _parse_calc_expressions(task)
        if calc_exprs:
            if len(calc_exprs) >= 2:
                for expr in calc_exprs:
                    steps.append(
                        _make_step(f"计算表达式 {expr}", tools_hint=["calculator"], order=len(steps))
                    )
                return steps
            # 单表达式：description 也带具体表达式（模型需要看到才能调计算器）
            steps.append(
                _make_step(f"计算表达式 {calc_exprs[0]}", tools_hint=["calculator"], order=0)
            )
            return steps

        # 4) 搜索类：单步带搜索提示（若搜索工具已注册）
        if _contains_search(task):
            steps.append(_make_step("搜索相关信息", tools_hint=["web_search"], order=0))
            return steps

        # 5) 兜底：单步无提示（交给 ReAct 自行决策）
        return [_make_step(f"完成任务：{task[:60]}", tools_hint=[], order=0)]

    # ------------------------------------------------------------------
    # llm：模型计划
    # ------------------------------------------------------------------
    async def _llm_plan(self, task: str, memory_context: list[str] | None = None) -> list[PlanStep]:
        """调用 LLM 生成计划步骤（教学演示；生产可用更专业的规划 Prompt）。"""
        tools_text = ", ".join(self.tool_names) if self.tool_names else "（无限制）"
        memory_text = ""
        if memory_context:
            memory_text = "\n相关记忆（可参考）：\n" + "\n".join(f"- {m}" for m in memory_context[:5])
        prompt = (
            "把下面的用户任务分解为按顺序执行的步骤。"
            f"可用工具：{tools_text}\n"
            "输出 JSON 数组，每个元素：{\"description\": \"步骤描述\", \"tools_hint\": [\"工具名\"]}\n"
            f"最多 {self.settings.max_plan_steps} 步。只输出 JSON，不要其他内容。"
            f"{memory_text}\n\n任务：{task}"
        )
        try:
            response = await self.llm.chat([{"role": "user", "content": prompt}], tools=None)
        except Exception:
            return self._stub_plan(task)  # LLM 失败降级规则分解

        raw = (response.content or "").strip()
        # 去掉可能的 markdown 代码块围栏
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()
        try:
            items = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # 宽松解析：按行提取
            items = _loose_parse(raw)

        steps: list[PlanStep] = []
        for it in items[: self.settings.max_plan_steps]:
            if isinstance(it, dict) and it.get("description"):
                steps.append(
                    _make_step(
                        it["description"],
                        tools_hint=[str(t) for t in (it.get("tools_hint") or []) if t],
                        order=len(steps),
                    )
                )
        return steps or self._stub_plan(task)


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _make_step(description: str, *, tools_hint: list[str], order: int) -> PlanStep:
    return PlanStep(
        step_id=f"plan_{uuid4().hex[:8]}",
        description=description,
        tools_hint=tools_hint,
        status="PLANNED",
        order=order,
    )


_WEATHER_CITY_RE = re.compile(r"[\u4e00-\u9fff]{2,4}(?=天气)")


def _parse_weather_cities(task: str) -> list[str]:
    """提取天气任务中的城市列表（"查询北京和上海天气" -> ["北京","上海"]）。"""
    if "天气" not in task:
        return []
    head = task.split("天气")[0]
    # 去掉动词前缀
    for pre in ("帮我查询", "查询一下", "查一下", "同时查询", "分别查询", "查询", "帮我查", "看看", "查"):
        if head.startswith(pre):
            head = head[len(pre):]
            break
    # 按并列分隔符拆
    parts = re.split(r"[和、与，,及\s]+", head)
    cities = [p for p in parts if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", p)]
    return cities[:4]


def _contains_weather(task: str) -> bool:
    return "天气" in task


def _contains_calc(task: str) -> bool:
    return bool(re.search(r"计算\s*[0-9+\-*/().%^ \t]", task))


_CALC_EXPR_RE = re.compile(
    r"计算\s*((?:[0-9+\-*/().%^ \t]){2,}?)(?=\s*(?:和|与|并|以及|、|,|，|$))"
)


def _parse_calc_expressions(task: str) -> list[str]:
    """解析任务中的多个计算表达式（"计算 12 * 34 和 56 + 7" -> ["12 * 34", "56 + 7"]）。

    策略：去掉"计算"前缀后，按中文并列词（和/与/以及/、/,/，）切分；
    表达式内部只有数字、运算符、空格，不含中文词，因此不会被误切。
    """
    if "计算" not in task:
        return []
    head = task.split("计算", 1)[1]
    segments = re.split(r"[和、与，,及以及]+", head)
    exprs: list[str] = []
    for seg in segments:
        seg = seg.strip()
        m = re.match(r"^[0-9+\-*/().%^ \t]+", seg)
        if m:
            expr = m.group(0).strip()
            # 必须是"数字开头"且含至少一个运算符（排除纯数字/空）
            if re.match(r"^\d", expr) and re.search(r"[+\-*/%^]", expr[1:]):
                exprs.append(expr)
        if len(exprs) >= 4:
            break
    return exprs


def _contains_search(task: str) -> bool:
    return any(k in task for k in ("搜索", "查找资料", "查询信息", "查一下资料", "了解一下"))


def _loose_parse(raw: str) -> list[dict]:
    """宽松解析 LLM 输出：按行拆 'description' 或 '步骤' 字段。"""
    items: list[dict] = []
    for line in raw.splitlines():
        line = line.strip().strip('",[]{}')
        if not line:
            continue
        m = re.search(r'"description"\s*:\s*"([^"]+)"', line)
        if m:
            items.append({"description": m.group(1), "tools_hint": []})
    return items
