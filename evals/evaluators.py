"""
Evaluators（评测器）：基于规则的结构化匹配 + Trace 指标。

设计原则：
    1. 第一版优先 Rule-Based Evaluation（结构化匹配 / 精确匹配 / Trace 指标），
       不依赖 LLM-as-a-Judge；
    2. 区分 Outcome Quality（最终结果对不对）与 Trajectory Quality（过程好不好）：
       - Outcome：任务成功、工具选对、参数正确
       - Trajectory：工具调用次数 / LLM 调用次数 / 无效工具 / 重试 / 策略违规 / 延迟
    3. 所有指标从 Trace（llm_call / tool.execute / agent.run 等 Span）与
       AgentTurnResult 中计算，可复现。
"""
from pydantic import BaseModel, Field

from app.tracing.models import SpanStatus


# ---------------------------------------------------------------------------
# 百分位数（P50 / P95）
# ---------------------------------------------------------------------------
def percentile(values: list[float], q: float) -> float:
    """
    线性插值百分位（与 numpy.percentile 默认方法一致，避免 off-by-one）。

    rank = (n - 1) * q / 100，在相邻排序值之间线性插值。
    例：data=[1..100]，p50 = (100-1)*0.5 = 49.5 -> 第 49 与第 50 个元素取中 -> 50.5
    """
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    if n == 1:
        return float(sorted_v[0])
    rank = (n - 1) * q / 100.0
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return round(sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * frac, 3)


# ---------------------------------------------------------------------------
# 单个 Case 的评测结果
# ---------------------------------------------------------------------------
class CaseResult(BaseModel):
    """一个评测用例的结果（同时保存 Outcome 与 Trajectory 信息）。"""

    case_id: str
    category: str = ""
    trace_id: str = ""
    checkpoint_id: str | None = None

    # ---- Outcome（结果质量）----
    task_success: bool = False
    tool_selection_correct: bool = False
    argument_correct: bool = False
    has_argument_expectation: bool = False
    unnecessary_tool: bool = False

    # ---- Trajectory（过程质量，来自 Trace）----
    tool_error_occurred: bool = False
    policy_violation: bool = False
    tool_call_count: int = 0
    llm_call_count: int = 0
    tokens_used: int = 0
    duration_ms: float = 0.0
    recovery_success: bool | None = None

    # ---- Stage 9 规划层（Plan）----
    plan_steps: int = 0
    plan_success_steps: int = 0
    plan_revisions: int = 0
    plan_accuracy: bool = False  # 计划是否完整覆盖任务（multi_step 用例专用）
    has_plan_expectation: bool = False

    # ---- 元信息 ----
    answer: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# 单个 Case 的评测逻辑
# ---------------------------------------------------------------------------
def evaluate_case(case: dict, result, spans: list) -> CaseResult:
    """根据 Case 期望 + AgentTurnResult + Trace Span 计算 CaseResult。"""
    expected = case.get("expected", {})
    called_tools = [tc.name for tc in result.tool_calls]

    cr = CaseResult(case_id=case["case_id"], category=case.get("category", ""))

    # ---- Tool Selection ----
    # 注意：multi_tool 用例用 expected.tools，无 expected.tool —— 不能误判为"不需要工具"
    expect_no_tool = expected.get("tool") is None and not expected.get("tools")
    if expect_no_tool:
        cr.tool_selection_correct = len(called_tools) == 0
        cr.unnecessary_tool = len(called_tools) > 0  # 不该调用却调用了
    elif expected.get("tools"):
        # Multi Tool：要求顺序与集合一致
        expected_tools = [t["tool"] for t in expected["tools"]]
        cr.tool_selection_correct = called_tools == expected_tools
    else:
        cr.tool_selection_correct = expected["tool"] in called_tools

    # ---- Argument Accuracy ----
    cr.has_argument_expectation = bool(expected.get("arguments") or expected.get("tools"))
    if expected.get("arguments"):
        calls = [tc for tc in result.tool_calls if tc.name == expected.get("tool")]
        cr.argument_correct = any(tc.arguments == expected["arguments"] for tc in calls)
    elif expected.get("tools"):
        expected_args = [t.get("arguments") for t in expected["tools"]]
        actual_args = [tc.arguments for tc in result.tool_calls]
        cr.argument_correct = all(
            e is None or a == e for e, a in zip(expected_args, actual_args)
        ) and len(expected_args) == len(actual_args)

    # ---- Trace 指标 ----
    cr.llm_call_count = len([s for s in spans if s.name == "llm_call"])
    cr.tool_call_count = len([s for s in spans if s.name == "tool.execute"])
    cr.tool_error_occurred = any(
        s.name == "tool.execute" and s.status == SpanStatus.ERROR for s in spans
    )
    cr.policy_violation = any(
        s.name == "tool_gateway"
        and (s.output or {}).get("error_type") in ("ToolPolicyError", "ToolPermissionError")
        for s in spans
    )
    cr.tokens_used = sum(
        s.attributes.get("total_tokens", 0) for s in spans if s.name == "llm_call"
    )
    agent_spans = [s for s in spans if s.name == "agent.run"]
    cr.duration_ms = agent_spans[0].duration_ms if agent_spans else 0.0

    # ---- Stage 9 规划层指标（来自 AgentTurnResult.plan）----
    plan = getattr(result, "plan", None) or []
    cr.plan_steps = len(plan)
    cr.plan_success_steps = sum(1 for s in plan if s.status == "SUCCEEDED")
    cr.plan_revisions = getattr(result, "plan_revisions", 0)

    # ---- Task Success（Outcome）----
    has_answer = bool(result.answer)
    if case.get("recovery"):
        # Checkpoint Recovery：恢复后能继续并给出答案即成功
        cr.recovery_success = has_answer and cr.tool_call_count > 0
        cr.task_success = bool(cr.recovery_success)
    elif case.get("expected", {}).get("tool_error"):
        # Tool Error Case：期望工具确实报错，且 Agent 给出了兜底回答
        cr.task_success = cr.tool_error_occurred and has_answer
    elif case.get("expected", {}).get("policy_reject"):
        # Policy Reject：期望被策略拦截（工具未真正执行）
        cr.task_success = cr.policy_violation and has_answer
    elif expect_no_tool:
        cr.task_success = cr.tool_selection_correct and has_answer
    elif case.get("category") == "multi_step":
        # Multi-Step：计划完整覆盖（计划步骤数 >= 期望步骤数）且任务成功
        expected_steps = len(expected.get("tools") or [])
        cr.has_plan_expectation = expected_steps > 0
        cr.plan_accuracy = cr.has_plan_expectation and cr.plan_steps >= expected_steps
        args_ok = cr.argument_correct if expected.get("tools") else True
        cr.task_success = cr.plan_accuracy and args_ok and has_answer
    else:
        args_ok = cr.argument_correct if expected.get("arguments") else True
        cr.task_success = cr.tool_selection_correct and args_ok and has_answer

    cr.answer = result.answer[:200]
    return cr


# ---------------------------------------------------------------------------
# 汇总指标
# ---------------------------------------------------------------------------
def compute_metrics(cases: list[CaseResult]) -> dict:
    """从全部 CaseResult 计算汇总指标。"""
    n = len(cases) or 1
    arg_cases = [c for c in cases if c.has_argument_expectation]
    metrics = {
        "task_success_rate": round(sum(c.task_success for c in cases) / n, 4),
        "tool_selection_accuracy": round(sum(c.tool_selection_correct for c in cases) / n, 4),
        # Argument Accuracy 只在"有参数期望"的用例上计算
        "argument_accuracy": (
            round(sum(c.argument_correct for c in arg_cases) / len(arg_cases), 4) if arg_cases else None
        ),
        "unnecessary_tool_rate": round(sum(c.unnecessary_tool for c in cases) / n, 4),
        "tool_error_rate": round(sum(c.tool_error_occurred for c in cases) / n, 4),
        "policy_violation_rate": round(sum(c.policy_violation for c in cases) / n, 4),
        "average_tool_calls": round(sum(c.tool_call_count for c in cases) / n, 3),
        "average_llm_calls": round(sum(c.llm_call_count for c in cases) / n, 3),
        "total_tokens": sum(c.tokens_used for c in cases),
        "average_tokens": round(sum(c.tokens_used for c in cases) / n, 1),
        "average_latency_ms": round(sum(c.duration_ms for c in cases) / n, 1),
        "p50_latency_ms": percentile([c.duration_ms for c in cases], 50),
        "p95_latency_ms": percentile([c.duration_ms for c in cases], 95),
        # ---- Stage 9 规划层指标 ----
        "plan_accuracy": round(sum(c.plan_accuracy for c in cases) / n, 4),
        "average_plan_steps": round(sum(c.plan_steps for c in cases) / n, 3),
        "plan_step_success_rate": (
            round(sum(c.plan_success_steps for c in cases) / max(1, sum(c.plan_steps for c in cases)), 4)
            if sum(c.plan_steps for c in cases)
            else None
        ),
        "average_plan_revisions": round(sum(c.plan_revisions for c in cases) / n, 3),
    }
    recovery_cases = [c for c in cases if c.recovery_success is not None]
    metrics["recovery_rate"] = (
        round(sum(c.recovery_success for c in recovery_cases) / len(recovery_cases), 4)
        if recovery_cases
        else None
    )
    # Plan Accuracy 只在"有规划期望"（multi_step）用例上计算
    plan_cases = [c for c in cases if c.has_plan_expectation]
    if plan_cases:
        metrics["plan_accuracy"] = round(sum(c.plan_accuracy for c in plan_cases) / len(plan_cases), 4)
    return metrics
