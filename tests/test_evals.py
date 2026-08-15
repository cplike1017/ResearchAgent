"""
Stage 6 测试：Evaluation（工具选择 / 参数 / P95 / 回归报告 / Runner 端到端）。
"""
import json

import pytest

from app.agent.models import AgentTurnResult
from app.llm.client import ToolCallRequest
from evals.evaluators import CaseResult, compute_metrics, evaluate_case, percentile
from evals.report import compare_runs
from evals.runner import load_dataset, run_evals

DATASET = "evals/datasets/basic_agent.jsonl"


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------
def test_dataset_distribution():
    cases = load_dataset(DATASET)
    assert len(cases) >= 30
    from collections import Counter

    dist = Counter(c["category"] for c in cases)
    assert dist["no_tool"] == 5
    assert dist["single_tool"] == 5
    assert dist["tool_args"] == 5
    assert dist["multi_tool"] == 5
    assert dist["tool_error"] == 4
    assert dist["policy_reject"] == 2
    assert dist["session_context"] == 2
    assert dist["checkpoint_recovery"] == 2
    assert dist["multi_step"] == 3  # Stage 9 规划层用例


# ---------------------------------------------------------------------------
# P50 / P95 百分位实现
# ---------------------------------------------------------------------------
def test_percentile_basic():
    data = list(range(1, 101))  # 1..100
    assert percentile(data, 50) == 50.5
    assert percentile(data, 95) == 95.05
    assert percentile(data, 0) == 1.0
    assert percentile(data, 100) == 100.0


def test_percentile_edge_cases():
    assert percentile([], 95) == 0.0
    assert percentile([42], 50) == 42.0
    assert percentile([3, 1, 2], 50) == 2.0  # 排序后取中位


# ---------------------------------------------------------------------------
# evaluate_case：工具选择 / 参数 / Trace 指标
# ---------------------------------------------------------------------------
def _make_result(tool_calls: list[ToolCallRequest], answer: str = "答案") -> AgentTurnResult:
    return AgentTurnResult(session_id="s", turn_id="t", answer=answer, steps=1, tool_calls=tool_calls)


def test_evaluate_tool_selection_correct():
    case = {"case_id": "x", "expected": {"tool": "get_weather", "arguments": {"city": "北京"}}}
    result = _make_result([ToolCallRequest(id="1", name="get_weather", arguments={"city": "北京"})])
    cr = evaluate_case(case, result, [])
    assert cr.tool_selection_correct is True
    assert cr.argument_correct is True
    assert cr.task_success is True


def test_evaluate_tool_selection_wrong():
    case = {"case_id": "x", "expected": {"tool": "get_weather"}}
    result = _make_result([ToolCallRequest(id="1", name="calculator", arguments={})])
    cr = evaluate_case(case, result, [])
    assert cr.tool_selection_correct is False
    assert cr.task_success is False


def test_evaluate_no_tool_expected():
    case = {"case_id": "x", "expected": {"tool": None}}
    cr = evaluate_case(case, _make_result([], answer="你好"), [])
    assert cr.tool_selection_correct is True
    assert cr.unnecessary_tool is False
    assert cr.task_success is True

    # 不该调用却调用了
    cr2 = evaluate_case(
        case, _make_result([ToolCallRequest(id="1", name="calculator", arguments={})]), []
    )
    assert cr2.unnecessary_tool is True
    assert cr2.task_success is False


def test_evaluate_argument_wrong():
    case = {"case_id": "x", "expected": {"tool": "get_weather", "arguments": {"city": "北京"}}}
    result = _make_result([ToolCallRequest(id="1", name="get_weather", arguments={"city": "上海"})])
    cr = evaluate_case(case, result, [])
    assert cr.tool_selection_correct is True
    assert cr.argument_correct is False
    assert cr.task_success is False


def test_evaluate_trace_metrics():
    """从 Trace 计算 llm/tool 调用次数、延迟、错误。"""
    from app.tracing.models import Span, SpanStatus

    spans = [
        Span(trace_id="t", span_id="s1", name="agent.run", span_type="agent", duration_ms=120.0),
        Span(trace_id="t", span_id="s2", name="llm_call", span_type="llm", attributes={"total_tokens": 10}),
        Span(trace_id="t", span_id="s3", name="llm_call", span_type="llm", attributes={"total_tokens": 8}),
        Span(trace_id="t", span_id="s4", name="tool.execute", span_type="tool", status=SpanStatus.OK, attributes={"success": True}),
    ]
    case = {"case_id": "x", "expected": {"tool": "get_weather"}}
    result = _make_result([ToolCallRequest(id="1", name="get_weather", arguments={})])
    cr = evaluate_case(case, result, spans)
    assert cr.llm_call_count == 2
    assert cr.tool_call_count == 1
    assert cr.tokens_used == 18
    assert cr.duration_ms == 120.0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
def test_compute_metrics():
    cases = [
        CaseResult(case_id="a", task_success=True, tool_selection_correct=True, argument_correct=True, tool_call_count=1, llm_call_count=2, tokens_used=10, duration_ms=50.0),
        CaseResult(case_id="b", task_success=False, tool_selection_correct=False, argument_correct=False, tool_call_count=0, llm_call_count=1, tokens_used=5, duration_ms=150.0),
    ]
    m = compute_metrics(cases)
    assert m["task_success_rate"] == 0.5
    assert m["average_tool_calls"] == 0.5
    assert m["average_llm_calls"] == 1.5
    assert m["average_latency_ms"] == 100.0
    assert m["p50_latency_ms"] == 100.0  # [50,150] 中位
    assert m["p95_latency_ms"] == 145.0  # 线性插值 (150-50)*0.95+50


# ---------------------------------------------------------------------------
# 回归报告
# ---------------------------------------------------------------------------
def test_regression_report_labels():
    baseline = {"eval_run_id": "base", "metrics": {
        "tool_selection_accuracy": 0.84, "task_success_rate": 0.82,
        "average_llm_calls": 2.7, "p95_latency_ms": 4200.0}}
    candidate = {"eval_run_id": "cand", "metrics": {
        "tool_selection_accuracy": 0.93, "task_success_rate": 0.79,
        "average_llm_calls": 2.1, "p95_latency_ms": 3400.0}}
    report = compare_runs(baseline, candidate)
    labels = {m.metric: m.label for m in report.metrics}
    assert labels["tool_selection_accuracy"] == "Improved"   # 0.84 -> 0.93 更高更好
    assert labels["task_success_rate"] == "Regressed"        # 0.82 -> 0.79 下降
    assert labels["average_llm_calls"] == "Improved"         # 2.7 -> 2.1 更低更好
    assert labels["p95_latency_ms"] == "Improved"            # 4200 -> 3400
    assert report.summary.startswith("3 improved, 1 regressed")


def test_regression_report_unchanged():
    baseline = {"eval_run_id": "b", "metrics": {"tool_selection_accuracy": 0.9}}
    candidate = {"eval_run_id": "c", "metrics": {"tool_selection_accuracy": 0.9001}}
    report = compare_runs(baseline, candidate)
    assert report.metrics[0].label == "Unchanged"


# ---------------------------------------------------------------------------
# Stage 9 规划层评测
# ---------------------------------------------------------------------------
def test_evaluate_plan_metrics():
    """multi_step 用例：计划覆盖 + 步骤成功数 + 重规划次数进入 CaseResult。"""
    from app.agent.models import PlanStep

    plan = [
        PlanStep(step_id="p1", description="查北京天气", status="SUCCEEDED", result="晴"),
        PlanStep(step_id="p2", description="查上海天气", status="SUCCEEDED", result="多云"),
    ]
    result = AgentTurnResult(
        session_id="s", turn_id="t",
        answer="北京晴，上海多云",
        steps=2,
        tool_calls=[ToolCallRequest(id="c1", name="get_weather", arguments={"city": "北京"}),
                    ToolCallRequest(id="c2", name="get_weather", arguments={"city": "上海"})],
        plan=plan,
        plan_revisions=1,
    )
    case = {
        "case_id": "ms_test", "category": "multi_step",
        "input": "查询北京和上海天气",
        "expected": {"tools": [
            {"tool": "get_weather", "arguments": {"city": "北京"}},
            {"tool": "get_weather", "arguments": {"city": "上海"}},
        ]},
    }
    cr = evaluate_case(case, result, spans=[])
    assert cr.plan_steps == 2
    assert cr.plan_success_steps == 2
    assert cr.plan_revisions == 1
    assert cr.plan_accuracy is True
    assert cr.has_plan_expectation is True
    assert cr.task_success is True


def test_compute_metrics_plan():
    """汇总指标包含 plan 指标且方向正确。"""
    from app.agent.models import PlanStep

    def mk(cid, ok, steps, succ, rev):
        return CaseResult(case_id=cid, category="multi_step", task_success=ok,
                          has_plan_expectation=True, plan_accuracy=ok,
                          plan_steps=steps, plan_success_steps=succ, plan_revisions=rev)

    cases = [mk("a", True, 2, 2, 0), mk("b", False, 1, 0, 2)]
    m = compute_metrics(cases)
    assert m["plan_accuracy"] == 0.5  # 1/2
    assert m["average_plan_steps"] == 1.5
    assert m["plan_step_success_rate"] == round(2 / 3, 4)
    assert m["average_plan_revisions"] == 1.0


def test_regression_report_plan_directions():
    """plan 指标方向进入回归对比。"""
    from evals.report import METRIC_DIRECTIONS

    assert METRIC_DIRECTIONS["plan_accuracy"] == "higher"
    assert METRIC_DIRECTIONS["average_plan_revisions"] == "lower"
    assert METRIC_DIRECTIONS["plan_step_success_rate"] == "higher"


# ---------------------------------------------------------------------------
# Runner 端到端（30 个用例全量跑）
# ---------------------------------------------------------------------------
async def test_runner_end_to_end(settings):
    """完整跑一遍数据集：指标可计算、Run 文件落盘、case 带 trace_id/checkpoint_id。"""
    run_data = await run_evals(settings, DATASET, tag="test")
    m = run_data["metrics"]
    # 关键指标都存在
    for key in ("task_success_rate", "tool_selection_accuracy", "argument_accuracy",
                "unnecessary_tool_rate", "tool_error_rate", "average_tool_calls",
                "average_llm_calls", "total_tokens", "average_latency_ms",
                "p50_latency_ms", "p95_latency_ms", "policy_violation_rate", "recovery_rate"):
        assert key in m

    # Stage 9 规划层指标
    for key in ("plan_accuracy", "average_plan_steps", "plan_step_success_rate",
                "average_plan_revisions"):
        assert key in m

    # Stub 模型 + 规则评测下，绝大多数用例应成功
    assert m["task_success_rate"] >= 0.8

    # 每个 case 都保存 trace_id；recovery 用例有 checkpoint_id
    cases = run_data["cases"]
    assert len(cases) == 33  # 30 基础 + 3 multi_step
    assert all(c["trace_id"] for c in cases)
    recovery_cases = [c for c in cases if c["category"] == "checkpoint_recovery"]
    assert all(c["checkpoint_id"] for c in recovery_cases)
    assert m["recovery_rate"] == 1.0

    # Run 文件落盘
    import os

    run_dir = settings.eval_run_dir
    files = [f for f in os.listdir(run_dir) if f.endswith(".json")]
    assert len(files) >= 1
