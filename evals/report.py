"""
Regression Report（回归报告）：Baseline vs Candidate。

防止只看单一指标：每个指标都标注 Improved / Regressed / Unchanged，
并区分"越高越好"（准确率）与"越低越好"（延迟 / 错误率）的指标方向。
"""
import json
import os
from datetime import datetime, timezone

from pydantic import BaseModel, Field

# 指标方向：higher=越高越好，lower=越低越好
METRIC_DIRECTIONS: dict[str, str] = {
    "task_success_rate": "higher",
    "tool_selection_accuracy": "higher",
    "argument_accuracy": "higher",
    "unnecessary_tool_rate": "lower",
    "tool_error_rate": "lower",
    "policy_violation_rate": "lower",
    "average_tool_calls": "lower",
    "average_llm_calls": "lower",
    "total_tokens": "lower",
    "average_tokens": "lower",
    "average_latency_ms": "lower",
    "p50_latency_ms": "lower",
    "p95_latency_ms": "lower",
    "recovery_rate": "higher",
    # Stage 9 规划层指标
    "plan_accuracy": "higher",
    "plan_step_success_rate": "higher",
    "average_plan_steps": "lower",       # 计划越精简越好（但需 >= 期望步骤数）
    "average_plan_revisions": "lower",   # 反思重规划越少越好
}

# 判定为"变化"的最小差值（避免把浮点噪音标成 Regressed）
DELTA_EPSILON = 0.001


class MetricDelta(BaseModel):
    """单个指标的对比结果。"""

    metric: str
    baseline: float
    candidate: float
    delta: float
    label: str = Field(description="Improved | Regressed | Unchanged")
    direction: str


class RegressionReport(BaseModel):
    """一次 Baseline vs Candidate 的回归对比。"""

    baseline_run_id: str = ""
    candidate_run_id: str = ""
    created_at: str = ""
    metrics: list[MetricDelta] = Field(default_factory=list)
    summary: str = ""

    def to_markdown(self) -> str:
        """生成人类可读的对比表格。"""
        lines = [
            f"# Regression Report",
            f"Baseline: `{self.baseline_run_id}` vs Candidate: `{self.candidate_run_id}`",
            "",
            "| Metric | Baseline | Candidate | Delta | Verdict |",
            "|---|---|---|---|---|",
        ]
        for m in self.metrics:
            lines.append(
                f"| {m.metric} | {m.baseline:g} | {m.candidate:g} | {m.delta:+.3g} | **{m.label}** |"
            )
        lines.append("")
        lines.append(f"**Summary**: {self.summary}")
        return "\n".join(lines)


def compare_runs(baseline_run: dict, candidate_run: dict) -> RegressionReport:
    """对比两份 Eval Run 的 metrics，输出带 Improved/Regressed 标注的报告。"""
    bm = baseline_run.get("metrics", {})
    cm = candidate_run.get("metrics", {})
    deltas: list[MetricDelta] = []

    improved, regressed, unchanged = 0, 0, 0
    for metric, direction in METRIC_DIRECTIONS.items():
        if metric not in bm or metric not in cm:
            continue
        base = bm[metric]
        cand = cm[metric]
        if base is None or cand is None:
            continue
        base = float(base)
        cand = float(cand)
        delta = cand - base
        if abs(delta) < DELTA_EPSILON:
            label = "Unchanged"
            unchanged += 1
        elif (direction == "higher" and delta > 0) or (direction == "lower" and delta < 0):
            label = "Improved"
            improved += 1
        else:
            label = "Regressed"
            regressed += 1
        deltas.append(MetricDelta(metric=metric, baseline=base, candidate=cand, delta=delta, label=label, direction=direction))

    summary = f"{improved} improved, {regressed} regressed, {unchanged} unchanged"
    return RegressionReport(
        baseline_run_id=baseline_run.get("eval_run_id", ""),
        candidate_run_id=candidate_run.get("eval_run_id", ""),
        created_at=datetime.now(timezone.utc).isoformat(),
        metrics=deltas,
        summary=summary,
    )


def save_regression_report(report: RegressionReport, path: str) -> None:
    """保存回归报告 JSON。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))
