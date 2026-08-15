"""
Eval Runner：运行评测数据集并输出指标报告。

用法：
    python -m evals.runner                          # 运行全部用例
    python -m evals.runner --tag candidate          # 打标签
    python -m evals.runner --compare evals/runs/eval_run_xxx.json   # 回归对比

流程：
    1. 加载 evals/datasets/basic_agent.jsonl（30+ 用例）
    2. 逐 case 执行 Agent（带 Tracing），收集 Trace
    3. evaluate_case 计算单个结果（含 trace_id / checkpoint_id）
    4. compute_metrics 汇总指标（Task Success / Tool Accuracy / P50 / P95 ...）
    5. 保存 Eval Run（含版本与 git commit），输出报告
    6. --compare 时输出 Baseline vs Candidate 回归报告

每个 CaseResult 保存 case_id / trace_id / checkpoint_id：
    Eval Failure -> Trace -> Checkpoint -> 恢复 State -> 复现
"""
import argparse
import asyncio
import json
import os
import subprocess
from datetime import datetime, timezone
from uuid import uuid4

from app.agent.react_loop import LoopHooks
from app.agent.runtime import AgentRuntime
from app.checkpoint.repository import SQLiteCheckpointRepository
from app.config import Settings, get_settings
from app.llm.client import create_llm_client
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolDefinition
from app.tools.schemas import UserContext
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span
from evals.evaluators import CaseResult, compute_metrics, evaluate_case
from evals.report import compare_runs, save_regression_report


class SimulatedCrash(RuntimeError):
    """模拟进程崩溃（Checkpoint Recovery 用例使用）。"""


def _crash(*args, **kwargs):
    raise SimulatedCrash("模拟进程崩溃")


def load_dataset(path: str) -> list[dict]:
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def git_commit_sha() -> str | None:
    """读取当前 git commit（不可用时返回 None）。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


# 评测默认用户：持有 weather:read，使普通天气用例通过权限校验；
# policy_reject 用例显式覆盖为无权限用户，验证 Permission/Policy 拦截。
_DEFAULT_EVAL_USER = UserContext(user_id="eval_user", permissions=["weather:read"])


async def run_single_case(
    runtime: AgentRuntime,
    recorder: TraceRecorder,
    case: dict,
    session_repo,
    checkpoint_repo,
) -> CaseResult:
    """执行一个用例并评测。"""
    case_id = case["case_id"]
    session_id = f"eval_{case_id}"

    # 预置会话历史（Session / Context 用例）
    if case.get("session_history"):
        session_repo.create_session(session_id)
        for m in case["session_history"]:
            session_repo.add_message(session_id, m)

    user = UserContext(**case["user"]) if case.get("user") else _DEFAULT_EVAL_USER

    async with trace_span("eval.case", "eval", input={"case_id": case_id}, recorder=recorder) as span:
        try:
            if case.get("recovery"):
                # Checkpoint Recovery：LLM 决策后崩溃 -> 从 Checkpoint 恢复
                try:
                    await runtime.run(
                        case["input"],
                        session_id=session_id,
                        user=user,
                        extra_hooks=LoopHooks(after_decision=_crash),
                    )
                except SimulatedCrash:
                    span.attributes.update(crashed=True)
                result = await runtime.resume(session_id, user=user)
            else:
                result = await runtime.run(case["input"], session_id=session_id, user=user)
        except Exception as exc:  # 用例执行失败也要产出可追踪的结果
            result = None
            cr = CaseResult(
                case_id=case_id,
                category=case.get("category", ""),
                trace_id=span.trace_id,
                checkpoint_id=None,
                error=f"{type(exc).__name__}: {exc}",
            )
            return cr

    spans = recorder.load_trace(result.trace_id or span.trace_id)
    cr = evaluate_case(case, result, spans)
    cr.trace_id = result.trace_id or span.trace_id
    cr.checkpoint_id = result.checkpoint_id
    return cr


def build_runner_runtime(settings: Settings, recorder: TraceRecorder) -> AgentRuntime:
    """构造评测用运行时（真实 Session / Checkpoint / Gateway）。

    评测注册表给 get_weather 加上 required_permission="weather:read"，
    使 policy_reject 用例（无权限用户）能真实触发 Permission 拦截。
    """
    session_repo = SQLiteSessionRepository(settings.database_url)
    checkpoint_repo = SQLiteCheckpointRepository(settings.database_url)
    llm = create_llm_client(settings)
    registry = build_default_registry()
    weather = registry.get("get_weather")
    registry.register(
        ToolDefinition(
            name="get_weather",
            description=weather.description,
            input_model=weather.input_model,
            handler=weather.handler,
            timeout_seconds=weather.timeout_seconds,
            required_permission="weather:read",
        ),
        overwrite=True,
    )
    gateway = ToolGateway(registry, settings=settings, recorder=recorder)
    return AgentRuntime(
        llm=llm,
        registry=registry,
        tool_gateway=gateway,
        session_repo=session_repo,
        checkpoint_repo=checkpoint_repo,
        recorder=recorder,
        settings=settings,
    )


async def run_evals(settings: Settings, dataset_path: str, tag: str = "candidate") -> dict:
    """执行一次完整 Eval Run，返回 Run 数据。"""
    cases = load_dataset(dataset_path)
    recorder = TraceRecorder(
        settings.trace_file,
        enabled=True,  # 评测强制开启 Trace（指标依赖）
        capture_content=settings.trace_capture_content,
    )
    runtime = build_runner_runtime(settings, recorder)

    print(f"评测数据集: {dataset_path}（{len(cases)} 个用例）")
    print(f"LLM: {settings.llm_provider_resolved} / {settings.llm_model}")

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        cr = await run_single_case(runtime, recorder, case, runtime.session_repo, runtime.checkpoint_repo)
        results.append(cr)
        status = "✓" if cr.task_success else "✗"
        print(f"  [{i:>2}/{len(cases)}] {cr.case_id:<12} {cr.category:<20} {status} "
              f"(tool={cr.tool_call_count}, llm={cr.llm_call_count}, trace={cr.trace_id[:20]}...)")

    metrics = compute_metrics(results)
    eval_run = {
        "eval_run_id": f"eval_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        "dataset": os.path.basename(dataset_path),
        "agent_version": settings.agent_version,
        "prompt_version": settings.prompt_version,
        "tool_schema_version": settings.tool_schema_version,
        "dataset_version": settings.dataset_version,
        "git_commit": git_commit_sha(),
        "metrics": metrics,
        "cases": [cr.model_dump() for cr in results],
    }

    os.makedirs(settings.eval_run_dir, exist_ok=True)
    run_path = os.path.join(settings.eval_run_dir, f"{eval_run['eval_run_id']}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(eval_run, f, ensure_ascii=False, indent=2)
    print(f"\nEval Run 已保存: {run_path}")
    print_metrics(metrics)
    return eval_run


def print_metrics(metrics: dict) -> None:
    """打印指标表格。"""
    print("\n================ 指标报告 ================")
    rows = [
        ("Task Success Rate（任务成功率）", metrics["task_success_rate"]),
        ("Tool Selection Accuracy（工具选择准确率）", metrics["tool_selection_accuracy"]),
        ("Argument Accuracy（参数准确率）", metrics["argument_accuracy"]),
        ("Unnecessary Tool Rate（多余工具率）", metrics["unnecessary_tool_rate"]),
        ("Tool Error Rate（工具错误率）", metrics["tool_error_rate"]),
        ("Policy Violation Rate（策略违规率）", metrics["policy_violation_rate"]),
        ("Recovery Rate（恢复成功率）", metrics["recovery_rate"]),
        ("Average Tool Calls（平均工具调用）", metrics["average_tool_calls"]),
        ("Average LLM Calls（平均 LLM 调用）", metrics["average_llm_calls"]),
        ("Total Tokens（总 token）", metrics["total_tokens"]),
        ("Average Latency（平均延迟 ms）", metrics["average_latency_ms"]),
        ("P50 Latency（P50 延迟 ms）", metrics["p50_latency_ms"]),
        ("P95 Latency（P95 延迟 ms）", metrics["p95_latency_ms"]),
        ("Plan Accuracy（计划覆盖率）", metrics.get("plan_accuracy")),
        ("Plan Step Success（计划步骤成功率）", metrics.get("plan_step_success_rate")),
        ("Avg Plan Revisions（平均重规划次数）", metrics.get("average_plan_revisions")),
    ]
    for name, value in rows:
        print(f"  {name:<40} {value}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="agent-runtime Eval Runner")
    parser.add_argument("--dataset", default="evals/datasets/basic_agent.jsonl")
    parser.add_argument("--tag", default="candidate")
    parser.add_argument("--compare", default=None, help="Baseline Eval Run JSON 路径")
    args = parser.parse_args()

    settings = get_settings()
    run_data = await run_evals(settings, args.dataset, args.tag)

    if args.compare:
        with open(args.compare, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        report = compare_runs(baseline, run_data)
        print("\n================ 回归报告（Baseline vs Candidate） ================")
        print(report.to_markdown())
        report_path = os.path.join(
            settings.eval_run_dir, f"regression_{run_data['eval_run_id']}.json"
        )
        save_regression_report(report, report_path)
        print(f"回归报告已保存: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
