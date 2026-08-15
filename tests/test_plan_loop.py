"""
Stage 9 测试：规划层（Planner / PlanExecutor / Reflector / AgentRuntime plan 模式）。

验收：stub 规则分解（多城市/计算/兜底）、llm 计划降级、PlanExecutor 逐步执行、
Reflector 失败反思与重规划上限、AgentRuntime plan 模式端到端（含 Checkpoint 状态扩展）。
"""
import pytest

from app.agent.models import PlanStep
from app.agent.planner import Planner, _parse_weather_cities
from app.agent.reflector import Reflector
from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.llm.client import StubLLMClient
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry


@pytest.fixture
def plan_settings(tmp_path) -> Settings:
    """plan 模式测试配置（stub 规划，离线）。"""
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/plan.db",
        trace_enabled=False,
        agent_mode="plan",
        planner_strategy="stub",
        max_plan_revisions=2,
        max_plan_steps=6,
    )


# ---------------------------------------------------------------------------
# Planner：stub 规则分解
# ---------------------------------------------------------------------------
def test_parse_weather_cities():
    assert _parse_weather_cities("查询北京和上海天气") == ["北京", "上海"]
    assert _parse_weather_cities("同时查询广州、深圳和成都天气") == ["广州", "深圳", "成都"]
    assert _parse_weather_cities("计算 1 + 1") == []


@pytest.mark.asyncio
async def test_stub_plan_multi_city(plan_settings):
    p = Planner(plan_settings)
    plan = await p.plan("查询北京和上海天气")
    assert len(plan) == 2
    assert all(s.tools_hint == ["get_weather"] for s in plan)
    assert plan[0].order == 0 and plan[1].order == 1
    assert plan[0].status == "PLANNED"


@pytest.mark.asyncio
async def test_stub_plan_calc(plan_settings):
    p = Planner(plan_settings)
    plan = await p.plan("计算 123 * 456")
    assert len(plan) == 1
    assert plan[0].tools_hint == ["calculator"]


@pytest.mark.asyncio
async def test_stub_plan_fallback(plan_settings):
    p = Planner(plan_settings)
    plan = await p.plan("你好")
    assert len(plan) == 1
    assert plan[0].tools_hint == []


@pytest.mark.asyncio
async def test_llm_plan_falls_back_on_failure(plan_settings):
    """LLM 策略无模型或失败时降级为 stub 规则分解。"""
    p = Planner(plan_settings.model_copy(update={"planner_strategy": "llm"}), llm=None)
    plan = await p.plan("查询北京和上海天气")
    assert len(plan) == 2  # 降级到规则分解


# ---------------------------------------------------------------------------
# Reflector
# ---------------------------------------------------------------------------
def test_reflect_failed_step():
    r = Reflector(max_revisions=2)
    failed = PlanStep(step_id="p1", description="查天气", status="FAILED", result="工具失败")
    ok = PlanStep(step_id="p2", description="查新闻", status="SUCCEEDED", result="新闻")
    d = r.reflect("任务", [failed, ok], revisions_so_far=0)
    assert d.need_replan is True
    assert "查天气" in d.revised_task


def test_reflect_all_ok():
    r = Reflector(max_revisions=2)
    ok = PlanStep(step_id="p1", description="查天气", status="SUCCEEDED", result="晴")
    d = r.reflect("任务", [ok], revisions_so_far=0)
    assert d.need_replan is False


def test_reflect_revision_limit():
    r = Reflector(max_revisions=2)
    failed = PlanStep(step_id="p1", description="查天气", status="FAILED", result="失败")
    d = r.reflect("任务", [failed], revisions_so_far=2)
    assert d.need_replan is False  # 已达上限，不再重规划


def test_reflect_empty_plan():
    r = Reflector(max_revisions=2)
    d = r.reflect("任务", [], revisions_so_far=0)
    assert d.need_replan is False


# ---------------------------------------------------------------------------
# AgentRuntime plan 模式端到端（Stub LLM + Stub 规划）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runtime_plan_mode_multi_city(tmp_path, plan_settings):
    """plan 模式：多城市天气任务应拆成多步并逐步执行。"""
    session_repo = SQLiteSessionRepository(plan_settings.database_url)
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        session_repo=session_repo,
        settings=plan_settings,
    )
    result = await runtime.run("查询北京和上海天气", session_id="s_plan")
    assert result.answer
    # Stub LLM 对多城市会一次性发两个 get_weather 调用
    assert any(tc.name == "get_weather" for tc in result.tool_calls)


@pytest.mark.asyncio
async def test_runtime_plan_mode_simple_task(tmp_path, plan_settings):
    """plan 模式：简单任务（寒暄）也应正常返回。"""
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        settings=plan_settings,
    )
    result = await runtime.run("你好", session_id="s_plan2")
    assert result.answer


@pytest.mark.asyncio
async def test_runtime_react_mode_backward_compat(tmp_path):
    """agent_mode=react 时行为与 Stage 8 完全一致（无 plan 字段）。"""
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/react.db",
        trace_enabled=False,
        agent_mode="react",
    )
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        settings=settings,
    )
    result = await runtime.run("查询北京天气", session_id="s_react")
    assert result.answer
    assert result.plan == []


@pytest.mark.asyncio
async def test_checkpoint_state_with_plan(tmp_path):
    """AgentState 扩展字段（plan / agent_mode）可被 Checkpoint 序列化恢复。"""
    from app.checkpoint.repository import SQLiteCheckpointRepository

    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/ckpt.db",
        trace_enabled=False,
        agent_mode="plan",
        planner_strategy="stub",
    )
    session_repo = SQLiteSessionRepository(settings.database_url)
    ckpt_repo = SQLiteCheckpointRepository(settings.database_url)
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        session_repo=session_repo,
        checkpoint_repo=ckpt_repo,
        settings=settings,
    )
    result = await runtime.run("查询北京和上海天气", session_id="s_ckpt")
    assert result.checkpoint_id
    latest = ckpt_repo.load_latest("s_ckpt")
    assert latest is not None
    # 恢复出的状态含 plan 字段（默认空列表也兼容）
    from app.agent.state import AgentState

    restored = AgentState(**latest.state)
    assert restored.agent_mode == "plan"
