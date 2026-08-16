"""
Stage 12 测试：多 Agent 编排（planner / executor / runner / delegate 工具 / Web 端点 / Trace 嵌套 / 多级编排）。

全部离线：脚本化 LLM（ScriptedLLM）精确控制规划、子 agent、合成三路输出。
"""
import pytest

from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.errors import LLMError
from app.llm.client import BaseLLMClient, LLMResponse, ToolCallRequest
from app.orchestrator.executor import SubAgentExecutor
from app.orchestrator.planner import OrchestratorPlanner
from app.orchestrator.profiles import get_profile
from app.orchestrator.runner import OrchestratorRunner
from app.orchestrator.tool import build_delegate_tool
from app.tools.builtin import build_default_registry
from app.tracing.recorder import TraceRecorder


# ---------------------------------------------------------------------------
# 脚本化 LLM：按消息内容分流（规划 / 合成 / 子 agent / 嵌套委派）
# ---------------------------------------------------------------------------
class ScriptedLLM(BaseLLMClient):
    """按 prompt 特征分发：
        - 含「可用子 Agent 档案」 → 规划（返回 plan_json）
        - 含「各子 Agent 结果」   → 合成（返回 synthesize_text）
        - 含「委派给」           → 子 agent 发起 delegate 工具调用（多级编排）
        - 其余                   → 子 agent 回合（返回子 agent 答案；命中 fail_tasks 抛错）
    """

    def __init__(self, plan_json=None, synthesize_text="【合成】", sub_answer="子Agent完成"):
        self.plan_json = plan_json
        self.synthesize_text = synthesize_text
        self.sub_answer = sub_answer
        self.fail_tasks: set[str] = set()
        self.calls: list[tuple[list[dict], list[dict] | None]] = []  # (messages, tools)
        self.model = "scripted"

    async def chat(self, messages, tools=None, **kwargs):
        self.calls.append((messages, tools))
        text = "\n".join(str(m.get("content") or "") for m in messages)
        if "可用子 Agent 档案" in text:
            return LLMResponse(content=self.plan_json, model=self.model)
        if "各子 Agent 结果" in text:
            return LLMResponse(content=self.synthesize_text, model=self.model)
        if any(t in text for t in self.fail_tasks):
            raise LLMError(f"scripted failure: {self.fail_tasks}")
        # 子 agent 回合（先取人设标记）
        system = messages[0].get("content", "") if messages else ""
        marker = "generalist"
        for keyword, name in (
            ("资深研究员", "researcher"),
            ("数据分析师", "analyst"),
            ("报告写手", "writer"),
        ):
            if keyword in system:
                marker = name
        # 已有工具结果（delegate 已返回）→ 收尾回答，绝不重复委派
        if any(m.get("role") == "tool" for m in messages):
            return LLMResponse(content=f"{self.sub_answer}({marker})", model=self.model)
        # 命中「委派给」→ 发起 delegate 工具调用（多级编排）
        if "委派给" in text:
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(
                        id="call_delegate_nested",
                        name="delegate",
                        arguments={"task": "细化子任务", "agents": ["writer"]},
                    )
                ],
                model=self.model,
            )
        # 子 agent 回合：直接最终回答（不调用工具）
        return LLMResponse(content=f"{self.sub_answer}({marker})", model=self.model)


@pytest.fixture
def orch_settings(tmp_path) -> Settings:
    return Settings(
        environment="test",
        trace_enabled=False,
        agent_mode="react",
        llm_provider="stub",
        orchestrator_enabled=True,
        orchestrator_planner_strategy="llm",
        orchestrator_max_steps=5,
        orchestrator_max_parallel=3,
        trace_file=str(tmp_path / "traces.jsonl"),
    )


@pytest.fixture
def registry():
    return build_default_registry()


# ---------------------------------------------------------------------------
# profiles：白名单与兜底
# ---------------------------------------------------------------------------
def test_get_profile_unknown_falls_back_to_generalist():
    p = get_profile("not_a_real_agent")
    assert p.name == "generalist"
    assert p.allowed_tools is None  # 全能


def test_filtered_registry_respects_whitelist(registry):
    """researcher 摸不到 run_code，analyst 摸不到 web_search，delegate 永远排除。"""
    executor = SubAgentExecutor(llm=ScriptedLLM(), master_registry=registry)
    research_tools = {t.name for t in executor._filtered_registry(get_profile("researcher")).all()}
    assert "web_search" in research_tools
    assert "run_code" not in research_tools
    assert "delegate" not in research_tools

    analyst_tools = {t.name for t in executor._filtered_registry(get_profile("analyst")).all()}
    assert "run_code" in analyst_tools
    assert "web_search" not in analyst_tools
    assert "delegate" not in analyst_tools


# ---------------------------------------------------------------------------
# planner：LLM 分工 / 降级 / 显式名单
# ---------------------------------------------------------------------------
PLAN_TWO = (
    '{"rationale": "先查资料再分析", "steps": ['
    '{"agent": "researcher", "task": "检索资料", "depends_on": []},'
    '{"agent": "analyst", "task": "分析数据", "depends_on": []}]}'
)


@pytest.mark.asyncio
async def test_planner_parses_llm_plan(orch_settings):
    llm = ScriptedLLM(plan_json=PLAN_TWO)
    plan = await OrchestratorPlanner(orch_settings, llm=llm).plan("调研课题")
    assert len(plan.steps) == 2
    assert plan.steps[0].agent == "researcher"
    assert plan.steps[1].agent == "analyst"
    assert plan.steps[1].depends_on == []


@pytest.mark.asyncio
async def test_planner_parses_code_fenced_json(orch_settings):
    llm = ScriptedLLM(plan_json=f"好的，这是计划：\n```json\n{PLAN_TWO}\n```\n请查收")
    plan = await OrchestratorPlanner(orch_settings, llm=llm).plan("调研课题")
    assert len(plan.steps) == 2


@pytest.mark.asyncio
async def test_planner_falls_back_on_garbage(orch_settings):
    """LLM 输出无法解析 → 降级为单步 generalist（不抛异常）。"""
    llm = ScriptedLLM(plan_json="抱歉，我无法规划这个任务。")
    plan = await OrchestratorPlanner(orch_settings, llm=llm).plan("调研课题")
    assert len(plan.steps) == 1
    assert plan.steps[0].agent == "generalist"


@pytest.mark.asyncio
async def test_planner_falls_back_on_llm_error(orch_settings):
    llm = ScriptedLLM()
    llm.fail_tasks.add("可用子 Agent 档案")  # 让规划调用抛错
    plan = await OrchestratorPlanner(orch_settings, llm=llm).plan("调研课题")
    assert len(plan.steps) == 1 and plan.steps[0].agent == "generalist"


@pytest.mark.asyncio
async def test_planner_explicit_agents_skips_llm(orch_settings):
    """显式指定名单时不再调 LLM，且未知档案名回退 generalist。"""
    llm = ScriptedLLM(plan_json=None)  # 若被调用会返回 None 内容
    plan = await OrchestratorPlanner(orch_settings, llm=llm).plan(
        "任务", agents=["researcher", "bogus"]
    )
    assert len(plan.steps) == 2
    assert plan.steps[0].agent == "researcher"
    assert plan.steps[1].agent == "generalist"


# ---------------------------------------------------------------------------
# runner：单步直返 / 并行 + 合成 / 依赖上下文 / 失败隔离
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runner_single_step_returns_directly(orch_settings, registry):
    """单步编排：直接返回子 agent 答案，不调用合成。"""
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    result = await runner.run("任务", agents=["generalist"])
    assert result.status == "SUCCEEDED"
    assert result.final_answer == "子Agent完成(generalist)"
    assert len(result.agent_results) == 1
    # 合成提示未被调用（single step 不合成）
    assert not any("各子 Agent 结果" in str(m) for m, _ in llm.calls)


@pytest.mark.asyncio
async def test_runner_parallel_steps_with_synthesis(orch_settings, registry):
    """两步并行执行，最后经 LLM 合成。"""
    llm = ScriptedLLM(plan_json=PLAN_TWO, synthesize_text="【整合后的最终答案】")
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    result = await runner.run("调研课题")
    assert result.status == "SUCCEEDED"
    assert result.final_answer == "【整合后的最终答案】"
    assert [r.agent for r in result.agent_results] == ["researcher", "analyst"]
    assert all(r.status == "SUCCEEDED" for r in result.agent_results)
    # 子 agent 工具 schema 按白名单过滤
    sub_calls = [(m, t) for m, t in llm.calls if "可用子 Agent 档案" not in str(m) and "各子 Agent 结果" not in str(m)]
    researcher_tools = {s["function"]["name"] for _, t in sub_calls[:1] for s in (t or [])}
    analyst_tools = {s["function"]["name"] for _, t in sub_calls[1:2] for s in (t or [])}
    assert "web_search" in researcher_tools and "run_code" not in researcher_tools
    assert "run_code" in analyst_tools and "web_search" not in analyst_tools


@pytest.mark.asyncio
async def test_runner_dependency_context_passed(orch_settings, registry):
    """依赖步骤的结果会拼进下游子 agent 的用户消息。"""
    plan = (
        '{"steps": [{"agent": "analyst", "task": "先算 1+1", "depends_on": []},'
        '{"agent": "analyst", "task": "再算 2+2", "depends_on": [0]}]}'
    )
    llm = ScriptedLLM(plan_json=plan, synthesize_text="合成")
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    result = await runner.run("连续计算")
    assert result.status == "SUCCEEDED"
    # 第二个子 agent 的用户消息包含步骤 0 的结果
    step1_user = ""
    for m, _ in llm.calls:
        text = "\n".join(str(x.get("content") or "") for x in m)
        if "再算 2+2" in text:
            step1_user = text
            break
    assert "步骤0" in step1_user and "子Agent完成" in step1_user
    assert "先算 1+1" not in step1_user  # 依赖上下文只带结果，不带完整任务


@pytest.mark.asyncio
async def test_runner_failure_isolation(orch_settings, registry):
    """一个子 agent 失败不影响另一个；整体 PARTIAL，合成照常。"""
    llm = ScriptedLLM(plan_json=PLAN_TWO, synthesize_text="部分完成")
    llm.fail_tasks.add("分析数据")
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    result = await runner.run("调研课题")
    assert result.status == "PARTIAL"
    statuses = {r.agent: r.status for r in result.agent_results}
    assert statuses["researcher"] == "SUCCEEDED"
    assert statuses["analyst"] == "FAILED"
    assert result.final_answer == "部分完成"


@pytest.mark.asyncio
async def test_runner_all_failed(orch_settings, registry):
    llm = ScriptedLLM(plan_json=PLAN_TWO)
    llm.fail_tasks.update(["检索资料", "分析数据"])
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    result = await runner.run("调研课题")
    assert result.status == "FAILED"
    assert all(r.status == "FAILED" for r in result.agent_results)


@pytest.mark.asyncio
async def test_runner_cycles_are_skipped(orch_settings, registry):
    """依赖环防御：无法推进的步骤标记 SKIPPED，不卡死。

    注：planner 的 _parse_plan 会过滤"向前依赖"（d >= i），LLM 输出造不出环；
    这里直接构造 OrchestrationPlan 验证 runner 自身的死锁兜底。
    """
    from app.orchestrator.models import OrchestrationPlan, SubTask

    plan = OrchestrationPlan(
        steps=[
            SubTask(agent="generalist", task="A", depends_on=[1]),
            SubTask(agent="generalist", task="B", depends_on=[0]),
        ]
    )
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    result = await runner._run_impl("环", plan, "", None, 0.0)
    assert all(r.status == "SKIPPED" for r in result.agent_results)
    assert result.status == "FAILED"


# ---------------------------------------------------------------------------
# executor：独立执行 / 异常包装
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_executor_runs_sub_agent(orch_settings, registry):
    llm = ScriptedLLM(sub_answer="研究报告")
    executor = SubAgentExecutor(llm=llm, master_registry=registry, settings=orch_settings)
    result = await executor.execute(get_profile("researcher"), "检索论文")
    assert result.status == "SUCCEEDED"
    assert result.answer == "研究报告(researcher)"
    assert result.steps >= 1


@pytest.mark.asyncio
async def test_executor_wraps_exceptions(orch_settings, registry):
    llm = ScriptedLLM()
    llm.fail_tasks.add("爆炸任务")
    executor = SubAgentExecutor(llm=llm, master_registry=registry, settings=orch_settings)
    result = await executor.execute(get_profile("generalist"), "爆炸任务")
    assert result.status == "FAILED"
    assert "LLMError" in result.error


# ---------------------------------------------------------------------------
# delegate 工具：注册进 runtime + 经网关执行
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delegate_tool_registered_in_runtime(orch_settings, registry):
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    runtime = AgentRuntime(llm=llm, registry=registry, settings=orch_settings, orchestrator=runner)
    tool = runtime.registry.get("delegate")
    assert tool.name == "delegate"
    assert "委派" in tool.description


@pytest.mark.asyncio
async def test_delegate_tool_executes_via_gateway(orch_settings, registry):
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    runtime = AgentRuntime(llm=llm, registry=registry, settings=orch_settings, orchestrator=runner)
    envelope = await runtime._execute_tool(
        "delegate", {"task": "整理报告", "agents": ["writer"], "context": "背景资料"}
    )
    assert envelope.success
    data = envelope.data
    assert data["status"] == "SUCCEEDED"
    assert data["final_answer"] == "子Agent完成(writer)"
    assert data["agent_results"][0]["agent"] == "writer"


# ---------------------------------------------------------------------------
# Trace 嵌套：orchestrator.run → agent.run → llm_call
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_orchestrator_trace_nesting(tmp_path, registry):
    settings = Settings(
        environment="test",
        trace_enabled=True,
        trace_file=str(tmp_path / "traces.jsonl"),
        agent_mode="react",
        llm_provider="stub",
        orchestrator_planner_strategy="llm",
    )
    recorder = TraceRecorder(settings.trace_file, enabled=True, capture_content=False)
    llm = ScriptedLLM(plan_json=PLAN_TWO, synthesize_text="合成答案")
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=settings, recorder=recorder)
    result = await runner.run("调研课题")
    assert result.trace_id
    tree = recorder.build_tree(result.trace_id)
    roots = tree["spans"]
    assert len(roots) == 1 and roots[0]["name"] == "orchestrator.run"
    kids = roots[0]["children"]
    agent_runs = [k for k in kids if k["name"] == "agent.run"]
    assert len(agent_runs) == 2
    # 每个子 agent 内部有 llm_call
    for a in agent_runs:
        assert any(c["name"] == "llm_call" for c in a["children"])


# ---------------------------------------------------------------------------
# 多级编排（子 agent 嵌套委派）
# ---------------------------------------------------------------------------
def _registry_with_delegate(registry, runner) -> None:
    """把 delegate 工具注入测试 registry（生产环境由 AgentRuntime 注入）。"""
    from app.orchestrator.tool import build_delegate_tool

    registry.register(build_delegate_tool(runner), overwrite=True)


@pytest.mark.asyncio
async def test_delegate_visibility_by_depth(orch_settings, registry):
    """delegate 可见性随深度变化：depth < max_depth 可见（可再委派），叶子层不可见。

    且不受档案白名单限制（编排层元能力）——researcher 也能看到 delegate。
    """
    from app.orchestrator.context import orchestration_depth

    runner = OrchestratorRunner(llm=ScriptedLLM(), registry=registry, settings=orch_settings)
    _registry_with_delegate(registry, runner)
    executor = SubAgentExecutor(llm=ScriptedLLM(), master_registry=registry, settings=orch_settings)

    # depth=0（主 agent 侧）：可见
    assert "delegate" in {t.name for t in executor._filtered_registry(get_profile("generalist")).all()}
    # 白名单豁免：researcher 也能看到 delegate（委派是元能力）
    assert "delegate" in {t.name for t in executor._filtered_registry(get_profile("researcher")).all()}

    # depth=1（max_depth=2）：仍可见，子 agent 可再委派
    token = orchestration_depth.set(1)
    try:
        tools1 = {t.name for t in executor._filtered_registry(get_profile("researcher")).all()}
        assert "delegate" in tools1
    finally:
        orchestration_depth.reset(token)

    # depth=2（叶子层）：delegate 被物理移除
    token2 = orchestration_depth.set(2)
    try:
        tools2 = {t.name for t in executor._filtered_registry(get_profile("researcher")).all()}
        assert "delegate" not in tools2
    finally:
        orchestration_depth.reset(token2)


@pytest.mark.asyncio
async def test_nested_delegation_two_levels(orch_settings, registry):
    """两级编排：researcher 子 agent 在任务中再委派给 writer（孙级），结果逐层回传。"""
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings)
    _registry_with_delegate(registry, runner)

    # 显式指定 researcher：researcher 的任务含「委派给」→ ScriptedLLM 触发 delegate 给 writer
    result = await runner.run("检索资料并委派给 writer 细化", agents=["researcher"])
    assert result.status == "SUCCEEDED"
    # 单步编排：最终答案 = researcher 的回答（含其内部委派后的收尾）
    assert result.agent_results[0].agent == "researcher"
    assert result.agent_results[0].tool_calls  # researcher 内部调用了 delegate
    assert any(t["name"] == "delegate" for t in result.agent_results[0].tool_calls)

    # writer 确实作为孙级执行过（llm.calls 中出现 writer 人设的 system 消息）
    writer_system_seen = any(
        any("报告写手" in str(m.get("content") or "") for m in msgs)
        for msgs, _ in llm.calls
    )
    assert writer_system_seen


@pytest.mark.asyncio
async def test_depth_limit_blocks_nested_delegation(orch_settings, registry):
    """max_depth=1：子 agent 看不到 delegate，委派调用变成失败的工具调用，孙级不执行。"""
    from app.orchestrator.context import orchestration_depth

    settings = orch_settings.model_copy(update={"orchestrator_max_depth": 1})
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=settings)
    _registry_with_delegate(registry, runner)

    # 直接验证叶子层过滤：depth=1 且 max_depth=1 → 无 delegate
    executor = SubAgentExecutor(llm=llm, master_registry=registry, settings=settings)
    token = orchestration_depth.set(1)
    try:
        tools = {t.name for t in executor._filtered_registry(get_profile("generalist")).all()}
        assert "delegate" not in tools
    finally:
        orchestration_depth.reset(token)

    # 端到端：researcher 试图委派 → delegate 未注册 → 工具调用失败 → 子 agent 收尾完成
    result = await runner.run("检索资料并委派给 writer 细化", agents=["researcher"])
    assert result.status == "SUCCEEDED"
    # 孙级 writer 从未执行（没有 writer 人设的调用）
    writer_system_seen = any(
        any("报告写手" in str(m.get("content") or "") for m in msgs)
        for msgs, _ in llm.calls
    )
    assert not writer_system_seen


@pytest.mark.asyncio
async def test_trace_nesting_two_levels(tmp_path, registry):
    """Trace 树呈现两级编排：orchestrator.run(depth1) → agent.run → delegate → orchestrator.run(depth2)。"""
    settings = Settings(
        environment="test",
        trace_enabled=True,
        trace_file=str(tmp_path / "traces.jsonl"),
        agent_mode="react",
        llm_provider="stub",
        orchestrator_planner_strategy="llm",
        orchestrator_max_depth=2,
    )
    recorder = TraceRecorder(settings.trace_file, enabled=True, capture_content=False)
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=settings, recorder=recorder)
    _registry_with_delegate(registry, runner)

    result = await runner.run("检索资料并委派给 writer 细化", agents=["researcher"])
    assert result.status == "SUCCEEDED"
    tree = recorder.build_tree(result.trace_id)
    roots = tree["spans"]
    assert len(roots) == 1 and roots[0]["name"] == "orchestrator.run"
    assert roots[0]["attributes"].get("depth") == 1

    def find(nodes, name):
        for n in nodes:
            if n["name"] == name:
                return n
            hit = find(n.get("children", []), name)
            if hit:
                return hit
        return None

    # 第一层 researcher agent.run
    level1_agent = find(roots[0]["children"], "agent.run")
    assert level1_agent is not None
    # 其下 delegate 工具调用里嵌套了第二层 orchestrator.run(depth=2)
    delegate_exec = find(level1_agent["children"], "tool.execute")
    assert delegate_exec is not None
    nested = find(delegate_exec["children"], "orchestrator.run")
    assert nested is not None
    assert nested["attributes"].get("depth") == 2
    # 第二层里有 writer 子 agent
    level2_agent = find(nested["children"], "agent.run")
    assert level2_agent is not None
    assert level2_agent["attributes"].get("agent_profile") == "writer"


# ---------------------------------------------------------------------------
# Web 端点（进程内直连）
# ---------------------------------------------------------------------------
def test_web_orchestrate_and_agents(tmp_path):
    from fastapi.testclient import TestClient

    import fakeredis.aioredis

    from app.main import create_app

    settings = Settings(
        environment="test",
        trace_enabled=False,
        memory_enabled=False,
        skills_enabled=False,
        agent_mode="react",
        llm_provider="stub",
        database_url=f"sqlite:///{tmp_path}/web.db",
        trace_file=str(tmp_path / "traces.jsonl"),
        eval_run_dir=str(tmp_path / "runs"),
        orchestrator_enabled=True,
        orchestrator_planner_strategy="stub",  # 确定性单步
    )
    app = create_app(settings, redis=fakeredis.aioredis.FakeRedis(decode_responses=True))
    with TestClient(app) as client:
        # 档案列表
        agents = client.get("/api/web/agents").json()
        assert agents["count"] == 4
        names = {a["name"] for a in agents["agents"]}
        assert {"researcher", "analyst", "writer", "generalist"} <= names

        # 直接编排（stub 规划 → 单步 generalist → stub LLM 天气）
        r = client.post("/api/web/orchestrate", json={"task": "查询北京天气"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "SUCCEEDED"
        assert data["agent_results"][0]["agent"] == "generalist"
        assert data["final_answer"]

        # 显式指定名单
        r2 = client.post("/api/web/orchestrate", json={"task": "分析数据", "agents": ["analyst"]})
        assert r2.status_code == 200
        assert r2.json()["agent_results"][0]["agent"] == "analyst"

        # 工具列表应包含 delegate
        tools = client.get("/api/web/tools").json()
        assert any(t["name"] == "delegate" for t in tools["tools"])


# ---------------------------------------------------------------------------
# 档案动态注册（ProfileRegistry）
# ---------------------------------------------------------------------------
def test_profile_registry_register_unregister(tmp_path):
    from app.orchestrator.profiles import AgentProfile
    from app.orchestrator.registry import ProfileRegistry, ProfileRegistryError

    settings = Settings(
        environment="test",
        agent_profiles_file=str(tmp_path / "profiles.json"),
        llm_provider="stub",
    )
    reg = ProfileRegistry(settings)
    assert reg.is_builtin("researcher")
    assert not reg.is_custom("researcher")

    # 注册自定义档案
    p = AgentProfile(
        name="frontend",
        description="前端专家：审查与修复前端代码",
        system_prompt="你是一名资深前端工程师。",
        allowed_tools=["file_read", "file_write"],
    )
    reg.register(p)
    assert reg.is_custom("frontend")
    assert "frontend" in reg.names()
    assert reg.get("frontend") is p

    # 内置档案不可覆盖 / 不可注销
    with pytest.raises(ProfileRegistryError):
        reg.register(AgentProfile(name="researcher", description="x", system_prompt="x"))
    with pytest.raises(ProfileRegistryError):
        reg.unregister("researcher")

    # 未知名字回退 generalist
    assert reg.get("not_exist").name == "generalist"

    # 持久化：重建注册表从文件恢复
    reg2 = ProfileRegistry(settings)
    assert reg2.is_custom("frontend")
    assert reg2.get("frontend").allowed_tools == ["file_read", "file_write"]

    # 注销 + 持久化
    assert reg2.unregister("frontend")
    reg3 = ProfileRegistry(settings)
    assert not reg3.is_custom("frontend")


def test_profile_registry_corrupt_file_ignored(tmp_path):
    from app.orchestrator.registry import ProfileRegistry

    f = tmp_path / "profiles.json"
    f.write_text("{ not valid json", encoding="utf-8")
    settings = Settings(environment="test", agent_profiles_file=str(f), llm_provider="stub")
    reg = ProfileRegistry(settings)
    assert reg.names() == ["researcher", "analyst", "writer", "generalist"]  # 内置不受影响


@pytest.mark.asyncio
async def test_runner_uses_dynamic_profile(orch_settings, registry, tmp_path):
    """动态注册的档案立即参与编排（planner 校验 + executor 执行）。"""
    from app.orchestrator.profiles import AgentProfile
    from app.orchestrator.registry import ProfileRegistry

    llm = ScriptedLLM(plan_json='{"steps": [{"agent": "frontend", "task": "审查前端", "depends_on": []}]}')
    reg = ProfileRegistry(orch_settings, profiles_file=str(tmp_path / "profiles.json"))
    reg.register(
        AgentProfile(
            name="frontend",
            description="前端专家",
            system_prompt="你是一名资深前端工程师。",
            allowed_tools=["calculator"],
        )
    )
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings, profile_registry=reg)
    result = await runner.run("审查前端代码")
    assert result.agent_results[0].agent == "frontend"
    assert result.agent_results[0].status == "SUCCEEDED"


# ---------------------------------------------------------------------------
# 委派结果持久化（SQLiteOrchestrationRepository）
# ---------------------------------------------------------------------------
def test_repository_save_and_query(tmp_path):
    from app.orchestrator.repository import OrchestrationRecord, SQLiteOrchestrationRepository
    from app.orchestrator.models import OrchestrationPlan, SubTask

    repo = SQLiteOrchestrationRepository(f"sqlite:///{tmp_path}/orch.db")
    rec = OrchestrationRecord(
        session_id="s_test",
        task="调研",
        status="SUCCEEDED",
        plan=OrchestrationPlan(
            rationale="r",
            steps=[SubTask(agent="researcher", task="查资料", depends_on=[])],
        ),
        final_answer="结论",
        duration_ms=123.4,
        trace_id="trace_1",
    )
    repo.save(rec)

    got = repo.get(rec.run_id)
    assert got is not None
    assert got.task == "调研"
    assert got.plan.steps[0].agent == "researcher"
    assert got.final_answer == "结论"

    by_session = repo.list_by_session("s_test")
    assert len(by_session) == 1
    assert by_session[0].run_id == rec.run_id

    # parent 链查询
    child = OrchestrationRecord(session_id="s_test", parent_run_id=rec.run_id, depth=2, task="子任务")
    repo.save(child)
    assert len(repo.list_children(rec.run_id)) == 1

    # 级联删除
    assert repo.delete_by_session("s_test") == 2
    assert repo.list_by_session("s_test") == []


@pytest.mark.asyncio
async def test_runner_persists_with_session(orch_settings, registry, tmp_path):
    """runner 注入 repository 后，编排结果按 session 持久化，嵌套编排挂 parent。"""
    from app.orchestrator.context import current_run_id, orchestration_depth
    from app.orchestrator.repository import SQLiteOrchestrationRepository

    repo = SQLiteOrchestrationRepository(f"sqlite:///{tmp_path}/orch.db")
    llm = ScriptedLLM()
    runner = OrchestratorRunner(llm=llm, registry=registry, settings=orch_settings, repository=repo)
    _registry_with_delegate(registry, runner)

    result = await runner.run("检索资料并委派给 writer 细化", agents=["researcher"], session_id="s_outer")
    assert result.run_id  # 持久化后回填

    # 顶层记录
    top = repo.get(result.run_id)
    assert top is not None
    assert top.session_id == "s_outer"
    assert top.depth == 1
    assert top.status == "SUCCEEDED"

    # 嵌套编排（writer 孙级）也持久化，且 parent_run_id 指向顶层
    children = repo.list_children(result.run_id)
    assert len(children) >= 1
    assert children[0].depth == 2
    assert children[0].session_id == "s_outer"

    # 按会话查询
    runs = repo.list_by_session("s_outer")
    assert len(runs) == 2
    assert {r.depth for r in runs} == {1, 2}


# ---------------------------------------------------------------------------
# Web：档案增删 + 编排记录查询
# ---------------------------------------------------------------------------
def test_web_agent_crud_and_orchestration_records(tmp_path):
    from fastapi.testclient import TestClient

    import fakeredis.aioredis

    from app.main import create_app

    settings = Settings(
        environment="test",
        trace_enabled=False,
        memory_enabled=False,
        skills_enabled=False,
        agent_mode="react",
        llm_provider="stub",
        database_url=f"sqlite:///{tmp_path}/web.db",
        trace_file=str(tmp_path / "traces.jsonl"),
        eval_run_dir=str(tmp_path / "runs"),
        orchestrator_enabled=True,
        orchestrator_planner_strategy="stub",
        agent_profiles_file=str(tmp_path / "profiles.json"),
    )
    app = create_app(settings, redis=fakeredis.aioredis.FakeRedis(decode_responses=True))
    with TestClient(app) as client:
        # 注册自定义档案
        r = client.post(
            "/api/web/agents",
            json={
                "name": "frontend",
                "description": "前端专家",
                "system_prompt": "你是一名资深前端工程师。",
                "allowed_tools": ["calculator"],
            },
        )
        assert r.status_code == 200
        assert "frontend" in r.json()["agents"]

        # 内置档案不可覆盖
        r409 = client.post(
            "/api/web/agents",
            json={"name": "researcher", "description": "x", "system_prompt": "x"},
        )
        assert r409.status_code == 409

        # 列表带 builtin 标记
        agents = client.get("/api/web/agents").json()
        by_name = {a["name"]: a for a in agents["agents"]}
        assert by_name["frontend"]["builtin"] is False
        assert by_name["researcher"]["builtin"] is True

        # 注销
        assert client.delete("/api/web/agents/frontend").status_code == 200
        assert client.delete("/api/web/agents/frontend").status_code == 404
        # 内置不可注销
        assert client.delete("/api/web/agents/researcher").status_code == 409

        # 编排记录：执行一次编排后按会话可查
        r = client.post("/api/web/orchestrate", json={"task": "查询北京天气", "agents": ["generalist"]})
        assert r.status_code == 200
        data = r.json()
        assert data["run_id"]

        runs = client.get("/api/web/orchestrations").json()
        assert runs["count"] >= 1
        assert runs["runs"][0]["run_id"] == data["run_id"]

        # 详情（含 plan + agent_results + children）
        detail = client.get(f"/api/web/orchestrations/{data['run_id']}").json()
        assert detail["task"] == "查询北京天气"
        assert detail["agent_results"][0]["agent"] == "generalist"
        assert "children" in detail

        # 不存在的 run
        assert client.get("/api/web/orchestrations/run_nope").status_code == 404
