"""
Stage 5 测试：Tool Gateway / Policy / 权限 / 超时 / 结果校验。

验收：分别测试 正常 / Schema 错误 / 权限拒绝 / 策略拒绝 / 超时 / 工具内部异常，
每种情况都有明确的结构化结果（ToolResult 信封）。
"""
import time

import pytest
from pydantic import BaseModel, Field

from app.agent.runtime import AgentRuntime
from app.errors import ToolExecutionError
from app.tools.builtin import build_default_registry
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolDefinition
from app.tools.schemas import UserContext


# ---------------------------------------------------------------------------
# 测试专用工具
# ---------------------------------------------------------------------------
class AdminArgs(BaseModel):
    message: str = Field(description="消息")


class DangerArgs(BaseModel):
    target: str = Field(description="目标")


class SlowArgs(BaseModel):
    delay: float = Field(description="延迟秒数")


class EmptyArgs(BaseModel):
    pass


class OutputModel(BaseModel):
    result: str = Field(description="必须为字符串")


def build_test_registry():
    """默认工具 + 权限/策略/超时/异常/瞬时重试等测试工具。"""
    reg = build_default_registry()

    reg.register(
        ToolDefinition(
            name="admin_tool",
            description="需要 admin 权限的工具",
            input_model=AdminArgs,
            handler=lambda message: f"admin 执行: {message}",
            required_permission="admin",
        )
    )
    reg.register(
        ToolDefinition(
            name="danger_tool",
            description="高风险工具（触发 Policy）",
            input_model=DangerArgs,
            handler=lambda target: f"危险操作: {target}",
            risk_level="high",
        )
    )
    reg.register(
        ToolDefinition(
            name="slow_tool",
            description="慢工具（触发超时）",
            input_model=SlowArgs,
            handler=lambda delay: (time.sleep(delay) or "slow done"),
            timeout_seconds=0.1,
        )
    )
    # 瞬时错误：前两次失败，第三次成功（演示 Gateway 重试）
    flaky_state = {"n": 0}

    def flaky_handler():
        flaky_state["n"] += 1
        if flaky_state["n"] <= 2:
            raise ToolExecutionError("外部服务抖动", transient=True)
        return "flaky ok"

    reg.register(ToolDefinition(name="flaky_tool", description="瞬时抖动工具", input_model=EmptyArgs, handler=flaky_handler))
    reg.register(
        ToolDefinition(
            name="explode_tool",
            description="内部抛未知异常的工具",
            input_model=EmptyArgs,
            handler=lambda: (_ for _ in ()).throw(ValueError("内部错误")),
        )
    )
    reg.register(
        ToolDefinition(
            name="bad_output_tool",
            description="返回值不符合声明的输出模型",
            input_model=EmptyArgs,
            handler=lambda: {"result": 123},  # 声明为 str，实际返回 int
            output_model=OutputModel,
        )
    )
    return reg


@pytest.fixture
def gateway(settings):
    return ToolGateway(build_test_registry(), settings=settings)


# ---------------------------------------------------------------------------
# 正常执行
# ---------------------------------------------------------------------------
async def test_normal_tool_success(gateway):
    result = await gateway.execute("get_weather", {"city": "北京"}, user=UserContext(user_id="u1"))
    assert result.success is True
    assert result.tool_name == "get_weather"
    assert "北京" in str(result.data) or "晴" in str(result.data)
    assert result.error is None
    assert "duration_ms" in result.metadata


# ---------------------------------------------------------------------------
# Schema 校验
# ---------------------------------------------------------------------------
async def test_schema_validation_error(gateway):
    """city=123 是错误参数，必须在 Gateway 层被拦截。"""
    result = await gateway.execute("get_weather", {"city": 123})
    assert result.success is False
    assert result.error.type == "ToolValidationError"
    assert result.data is None


async def test_missing_required_field(gateway):
    result = await gateway.execute("get_weather", {})
    assert result.success is False
    assert result.error.type == "ToolValidationError"


async def test_unknown_tool(gateway):
    result = await gateway.execute("no_such_tool", {})
    assert result.success is False
    assert result.error.type == "ToolError"


# ---------------------------------------------------------------------------
# 权限（Permission）
# ---------------------------------------------------------------------------
async def test_permission_denied(gateway):
    result = await gateway.execute(
        "admin_tool", {"message": "hi"}, user=UserContext(user_id="u1", permissions=["user"])
    )
    assert result.success is False
    assert result.error.type == "ToolPermissionError"
    assert "admin" in result.error.message


async def test_permission_allowed(gateway):
    result = await gateway.execute(
        "admin_tool", {"message": "hi"}, user=UserContext(user_id="admin", permissions=["admin"])
    )
    assert result.success is True
    assert result.data == "admin 执行: hi"


async def test_permission_ignored_when_no_requirement(gateway):
    """无 required_permission 的工具不受权限限制。"""
    result = await gateway.execute("calculator", {"expression": "1+1"}, user=None)
    assert result.success is True


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
async def test_policy_high_risk_requires_confirmation(gateway, settings):
    """高风险工具（配置 high 需确认）-> REQUIRE_CONFIRMATION -> 无确认通道按拒绝处理。"""
    from app.tools.policy import PolicyEngine

    reg = build_test_registry()
    gw = ToolGateway(
        reg,
        policy_engine=PolicyEngine(
            require_confirmation_risks=["high"],
            settings=settings.model_copy(update={"policy_require_confirmation_risks": "high"}),
        ),
        settings=settings.model_copy(update={"policy_require_confirmation_risks": "high"}),
    )
    result = await gw.execute("danger_tool", {"target": "x"})
    assert result.success is False
    assert result.error.type == "ToolPolicyError"
    assert "确认" in result.error.message


async def test_policy_deny_list(gateway):
    """黑名单策略：构造一个被禁的工具。"""
    reg = build_test_registry()
    from app.tools.policy import PolicyEngine

    gw = ToolGateway(reg, policy_engine=PolicyEngine(denied_tools=["calculator"]), settings=None)
    result = await gw.execute("calculator", {"expression": "1+1"})
    assert result.success is False
    assert result.error.type == "ToolPolicyError"
    assert "黑名单" in result.error.message


# ---------------------------------------------------------------------------
# 超时（Timeout）
# ---------------------------------------------------------------------------
async def test_tool_timeout(gateway):
    """slow_tool 睡 0.5s，超时 0.1s -> ToolTimeoutError。"""
    result = await gateway.execute("slow_tool", {"delay": 0.5})
    assert result.success is False
    assert result.error.type == "ToolTimeoutError"
    assert "0.1s" in result.error.message
    # 超时时间应接近 timeout_seconds
    assert result.metadata["duration_ms"] < 500


# ---------------------------------------------------------------------------
# 工具内部异常
# ---------------------------------------------------------------------------
async def test_tool_internal_exception_wrapped(gateway):
    """未知异常被包装为 ToolExecutionError，不向上抛。"""
    result = await gateway.execute("explode_tool", {})
    assert result.success is False
    assert result.error.type == "ToolExecutionError"
    assert "内部错误" in result.error.message


async def test_business_error_non_transient_no_retry(gateway):
    """确定性业务错误（未知城市）不重试，直接失败。"""
    result = await gateway.execute("get_weather", {"city": "东京"})
    assert result.success is False
    assert result.error.type == "ToolExecutionError"
    assert result.metadata.get("retries") == 0


# ---------------------------------------------------------------------------
# 瞬时错误重试
# ---------------------------------------------------------------------------
async def test_transient_error_retry(gateway):
    result = await gateway.execute("flaky_tool", {})
    assert result.success is True
    assert result.data == "flaky ok"
    assert result.metadata.get("retries") == 2  # 重试了 2 次


# ---------------------------------------------------------------------------
# Result Validation
# ---------------------------------------------------------------------------
async def test_result_validation_failure(gateway):
    result = await gateway.execute("bad_output_tool", {})
    assert result.success is False
    assert result.error.type == "ToolExecutionError"
    assert "返回值" in result.error.message


# ---------------------------------------------------------------------------
# 通过 Runtime 端到端：权限随用户上下文生效
# ---------------------------------------------------------------------------
def _registry_with_weather_permission():
    """给内置 get_weather 加 required_permission="weather:read"。"""
    reg = build_test_registry()
    weather = reg.get("get_weather")
    reg.register(
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
    return reg


async def test_runtime_permission_denied_end_to_end(settings, stub_llm):
    """用户无 weather:read 权限 -> Agent 拿到权限拒绝的工具结果并如实回答。"""
    reg = _registry_with_weather_permission()
    rt = AgentRuntime(llm=stub_llm, registry=reg, settings=settings)
    result = await rt.run("查询北京天气", user=UserContext(user_id="u1", permissions=[]))
    assert len(result.tool_calls) == 1
    assert "缺少权限" in result.answer


async def test_runtime_permission_allowed_end_to_end(settings, stub_llm):
    reg = _registry_with_weather_permission()
    rt = AgentRuntime(llm=stub_llm, registry=reg, settings=settings)
    result = await rt.run("查询北京天气", user=UserContext(user_id="u1", permissions=["weather:read"]))
    assert "北京" in result.answer
