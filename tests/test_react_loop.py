"""
Stage 1 测试：ReAct / Tool Loop。

覆盖验收点：
    你好            -> 不调用工具
    计算 123 * 456  -> 调用 calculator
    查询北京天气     -> 调用 get_weather
以及安全计算器、Schema 校验、死循环防护。
"""
import json

import pytest

from app.errors import AgentError, ToolExecutionError
from app.llm.client import LLMResponse, ToolCallRequest
from app.tools.builtin.calculator import safe_evaluate


# ---------------------------------------------------------------------------
# ReAct 循环：三种验收输入
# ---------------------------------------------------------------------------
async def test_greeting_does_not_call_tool(runtime):
    """「你好」不应调用任何工具，直接给出最终回答。"""
    result = await runtime.run("你好")
    assert result.tool_calls == []
    assert result.answer
    assert result.steps == 1  # 只调了一次 LLM


async def test_calculator_tool_call(runtime):
    """「计算 123 * 456」应调用 calculator 且参数正确。"""
    result = await runtime.run("计算 123 * 456")
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "calculator"
    assert tc.arguments == {"expression": "123 * 456"}
    assert "56088" in result.answer  # 123 * 456 = 56088
    assert result.steps == 2  # 一次工具调用 + 一次最终回答


async def test_weather_tool_call(runtime):
    """「查询北京天气」应调用 get_weather 且城市为北京。"""
    result = await runtime.run("查询北京天气")
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "北京"}
    assert "北京" in result.answer
    assert result.steps == 2


async def test_messages_evolution(runtime):
    """验证 Tool Loop 中 Messages 的完整演变序列。"""
    result = await runtime.run("查询北京天气")
    roles = [m["role"] for m in result.messages]
    # user -> assistant(带 tool_calls) -> tool -> assistant(最终回答)
    assert roles == ["user", "assistant", "tool", "assistant"]
    assistant_msg = result.messages[1]
    assert "tool_calls" in assistant_msg
    tool_msg = result.messages[2]
    assert tool_msg["tool_call_id"] == assistant_msg["tool_calls"][0]["id"]
    # 工具结果是统一信封 JSON
    envelope = json.loads(tool_msg["content"])
    assert envelope["success"] is True
    assert envelope["tool_name"] == "get_weather"


async def test_multi_tool_call(runtime):
    """多城市请求应产生多个工具调用（Stub 支持）。"""
    result = await runtime.run("同时查询北京和上海天气")
    assert [tc.name for tc in result.tool_calls] == ["get_weather", "get_weather"]
    assert [tc.arguments["city"] for tc in result.tool_calls] == ["北京", "上海"]


# ---------------------------------------------------------------------------
# 死循环防护
# ---------------------------------------------------------------------------
async def test_max_steps_protection(runtime, stub_llm):
    """模型永远发起工具调用时，循环必须在 max_steps 处终止。"""

    class AlwaysToolLLM:
        """模拟失控模型：永远要求调用工具。"""

        async def chat(self, messages, tools=None, **kwargs):
            return LLMResponse(
                tool_calls=[ToolCallRequest(id="x", name="calculator", arguments={"expression": "1+1"})],
                finish_reason="tool_calls",
            )

    runtime.llm = AlwaysToolLLM()
    with pytest.raises(AgentError) as exc:
        await runtime.run("计算 1+1")
    assert exc.value.code == "MAX_STEPS_EXCEEDED"


# ---------------------------------------------------------------------------
# 安全计算器（禁止 eval/exec/shell）
# ---------------------------------------------------------------------------
def test_safe_evaluate_basic():
    assert safe_evaluate("123 * 456") == 56088.0
    assert safe_evaluate("1 + 2 * 3") == 7.0
    assert safe_evaluate("(1 + 2) ** 2") == 9.0
    assert safe_evaluate("10 // 3") == 3.0
    assert safe_evaluate("10 % 3") == 1.0


def test_safe_evaluate_rejects_dangerous_input():
    # 函数调用 / 属性访问 / 变量名全部拒绝
    with pytest.raises(ToolExecutionError):
        safe_evaluate("__import__('os').system('echo hi')")
    with pytest.raises(ToolExecutionError):
        safe_evaluate("os.system('echo hi')")
    with pytest.raises(ToolExecutionError):
        safe_evaluate("[1,2,3]")
    with pytest.raises(ToolExecutionError):
        safe_evaluate("lambda: 1")


def test_safe_evaluate_division_by_zero():
    with pytest.raises(ToolExecutionError):
        safe_evaluate("1 / 0")
    with pytest.raises(ToolExecutionError):
        safe_evaluate("1 // 0")


def test_safe_evaluate_syntax_error():
    with pytest.raises(ToolExecutionError):
        safe_evaluate("1 +")


# ---------------------------------------------------------------------------
# Registry 层 Schema 校验（Gateway 前置形态）
# ---------------------------------------------------------------------------
async def test_registry_schema_validation(registry):
    """错误参数（city 为数字）必须在入口被拦截。"""
    result = await registry.execute("get_weather", {"city": 123})
    assert result.success is False
    assert result.error.type == "ToolValidationError"


async def test_registry_unknown_tool(registry):
    result = await registry.execute("no_such_tool", {})
    assert result.success is False
    assert result.error.type == "ToolError"


async def test_registry_weather_unknown_city(registry):
    """未知城市触发业务错误（Tool Error 场景）。"""
    result = await registry.execute("get_weather", {"city": "东京"})
    assert result.success is False
    assert result.error.type == "ToolExecutionError"
    assert "东京" in result.error.message


async def test_registry_calculator_ok(registry):
    result = await registry.execute("calculator", {"expression": "2 ** 10"})
    assert result.success is True
    assert result.data == 1024.0
