"""
修复验证：tool 配对完整性（No tool output found 缺陷）。

覆盖：
    - _truncate_incomplete_tool_block：持久化时截断不完整块
    - _repair_tool_pairing：加载历史时补充缺失的 tool 占位
    - react_loop 取消时补 tool 失败消息（配对完整）
"""
import asyncio

import pytest

from app.agent.runtime import AgentRuntime, _repair_tool_pairing, _truncate_incomplete_tool_block
from app.config import Settings
from app.llm.client import StubLLMClient
from app.tools.builtin import build_default_registry


def _tc_msg(n, declared=4, present=3):
    """构造 assistant(tool_calls=n) + present 条 tool 的消息块。"""
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}
            for i in range(declared)
        ]}
    ]
    for i in range(present):
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "web_search", "content": "{}"})
    return msgs


# ---------------------------------------------------------------------------
# _truncate_incomplete_tool_block
# ---------------------------------------------------------------------------
def test_truncate_keeps_complete_block():
    msgs = [{"role": "user", "content": "hi"}] + _tc_msg(2, declared=2, present=2)
    out = _truncate_incomplete_tool_block(msgs)
    assert len(out) == 4  # user + assistant + 2 tool，完整保留


def test_truncate_removes_incomplete_block():
    msgs = [{"role": "user", "content": "hi"}] + _tc_msg(2, declared=4, present=3)
    out = _truncate_incomplete_tool_block(msgs)
    assert len(out) == 1  # 只保留 user（不完整块被截断）


def test_truncate_no_tool_messages():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert _truncate_incomplete_tool_block(msgs) == msgs


# ---------------------------------------------------------------------------
# _repair_tool_pairing
# ---------------------------------------------------------------------------
def test_repair_missing_tool_output():
    """assistant 声明 4 个调用但只有 3 条 tool -> 补 1 条失败占位。"""
    msgs = [{"role": "user", "content": "调研"}] + _tc_msg(1, declared=4, present=3)
    out = _repair_tool_pairing(msgs)
    # user + assistant + 4 tool
    assert len(out) == 6
    tool_ids = [m.get("tool_call_id") for m in out if m.get("role") == "tool"]
    assert set(tool_ids) == {"c0", "c1", "c2", "c3"}  # 4 个都配对
    # 补充的 c3 占位是失败信封
    import json

    placeholder = [m for m in out if m.get("role") == "tool" and m.get("tool_call_id") == "c3"][0]
    env = json.loads(placeholder["content"])
    assert env["success"] is False
    assert env["error"]["code"] == "MISSING_TOOL_OUTPUT"


def test_repair_complete_block_unchanged():
    msgs = [{"role": "user", "content": "hi"}] + _tc_msg(1, declared=2, present=2)
    out = _repair_tool_pairing(msgs)
    assert len(out) == len(msgs)


def test_repair_no_tool_calls_unchanged():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "bye"}]
    assert _repair_tool_pairing(msgs) == msgs


def test_repair_merges_consecutive_users():
    """连续 user 消息（多轮"继续"）合并为一条，保证 user/assistant 交替。"""
    msgs = [
        {"role": "user", "content": "介绍SAC"},
        {"role": "assistant", "content": "SAC是..."},
        {"role": "user", "content": "继续"},
        {"role": "user", "content": "继续"},
        {"role": "user", "content": "继续"},
    ]
    out = _repair_tool_pairing(msgs)
    roles = [m["role"] for m in out]
    assert roles == ["user", "assistant", "user"]
    # 保留最后一条 user（"继续"）
    assert out[-1]["content"] == "继续"


def test_repair_full_polluted_history():
    """完整污染场景：tool 缺失 + 连续 user 同时修复。"""
    msgs = (
        [{"role": "user", "content": "介绍SAC"}, {"role": "assistant", "content": "SAC是..."}]
        + _tc_msg(1, declared=4, present=3)  # assistant(tool_calls=4) + 3 tool
        + [{"role": "user", "content": "继续"}, {"role": "user", "content": "继续"}]
    )
    out = _repair_tool_pairing(msgs)
    roles = [m["role"] for m in out]
    # 期望：user, assistant, assistant(tool_calls), tool×4, user
    assert roles == ["user", "assistant", "assistant", "tool", "tool", "tool", "tool", "user"]
    # tool 配对完整
    assistant = [m for m in out if m.get("tool_calls")][0]
    tools = [m for m in out if m.get("role") == "tool"]
    assert len(tools) == len(assistant["tool_calls"]) == 4


# ---------------------------------------------------------------------------
# react_loop：工具执行被取消时补 tool 失败消息
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_react_loop_cancel_completes_pairing():
    """工具执行抛 CancelledError 时，补一条 tool 失败消息再重抛。"""
    from app.agent.react_loop import run_react_loop
    from app.llm.client import LLMResponse, ToolCallRequest

    class CancelLLM(StubLLMClient):
        """第一次返回 4 个工具调用，之后不再被调用（会先被取消）。"""

        async def chat(self, messages, tools=None, **kwargs):
            return LLMResponse(
                tool_calls=[
                    ToolCallRequest(id=f"c{i}", name="web_search", arguments={}) for i in range(4)
                ],
                finish_reason="tool_calls",
            )

    calls = 0

    async def execute_tool(name, args):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise asyncio.CancelledError()  # 第 4 个被取消
        from app.tools.schemas import ToolResult

        return ToolResult.ok(name, "result")

    messages = [{"role": "user", "content": "调研"}]
    with pytest.raises(asyncio.CancelledError):
        await run_react_loop(
            llm=CancelLLM(),
            tools_schema=[],
            messages=messages,
            execute_tool=execute_tool,
            max_steps=5,
        )

    # 断言：assistant(tool_calls=4) + 4 条 tool（第 4 条是失败占位）
    assistant = [m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")]
    tools = [m for m in messages if m.get("role") == "tool"]
    assert len(assistant) == 1
    assert len(assistant[0]["tool_calls"]) == 4
    assert len(tools) == 4  # 配对完整！
    import json

    env = json.loads(tools[-1]["content"])
    assert env["success"] is False
    assert env["error"]["code"] == "CANCELLED"
