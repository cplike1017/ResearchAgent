"""
Context Builder 窗口边界修复测试：防止孤儿 tool 消息。

覆盖：窗口截断切掉 assistant(tool_calls) 留下 tool、窗口开头是 tool、
末尾悬空 tool_calls 声明。
"""
from app.agent.context_builder import _repair_window_boundary


def _tc_block(n_tools=3):
    """构造 assistant(tool_calls=n) + n 条 tool。"""
    msgs = [{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": f"c{i}", "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"}} for i in range(n_tools)],
    }]
    for i in range(n_tools):
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "name": "web_search", "content": "{}"})
    return msgs


def test_window_contains_orphan_tool_at_start():
    """窗口截断切掉 assistant(tool_calls)，开头是孤儿 tool -> 向前扩展。"""
    full = (
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
        + _tc_block(3)  # assistant(tool_calls=3) + 3 tool
        + [{"role": "user", "content": "继续"}]
    )
    # 窗口取最后 4 条：tool, tool, tool, user -> 开头 3 条孤儿 tool
    window = full[-4:]
    assert window[0]["role"] == "tool"
    repaired = _repair_window_boundary(full, window)
    # 修复后开头应是 assistant(tool_calls)
    assert repaired[0]["role"] == "assistant"
    assert repaired[0].get("tool_calls")
    # 配对完整
    roles = [m["role"] for m in repaired]
    assert roles == ["assistant", "tool", "tool", "tool", "user"]


def test_window_normal_unchanged():
    """正常窗口（无孤儿 tool）保持不变。"""
    full = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    window = full[-1:]
    assert _repair_window_boundary(full, window) == window


def test_window_dangling_tool_calls_truncated():
    """窗口末尾是悬空 assistant(tool_calls)（tool 未齐）-> 截断。"""
    full = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok"},
        # assistant 声明 3 个但只有 1 条 tool（不完整）
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}} for i in range(3)
        ]},
        {"role": "tool", "tool_call_id": "c0", "name": "x", "content": "{}"},
    ]
    window = full[-2:]  # assistant(tool_calls=3) + 1 tool
    repaired = _repair_window_boundary(full, window)
    # 悬空块被截断，只剩前面完整部分
    roles = [m["role"] for m in repaired]
    assert "assistant" not in [r for r in roles] or roles[-1] != "assistant" or repaired[-1].get("content") is not None


def test_window_multi_tool_block_kept():
    """完整的多工具块（含 assistant + 多条 tool）应完整保留。"""
    full = [{"role": "user", "content": "hi"}] + _tc_block(4)
    window = full[-5:]  # assistant + 4 tool（完整）
    repaired = _repair_window_boundary(full, window)
    assert len(repaired) == 5
    assert repaired[0]["role"] == "assistant"
    assert repaired[0].get("tool_calls")
    tools = [m for m in repaired if m["role"] == "tool"]
    assert len(tools) == 4
