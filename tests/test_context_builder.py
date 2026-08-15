"""
Stage 2 测试：Context Builder（滑动窗口 + 压缩 + token 估算）。

核心验收：Session History != LLM Context，且历史越来越长时 LLM Context 不无限增长。
"""
import pytest

from app.agent.context_builder import (
    ContextBuilder,
    estimate_messages_tokens,
)
from app.agent.runtime import AgentRuntime
from app.config import Settings


def _make_history(n: int) -> list[dict]:
    """构造 n 条交替 user/assistant 的历史消息。"""
    history = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        history.append({"role": role, "content": f"消息第 {i} 条"})
    return history


def _builder(settings: Settings) -> ContextBuilder:
    return ContextBuilder(settings)


# ---------------------------------------------------------------------------
# 滑动窗口
# ---------------------------------------------------------------------------
async def test_window_selects_recent_messages():
    """窗口 N=10 时只选中最近 10 条。"""
    settings = Settings(max_context_messages=10, context_summary_threshold=100)  # 阈值调高，避免压缩干扰
    builder = _builder(settings)
    history = _make_history(20)
    built = await builder.build(history, [])
    assert built.total_history == 20
    assert built.selected == 10
    assert len(built.messages) == 1 + 10  # system + 窗口内 10 条
    assert built.summary is None
    # 最后一条历史消息必须出现在上下文中
    assert built.messages[-1] == history[-1]


async def test_window_tail_order_preserved():
    """窗口内消息顺序必须与历史一致。"""
    settings = Settings(max_context_messages=3, context_summary_threshold=100)
    builder = _builder(settings)
    history = _make_history(10)
    built = await builder.build(history, [])
    selected = built.messages[1:]
    assert [m["content"] for m in selected] == [f"消息第 {i} 条" for i in range(7, 10)]


# ---------------------------------------------------------------------------
# 压缩
# ---------------------------------------------------------------------------
async def test_compression_triggered_over_threshold():
    """历史条数超过阈值 -> 生成 summary。"""
    settings = Settings(max_context_messages=10, context_summary_threshold=16)
    builder = _builder(settings)
    history = _make_history(20)
    built = await builder.build(history, [])
    assert built.summary is not None
    assert "历史共 20 条消息" in built.summary
    # 摘要作为一条 system 消息进入上下文
    assert any(m.get("role") == "system" and "[历史摘要]" in m["content"] for m in built.messages)


async def test_no_compression_below_threshold():
    settings = Settings(max_context_messages=10, context_summary_threshold=100)
    builder = _builder(settings)
    built = await builder.build(_make_history(12), [])
    assert built.summary is None


async def test_compression_strategy_off():
    """strategy=off 时即使超阈值也不压缩。"""
    settings = Settings(max_context_messages=10, context_summary_threshold=5, context_summary_strategy="off")
    builder = _builder(settings)
    built = await builder.build(_make_history(20), [])
    assert built.summary is None


async def test_stub_summary_deterministic():
    """确定性摘要：相同输入 -> 相同输出。"""
    settings = Settings(max_context_messages=10, context_summary_threshold=2)
    builder = _builder(settings)
    history = _make_history(20)
    s1 = await builder.compress_history(history)
    s2 = await builder.compress_history(history)
    assert s1 == s2
    assert "user" in s1 and "assistant" in s1


# ---------------------------------------------------------------------------
# 核心验收：历史增长时上下文有界
# ---------------------------------------------------------------------------
async def test_context_does_not_grow_with_history():
    """历史从 10 条涨到 200 条，送入模型的上下文消息数保持有界。"""
    settings = Settings(max_context_messages=5, context_summary_threshold=8)
    builder = _builder(settings)
    sizes = []
    for n in (10, 20, 50, 100, 200):
        built = await builder.build(_make_history(n), [])
        sizes.append(len(built.messages))
        assert built.total_history == n
        # system + 摘要 + 窗口(5) => 恒 <= 7
        assert len(built.messages) <= 7
    # 上下文大小几乎恒定，而历史在增长
    assert max(sizes) - min(sizes) <= 1


# ---------------------------------------------------------------------------
# token 估算
# ---------------------------------------------------------------------------
def test_token_estimation():
    assert estimate_messages_tokens([]) == 0
    assert estimate_messages_tokens([{"role": "user", "content": "你好"}]) >= 1
    long_messages = [{"role": "user", "content": "x" * 400}]
    short_messages = [{"role": "user", "content": "x"}]
    assert estimate_messages_tokens(long_messages) > estimate_messages_tokens(short_messages)


# ---------------------------------------------------------------------------
# 与运行时集成：每轮 Tool Call 后重新构建上下文
# ---------------------------------------------------------------------------
async def test_builder_rebuilds_each_loop_iteration(runtime, stub_llm, settings):
    """证明 Context Builder 在每轮 Tool Call 之后重新运行（模型能看到工具结果）。"""
    recorded: list[list[dict]] = []

    class RecordingLLM:
        """记录每次真正送入模型的 messages。"""

        async def chat(self, messages, tools=None, **kwargs):
            recorded.append([dict(m) for m in messages])
            return await stub_llm.chat(messages, tools, **kwargs)

    builder = ContextBuilder(settings)
    rt = AgentRuntime(llm=RecordingLLM(), registry=runtime.registry, context_builder=builder, settings=settings)

    result = await rt.run("查询北京天气")
    assert result.steps == 2  # 两次 LLM 调用
    assert len(recorded) == 2
    # 第一次请求：只有 user + system
    assert recorded[0][0]["role"] == "system"
    assert recorded[0][-1]["role"] == "user"
    # 第二次请求：工具结果已经进入上下文（说明 builder 在 Tool Call 后重新运行）
    assert any(m["role"] == "tool" for m in recorded[1])
    assert recorded[1][0]["role"] == "system"


async def test_system_prompt_present(runtime):
    """最终上下文第一条必须是 system 提示。"""
    result = await runtime.run("你好")
    assert result.messages[0]["role"] == "user"  # 历史里是 user 开头
    # 通过 builder 构建后第一条是 system
    built = await runtime.context_builder.build(result.messages, runtime.registry.schemas())
    assert built.messages[0]["role"] == "system"
    assert "可用工具" in built.messages[0]["content"]
