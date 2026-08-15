"""
Stage 2 Demo：Context Builder。

运行：python -m demos.stage2_demo

展示目标：
    1. 完整历史消息数量（Session History 全量）
    2. Context Builder 实际选中数量（滑动窗口）
    3. Summary（历史超阈值时生成的摘要）
    4. Recent Messages（窗口内消息）
    5. 最终发送给 LLM 的 messages
    6. 估算 token 数

用真实 Agent 跑 12 轮对话累积历史，再交给 Context Builder 构建上下文，
证明 Session History != LLM Context。
"""
import asyncio
import json

from app.agent.context_builder import ContextBuilder
from app.agent.runtime import AgentRuntime
from app.config import get_settings
from app.llm.client import create_llm_client
from app.tools.builtin import build_default_registry

# 12 轮对话输入（混合无需工具 / 单工具 / 双工具场景）
TURN_INPUTS = [
    "你好",
    "计算 12 * 34",
    "查询北京天气",
    "你好",
    "计算 2 ** 10",
    "查询上海天气",
    "你好",
    "计算 100 / 4",
    "查询广州天气",
    "你好",
    "计算 7 * 8",
    "查询成都天气",
]


async def main() -> None:
    settings = get_settings()
    llm = create_llm_client(settings)
    registry = build_default_registry()
    builder = ContextBuilder(settings, llm=llm)
    runtime = AgentRuntime(llm=llm, registry=registry, context_builder=builder, settings=settings)

    # 1) 逐轮运行真实 Agent，累积完整历史（同一 session）
    history: list[dict] = []
    print(f"LLM Provider: {settings.llm_provider_resolved}")
    print(f"滑动窗口 N = {settings.max_context_messages}，压缩阈值 = {settings.context_summary_threshold}\n")
    for i, text in enumerate(TURN_INPUTS, 1):
        result = await runtime.run(text, session_id="stage2_demo_session")
        history.extend(result.messages)
        print(f"[第 {i:>2} 轮] 输入: {text:<12} -> 回合消息 {len(result.messages)} 条，历史累计 {len(history)} 条")

    print("\n" + "=" * 64)
    print(f"① 完整历史消息数量（Session History）: {len(history)} 条")

    # 2) 交给 Context Builder 构建模型输入
    built = await builder.build(history, registry.schemas())
    print(f"② Context Builder 实际选中数量（窗口内）: {built.selected} 条")
    print(f"③ Summary: {built.summary}")
    print(f"④ 估算 token 数: {built.estimated_tokens}")

    # 3) 展示窗口内的 Recent Messages
    print("\n⑤ Recent Messages（最近 {0} 条）:".format(built.selected))
    for m in built.messages[1 if built.summary is None else 2:]:
        content = m.get("content")
        if isinstance(content, str):
            print(f"   [{m['role']}] {content[:50]}")
        else:
            print(f"   [{m['role']}] (tool_calls 消息)")

    # 4) 展示最终发送给 LLM 的 messages 结构
    print(f"\n⑥ 最终发送给 LLM 的 messages（共 {len(built.messages)} 条）:")
    for i, m in enumerate(built.messages):
        content = m.get("content")
        preview = content[:40] + "..." if isinstance(content, str) and len(content) > 40 else content
        print(f"   [{i}] role={m['role']:<9} content={preview!r}")

    # 5) 证明：历史增长，但上下文有界
    print("\n" + "=" * 64)
    print("⑦ 结论：Session History 与 LLM Context 是两个东西。")
    print(f"   历史 {len(history)} 条 -> 上下文 {len(built.messages)} 条（system + 摘要 + 窗口）")


if __name__ == "__main__":
    asyncio.run(main())
