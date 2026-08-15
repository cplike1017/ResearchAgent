"""
Stage 8 Demo：记忆层（Memory / RAG）。

运行：python -m demos.stage8_demo

展示：
    1. Embedding 客户端（真实 qwen3.7-text-embedding 或 Stub）
    2. 回合结束后自动提炼事实写入记忆
    3. 第二轮检索到第一轮的记忆并注入 Context（system prompt 含参考资料）
    4. Trace 中可见 memory.remember / memory.retrieve Span（若开启 Tracing）

离线跑：不配置 .env 的 embedding 时自动用 Stub（hash 伪向量），
真实语义检索需在 .env 配置 EMBEDDING_BASE_URL / EMBEDDING_API_KEY / EMBEDDING_MODEL。
"""
import asyncio

from app.agent.runtime import AgentRuntime
from app.config import get_settings
from app.llm.client import create_llm_client
from app.memory.store import MemoryStore
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry
from app.tracing.recorder import TraceRecorder

SEPARATOR = "=" * 64


async def main() -> None:
    settings = get_settings()
    recorder = TraceRecorder(
        settings.trace_file, enabled=True, capture_content=settings.trace_capture_content
    )

    print(SEPARATOR)
    print("Stage 8 记忆层 Demo")
    print(SEPARATOR)
    print(f"LLM provider     : {settings.llm_provider_resolved} / {settings.llm_model}")
    print(f"Embedding provider: {settings.embedding_provider_resolved} / {settings.embedding_model}")
    print(f"Memory enabled   : {settings.memory_enabled}")
    if settings.embedding_provider_resolved == "stub":
        print("  ⚠ 当前用 Stub 伪向量（离线可跑），真实语义检索请配置 .env 的 EMBEDDING_*")
    print()

    # 组装完整运行时（Session + 记忆）
    llm = create_llm_client(settings)
    # 记忆层：llm 传入用于 LLM 重排策略（memory_rerank_strategy=llm 时生效）
    memory = MemoryStore(settings=settings, recorder=recorder, llm=llm)
    session_repo = SQLiteSessionRepository(settings.database_url)
    runtime = AgentRuntime(
        llm=llm,
        registry=build_default_registry(),
        session_repo=session_repo,
        memory=memory,
        recorder=recorder,
        settings=settings,
    )

    session_id = "stage8_demo_session"

    # ---- 第一轮：正常问答（回合结束后自动写入记忆）----
    print(SEPARATOR)
    print("1) 第一轮对话（回合结束后自动提炼事实写入记忆）")
    print(SEPARATOR)
    r1 = await runtime.run("查询北京天气", session_id=session_id)
    print(f"回答: {r1.answer}")
    print(f"记忆条数: {memory.repository.count()}")
    print("\n已存记忆：")
    for m in memory.repository.list_all():
        print(f"  [{m.memory_type}] {m.text}")

    # ---- 第二轮：换一个相关话题，验证记忆注入 ----
    print("\n" + SEPARATOR)
    print("2) 第二轮对话（构建上下文前检索记忆并注入）")
    print(SEPARATOR)
    r2 = await runtime.run("那上海呢", session_id=session_id)
    print(f"回答: {r2.answer}")

    # ---- 展示检索分数 ----
    print("\n" + SEPARATOR)
    print("3) 检索结果（含相似度分数，验证记忆被找回）")
    print(SEPARATOR)
    print(f"重排策略: {settings.memory_rerank_strategy}（粗召回 × {settings.memory_rerank_candidate_multiplier} 再精排）")
    records = await memory.retrieve_records("查询北京天气", top_k=5)
    for r in records:
        print(f"  score={r.score:.4f}  {r.text}")

    print("\nDemo 完成。")
    memory.close()


if __name__ == "__main__":
    asyncio.run(main())
