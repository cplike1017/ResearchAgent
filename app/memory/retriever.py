"""
记忆检索器（Retriever）：把"当前用户输入"变成"相关记忆"。

职责（Stage 8 Level 1 + Level 2）：
    1. 把查询文本向量化（embedding client）；
    2. 在记忆仓库粗召回 Top-K × candidate_multiplier（向量相似度 + 时间衰减）；
    3. 重排器精排（Level 2）：规则重排 / LLM 重排 / 跳过；
    4. 取最终 Top-K，过滤低分噪音（min_score）；
    5. 返回可直接注入 Context Builder `retrieved_docs` 的文本列表。

设计原则：
    - Retriever 只做"检索 + 精排"，不关心上下文怎么拼（那是 Context Builder 的事）；
    - 阈值 / Top-K / 半衰期 / 重排策略全部走 Settings，与项目配置风格一致。
"""
from app.config import Settings
from app.llm.client import BaseLLMClient
from app.memory.embedding import BaseEmbeddingClient
from app.memory.models import MemoryRecord
from app.memory.repository import SQLiteVecMemoryRepository
from app.memory.rerank import BaseReranker, create_reranker


class MemoryRetriever:
    """语义记忆检索器（粗召回 + 可选重排）。"""

    def __init__(
        self,
        repository: SQLiteVecMemoryRepository,
        embedding: BaseEmbeddingClient,
        settings: Settings | None = None,
        reranker: BaseReranker | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.repository = repository
        self.embedding = embedding
        self.settings = settings or Settings()
        # 重排器：显式注入优先，否则按配置创建（off -> None 跳过重排）
        self.reranker = reranker
        if self.reranker is None and self.settings.memory_rerank_strategy != "off":
            self.reranker = create_reranker(self.settings, llm=llm)

    # ------------------------------------------------------------------
    async def retrieve(
        self, query: str, *, top_k: int | None = None, session_id: str | None = None
    ) -> list[str]:
        """检索与查询相关的记忆，返回文本列表（供 retrieved_docs 注入）。

        分层语义：session_id 给定 → 全局记忆 + 本会话会话级记忆；
        session_id 为空 → 仅全局记忆（跨会话共享层）。
        """
        records = await self.retrieve_records(query, top_k=top_k, session_id=session_id)
        return [r.text for r in records]

    async def retrieve_records(
        self, query: str, *, top_k: int | None = None, session_id: str | None = None
    ) -> list[MemoryRecord]:
        """检索并返回完整记录（含分数，供 demo / 测试观察）。"""
        if not query or not query.strip():
            return []
        top_k = top_k or self.settings.memory_top_k
        vectors = await self.embedding.embed([query])
        if not vectors:
            return []

        # 1) 粗召回：top_k × candidate_multiplier，不做阈值过滤
        #    （min_score 过滤推迟到重排之后——重排本可以救回向量分低但关键词强相关的候选，
        #     过早过滤会把它们提前丢掉）
        multiplier = self.settings.memory_rerank_candidate_multiplier
        raw_top = max(top_k, top_k * multiplier)
        # 分层视角：有 session_id → global + 本会话 session 级；
        # 无 session_id → 仅 global 层（跨会话共享记忆）
        if session_id:
            candidates = self.repository.search(
                vectors[0],
                top_k=raw_top,
                min_score=-1.0,  # 粗召回阶段不过滤
                session_id=session_id,
            )
        else:
            candidates = self.repository.search(
                vectors[0],
                top_k=raw_top,
                min_score=-1.0,
                scope="global",
            )
        if not candidates:
            return []

        # 2) 精排（Level 2）：有重排器则重排，否则保持粗召回顺序
        if self.reranker is not None:
            candidates = await self.reranker.rerank(query, candidates)

        # 3) 精排后按阈值过滤噪音，再取最终 top_k
        threshold = self.settings.memory_min_score
        filtered = [c for c in candidates if c.score >= threshold]
        return filtered[:top_k]
