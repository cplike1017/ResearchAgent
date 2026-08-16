"""
MemoryStore：记忆层的统一门面（AgentRuntime 只依赖这一个对象）。

组装：
    - embedding client（向量化）
    - repository（sqlite-vec 存储 / 检索）
    - extractor（回合后提炼事实）
    - retriever（检索）

AgentRuntime 视角只需两个操作：
    - remember(messages, session_id, turn_id) —— 回合结束后写入记忆
    - retrieve(query)                          —— 构建上下文前取回相关记忆
"""
from app.config import Settings
from app.llm.client import BaseLLMClient
from app.memory.embedding import BaseEmbeddingClient, create_embedding_client
from app.memory.extractor import MemoryExtractor
from app.memory.models import MemoryRecord
from app.memory.repository import SQLiteVecMemoryRepository
from app.memory.rerank import BaseReranker, create_reranker
from app.memory.retriever import MemoryRetriever
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span


class MemoryStore:
    """记忆层门面：写入 + 检索 + 可追踪。"""

    def __init__(
        self,
        repository: SQLiteVecMemoryRepository | None = None,
        embedding: BaseEmbeddingClient | None = None,
        extractor: MemoryExtractor | None = None,
        settings: Settings | None = None,
        recorder: TraceRecorder | None = None,
        reranker: BaseReranker | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.recorder = recorder  # None = 不追踪
        self.embedding = embedding or create_embedding_client(self.settings)
        # 仓库需要向量维度：优先取 embedding.dim（实际维度），兜底用配置
        dim = getattr(self.embedding, "dim", None) or self.settings.embedding_dim
        self.repository = repository or SQLiteVecMemoryRepository(
            self.settings.database_url, dim=dim
        )
        # extractor：显式注入优先；否则按配置创建（llm 策略需要主 LLM）
        self.extractor = extractor or MemoryExtractor(self.settings, llm=llm)
        # 重排器：显式注入优先；llm 策略需要主 LLM（AgentRuntime 注入）
        self.reranker = reranker
        if self.reranker is None and self.settings.memory_rerank_strategy != "off":
            self.reranker = create_reranker(self.settings, llm=llm)
        self.retriever = MemoryRetriever(
            self.repository, self.embedding, self.settings, reranker=self.reranker, llm=llm
        )
        # 已写入记忆的 turn_id 集合（幂等：防重试导致重复记忆）
        self._remembered_turns: set[str] = set()

    @property
    def enabled(self) -> bool:
        return self.settings.memory_enabled

    # ------------------------------------------------------------------
    # 写入：回合结束后调用
    # ------------------------------------------------------------------
    async def remember(
        self,
        messages: list[dict],
        *,
        session_id: str = "",
        turn_id: str = "",
    ) -> list[MemoryRecord]:
        """提炼回合事实并写入记忆（幂等：同 turn 不重复写）。"""
        if not self.enabled:
            return []
        # 已写过的 turn 跳过（防重试 / 重复调用导致重复记忆）
        if turn_id and turn_id in self._remembered_turns:
            return []
        if turn_id:
            self._remembered_turns.add(turn_id)

        facts = await self.extractor.extract(messages)
        if not facts:
            return []

        try:
            if self.recorder is None or not self.recorder.enabled:
                return await self._write_facts(facts, session_id, turn_id)

            async with trace_span(
                "memory.remember",
                "memory",
                input={"session_id": session_id, "turn_id": turn_id, "fact_count": len(facts)},
                recorder=self.recorder,
            ) as span:
                records = await self._write_facts(facts, session_id, turn_id)
                span.output = {"stored": len(records)}
                return records
        except Exception:
            return []  # 降级：写入失败不影响主流程

    async def _write_facts(
        self, facts: list[dict], session_id: str, turn_id: str
    ) -> list[MemoryRecord]:
        """把分类事实句向量化并批量写入仓库。

        :param facts: [{"text": str, "memory_type": str, "scope": "session|global"}]
        """
        texts = [f["text"] for f in facts]
        vectors = await self.embedding.embed(texts)
        records = [
            {
                "text": text,
                "embedding": vec,
                "memory_type": f.get("memory_type", "fact"),
                "scope": f.get("scope", "session"),
                "source_session_id": session_id,
                "source_turn_id": turn_id,
            }
            for text, vec, f in zip(texts, vectors, facts)
        ]
        return self.repository.add_batch(records)

    # ------------------------------------------------------------------
    # 检索：构建上下文前调用
    # ------------------------------------------------------------------
    async def retrieve(
        self, query: str, *, top_k: int | None = None, session_id: str | None = None
    ) -> list[str]:
        """返回相关记忆文本列表（注入 Context Builder retrieved_docs）。

        分层语义：session_id 给定 → 全局记忆 + 本会话会话级记忆；
        session_id 为空 → 仅全局记忆。

        降级语义：检索（embedding 服务 / 仓库）失败时不抛异常、返回空列表，
        保证记忆层是"可选增强"而非"单点故障" —— Agent 回合不应因记忆
        服务不可用而中断（记忆失败 → 本轮无记忆，继续执行）。
        """
        if not self.enabled:
            return []
        try:
            if self.recorder is None or not self.recorder.enabled:
                return await self.retriever.retrieve(query, top_k=top_k, session_id=session_id)

            async with trace_span(
                "memory.retrieve",
                "memory",
                input={"query": query, "session_id": session_id},
                recorder=self.recorder,
            ) as span:
                docs = await self.retriever.retrieve(query, top_k=top_k, session_id=session_id)
                span.output = {"hits": len(docs)}
                return docs
        except Exception:
            return []  # 降级：记忆不可用不影响主流程

    async def retrieve_records(
        self, query: str, *, top_k: int | None = None, session_id: str | None = None
    ) -> list[MemoryRecord]:
        """检索完整记录（含分数，测试 / demo 用）。"""
        if not self.enabled:
            return []
        return await self.retriever.retrieve_records(query, top_k=top_k, session_id=session_id)

    def close(self) -> None:
        """关闭仓库连接（embedding 连接池由上层负责）。"""
        self.repository.close()
