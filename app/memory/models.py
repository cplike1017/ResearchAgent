"""记忆数据模型。"""
from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    """一条语义记忆。

    - memory_id：唯一 ID
    - text：事实句 / 记忆内容（人类可读）
    - embedding：向量（由 EmbeddingClient 生成；存储时序列化为 bytes）
    - source_session_id / source_turn_id：来源（可追溯）
    - memory_type：fact(事实) | preference(偏好) | conclusion(结论)
    - scope：session(会话级，仅本会话检索可见) | global(全局，跨会话共享)
    - score：检索时填充的相似度（存储时为 0）
    - created_at：创建时间（时间衰减用）
    """

    memory_id: str
    text: str
    embedding: list[float] = Field(default_factory=list)
    source_session_id: str = ""
    source_turn_id: str = ""
    memory_type: str = "fact"  # fact | preference | conclusion
    scope: str = "session"  # session | global
    score: float = 0.0
    created_at: str = ""

    def to_search_result(self) -> "MemoryRecord":
        """检索结果只携带文本与分数（不携带大向量，节省返回体）。"""
        return MemoryRecord(
            memory_id=self.memory_id,
            text=self.text,
            source_session_id=self.source_session_id,
            source_turn_id=self.source_turn_id,
            memory_type=self.memory_type,
            scope=self.scope,
            score=self.score,
            created_at=self.created_at,
        )
