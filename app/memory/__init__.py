"""记忆层（Memory）包：语义记忆 / RAG。

Stage 8 目标：把 Context Builder 预留的 `retrieved_docs` 空位填上——
    - embedding.py   统一 Embedding 客户端（OpenAI-compatible + Stub 工厂，同 llm/client.py）
    - models.py      记忆记录模型
    - repository.py  向量存储 / 检索（sqlite-vec，复用现有 SQLite 思路）
    - extractor.py   回合结束后把消息提炼为事实句写入记忆
    - retriever.py   检索：Top-K 余弦相似度 + 时间衰减
"""
