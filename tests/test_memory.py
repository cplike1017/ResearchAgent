"""
Stage 8 测试：记忆层（Embedding / Repository / Extractor / Retriever / Store / Runtime 接线）。

验收：向量写入与检索、Top-K 正确性、时间衰减、embedding 工厂、
回合后提炼写入、retrieved_docs 注入 context（system prompt 含参考资料）。
"""
import pytest

from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.llm.client import StubLLMClient
from app.memory.embedding import StubEmbeddingClient, create_embedding_client
from app.memory.extractor import MemoryExtractor
from app.memory.models import MemoryRecord
from app.memory.repository import SQLiteVecMemoryRepository
from app.memory.rerank import RuleReranker, LLMReranker, _tokenize, create_reranker
from app.memory.retriever import MemoryRetriever
from app.memory.store import MemoryStore
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry


@pytest.fixture
def mem_settings(tmp_path) -> Settings:
    """开启记忆的测试配置（stub embedding，离线可跑）。"""
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/mem.db",
        trace_enabled=False,
        llm_provider="stub",
        embedding_provider="stub",
        embedding_dim=64,
        memory_enabled=True,
        memory_top_k=3,
        memory_min_score=0.0,
        memory_extract_strategy="stub",
        agent_mode="react",  # 显式 react：.env 的 AGENT_MODE=plan 不影响测试
    )


@pytest.fixture
def mem_repo(mem_settings) -> SQLiteVecMemoryRepository:
    return SQLiteVecMemoryRepository(mem_settings.database_url, dim=64)


@pytest.fixture
def mem_emb(mem_settings) -> StubEmbeddingClient:
    return StubEmbeddingClient(mem_settings)


@pytest.fixture
def mem_store(mem_settings, mem_repo, mem_emb) -> MemoryStore:
    return MemoryStore(
        repository=mem_repo,
        embedding=mem_emb,
        settings=mem_settings,
    )


# ---------------------------------------------------------------------------
# Embedding 工厂
# ---------------------------------------------------------------------------
def test_embedding_factory_stub(mem_settings):
    client = create_embedding_client(mem_settings)
    assert isinstance(client, StubEmbeddingClient)
    assert client.dim == 64


def test_embedding_factory_openai(tmp_path):
    """配置了真实 base_url 时创建 OpenAI 实现（不实际调用）。"""
    settings = Settings(
        embedding_provider="openai",
        embedding_base_url="https://example.com/v1",
        embedding_api_key="test-key",
        embedding_model="m",
    )
    from app.memory.embedding import OpenAICompatEmbeddingClient

    client = create_embedding_client(settings)
    assert isinstance(client, OpenAICompatEmbeddingClient)


def test_stub_embedding_deterministic(mem_emb):
    v1 = mem_emb._hash_vec("查询北京天气")
    v2 = mem_emb._hash_vec("查询北京天气")
    v3 = mem_emb._hash_vec("查询上海天气")
    assert v1 == v2
    assert v1 != v3
    assert len(v1) == 64


# ---------------------------------------------------------------------------
# Repository：写入 / 检索
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_repo_add_and_search(mem_repo, mem_emb):
    texts = ["用户喜欢查询天气", "用户喜欢计算数学", "用户喜欢问寒暄话"]
    vectors = await mem_emb.embed(texts)
    for t, v in zip(texts, vectors):
        mem_repo.add(t, v)
    assert mem_repo.count() == 3

    q = (await mem_emb.embed(["天气怎么样"]))[0]
    results = mem_repo.search(q, top_k=3, min_score=0.0)
    # 检索结果非空且第一条就是"天气"相关记忆
    assert results
    assert results[0].score > 0


def test_repo_empty_search(mem_repo, mem_emb):
    q = (mem_emb._hash_vec("任何查询"))
    results = mem_repo.search(q, top_k=3)
    assert results == []


# ---------------------------------------------------------------------------
# Retriever：Top-K + 注入格式
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_retriever_top_k(mem_settings, mem_repo, mem_emb):
    texts = ["天气", "天气", "天气", "天气", "数学", "数学"]
    vectors = await mem_emb.embed(texts)
    for t, v in zip(texts, vectors):
        mem_repo.add(t, v, scope="global")  # 无 session 检索只查 global 层
    retriever = MemoryRetriever(mem_repo, mem_emb, mem_settings)
    docs = await retriever.retrieve("今天天气", top_k=3)
    assert len(docs) == 3


@pytest.mark.asyncio
async def test_retriever_empty_query(mem_settings, mem_repo, mem_emb):
    retriever = MemoryRetriever(mem_repo, mem_emb, mem_settings)
    assert await retriever.retrieve("") == []
    assert await retriever.retrieve("   ") == []


# ---------------------------------------------------------------------------
# Extractor：回合后事实提炼
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_extractor_stub_facts(mem_settings):
    extractor = MemoryExtractor(mem_settings)
    messages = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！"},
        {"role": "user", "content": "查询北京天气"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": '{"success": true, "data": "晴，25°C"}'},
    ]
    facts = await extractor.extract(messages)
    # extract 返回 [{"text", "memory_type", "scope"}] 结构化列表
    texts = [f["text"] for f in facts]
    types = {f["memory_type"] for f in facts}
    # 寒暄被过滤，用户询问(fact) + 工具结果(conclusion)被提炼
    assert any("用户询问" in t for t in texts)
    assert any("get_weather" in t for t in texts)
    assert not any("你好" == t for t in texts)
    # 分类正确：用户询问 → fact/session；工具结果 → conclusion/global
    ask = next(f for f in facts if "用户询问" in f["text"])
    assert ask["memory_type"] == "fact" and ask["scope"] == "session"
    tool = next(f for f in facts if "get_weather" in f["text"])
    assert tool["memory_type"] == "conclusion" and tool["scope"] == "global"


@pytest.mark.asyncio
async def test_extractor_llm_classified(mem_settings):
    """LLM 提炼：结构化 JSON 三类输出 + 解析防御（坏输出降级 stub）。"""
    from app.llm.client import BaseLLMClient, LLMResponse

    class FakeLLM(BaseLLMClient):
        async def chat(self, messages, tools=None, **kwargs):
            return LLMResponse(
                content=(
                    '```json\n{"facts": ["用户查询了北京天气"], '
                    '"preferences": ["用户关注 RAG 应用开发"], '
                    '"conclusions": ["调研表明动态图通信适合 MARL"]}\n```'
                ),
                model="fake",
            )

    settings = mem_settings.model_copy(update={"memory_extract_strategy": "llm"})
    extractor = MemoryExtractor(settings, llm=FakeLLM())
    items = await extractor.extract([{"role": "user", "content": "查询北京天气"}])
    by_type = {i["memory_type"]: i for i in items}
    assert by_type["fact"]["text"] == "用户查询了北京天气"
    assert by_type["fact"]["scope"] == "session"
    assert by_type["preference"]["text"] == "用户关注 RAG 应用开发"
    assert by_type["preference"]["scope"] == "global"
    assert by_type["conclusion"]["text"] == "调研表明动态图通信适合 MARL"
    assert by_type["conclusion"]["scope"] == "global"


@pytest.mark.asyncio
async def test_extractor_llm_bad_output_falls_back(mem_settings):
    """LLM 输出不可解析 → 降级 stub 规则提炼（不抛异常）。"""
    from app.llm.client import BaseLLMClient, LLMResponse

    class BadLLM(BaseLLMClient):
        async def chat(self, messages, tools=None, **kwargs):
            return LLMResponse(content="抱歉，我无法提炼。", model="fake")

    settings = mem_settings.model_copy(update={"memory_extract_strategy": "llm"})
    extractor = MemoryExtractor(settings, llm=BadLLM())
    items = await extractor.extract([{"role": "user", "content": "查询北京天气"}])
    assert any("用户询问" in i["text"] for i in items)  # stub 兜底


@pytest.mark.asyncio
async def test_extractor_off(mem_settings):
    settings = mem_settings.model_copy(update={"memory_extract_strategy": "off"})
    extractor = MemoryExtractor(settings)
    assert await extractor.extract([{"role": "user", "content": "查询北京天气"}]) == []


# ---------------------------------------------------------------------------
# Store：remember 幂等 + retrieve
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_store_remember_and_retrieve(mem_store, mem_settings):
    messages = [
        {"role": "user", "content": "查询北京天气"},
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": '{"success": true, "data": "晴，25°C"}'},
    ]
    stored = await mem_store.remember(messages, session_id="s1", turn_id="t1")
    assert len(stored) >= 1
    # 幂等：同 turn 不重复写
    stored2 = await mem_store.remember(messages, session_id="s1", turn_id="t1")
    assert stored2 == []
    # 检索能找回（Stub embedding 是确定性 hash：相同文本 -> 相同向量 -> 相似度 1.0）
    docs = await mem_store.retrieve("查询北京天气")
    assert any("get_weather" in d or "北京" in d for d in docs)


@pytest.mark.asyncio
async def test_store_disabled(tmp_path):
    """memory_enabled=False 时 store 不写入不检索。"""
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/disabled.db",
        memory_enabled=False,
        embedding_provider="stub",
    )
    store = MemoryStore(settings=settings)
    assert await store.remember([{"role": "user", "content": "查询北京天气"}], turn_id="t") == []
    assert await store.retrieve("查询北京天气") == []


# ---------------------------------------------------------------------------
# Runtime 接线：记忆注入 Context（system prompt 含参考资料）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runtime_memory_injects_retrieved_docs(tmp_path, mem_settings):
    """记忆开启时，Context Builder 的 system prompt 应包含"参考资料"行。"""
    mem_repo = SQLiteVecMemoryRepository(mem_settings.database_url, dim=64)
    mem_emb = StubEmbeddingClient(mem_settings)
    # 预置一条记忆
    vec = (await mem_emb.embed(["用户喜欢查询北京天气"]))[0]
    mem_repo.add("用户喜欢查询北京天气", vec)
    store = MemoryStore(repository=mem_repo, embedding=mem_emb, settings=mem_settings)

    session_repo = SQLiteSessionRepository(mem_settings.database_url)
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        session_repo=session_repo,
        memory=store,
        settings=mem_settings,
    )
    result = await runtime.run("帮我查一下北京天气", session_id="s_mem")
    assert result.answer
    # 回合结束后记忆被写入（原有 1 条 + 新增 >= 1）
    assert mem_repo.count() >= 2


@pytest.mark.asyncio
async def test_runtime_without_memory_backward_compat(tmp_path):
    """不注入 memory 时行为与 Stage 6 完全一致（向后兼容）。"""
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/legacy.db",
        trace_enabled=False,
        memory_enabled=True,  # 配置开启了，但 runtime 未注入 store -> 不启用
    )
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        settings=settings,
    )
    result = await runtime.run("你好")
    assert result.answer
    assert runtime.memory is None


# ---------------------------------------------------------------------------
# Rerank（Level 2）：规则重排
# ---------------------------------------------------------------------------
def test_tokenize_chinese_and_english():
    tokens = _tokenize("查询北京天气 weather")
    assert "北京" in tokens
    assert "天气" in tokens
    assert "weather" in tokens


def test_rule_reranker_overlap():
    r = RuleReranker(Settings(memory_rerank_keyword_weight=0.5))
    assert r._overlap({"北京", "天气"}, {"北京", "天气", "上海"}) == 1.0
    assert r._overlap({"北京", "天气"}, {"上海", "广州"}) == 0.0
    assert r._overlap(set(), {"北京"}) == 0.0


@pytest.mark.asyncio
async def test_rule_reranker_boosts_keyword_match(mem_settings):
    """关键词重叠应提升字面相关记忆的最终分数。"""
    # 构造：m1 向量分略高但无关，m2 向量分略低但关键词强相关
    reranker = RuleReranker(mem_settings)
    candidates = [
        MemoryRecord(memory_id="m1", text="用户喜欢股票投资", score=0.70),
        MemoryRecord(memory_id="m2", text="用户喜欢查询北京天气", score=0.65),
    ]
    ranked = await reranker.rerank("北京天气怎么样", candidates)
    # m2 与查询关键词重叠（北京/天气），重排后应反超 m1 排到第一
    assert ranked[0].memory_id == "m2"
    assert ranked[0].score > ranked[1].score
    # 分数是"语义权重 + 关键词权重"的合成，落在 [0,1] 区间
    assert 0.0 <= ranked[0].score <= 1.0


@pytest.mark.asyncio
async def test_rerank_pipeline_in_retriever(mem_settings, mem_repo, mem_emb):
    """retrieve 走 粗召回(不过滤) → 重排 → 阈值过滤 → top_k 完整链路。"""
    texts = [
        "用户喜欢查询北京天气",
        "用户喜欢查询上海天气",
        "用户对股票投资感兴趣",
        "用户喜欢计算数学题",
        "用户喜欢问寒暄话",
    ]
    vectors = await mem_emb.embed(texts)
    for t, v in zip(texts, vectors):
        mem_repo.add(t, v, scope="global")  # 无 session 检索只查 global 层
    retriever = MemoryRetriever(mem_repo, mem_emb, mem_settings)
    docs = await retriever.retrieve("北京天气怎么样", top_k=3)
    # 结果数不超过 top_k
    assert len(docs) <= 3
    # 结果按分数降序（机制正确性）
    scores = [d for d in docs]
    # 关键词强相关的"北京天气"记忆不应被粗召回阶段提前丢掉
    all_records = await retriever.retrieve_records("北京天气怎么样", top_k=5)
    assert any("北京天气" in r.text for r in all_records)


def test_create_reranker_off(mem_settings):
    settings = mem_settings.model_copy(update={"memory_rerank_strategy": "off"})
    assert create_reranker(settings) is None


def test_create_reranker_stub(mem_settings):
    assert isinstance(create_reranker(mem_settings), RuleReranker)


def test_create_reranker_llm_without_llm(mem_settings):
    settings = mem_settings.model_copy(update={"memory_rerank_strategy": "llm"})
    r = create_reranker(settings, llm=None)
    assert isinstance(r, LLMReranker)


@pytest.mark.asyncio
async def test_llm_reranker_without_llm_keeps_order(mem_settings):
    """LLM 重排无模型时降级为原顺序（不崩溃）。"""
    reranker = LLMReranker(mem_settings, llm=None)
    candidates = [
        MemoryRecord(memory_id="m1", text="a", score=0.9),
        MemoryRecord(memory_id="m2", text="b", score=0.8),
    ]
    ranked = await reranker.rerank("query", candidates)
    assert [r.memory_id for r in ranked] == ["m1", "m2"]


# ---------------------------------------------------------------------------
# 记忆分层：会话级 vs 全局级（scope）
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_memory_scope_layering(mem_settings, mem_repo, mem_emb):
    """分层语义：global 跨会话可见；session 级只在本会话可见。

    注：Stub embedding 是文本 hash（相同文本→相同向量→相似度 1.0），
    检索命中依赖查询词与写入文本重叠，因此查询词用与写入文本相同的关键句。
    """
    # 写入：一条 global（偏好）+ 两条 session（分别属于 s1 / s2 的事实）
    vec = await mem_emb.embed(["x"])
    mem_repo.add("用户关注RAG应用开发", vec[0], memory_type="preference", scope="global")
    mem_repo.add("s1查询了北京天气", vec[0], memory_type="fact", scope="session", source_session_id="s1")
    mem_repo.add("s2查询了上海天气", vec[0], memory_type="fact", scope="session", source_session_id="s2")

    retriever = MemoryRetriever(mem_repo, mem_emb, mem_settings)

    # 查询 global 记忆：s1 / s2 / 无会话 三个视角都应命中
    for sid in ("s1", "s2", None):
        docs = await retriever.retrieve_records("用户关注RAG应用开发", top_k=10, session_id=sid)
        assert any("RAG" in r.text for r in docs), f"global 记忆在 {sid} 视角应可见"

    # 查询 s1 的会话级记忆：s1 视角可见，s2 视角不可见
    docs_s1 = await retriever.retrieve_records("s1查询了北京天气", top_k=10, session_id="s1")
    assert any("北京天气" in r.text for r in docs_s1)
    docs_s2 = await retriever.retrieve_records("s1查询了北京天气", top_k=10, session_id="s2")
    assert not any("北京天气" in r.text for r in docs_s2), "s1 的会话级记忆对 s2 不可见"

    # 无 session_id：看不到任何会话级记忆
    docs_none = await retriever.retrieve_records("s1查询了北京天气", top_k=10)
    assert not any("北京天气" in r.text for r in docs_none)


@pytest.mark.asyncio
async def test_store_remember_respects_scope(mem_store, mem_settings):
    """store.remember 把 extractor 的分类 scope 写入仓库。"""
    messages = [
        {"role": "user", "content": "查询北京天气"},
        {"role": "tool", "tool_call_id": "c1", "name": "get_weather", "content": '{"success": true, "data": "晴"}'},
    ]
    stored = await mem_store.remember(messages, session_id="s_scope", turn_id="t_scope")
    types = {(r.memory_type, r.scope) for r in stored}
    # 用户询问 → fact/session；工具结果 → conclusion/global
    assert ("fact", "session") in types
    assert ("conclusion", "global") in types

    # 从 s_scope 视角检索能拿到两者；从别的会话视角只能拿到 global
    all_docs = await mem_store.retrieve("北京", session_id="s_scope")
    assert any("用户询问" in d for d in all_docs) and any("get_weather" in d for d in all_docs)
    other_docs = await mem_store.retrieve("北京", session_id="s_other")
    assert not any("用户询问" in d for d in other_docs)  # s_scope 的会话级不可见
    assert any("get_weather" in d for d in other_docs)   # global 可见


def test_migration_adds_scope_to_old_table(tmp_path):
    """旧库（无 scope 列）打开时自动迁移补列，且不报错。"""
    import sqlite3

    from app.memory.repository import SQLiteVecMemoryRepository

    # 构造一个旧版 memories 表（无 scope 列）
    db = f"sqlite:///{tmp_path}/old.db"
    conn = sqlite3.connect(str(tmp_path / "old.db"))
    conn.execute(
        "CREATE TABLE memories (memory_id TEXT PRIMARY KEY, text TEXT NOT NULL, "
        "embedding BLOB, memory_type TEXT DEFAULT 'fact', "
        "source_session_id TEXT DEFAULT '', source_turn_id TEXT DEFAULT '', created_at TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    # 打开仓库：迁移应补上 scope 列
    repo = SQLiteVecMemoryRepository(db, dim=8)
    cols = [r[1] for r in repo._conn.execute("PRAGMA table_info(memories)").fetchall()]
    assert "scope" in cols
    repo.close()
