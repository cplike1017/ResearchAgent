"""
pytest 共享夹具（fixtures）。

所有测试默认离线运行：
    - LLM 使用确定性 Stub（StubLLMClient）
    - SQLite / Trace 文件使用 pytest 的 tmp_path 隔离
    - Redis 使用 fakeredis（内存假实现）
"""
import pytest

from app.agent.runtime import AgentRuntime
from app.checkpoint.repository import SQLiteCheckpointRepository
from app.config import Settings
from app.llm.client import StubLLMClient
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry


@pytest.fixture
def settings(tmp_path) -> Settings:
    """隔离的测试配置：临时数据库 / 临时 trace 文件。

    强制 stub provider：测试绝不依赖 .env 中的真实 LLM / Embedding 配置，
    保证 CI 与本地行为一致（.env 配了真实 Key 也不会串进测试）。
    """
    return Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/test.db",
        trace_file=str(tmp_path / "traces.jsonl"),
        trace_enabled=False,
        redis_url="redis://localhost:6379/15",
        eval_run_dir=str(tmp_path / "runs"),
        llm_provider="stub",
        embedding_provider="stub",
        agent_mode="react",  # 测试默认 react 模式（.env 的 AGENT_MODE=plan 不影响测试）
        agent_profiles_file=str(tmp_path / "profiles.json"),  # 隔离动态档案，防污染默认文件
    )


@pytest.fixture
def stub_llm() -> StubLLMClient:
    return StubLLMClient()


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def session_repo(settings) -> SQLiteSessionRepository:
    return SQLiteSessionRepository(settings.database_url)


@pytest.fixture
def checkpoint_repo(settings) -> SQLiteCheckpointRepository:
    return SQLiteCheckpointRepository(settings.database_url)


@pytest.fixture
def runtime(settings, stub_llm, registry) -> AgentRuntime:
    """最小运行时（Stage 1/2 形态：无 Session / Checkpoint）。"""
    return AgentRuntime(llm=stub_llm, registry=registry, settings=settings)


@pytest.fixture
def full_runtime(settings, stub_llm, registry, session_repo, checkpoint_repo) -> AgentRuntime:
    """完整运行时（Stage 3 形态：带 Session + Checkpoint 持久化）。"""
    return AgentRuntime(
        llm=stub_llm,
        registry=registry,
        session_repo=session_repo,
        checkpoint_repo=checkpoint_repo,
        settings=settings,
    )
