"""
Stage 6 测试：Tracing（Span 嵌套 / 异常 / ContextVar 恢复 / 脱敏 / 跨进程传播 / API）。
"""
import pytest

import fakeredis.aioredis

from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.llm.client import StubLLMClient
from app.queue.models import Job, JobStatus
from app.queue.producer import RedisJobQueue, utc_now
from app.queue.consumer import process_job
from app.tracing.context import current_span_id, current_trace_id, get_trace_context
from app.tracing.models import SpanStatus
from app.tracing.recorder import TraceRecorder, redact
from app.tracing.span import trace_span, trace_span_sync
from app.tools.builtin import build_default_registry


@pytest.fixture
def recorder(tmp_path) -> TraceRecorder:
    """启用的 Trace Recorder（写入临时 JSONL）。"""
    return TraceRecorder(str(tmp_path / "traces.jsonl"), enabled=True, capture_content=False)


# ---------------------------------------------------------------------------
# Span 嵌套 / 树重建
# ---------------------------------------------------------------------------
async def test_span_nesting_and_tree(recorder):
    """父 -> 子 -> 孙，parent_span_id 正确串联，树可重建。"""
    async with trace_span("root", "test", recorder=recorder) as root:
        async with trace_span("child", "test", recorder=recorder) as child:
            async with trace_span("grandchild", "test", recorder=recorder) as grand:
                assert grand.parent_span_id == child.span_id
            assert child.parent_span_id == root.span_id
        assert root.parent_span_id is None

    spans = recorder.load_trace(root.trace_id)
    assert len(spans) == 3
    tree = recorder.build_tree(root.trace_id)
    assert tree["spans"][0]["name"] == "root"
    assert tree["spans"][0]["children"][0]["name"] == "child"
    assert tree["spans"][0]["children"][0]["children"][0]["name"] == "grandchild"


async def test_span_error_handling(recorder):
    """Span 内抛异常 -> status=ERROR + error 记录 + 异常继续向上抛。"""
    with pytest.raises(ValueError, match="boom"):
        async with trace_span("failing", "test", recorder=recorder):
            raise ValueError("boom")

    # 从文件中找到该 Span 的 trace_id
    import json as _json

    trace_id = None
    for line in open(recorder.trace_file, encoding="utf-8"):
        span = _json.loads(line)
        if span["name"] == "failing":
            trace_id = span["trace_id"]
            break
    assert trace_id is not None

    failing = [s for s in recorder.load_trace(trace_id) if s.name == "failing"][0]
    assert failing.status == SpanStatus.ERROR
    assert failing.error["type"] == "ValueError"
    assert failing.duration_ms is not None


# ---------------------------------------------------------------------------
# ContextVar 恢复
# ---------------------------------------------------------------------------
async def test_contextvar_restored_after_span(recorder):
    async with trace_span("outer", "test", recorder=recorder) as outer:
        async with trace_span("inner", "test", recorder=recorder):
            pass
        # inner 退出后，current_span_id 应恢复为 outer
        assert current_span_id.get() == outer.span_id
    # 全部退出后恢复为 None
    assert current_trace_id.get() is None
    assert current_span_id.get() is None


def test_sync_span_restores_context(recorder):
    with trace_span_sync("sync_outer", "test", recorder=recorder) as outer:
        with trace_span_sync("sync_inner", "test", recorder=recorder):
            pass
        assert current_span_id.get() == outer.span_id
    assert current_span_id.get() is None


# ---------------------------------------------------------------------------
# 脱敏 / 内容省略
# ---------------------------------------------------------------------------
def test_redaction_sensitive_keys():
    value = {"api_key": "sk-123", "city": "北京", "nested": {"password": "pwd", "ok": 1}}
    out = redact(value)
    assert out["api_key_redacted"] == "[REDACTED]"
    assert out["nested"]["password_redacted"] == "[REDACTED]"
    assert out["city"] == "北京"
    assert out["nested"]["ok"] == 1


async def test_content_omitted_when_capture_disabled(recorder):
    """TRACE_CAPTURE_CONTENT=false：llm span 不保存完整 Prompt。"""
    async with trace_span("llm_call", "llm", input=[{"role": "user", "content": "秘密内容" * 50}], recorder=recorder):
        pass
    spans = recorder.load_trace("")
    # 读取文件拿到 trace_id
    import json as _json

    trace_id = None
    for line in open(recorder.trace_file, encoding="utf-8"):
        span = _json.loads(line)
        trace_id = span["trace_id"]
        break
    llm_span = [s for s in recorder.load_trace(trace_id) if s.name == "llm_call"][0]
    assert llm_span.input["omitted"] is True
    assert "秘密内容" not in llm_span.model_dump_json()


async def test_content_captured_when_enabled(tmp_path):
    recorder = TraceRecorder(str(tmp_path / "traces.jsonl"), enabled=True, capture_content=True)
    async with trace_span("llm_call", "llm", input=[{"role": "user", "content": "完整内容"}], recorder=recorder):
        pass
    import json as _json

    trace_id = None
    for line in open(recorder.trace_file, encoding="utf-8"):
        span = _json.loads(line)
        trace_id = span["trace_id"]
        break
    spans = recorder.load_trace(trace_id)
    assert spans[0].input == [{"role": "user", "content": "完整内容"}]


# ---------------------------------------------------------------------------
# Agent 运行时全链路 Span
# ---------------------------------------------------------------------------
async def test_agent_run_trace_spans(settings, stub_llm, recorder):
    rt = AgentRuntime(llm=stub_llm, registry=build_default_registry(), settings=settings, recorder=recorder)
    result = await rt.run("查询北京天气", session_id="session_trace_1")
    assert result.trace_id is not None

    spans = recorder.load_trace(result.trace_id)
    names = [s.name for s in spans]
    # 必备组件全部被追踪
    assert "agent.run" in names
    assert "context_builder" in names
    assert "llm_call" in names
    assert "tool_gateway" in names
    assert "tool.execute" in names

    # llm span 记录 model / tokens / finish_reason
    llm_spans = [s for s in spans if s.name == "llm_call"]
    assert len(llm_spans) == 2  # 一次工具调用决策 + 一次最终回答
    assert llm_spans[0].attributes["model"]
    assert "total_tokens" in llm_spans[0].attributes
    assert llm_spans[0].attributes["finish_reason"] == "tool_calls"

    # tool span 记录 tool_name / success
    tool_spans = [s for s in spans if s.name == "tool.execute"]
    assert tool_spans[0].attributes["tool_name"] == "get_weather"
    assert tool_spans[0].attributes["success"] is True


async def test_checkpoint_spans_in_trace(settings, stub_llm, recorder, session_repo, checkpoint_repo):
    from app.session.repository import SQLiteSessionRepository
    from app.checkpoint.repository import SQLiteCheckpointRepository

    rt = AgentRuntime(
        llm=stub_llm,
        registry=build_default_registry(),
        settings=settings,
        recorder=recorder,
        session_repo=session_repo,
        checkpoint_repo=checkpoint_repo,
    )
    result = await rt.run("你好", session_id="session_trace_ckpt")
    spans = recorder.load_trace(result.trace_id)
    names = [s.name for s in spans]
    assert "checkpoint.save" in names
    save_spans = [s for s in spans if s.name == "checkpoint.save"]
    assert save_spans[0].attributes["version"] == 1


# ---------------------------------------------------------------------------
# 跨进程 Trace 传播：Gateway -> Redis -> Worker -> Agent
# ---------------------------------------------------------------------------
async def test_trace_propagation_through_queue(settings, stub_llm, recorder):
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    queue = RedisJobQueue(redis, settings, recorder=recorder)
    # 运行时也要带 recorder，否则 agent.run 等 Span 不会被记录
    rt = AgentRuntime(llm=stub_llm, registry=build_default_registry(), settings=settings, recorder=recorder)

    # 1) Gateway 进程：创建根 Span，把 trace_context 写入 Job
    async with trace_span("gateway.request", "gateway", recorder=recorder) as root:
        job = Job(
            job_id="job_prop",
            request_id="req_prop",
            session_id="session_prop",
            input={"message": "查询北京天气"},
            trace_context=get_trace_context(),
            created_at=utc_now(),
        )
        await queue.enqueue(job)
        root_span_id = root.span_id
        trace_id = root.trace_id

    # Job 中携带了 Trace 上下文
    saved = await queue.get_job("job_prop")
    assert saved.trace_context["trace_id"] == trace_id
    assert saved.trace_context["parent_span_id"] == root_span_id

    # 2) Worker 进程：消费并恢复 Trace 上下文
    popped = await queue.pop(timeout=0.1)
    done = await process_job(queue, lambda: rt, popped, recorder=recorder)
    assert done.status == JobStatus.SUCCEEDED
    assert done.result["trace_id"] == trace_id  # 结果里的 trace_id 与 Gateway 一致

    # 3) 同一 trace_id 下的完整链路
    spans = recorder.load_trace(trace_id)
    names = [s.name for s in spans]
    assert "gateway.request" in names
    assert "redis.enqueue" in names
    assert "worker.process" in names
    assert "agent.run" in names
    assert all(s.trace_id == trace_id for s in spans)

    # 4) 树结构：gateway.request 下挂 redis.enqueue 与 worker.process
    tree = recorder.build_tree(trace_id)
    root_node = tree["spans"][0]
    assert root_node["name"] == "gateway.request"
    child_names = [c["name"] for c in root_node["children"]]
    assert "redis.enqueue" in child_names
    assert "worker.process" in child_names
    # worker.process 下应有 agent.run -> llm_call / tool_gateway
    worker_node = next(c for c in root_node["children"] if c["name"] == "worker.process")
    agent_node = worker_node["children"][0]
    assert agent_node["name"] == "agent.run"
    agent_child_names = [c["name"] for c in agent_node["children"]]
    assert "llm_call" in agent_child_names
    assert "tool_gateway" in agent_child_names

    await redis.aclose()


# ---------------------------------------------------------------------------
# Trace API
# ---------------------------------------------------------------------------
def test_trace_api_endpoint(tmp_path, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/api.db",
        trace_file=str(tmp_path / "api_traces.jsonl"),
        trace_enabled=True,
        redis_url="redis://localhost:6379/15",
        agent_mode="react",  # 显式 react：.env 的 AGENT_MODE=plan 不影响测试
    )
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    app = create_app(settings, redis=fake_redis)
    with TestClient(app) as client:
        resp = client.post("/api/chat", json={"message": "你好"})
        assert resp.status_code == 202
        body = resp.json()

        # 从 trace 文件读取刚写入的 trace_id
        import json as _json

        trace_id = None
        with open(settings.trace_file, encoding="utf-8") as f:
            for line in f:
                span = _json.loads(line)
                trace_id = span["trace_id"]
                break
        assert trace_id is not None

        tree = client.get(f"/api/traces/{trace_id}")
        assert tree.status_code == 200
        data = tree.json()
        assert data["trace_id"] == trace_id
        assert data["spans"][0]["name"] == "gateway.request"
        assert "redis.enqueue" in [c["name"] for c in data["spans"][0]["children"]]

        # 不存在的 trace -> 404
        assert client.get("/api/traces/nope").status_code == 404
