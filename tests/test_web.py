"""
Stage 12 测试：Web UI（进程内直连 + SSE 流式）。

覆盖：首页静态资源、工具/技能/MCP 列表、同步聊天、SSE 流式事件、会话历史。
使用 fakeredis + stub LLM + test 环境（跳过 MCP 连接），完全离线。
"""
import fakeredis.aioredis
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _make_app(tmp_path):
    settings = Settings(
        environment="test",
        trace_enabled=False,
        memory_enabled=False,
        skills_enabled=True,
        agent_mode="react",
        llm_provider="stub",
        database_url=f"sqlite:///{tmp_path}/web.db",
        trace_file=str(tmp_path / "traces.jsonl"),
        eval_run_dir=str(tmp_path / "runs"),
    )
    fake_redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    return create_app(settings, redis=fake_redis)


def test_web_index(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        r = client.get("/")
        assert r.status_code == 200
        assert "Agent Runtime" in r.text


def test_web_capabilities(tmp_path):
    with TestClient(_make_app(tmp_path)) as client:
        tools = client.get("/api/web/tools").json()
        assert tools["count"] >= 10
        names = {t["name"] for t in tools["tools"]}
        assert {"calculator", "get_weather", "send_email"} <= names

        skills = client.get("/api/web/skills").json()
        assert skills["count"] >= 2

        mcp = client.get("/api/web/mcp").json()
        assert mcp["count"] == 0  # test 环境跳过 MCP


def test_web_chat_sync(tmp_path):
    """同步聊天：stub LLM 调用 get_weather。"""
    with TestClient(_make_app(tmp_path)) as client:
        r = client.post("/api/web/chat", json={"message": "查询北京天气", "agent_mode": "react"})
        assert r.status_code == 200
        data = r.json()
        assert data["answer"]
        assert any(t["name"] == "get_weather" for t in data["tool_calls"])
        assert data["session_id"]


def test_web_chat_plan_mode(tmp_path):
    """plan 模式聊天：结果含 plan 字段。"""
    with TestClient(_make_app(tmp_path)) as client:
        r = client.post("/api/web/chat", json={"message": "查询北京和上海天气", "agent_mode": "plan"})
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "plan"
        assert data["answer"]


def test_web_chat_stream_sse(tmp_path):
    """SSE 流式：应包含 step / tool_result / done 事件。"""
    with TestClient(_make_app(tmp_path)) as client:
        with client.stream(
            "POST", "/api/web/chat/stream",
            json={"message": "计算 123 * 456", "agent_mode": "react"},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
            assert "event: step" in body
            assert "event: tool_result" in body
            assert "event: done" in body


def test_web_sessions(tmp_path):
    """会话历史：聊天后会话可查。"""
    with TestClient(_make_app(tmp_path)) as client:
        client.post("/api/web/chat", json={"message": "你好", "agent_mode": "react"})
        r = client.get("/api/web/sessions")
        assert r.status_code == 200
        assert len(r.json()["sessions"]) >= 1


def test_web_session_messages(tmp_path):
    """会话消息历史。"""
    with TestClient(_make_app(tmp_path)) as client:
        data = client.post("/api/web/chat", json={"message": "你好", "agent_mode": "react"}).json()
        sid = data["session_id"]
        r = client.get(f"/api/web/sessions/{sid}/messages")
        assert r.status_code == 200
        msgs = r.json()["messages"]
        assert any(m["role"] == "user" for m in msgs)
