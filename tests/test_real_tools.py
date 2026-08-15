"""
Stage 10 测试：真实工具（天气 API / sqlite 只读 / 文件沙箱 / web 工具）。

验收：真实天气 API（Open-Meteo，需网络，失败回退 Stub）、sqlite 只读拦截、
文件沙箱目录穿越拦截、web_search/http_get 配置依赖、注册表完整性。
"""
import os
import sqlite3

import pytest

from app.config import Settings, get_settings
from app.errors import ToolExecutionError
from app.tools.builtin import build_default_registry
from app.tools.builtin.data import (
    _resolve_in_sandbox,
    file_read_handler,
    file_write_handler,
    sqlite_query_handler,
)
from app.tools.builtin.web import (
    DateArgs,
    HttpGetArgs,
    TimeArgs,
    WebSearchArgs,
    get_date_handler,
    get_time_handler,
    http_get_handler,
    web_search_handler,
)


# ---------------------------------------------------------------------------
# 注册表完整性
# ---------------------------------------------------------------------------
def test_registry_has_all_tools():
    registry = build_default_registry()
    names = {t.name for t in registry.all()}
    assert {
        "calculator", "get_weather", "web_search", "http_get",
        "get_time", "get_date", "sqlite_query", "file_read", "file_write",
    } <= names


# ---------------------------------------------------------------------------
# sqlite_query：只读约束
# ---------------------------------------------------------------------------
def test_sqlite_query_read(tmp_path):
    db = tmp_path / "t.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
    conn.execute("INSERT INTO t VALUES (1, 'alice')")
    conn.commit()
    conn.close()
    out = sqlite_query_handler(str(db), "SELECT * FROM t")
    assert "alice" in out


def test_sqlite_query_blocks_writes(tmp_path):
    db = tmp_path / "t2.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (id INTEGER)")
    conn.commit()
    conn.close()
    with pytest.raises(ToolExecutionError):
        sqlite_query_handler(str(db), "DELETE FROM t")
    with pytest.raises(ToolExecutionError):
        sqlite_query_handler(str(db), "INSERT INTO t VALUES (1)")
    with pytest.raises(ToolExecutionError):
        sqlite_query_handler(str(db), "DROP TABLE t")
    # 多语句也被拦截
    with pytest.raises(ToolExecutionError):
        sqlite_query_handler(str(db), "SELECT * FROM t; DROP TABLE t")


def test_sqlite_query_missing_db():
    with pytest.raises(ToolExecutionError):
        sqlite_query_handler("/nonexistent/x.db", "SELECT 1")


# ---------------------------------------------------------------------------
# 文件沙箱
# ---------------------------------------------------------------------------
def test_file_sandbox_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_DIR", str(tmp_path / "sbox"))
    out = file_write_handler("notes/hello.txt", "hello agent")
    assert "已写入" in out
    assert file_read_handler("notes/hello.txt") == "hello agent"


def test_file_sandbox_traversal_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_DIR", str(tmp_path / "sbox"))
    with pytest.raises(ToolExecutionError):
        file_read_handler("../../etc/passwd")
    with pytest.raises(ToolExecutionError):
        file_write_handler("../escape.txt", "x")


def test_sandbox_resolve_creates_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_DIR", str(tmp_path / "sbox"))
    p = _resolve_in_sandbox("a/b.txt")
    assert p.exists() is False
    # 解析后的路径必须位于沙箱内（用 commonpath 避免 Windows 分隔符差异）
    sandbox = (tmp_path / "sbox").resolve()
    assert os.path.commonpath([str(sandbox), str(p)]) == str(sandbox)


# ---------------------------------------------------------------------------
# 天气：真实 API 失败回退 Stub（离线测试不断言真实数据）
# ---------------------------------------------------------------------------
def test_weather_stub_fallback(monkeypatch):
    """无网络或 API 失败时应回退 Stub（不抛异常）。"""
    from app.tools.builtin import weather as weather_mod
    from app.tools.builtin.weather import weather_handler

    # 强制关闭真实 API + QWeather -> 用 Stub 数据（weather.py 用 Settings() 而非 get_settings()）
    monkeypatch.setattr(
        weather_mod, "Settings",
        lambda: Settings(weather_use_real_api=False, qweather_host="", qweather_api_key=""),
    )
    out = weather_handler("北京")
    assert "晴" in out or "多云" in out


def test_weather_unknown_city(monkeypatch):
    from app.tools.builtin import weather as weather_mod
    from app.tools.builtin.weather import weather_handler

    # 清空 QWeather + 关闭真实 API，确保走 Stub 分支
    monkeypatch.setattr(
        weather_mod, "Settings",
        lambda: Settings(weather_use_real_api=False, qweather_host="", qweather_api_key=""),
    )
    with pytest.raises(ToolExecutionError):
        weather_handler("东京")


# ---------------------------------------------------------------------------
# 时间 / 日期工具
# ---------------------------------------------------------------------------
def test_time_date_tools():
    t = get_time_handler()
    assert "UTC" in t and len(t) >= 16
    d = get_date_handler()
    assert "-" in d and len(d) == 10


# ---------------------------------------------------------------------------
# web 工具：依赖配置的行为
# ---------------------------------------------------------------------------
def test_web_search_requires_key(monkeypatch):
    """无 TAVILY_API_KEY 时返回结构化失败（ToolExecutionError）。"""
    from app.tools.builtin import web as web_mod

    monkeypatch.setattr(web_mod, "get_settings", lambda: Settings(tavily_api_key=""))
    with pytest.raises(ToolExecutionError):
        web_search_handler("test", max_results=2)


def test_http_get_rejects_non_http():
    with pytest.raises(ToolExecutionError):
        http_get_handler("file:///etc/passwd")


def test_web_tool_schemas():
    assert WebSearchArgs.model_json_schema()["properties"]["query"]
    assert HttpGetArgs.model_json_schema()["properties"]["url"]
    assert TimeArgs.model_json_schema() is not None
    assert DateArgs.model_json_schema() is not None


# ---------------------------------------------------------------------------
# 邮件工具（配置依赖，不真正发信）
# ---------------------------------------------------------------------------
def test_email_missing_config(monkeypatch):
    """SMTP 未配置时返回结构化失败。"""
    from app.tools.builtin import mail as mail_mod
    from app.tools.builtin.mail import send_email_handler

    monkeypatch.setattr(mail_mod, "Settings", lambda: Settings(smtp_host="", smtp_user="", smtp_password=""))
    with pytest.raises(ToolExecutionError):
        send_email_handler("a@b.com", "主题", "正文")


def test_email_invalid_recipient():
    from app.tools.builtin.mail import send_email_handler

    with pytest.raises(ToolExecutionError):
        send_email_handler("not-an-email", "主题", "正文")


def test_email_schema():
    from app.tools.builtin.mail import SendEmailArgs

    schema = SendEmailArgs.model_json_schema()
    assert "to" in schema["properties"]
    assert "subject" in schema["properties"]
    assert "body" in schema["properties"]


# ---------------------------------------------------------------------------
# 天气：QWeather 未配置时回退（不抛异常）
# ---------------------------------------------------------------------------
def test_weather_qweather_missing_falls_back(monkeypatch):
    """QWeather 未配置 key 时应跳过它，继续走 Open-Meteo / Stub。"""
    from app.tools.builtin import weather as weather_mod
    from app.tools.builtin.weather import weather_handler

    # QWeather 无 key + 关闭真实 API -> Stub 数据
    monkeypatch.setattr(
        weather_mod, "Settings",
        lambda: Settings(qweather_host="m33h2tjaf9.re.qweatherapi.com", qweather_api_key="",
                         weather_use_real_api=False),
    )
    out = weather_handler("北京")
    assert "晴" in out or "多云" in out
