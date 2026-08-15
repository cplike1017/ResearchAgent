"""
MCP 接入测试：连接本地 stdio server -> 注册 -> 走 Gateway 调用。

说明：MCP SDK 的 stdio_client 使用 anyio task group，要求进入/退出
cancel scope 在同一事件循环任务。pytest-asyncio 的 loop 与 SDK 后台任务
管理有兼容噪音（teardown 时报 GeneratorExit / cancel scope 跨任务）。
因此这里用 asyncio.run() 独立 loop 跑 MCP 逻辑（与真实运行环境一致），
断言在同步测试函数内完成。
"""
import asyncio
import json
import sys

import pytest

from app.config import Settings
from app.mcp.bridge import MCPBridge, _schema_to_pydantic
from app.mcp.client import MCPClientManager, MCPServerError
from app.tools.builtin import build_default_registry
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolDefinition
from app.tools.schemas import UserContext


def _make_server_script(tmp_path):
    script = tmp_path / "mcp_server.py"
    script.write_text(
        'from mcp.server.fastmcp import FastMCP\n'
        'mcp = FastMCP("test")\n'
        '@mcp.tool()\n'
        'def echo(text: str) -> str:\n'
        '    """原样返回文本"""\n'
        '    return f"echo: {text}"\n'
        '@mcp.tool()\n'
        'def add(a: float, b: float) -> float:\n'
        '    """两数之和"""\n'
        '    return a + b\n'
        'if __name__ == "__main__":\n'
        '    mcp.run()\n',
        encoding="utf-8",
    )
    return str(script)


def _settings_with_server(server_script: str) -> Settings:
    return Settings(
        mcp_servers=json.dumps([
            {"name": "demo", "transport": "stdio",
             "command": sys.executable, "args": [server_script]}
        ]),
        mcp_connect_timeout_seconds=20,
    )


# ---------------------------------------------------------------------------
# schema 转换（纯同步，无 MCP 连接）
# ---------------------------------------------------------------------------
def test_schema_to_pydantic():
    m = _schema_to_pydantic({
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "文本"},
            "count": {"type": "integer", "description": "数量"},
        },
        "required": ["text"],
    })
    inst = m(text="hi")
    assert inst.text == "hi"
    assert inst.count is None
    assert m.model_json_schema()["properties"]["text"]


def test_schema_to_pydantic_empty():
    m = _schema_to_pydantic({"type": "object", "properties": {}})
    assert m() is not None


def test_tool_name_normalized():
    """MCP 工具名规范化为符合 OpenAI 命名规则（去点号）。"""
    assert MCPBridge.normalize_name("demo", "echo") == "demo_echo"
    assert MCPBridge.normalize_name("my.server", "read.file") == "my_server_read_file"


def test_mcp_config_invalid():
    settings = Settings(mcp_servers="not-json")
    with pytest.raises(MCPServerError):
        MCPClientManager(settings).parse_config()


# ---------------------------------------------------------------------------
# 端到端（asyncio.run 独立 loop，避免 pytest-asyncio 与 SDK task group 冲突）
# ---------------------------------------------------------------------------
def test_mcp_connect_and_list(tmp_path):
    async def _run():
        settings = _settings_with_server(_make_server_script(tmp_path))
        client = MCPClientManager(settings)
        try:
            connected = await client.connect_all()
            assert "demo" in connected
            names = {(s, t) for s, t, _ in client.list_all_tools()}
            assert ("demo", "echo") in names
            assert ("demo", "add") in names
        finally:
            await client.close()
    asyncio.run(_run())


def test_mcp_bridge_registers_tools(tmp_path):
    async def _run():
        settings = _settings_with_server(_make_server_script(tmp_path))
        client = MCPClientManager(settings)
        try:
            await client.connect_all()
            registry = build_default_registry()
            count = MCPBridge(client).register_all(registry)
            assert count == 2
            names = {t.name for t in registry.all()}
            assert "demo_echo" in names and "demo_add" in names
        finally:
            await client.close()
    asyncio.run(_run())


def test_mcp_tool_high_risk_blocked_by_policy(tmp_path):
    async def _run():
        settings = _settings_with_server(_make_server_script(tmp_path))
        client = MCPClientManager(settings)
        try:
            await client.connect_all()
            registry = build_default_registry()
            MCPBridge(client).register_all(registry)
            gateway = ToolGateway(registry, settings=settings)
            r = await gateway.execute("demo_echo", {"text": "hi"}, user=UserContext(user_id="u"))
            assert r.success is False
            assert r.error is not None and "ToolPolicyError" in r.error.type
        finally:
            await client.close()
    asyncio.run(_run())


def test_mcp_tool_executes_when_allowed(tmp_path):
    async def _run():
        settings = _settings_with_server(_make_server_script(tmp_path))
        client = MCPClientManager(settings)
        try:
            await client.connect_all()
            registry = build_default_registry()
            MCPBridge(client).register_all(registry)
            for name in ("demo_echo", "demo_add"):
                t = registry.get(name)
                registry.register(
                    ToolDefinition(name=t.name, description=t.description, input_model=t.input_model,
                                   handler=t.handler, timeout_seconds=t.timeout_seconds, risk_level="low"),
                    overwrite=True,
                )
            gateway = ToolGateway(registry, settings=settings)
            r1 = await gateway.execute("demo_echo", {"text": "hello"}, user=UserContext(user_id="u"))
            assert r1.success is True
            assert r1.data == "echo: hello"
            r2 = await gateway.execute("demo_add", {"a": 2, "b": 3}, user=UserContext(user_id="u"))
            assert r2.success is True
            # MCP call_tool 返回的是文本内容（content[].text），数字会转成字符串
            assert r2.data in ("5.0", "5")
        finally:
            await client.close()
    asyncio.run(_run())


def test_mcp_no_servers():
    async def _run():
        client = MCPClientManager(Settings(mcp_servers="[]"))
        try:
            connected = await client.connect_all()
            assert connected == []
        finally:
            await client.close()
    asyncio.run(_run())
