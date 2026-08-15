"""
Stage 10 Demo：MCP（Model Context Protocol）外部工具接入。

运行：python -m demos.stage10_mcp_demo

展示：
    1. 连接 MCP Server（stdio 子进程 / sse 远程），列出其工具
    2. 把 MCP 工具桥接进 ToolRegistry（复用 Gateway 治理）
    3. 通过 Agent 循环让真实 LLM 调用 MCP 工具

配置：在 .env 的 MCP_SERVERS 配置一个 server 后运行本 demo；
未配置时本 demo 会自动起一个内置 echo 测试 server 演示流程。
"""
import asyncio
import json
import sys

from app.agent.runtime import AgentRuntime
from app.config import get_settings
from app.llm.client import create_llm_client
from app.mcp.client import MCPClientManager
from app.mcp.bridge import MCPBridge
from app.session.repository import SQLiteSessionRepository
from app.tools.builtin import build_default_registry
from app.tools.registry import ToolDefinition

SEPARATOR = "=" * 64

# 内置 echo 测试 server（未配置 MCP_SERVERS 时用于演示）
_ECHO_SERVER_CODE = (
    'from mcp.server.fastmcp import FastMCP\n'
    'mcp = FastMCP("demo")\n'
    '@mcp.tool()\n'
    'def echo(text: str) -> str:\n'
    '    """原样返回输入的文本"""\n'
    '    return f"echo: {text}"\n'
    '@mcp.tool()\n'
    'def add(a: float, b: float) -> float:\n'
    '    """计算两个数字之和"""\n'
    '    return a + b\n'
    'if __name__ == "__main__":\n'
    '    mcp.run()\n'
)


async def main() -> None:
    settings = get_settings()

    # 1) 确定 MCP 配置：有则用配置，无则用内置 echo server
    servers = json.loads(settings.mcp_servers or "[]")
    demo_script = None
    if not servers:
        demo_script = "data/demo_mcp_server.py"
        import pathlib

        pathlib.Path(demo_script).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(demo_script).write_text(_ECHO_SERVER_CODE, encoding="utf-8")
        servers = [
            {"name": "demo", "transport": "stdio",
             "command": sys.executable, "args": [demo_script]}
        ]

    print(SEPARATOR)
    print("Stage 10 MCP Demo")
    print(SEPARATOR)
    print(f"MCP_SERVERS: {len(servers)} 个 server")

    client = MCPClientManager(settings.model_copy(update={"mcp_servers": json.dumps(servers)}))
    connected = await client.connect_all()
    print(f"已连接: {connected}")

    # 2) 桥接进 registry
    registry = build_default_registry()
    count = MCPBridge(client).register_all(registry)
    print(f"注册 MCP 工具: {count} 个")
    mcp_tools = [t for t in registry.all() if "_" in t.name and t.name.split("_")[0] in {s.get('name','') for s in servers}]
    # 降风险放行（演示用；生产由 Policy 审核）
    for name in [t.name for t in registry.all() if "demo" in t.name]:
        t = registry.get(name)
        registry.register(
            ToolDefinition(name=t.name, description=t.description, input_model=t.input_model,
                           handler=t.handler, timeout_seconds=t.timeout_seconds, risk_level="low"),
            overwrite=True,
        )

    # 3) 让 Agent 调用 MCP 工具
    print("\n" + SEPARATOR)
    print("3) Agent 调用 MCP 工具（真实 LLM）")
    print(SEPARATOR)
    llm = create_llm_client(settings)
    runtime = AgentRuntime(
        llm=llm, registry=registry,
        session_repo=SQLiteSessionRepository(settings.database_url),
        mcp_client=client, settings=settings,
    )
    r = await runtime.run(
        "请调用 echo 工具回复一句问候，并告诉我 12 + 30 等于多少",
        session_id="s_mcp_demo",
    )
    print(f"回答: {r.answer}")
    print(f"工具调用: {[(tc.name, tc.arguments) for tc in r.tool_calls]}")

    await client.close()
    if demo_script:
        import os

        os.remove(demo_script)
    print("\nDemo 完成。")


if __name__ == "__main__":
    asyncio.run(main())
