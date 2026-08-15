"""MCP 客户端管理器：连接多个 MCP Server，提供统一的工具列表与调用入口。

支持两种 transport（与 MCP 官方 SDK 一致）：
    - stdio：本地子进程（如 npx 启动的官方 server）
    - sse：远程 HTTP Server（URL 方式）

设计：
    - MCPClientManager 持有多个 ServerConnection（每个 = 一个 server 的连接上下文）；
    - list_all_tools()：汇总所有 server 的工具（含 server 前缀命名，避免冲突）；
    - call_tool()：按 server_name + tool_name 转发调用。

生命周期：connect_all() 建立连接；close() 关闭全部。
"""
import asyncio
import json
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from app.config import Settings


class MCPServerError(Exception):
    """MCP Server 连接 / 调用错误。"""


class ServerConnection:
    """单个 MCP Server 的连接上下文。"""

    def __init__(self, name: str, config: dict, settings: Settings | None = None) -> None:
        self.name = name
        self.config = config
        self.settings = settings or Settings()
        self.transport = config.get("transport", "stdio")
        self._client_ctx = None  # stdio_client/sse_client 的上下文管理器
        self._session: ClientSession | None = None
        self.tools: list[dict] = []  # 该 server 的工具清单（MCP 原始格式）

    async def connect(self, timeout_seconds: float) -> None:
        """建立连接并拉取工具清单。"""
        try:
            if self.transport == "sse":
                url = self.config.get("url")
                if not url:
                    raise MCPServerError(f"MCP server [{self.name}] sse 模式缺少 url")
                self._client_ctx = sse_client(url)
                read, write = await asyncio.wait_for(
                    self._client_ctx.__aenter__(), timeout=timeout_seconds
                )
            else:  # stdio
                from mcp.client.stdio import StdioServerParameters

                # 注入子进程环境变量：显式配置的 env 优先，否则自动补充
                # GITHUB_TOKEN（github server 需要）等已知变量
                env = dict(self.config.get("env") or {})
                if "GITHUB_TOKEN" not in env and self.settings.github_token:
                    env["GITHUB_TOKEN"] = self.settings.github_token

                params = StdioServerParameters(
                    command=self.config.get("command", ""),
                    args=self.config.get("args", []),
                    env=env or None,
                )
                if not params.command:
                    raise MCPServerError(f"MCP server [{self.name}] stdio 模式缺少 command")
                self._client_ctx = stdio_client(params)
                read, write = await asyncio.wait_for(
                    self._client_ctx.__aenter__(), timeout=timeout_seconds
                )

            self._session = await asyncio.wait_for(
                ClientSession(read, write).__aenter__(), timeout=timeout_seconds
            )
            await asyncio.wait_for(self._session.initialize(), timeout=timeout_seconds)

            # 拉取工具清单
            result = await self._session.list_tools()
            self.tools = [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in result.tools]
        except asyncio.TimeoutError as exc:
            # 超时后：尽力清理半开的连接（吞掉清理噪音）
            await self._cleanup_quiet()
            raise MCPServerError(f"MCP server [{self.name}] 连接超时") from exc
        except Exception as exc:
            await self._cleanup_quiet()
            raise MCPServerError(f"MCP server [{self.name}] 连接失败: {exc}") from exc

    async def _cleanup_quiet(self) -> None:
        """尽力清理连接资源（忽略 SDK 跨 task 关闭噪音）。

        说明：MCP SDK 的 stdio_client 用 anyio task group 管理子进程，
        __aexit__ 必须在进入时的同一 task 调用（uvicorn lifespan 的
        startup/shutdown 在同一 task，直接 await 即可干净关闭）。
        注意：绝不能包 asyncio.shield —— shield 会创建新 task 执行
        __aexit__，反而造成跨 task 的 cancel scope 错误！
        """
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except (BaseExceptionGroup, Exception, asyncio.CancelledError):
                pass
        if self._client_ctx is not None:
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except (BaseExceptionGroup, Exception, asyncio.CancelledError):
                pass
        self._session = None
        self._client_ctx = None

    async def call_tool(self, tool_name: str, arguments: dict) -> Any:
        """调用该 server 上的一个工具。"""
        if self._session is None:
            raise MCPServerError(f"MCP server [{self.name}] 未连接")
        try:
            result = await self._session.call_tool(tool_name, arguments or {})
            # 提取文本内容
            if hasattr(result, "content"):
                texts = [
                    c.text for c in result.content
                    if getattr(c, "type", "") == "text" and getattr(c, "text", None)
                ]
                return "\n".join(texts) if texts else result.model_dump()
            return result
        except Exception as exc:
            raise MCPServerError(f"MCP tool [{self.name}/{tool_name}] 调用失败: {exc}") from exc

    async def close(self) -> None:
        await self._cleanup_quiet()


class MCPClientManager:
    """多 MCP Server 管理器。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.connections: list[ServerConnection] = []

    def parse_config(self) -> list[dict]:
        """解析 MCP_SERVERS 配置（JSON 字符串 -> 列表）。"""
        raw = self.settings.mcp_servers or "[]"
        try:
            servers = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MCPServerError(f"MCP_SERVERS 配置非法: {exc}") from exc
        if not isinstance(servers, list):
            raise MCPServerError("MCP_SERVERS 必须是 JSON 数组")
        return servers

    async def connect_all(self) -> list[str]:
        """连接所有配置的 server，返回成功连接的名字列表（失败跳过不阻塞）。"""
        servers = self.parse_config()
        connected: list[str] = []
        for cfg in servers:
            name = cfg.get("name", "mcp_server")
            conn = ServerConnection(name, cfg, self.settings)
            try:
                await conn.connect(self.settings.mcp_connect_timeout_seconds)
            except MCPServerError as exc:
                print(f"[mcp] 跳过 server [{name}]: {exc}", flush=True)
                continue
            self.connections.append(conn)
            connected.append(name)
            print(f"[mcp] 已连接 server [{name}]（{len(conn.tools)} 个工具）", flush=True)
        return connected

    def list_all_tools(self) -> list[tuple[str, str, dict]]:
        """
        汇总所有 server 的工具。

        :return: [(server_name, tool_name, tool_schema_dict), ...]
        """
        out: list[tuple[str, str, dict]] = []
        for conn in self.connections:
            for t in conn.tools:
                out.append((conn.name, t.get("name", ""), t))
        return out

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict) -> Any:
        """按 server + tool 调用。"""
        for conn in self.connections:
            if conn.name == server_name:
                return await conn.call_tool(tool_name, arguments)
        raise MCPServerError(f"未知 MCP server: {server_name}")

    async def close(self) -> None:
        for conn in self.connections:
            await conn.close()
        self.connections.clear()
