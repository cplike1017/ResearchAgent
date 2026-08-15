"""MCP 接入包：把外部 MCP Server 的工具桥接进现有 ToolRegistry / ToolGateway。

用法：
    from app.mcp.client import MCPClientManager
    from app.mcp.bridge import MCPBridge

    client = MCPClientManager(settings)
    await client.connect_all()
    registry = build_default_registry()
    count = MCPBridge(client).register_all(registry)   # 复用 Gateway 治理
"""
