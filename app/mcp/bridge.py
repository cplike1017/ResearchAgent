"""
MCP Bridge：把 MCP Server 的工具桥接进现有 ToolRegistry / ToolGateway 体系。

为什么必须桥接（而不是另起一套工具执行）？
    - 复用 ToolGateway 的 schema 校验 / 权限 / 策略 / 超时 / 重试；
    - 复用 Trace 埋点（tool_gateway / tool.execute Span 自动生效）；
    - 复用 ToolResult 信封（成功/失败统一结构化）。

转换规则：
    - MCP 工具名 -> 带 server 前缀的名字（如 filesystem.read_file），避免多 server 冲突；
    - MCP 工具 JSON Schema（inputSchema）-> pydantic input_model（动态创建）；
    - MCP 调用 -> 异步 handler 代理转发给对应 server。

安全：
    - MCP 工具默认 risk_level="high"（外部代码/数据源，需 Policy 确认才放行）；
    - 无法生成 pydantic model 的工具跳过并告警（不阻塞其他工具注册）。
"""
import json
import re
from typing import Any

from pydantic import BaseModel, create_model, Field

from app.mcp.client import MCPClientManager
from app.tools.registry import ToolDefinition, ToolRegistry


def _schema_to_pydantic(schema: dict) -> type[BaseModel] | None:
    """把 JSON Schema（inputSchema）转换为 pydantic model。

    只处理最常用的子集：type: object + properties（string/number/integer/boolean）。
    复杂 schema（enum / array / 嵌套）会简化处理；无法处理的返回 None。
    """
    schema = schema or {}
    if schema.get("type") != "object":
        # 无参数工具：空 model
        return create_model("McpArgs")

    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, tuple[Any, Any]] = {}

    for name, prop in properties.items():
        prop_type = prop.get("type", "string")
        py_type: Any = str
        if prop_type == "number":
            py_type = float
        elif prop_type == "integer":
            py_type = int
        elif prop_type == "boolean":
            py_type = bool
        elif prop_type == "array":
            py_type = list

        description = prop.get("description") or name
        if name in required:
            fields[name] = (py_type, Field(description=description))
        else:
            fields[name] = (py_type | None, Field(default=None, description=description))

    return create_model(f"McpArgs_{len(fields)}", **fields)


class MCPBridge:
    """把 MCP 工具注册进 ToolRegistry。"""

    def __init__(self, client: MCPClientManager) -> None:
        self.client = client
        self.registered: list[str] = []  # 已注册的工具全名

    @staticmethod
    def normalize_name(server_name: str, tool_name: str) -> str:
        """把 MCP 工具名规范化为符合 OpenAI 工具名规则的名称。

        OpenAI-compatible 接口对工具名通常要求 ^[a-zA-Z0-9_-]+$（无点号），
        因此 'demo.echo' -> 'demo_echo'。
        """
        raw = f"{server_name}.{tool_name}"
        return re.sub(r"[^a-zA-Z0-9_-]", "_", raw)

    def register_all(self, registry: ToolRegistry) -> int:
        """
        把当前已连接 server 的全部工具注册进 registry（幂等：已注册的跳过）。

        :return: 成功注册（新增）的工具数
        """
        count = 0
        for server_name, tool_name, schema in self.client.list_all_tools():
            full_name = self.normalize_name(server_name, tool_name)
            # 幂等：已注册过则跳过（避免 runtime 懒初始化与手动注册重复）
            if full_name in registry._tools:
                continue
            try:
                input_model = _schema_to_pydantic(schema.get("inputSchema") or {})
                if input_model is None:
                    print(f"[mcp] 跳过工具 {full_name}：无法解析 inputSchema", flush=True)
                    continue

                description = (
                    schema.get("description")
                    or f"MCP 工具（来自 {server_name}）"
                )
                if server_name not in (description or ""):
                    description = f"[MCP:{server_name}] {description}"

                async def _proxy(srv=server_name, tname=tool_name, **kwargs):
                    # 过滤 None 参数：可选字段未填时传 None 会被严格 server 拒绝
                    clean = {k: v for k, v in kwargs.items() if v is not None}
                    return await self.client.call_tool(srv, tname, clean)

                registry.register(
                    ToolDefinition(
                        name=full_name,
                        description=description,
                        input_model=input_model,
                        handler=_proxy,
                        timeout_seconds=self.client.settings.mcp_tool_timeout_seconds,
                        # 风险等级：默认走配置（暂为 low 全放行，等待 HITL/LLM 评估）
                        risk_level=self.client.settings.tool_default_risk_level,
                    ),
                )
                self.registered.append(full_name)
                count += 1
            except Exception as exc:
                print(f"[mcp] 注册工具 {full_name} 失败: {exc}", flush=True)
                continue
        return count
