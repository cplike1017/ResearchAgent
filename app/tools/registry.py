"""
Tool Registry（工具注册表）：所有工具的单一登记处。

安全说明：计算器不使用 eval()/exec()/shell，而是用 ast 手工求值，
只允许极小的算术子集，见 builtin/calculator.py。
"""
import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from app.errors import ToolError, ToolExecutionError, ToolValidationError
from app.tools.schemas import ToolResult


@dataclass
class ToolDefinition:
    """一个工具的完整定义（Stage 5 的 Gateway / Policy 将消费这些字段）。"""

    name: str
    description: str
    # 参数模型：pydantic BaseModel，用于生成 JSON Schema 与参数校验
    input_model: type[BaseModel]
    # 处理器：同步或异步函数，签名 handler(**args) -> Any
    handler: Callable[..., Any]
    # 超时秒数：Gateway 统一执行超时（Stage 5 起强制生效）
    timeout_seconds: float = 10.0
    # 风险等级：low | medium | high（Stage 5 Policy 使用）
    risk_level: str = "low"
    # 调用该工具所需的最小权限（Stage 5 Permission 校验使用）
    required_permission: str | None = None
    # 输出模型（可选）：Gateway 对返回值做 Result Validation
    output_model: type[BaseModel] | None = None
    # 注册时补充的元信息
    extra: dict = field(default_factory=dict)

    def input_schema(self) -> dict:
        """生成 JSON Schema（OpenAI function calling 的 parameters 字段）。"""
        return self.input_model.model_json_schema()

    def to_openai_schema(self) -> dict:
        """转换为 OpenAI tools 格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema(),
            },
        }

    async def invoke(self, args: dict) -> Any:
        """直接调用处理器（Gateway 之前使用；Stage 5 起统一走 Gateway）。"""
        if inspect.iscoroutinefunction(self.handler):
            return await self.handler(**args)
        return await asyncio.to_thread(self.handler, **args)


class ToolRegistry:
    """工具注册表：注册 / 查找 / 生成 Schema / 直接执行。"""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition, *, overwrite: bool = False) -> "ToolRegistry":
        if tool.name in self._tools and not overwrite:
            raise ToolError(f"工具重复注册: {tool.name}")
        self._tools[tool.name] = tool
        return self

    def get(self, name: str) -> ToolDefinition:
        if name not in self._tools:
            raise ToolError(f"未注册的工具: {name}")
        return self._tools[name]

    def all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def schemas(self) -> list[dict]:
        """供 LLM 使用的 tools 参数列表。"""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, args: dict) -> ToolResult:
        """
        Stage 1 的直接执行入口：参数校验（pydantic）→ 调用处理器 → 统一信封。
        Stage 5 起由 Tool Gateway 取代本方法成为唯一执行入口。
        """
        start = time.perf_counter()
        metadata: dict = {"args": args}
        try:
            tool = self.get(name)
        except ToolError as exc:
            return ToolResult.fail(name, exc, metadata={"duration_ms": _ms(start)})
        # 1) Schema 校验：错误参数在入口被拦截，不让底层 Tool 自行报错
        try:
            validated = tool.input_model(**args)
        except ValidationError as exc:
            err = ToolValidationError(f"参数校验失败: {exc.errors()}")
            return ToolResult.fail(name, err, metadata={"duration_ms": _ms(start)})
        # 2) 执行
        try:
            data = await tool.invoke(validated.model_dump())
            return ToolResult.ok(name, data, metadata={"duration_ms": _ms(start), "args": args})
        except ToolExecutionError as exc:
            return ToolResult.fail(name, exc, metadata={"duration_ms": _ms(start)})
        except Exception as exc:  # 未知异常也包装成结构化失败
            return ToolResult.fail(name, ToolExecutionError(str(exc)), metadata={"duration_ms": _ms(start)})


def _ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
