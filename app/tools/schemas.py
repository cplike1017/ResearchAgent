"""
统一 Tool 结果信封（Tool Result Envelope）。

Stage 1 起所有工具执行结果都包装成统一结构写回 Messages，
这样 Agent 循环、Stub 模型解析、Stage 5 Gateway、Stage 6 Trace 都用同一份数据格式。
（Stage 5 会在此基础上补充 Schema 校验 / 权限 / Policy / 超时等治理能力。）
"""
from typing import Any

from pydantic import BaseModel, Field

from app.errors import error_to_dict


class UserContext(BaseModel):
    """调用方的身份上下文（Stage 5 Permission / Policy 使用）。"""

    user_id: str = ""
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)


class ToolErrorInfo(BaseModel):
    """结构化错误信息（与 app.errors 中的异常类型一一对应）。"""

    type: str = Field(description="异常类型名，如 ToolValidationError / ToolTimeoutError")
    message: str = Field(description="人类可读的错误描述")
    code: str = ""


class ToolResult(BaseModel):
    """工具执行的统一返回结构。"""

    success: bool
    tool_name: str
    data: Any = None
    error: ToolErrorInfo | None = None
    metadata: dict = Field(default_factory=dict, description="耗时、入参等元信息")

    def to_json(self) -> str:
        """序列化为写入 Messages 的字符串内容。"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, raw: str) -> "ToolResult":
        return cls.model_validate_json(raw)

    @classmethod
    def ok(cls, tool_name: str, data: Any, metadata: dict | None = None) -> "ToolResult":
        return cls(success=True, tool_name=tool_name, data=data, metadata=metadata or {})

    @classmethod
    def fail(cls, tool_name: str, exc: BaseException, metadata: dict | None = None) -> "ToolResult":
        """把任意异常包装为失败结果（统一走 error_to_dict，避免堆栈外泄）。"""
        return cls(
            success=False,
            tool_name=tool_name,
            error=ToolErrorInfo(**error_to_dict(exc)),
            metadata=metadata or {},
        )
