"""Session 数据模型：会话与消息。"""
from enum import Enum

from pydantic import BaseModel, Field


class SessionStatus(str, Enum):
    """会话状态。"""

    ACTIVE = "ACTIVE"   # 进行中
    CLOSED = "CLOSED"   # 已关闭


class SessionRecord(BaseModel):
    """一条会话记录。"""

    session_id: str
    created_at: str
    updated_at: str
    status: SessionStatus = SessionStatus.ACTIVE


class MessageRecord(BaseModel):
    """一条消息记录。

    content 字段保存"完整消息 dict 的 JSON 序列化"（可能含 tool_calls 结构），
    role 字段单独冗余一份便于查询与统计。
    """

    message_id: str
    session_id: str
    role: str
    content: str = Field(description="完整消息 dict 的 JSON 字符串")
    created_at: str
