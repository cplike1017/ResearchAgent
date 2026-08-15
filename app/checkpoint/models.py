"""Checkpoint 数据模型。"""
from pydantic import BaseModel, Field


class CheckpointRecord(BaseModel):
    """一个检查点记录。

    保存的是 AgentState 的完整快照：
        session_id / turn_id / step / status / messages / pending_tool_calls / last_tool_result

    与数据库 Session 的区别（面试点）：
        - Session 是"业务数据"：会话元信息 + 消息历史；
        - Checkpoint 是"执行快照"：某一瞬间 Agent 的完整状态机，
          用于进程崩溃后从断点继续执行，而不是从头重放。
    """

    checkpoint_id: str
    session_id: str
    turn_id: str
    step: int
    version: int = Field(description="版本号，同一会话内 1, 2, 3... 递增")
    state: dict = Field(description="AgentState 的 JSON 序列化")
    created_at: str
