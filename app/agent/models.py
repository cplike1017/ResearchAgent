"""Agent 层共享数据模型。"""
from pydantic import BaseModel, Field

from app.llm.client import ToolCallRequest


class AgentTurnResult(BaseModel):
    """一次 Agent 回合（turn）的最终结果。"""

    session_id: str = ""
    turn_id: str = ""
    answer: str = Field(description="最终回答文本")
    steps: int = Field(description="ReAct 循环实际执行的步数（LLM 调用次数）")
    tool_calls: list[ToolCallRequest] = Field(default_factory=list, description="本回合全部工具调用")
    messages: list[dict] = Field(default_factory=list, description="本回合完整消息演变")
    trace_id: str | None = None
    checkpoint_id: str | None = None
    # ---- Stage 9 规划层 ----
    plan: list["PlanStep"] = Field(default_factory=list, description="本次执行使用的计划（plan 模式）")
    plan_revisions: int = Field(default=0, description="反思触发的重规划次数（越低越好，Eval 指标）")


class PlanStep(BaseModel):
    """计划中的一步。

    状态机：PLANNED -> RUNNING -> SUCCEEDED | FAILED | SKIPPED
    """

    step_id: str = Field(description="步骤唯一 ID，如 plan_0001")
    description: str = Field(description="这一步要做什么（给 LLM 看的人类可读描述）")
    tools_hint: list[str] = Field(default_factory=list, description="建议使用的工具名（可空）")
    status: str = "PLANNED"  # PLANNED | RUNNING | SUCCEEDED | FAILED | SKIPPED
    result: str = Field(default="", description="执行结果摘要（工具结果 / 失败原因）")
    order: int = Field(default=0, description="执行顺序（0 起）")
