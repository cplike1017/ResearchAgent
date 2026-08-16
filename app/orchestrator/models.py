"""多 Agent 编排层共享数据模型（Stage 12）。

编排语义：
    OrchestrationPlan    —— 一次编排的计划（步骤 + 依赖 + 理由）
    SubTask              —— 计划中的一步：交给哪个子 agent、做什么、依赖哪些步骤
    AgentRunResult       —— 一个子 agent 的执行结果（答案 / 工具调用 / 耗时 / 状态）
    OrchestrationResult  —— 一次编排的完整结果（计划 + 各子 agent 结果 + 最终合成答案）

状态机（AgentRunResult.status）：
    SUCCEEDED  子 agent 正常给出答案
    FAILED     子 agent 循环异常（超步数 / LLM 失败 / 工具全失败）
    SKIPPED    依赖步骤失败，本步未执行
"""
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """计划中的一步：委派给某个 profile 的子 agent。"""

    agent: str = Field(description="子 agent profile 名（researcher/analyst/writer/generalist）")
    task: str = Field(description="这一步要完成的具体任务（给子 agent 的指令）")
    depends_on: list[int] = Field(
        default_factory=list, description="依赖的步骤下标（0 起）；依赖完成后其结果为本文 context")
    context: str = Field(default="", description="依赖步骤结果拼接后的上下文（执行前由 Runner 填充）")


class AgentRunResult(BaseModel):
    """单个子 agent 的执行结果。"""

    agent: str
    task: str
    answer: str = ""
    tool_calls: list[dict] = Field(default_factory=list, description="子 agent 实际调用的工具 [{'name','arguments'}]")
    steps: int = 0
    duration_ms: float = 0.0
    status: str = "SUCCEEDED"  # SUCCEEDED | FAILED | SKIPPED
    error: str = ""


class OrchestrationPlan(BaseModel):
    """编排计划（planner 输出）。"""

    rationale: str = Field(default="", description="为什么这样拆（LLM 说明）")
    steps: list[SubTask] = Field(default_factory=list)


class OrchestrationResult(BaseModel):
    """一次编排的完整结果。"""

    task: str = ""
    plan: OrchestrationPlan = Field(default_factory=OrchestrationPlan)
    agent_results: list[AgentRunResult] = Field(default_factory=list)
    final_answer: str = ""
    duration_ms: float = 0.0
    status: str = "SUCCEEDED"  # SUCCEEDED | PARTIAL | FAILED
    trace_id: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "SUCCEEDED"
