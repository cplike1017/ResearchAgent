"""delegate 工具：主 agent 的多 Agent 委派入口。

主 agent 在对话中可以直接调用：
    delegate(task="调研 X 并写分析报告", agents=["researcher", "analyst"])
返回结构化编排结果（plan + 各子 agent 结果 + 最终合成答案），
作为工具结果重新进入主 agent 的消息流 —— 主 agent 看到的是"一次工具调用"，
而 Trace 树里能看到完整的 orchestrator.run → agent.run → llm_call/tool.execute 层级。

风险与超时：编排可能运行较久（多个子 agent 各自跑 ReAct，每步一次 LLM 调用），
timeout_seconds 设为 600s（实测 3 个子 agent 全链路约 3~6 分钟）；risk_level 保持 low
（工具本身只读性质，子 agent 的能力边界由其档案白名单决定）。
"""
from pydantic import BaseModel, Field

from app.orchestrator.runner import OrchestratorRunner
from app.tools.registry import ToolDefinition

TOOL_TIMEOUT_SECONDS = 600.0


class DelegateArgs(BaseModel):
    """delegate 工具参数。"""

    task: str = Field(description="要委派给子 Agent 的完整任务描述（自包含，子 Agent 看不到主对话历史）")
    agents: list[str] | None = Field(
        default=None,
        description="指定参与的子 Agent 名单（researcher/analyst/writer/generalist）；缺省由编排规划器自动分工",
    )
    context: str = Field(default="", description="主 Agent 提供的背景信息（如已收集的资料摘要）")


def build_delegate_tool(orchestrator: OrchestratorRunner) -> ToolDefinition:
    """构造 delegate 工具（闭包绑定编排器实例）。"""

    async def handler(task: str, agents: list[str] | None = None, context: str = "") -> dict:
        # 第二道防线：即使工具可见性被绕过，深度超限也拒绝委派
        from app.orchestrator.context import current_session_id, orchestration_depth

        if orchestration_depth.get() >= orchestrator.settings.orchestrator_max_depth:
            return {
                "status": "FAILED",
                "error": (
                    f"编排深度已达上限（max_depth={orchestrator.settings.orchestrator_max_depth}），"
                    "本层子 Agent 不能再向下委派。请在本层完成任务。"
                ),
                "plan": {"rationale": "", "steps": []},
                "agent_results": [],
                "final_answer": "",
                "duration_ms": 0.0,
                "trace_id": None,
            }
        # 携带会话上下文：委派结果持久化到当前会话（主 agent 由 AgentRuntime 注入）
        session_id = current_session_id.get() or ""
        result = await orchestrator.run(task, agents=agents, context=context, session_id=session_id)
        return result.model_dump(mode="json")

    return ToolDefinition(
        name="delegate",
        description=(
            "把复杂任务委派给多个专业子 Agent（研究员/数据分析师/报告写手等）并行或串行协作，"
            "返回整合后的最终结果。适合需要『检索资料 + 数据分析 + 撰写报告』等多能力组合的任务。"
        ),
        input_model=DelegateArgs,
        handler=handler,
        timeout_seconds=TOOL_TIMEOUT_SECONDS,
        risk_level="low",
    )
