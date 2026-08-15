"""
Policy Engine（策略引擎，第五阶段）。

回答一个面试必问题：Permission（权限）和 Policy（策略）有什么区别？
    - Permission：这个"人"能不能用这个工具？（身份维度，如 roles / permissions）
    - Policy：这个"调用"该不该放行？（规则维度，如风险等级、黑名单、确认要求）

PolicyDecision 是策略评估的结构化结果：
    decision      ALLOW | DENY | REQUIRE_CONFIRMATION
    reason        原因（写入 Trace / Job / API，可解释）
    policy_name   命中的策略名
"""
from pydantic import BaseModel, Field

from app.tools.registry import ToolDefinition
from app.tools.schemas import UserContext


class PolicyDecision(BaseModel):
    """一次策略评估的结果。"""

    decision: str = Field(description="ALLOW | DENY | REQUIRE_CONFIRMATION")
    reason: str = Field(description="人类可读的原因")
    policy_name: str = Field(description="命中的策略名")


class PolicyEngine:
    """最小规则 Policy Engine（后续可扩展为规则文件 / 动态加载）。"""

    def __init__(
        self,
        denied_tools: list[str] | None = None,
        require_confirmation_risks: list[str] | None = None,
    ) -> None:
        # 工具黑名单：命中即拒绝
        self.denied_tools = denied_tools or []
        # 达到该风险等级的工具需要人工确认
        self.require_confirmation_risks = require_confirmation_risks or ["high"]

    def evaluate(self, tool: ToolDefinition, user: UserContext | None = None) -> PolicyDecision:
        """对一次工具调用做策略评估。"""
        # 规则 1：黑名单
        if tool.name in self.denied_tools:
            return PolicyDecision(
                decision="DENY",
                reason=f"工具 {tool.name} 被策略黑名单禁止",
                policy_name="deny_list",
            )
        # 规则 2：风险等级确认
        if tool.risk_level in self.require_confirmation_risks:
            return PolicyDecision(
                decision="REQUIRE_CONFIRMATION",
                reason=f"高风险工具 {tool.name}（risk_level={tool.risk_level}）需要人工确认",
                policy_name="risk_level_confirmation",
            )
        # 规则 3：默认放行
        return PolicyDecision(
            decision="ALLOW",
            reason="未命中任何限制策略，默认放行",
            policy_name="default_allow",
        )
