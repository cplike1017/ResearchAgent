"""
Policy Engine（策略引擎，第五阶段）。

回答一个面试必问题：Permission（权限）和 Policy（策略）有什么区别？
    - Permission：这个"人"能不能用这个工具？（身份维度，如 roles / permissions）
    - Policy：这个"调用"该不该放行？（规则维度，如风险等级、黑名单、确认要求）

PolicyDecision 是策略评估的结构化结果：
    decision      ALLOW | DENY | REQUIRE_CONFIRMATION
    reason        原因（写入 Trace / Job / API，可解释）
    policy_name   命中的策略名

当前决策链（Stage 10 调整）：
    1. 黑名单（denied_tools）-> DENY
    2. 风险等级确认（require_confirmation_risks，默认从 Settings 读，暂为空=全放行）
    3. （预留）RiskEvaluator（LLM 危险度评估）-> 决定 ALLOW / REQUIRE_CONFIRMATION
    4. 默认 ALLOW

设计说明（用户决策）：
    目前无人工确认通道，require_confirmation_risks 默认空（全部放行），
    避免 MCP 外部工具被策略误伤；后续引入 LLM 危险度评估器（RiskEvaluator）
    根据工具实际行为决定放行或转入人工确认。
"""
from typing import Any, Callable

from pydantic import BaseModel, Field

from app.config import Settings
from app.tools.registry import ToolDefinition
from app.tools.schemas import UserContext


class PolicyDecision(BaseModel):
    """一次策略评估的结果。"""

    decision: str = Field(description="ALLOW | DENY | REQUIRE_CONFIRMATION")
    reason: str = Field(description="人类可读的原因")
    policy_name: str = Field(description="命中的策略名")


# 风险评估器签名：async (tool, user, args) -> ("allow" | "confirm" | "deny", reason)
RiskEvaluator = Callable[[ToolDefinition, UserContext | None, dict], Any]


class PolicyEngine:
    """规则 Policy Engine（可扩展 LLM 风险评估）。"""

    def __init__(
        self,
        denied_tools: list[str] | None = None,
        require_confirmation_risks: list[str] | None = None,
        settings: Settings | None = None,
        risk_evaluator: RiskEvaluator | None = None,
    ) -> None:
        self.settings = settings or Settings()
        # 工具黑名单：命中即拒绝
        self.denied_tools = denied_tools or []
        # 达到该风险等级的工具需要人工确认
        # 优先级：显式参数 > Settings（默认空 = 全部放行，等待 HITL 通道）
        if require_confirmation_risks is not None:
            self.require_confirmation_risks = require_confirmation_risks
        else:
            raw = self.settings.policy_require_confirmation_risks
            self.require_confirmation_risks = (
                [r.strip() for r in raw.split(",") if r.strip()] if raw else []
            )
        # 预留：LLM 危险度评估器（后续实现）
        self.risk_evaluator = risk_evaluator

    def evaluate(self, tool: ToolDefinition, user: UserContext | None = None) -> PolicyDecision:
        """对一次工具调用做策略评估。"""
        # 规则 1：黑名单
        if tool.name in self.denied_tools:
            return PolicyDecision(
                decision="DENY",
                reason=f"工具 {tool.name} 被策略黑名单禁止",
                policy_name="deny_list",
            )
        # 规则 2：风险等级确认（当前配置为空 = 不触发）
        if tool.risk_level in self.require_confirmation_risks:
            return PolicyDecision(
                decision="REQUIRE_CONFIRMATION",
                reason=f"高风险工具 {tool.name}（risk_level={tool.risk_level}）需要人工确认",
                policy_name="risk_level_confirmation",
            )
        # 规则 3：默认放行（暂定；后续由 risk_evaluator 接管）
        return PolicyDecision(
            decision="ALLOW",
            reason="未命中任何限制策略，默认放行",
            policy_name="default_allow",
        )
