"""
Reflector（反思器，Stage 9 核心）：执行后检查结果，决定是否需要重新规划。

回答的问题：
    - 这一步真的完成了吗？（步骤 FAILED 或结果为空 → 需要反思）
    - 计划的步骤覆盖了用户任务吗？（漏步骤 → 需要补计划）
    - 要不要重规划？（受 max_plan_revisions 限制，防死循环）

输出 ReflectDecision：
    - need_replan: 是否重新规划
    - reason: 反思理由（写入 Trace / 结果，可解释）
    - revised_task: 重规划时使用的修订后任务（把失败信息并入，指导新计划）

设计原则：
    - Reflector 只做"判断"，不执行（重规划由 Plan Loop 调用 Planner 完成）；
    - 判断规则确定性（不调 LLM）：步骤级失败 / 空结果 / 计划为空 都算需要反思；
    - LLM 反思作为后续增强项（判断"结果质量"需要模型判断）。
"""
from app.agent.models import PlanStep


class ReflectDecision:
    """一次反思的结论。"""

    def __init__(self, need_replan: bool, reason: str, revised_task: str | None = None) -> None:
        self.need_replan = need_replan
        self.reason = reason
        self.revised_task = revised_task

    def to_dict(self) -> dict:
        return {
            "need_replan": self.need_replan,
            "reason": self.reason,
            "revised_task": self.revised_task,
        }


class Reflector:
    """基于步骤结果的确定性反思器。"""

    def __init__(self, max_revisions: int = 2) -> None:
        self.max_revisions = max_revisions

    def reflect(
        self,
        task: str,
        plan: list[PlanStep],
        *,
        revisions_so_far: int = 0,
    ) -> ReflectDecision:
        """对一次执行结果做反思。"""
        # 1) 无计划（off 策略降级执行）——不需要反思
        if not plan:
            return ReflectDecision(False, "无计划，直接执行", task)

        # 2) 重规划次数超限——不再重规划，接受现状
        if revisions_so_far >= self.max_revisions:
            return ReflectDecision(False, f"已达最大重规划次数 {self.max_revisions}，接受当前结果", task)

        # 3) 统计失败步骤
        failed = [s for s in plan if s.status == "FAILED"]
        empty = [s for s in plan if s.status == "SUCCEEDED" and not s.result]

        if failed:
            failed_desc = "；".join(s.description for s in failed)
            revised = f"{task}\n（注意：以下步骤此前失败了，请重新规划如何处理：{failed_desc}）"
            return ReflectDecision(True, f"{len(failed)} 个步骤失败：{failed_desc}", revised)

        if empty:
            desc = "；".join(s.description for s in empty)
            revised = f"{task}\n（注意：以下步骤没有产出结果，请重新规划：{desc}）"
            return ReflectDecision(True, f"{len(empty)} 个步骤结果为空：{desc}", revised)

        # 4) 全部成功——结束
        return ReflectDecision(False, f"全部 {len(plan)} 个步骤成功", task)
