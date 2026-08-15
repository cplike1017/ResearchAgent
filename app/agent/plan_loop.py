"""
Plan Loop（规划执行总控，Stage 9 核心）。

流程：
    Planner.plan(task) → 计划步骤列表
    ↓
    逐 step 执行（每步复用 ReAct 循环，但上下文注入"当前步骤指令"，
    让模型聚焦单步目标，不会跑偏到整任务）
    ↓
    每步结果写入 PlanStep.result / status
    ↓
    全部步骤完成 → 汇总最终回答（把各步结果交给 LLM 组织）

与 react 模式的区别：
    - react：整任务一次性进入循环，模型自己决定先后；
    - plan：任务先分解，每步一个小循环，步骤状态可追踪、可恢复。

关键设计（修复 OpenAI 协议非法序列）：
    每步执行使用【完全独立】的 messages（step_messages），与全局历史隔离：
      - 每步内的 ReAct 循环产生的 assistant(tool_calls) + tool 消息
        只在 step_messages 内合法交替，绝不写入全局；
      - 每步结束后，仅把【干净的步骤结果】以纯文本 assistant 消息写回
        全局 messages（user 提问 → assistant 结果交替），保证全局序列
        永远满足 OpenAI 协议（无孤儿 tool 消息、无乱序）；
      - 重规划时基于干净的全局历史重新执行，不会残留非法片段。
"""
import json

from app.agent.models import PlanStep
from app.agent.planner import Planner
from app.agent.react_loop import LoopHooks, run_react_loop
from app.llm.client import BaseLLMClient, ToolCallRequest
from app.tools.registry import ToolRegistry


class PlanExecutor:
    """按计划逐步执行，返回各步结果与最终回答。"""

    def __init__(
        self,
        llm: BaseLLMClient,
        registry: ToolRegistry,
        execute_tool,
        planner: Planner,
        *,
        max_steps_per_step: int = 4,
        context_builder=None,
        hooks: LoopHooks | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.execute_tool = execute_tool
        self.planner = planner
        self.max_steps_per_step = max_steps_per_step
        self.context_builder = context_builder
        self.hooks = hooks

    # ------------------------------------------------------------------
    async def execute(
        self,
        task: str,
        messages: list[dict],
        memory_context: list[str] | None = None,
    ) -> tuple[list[PlanStep], str, list[ToolCallRequest]]:
        """
        执行计划。

        :param task: 用户原始任务（用于规划）
        :param messages: 全局会话消息（仅追加干净的 user/assistant 结果，保持协议合法）
        :param memory_context: Stage 8 记忆层检索到的相关记忆（可空，供规划参考）
        :return: (计划步骤(含结果), 最终回答, 全部工具调用)
        """
        # 1) 规划（传入记忆上下文）
        plan = await self.planner.plan(task, memory_context)
        if not plan:
            # 无计划（off 策略）：降级为直接 ReAct 完整执行
            final_messages, answer, _, calls = await run_react_loop(
                llm=self.llm,
                tools_schema=self.registry.schemas(),
                messages=messages,
                execute_tool=self.execute_tool,
                max_steps=self.max_steps_per_step * 2,
                context_builder=self.context_builder,
                hooks=self.hooks,
            )
            return [], answer, calls

        all_calls: list[ToolCallRequest] = []
        step_results: list[str] = []

        # 2) 逐步执行（每步独立 messages）
        total = len(plan)
        for step in plan:
            step.status = "RUNNING"
            # 每步构造【完全独立】的消息视图：
            #   system: 步骤指令（"只做本步"）
            #   user:   该步骤的独立子任务描述
            #   （可选）assistant: 上一步结果摘要（纯文本，协议合法）
            step_messages: list[dict] = []
            if step_results:
                step_messages.append(
                    {"role": "assistant", "content": f"已完成的上一步结果：{'；'.join(step_results[-2:])}"}
                )
            step_messages.append({"role": "system", "content": _step_prompt(task, step, total)})
            step_messages.append({"role": "user", "content": step.description})

            try:
                _, step_answer, _, step_calls = await run_react_loop(
                    llm=self.llm,
                    tools_schema=self.registry.schemas(),
                    messages=step_messages,
                    execute_tool=self.execute_tool,
                    max_steps=self.max_steps_per_step,
                    context_builder=self.context_builder,
                    hooks=self.hooks,
                )
                all_calls.extend(step_calls)
                # 步骤成功标准：有回答 且（无工具要求 或 至少调用了一次工具）
                success = bool(step_answer) and (not step.tools_hint or bool(step_calls))
                step.status = "SUCCEEDED" if success else "FAILED"
                step.result = step_answer[:120] if step_answer else "无回答"
                step_results.append(step.result)
            except Exception as exc:
                step.status = "FAILED"
                step.result = f"{type(exc).__name__}: {str(exc)[:100]}"
                step_results.append(step.result)

            # 每步结束后：仅把【干净的步骤结果】写回全局消息
            # （user 提问 + assistant 结果交替，保证协议合法、可持久化）
            # 每步结束后：仅把【干净的步骤结果】写回全局消息
            # （原始 user 消息已由 runtime 的 _prepare_messages 追加，
            #  这里只追加 assistant 结果，保证 user/assistant 交替、协议合法）
            messages.append({"role": "assistant", "content": f"[{step.description}] {step.result}"})

        # 3) 汇总最终回答（独立上下文，不污染全局消息）
        answer = await self._summarize(task, plan)
        messages.append({"role": "assistant", "content": answer})
        return plan, answer, all_calls

    # ------------------------------------------------------------------
    async def _summarize(self, task: str, plan: list[PlanStep]) -> str:
        """把各步结果汇总为最终回答（LLM 组织；失败则拼接各步结果）。"""
        step_lines = "\n".join(
            f"- [{s.status}] {s.description}：{s.result}" for s in plan
        )
        prompt = (
            f"用户任务：{task}\n\n"
            f"各步骤执行结果：\n{step_lines}\n\n"
            "请基于以上结果，给用户一个完整、自然的最终回答。直接输出回答内容。"
        )
        try:
            response = await self.llm.chat([{"role": "user", "content": prompt}], tools=None)
            if response.content:
                return response.content
        except Exception:
            pass
        # 降级：拼接各步结果
        return "；".join(f"{s.description}：{s.result}" for s in plan if s.result)


def _step_prompt(task: str, step: PlanStep, total: int) -> str:
    """构造当前步骤的指令（注入每步 ReAct 循环的上下文）。"""
    hint = f"，建议使用工具: {', '.join(step.tools_hint)}" if step.tools_hint else ""
    return (
        f"[当前子任务 {step.order + 1}/{total}] {step.description}{hint}。"
        f"只完成这一个子任务，完成后直接给出该子任务的结果，不要执行其他步骤。"
    )
