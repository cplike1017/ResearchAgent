"""子 Agent 执行器（SubAgentExecutor）：让一个子 agent 独立跑一轮 ReAct 循环。

隔离原则（子 agent 与主 agent 的关键差异）：
    - 工具隔离：子 agent 只拿到自己档案白名单内的工具（过滤后的 registry + gateway）
    - 上下文隔离：只看到 system_prompt + 委派任务（+ 依赖步骤结果），看不到主 agent 历史
    - 无持久化：子 agent 是"临时工"，不写 Session / Checkpoint / Memory，
      结果以 AgentRunResult 返回给编排层（编排层才负责持久化）
    - 追踪继承：子 agent 的 agent.run span 嵌套在 orchestrator.run span 之下，
      Web UI 的 Trace 树因此能看到"主管 → 员工 → 工具"的完整层级

工具过滤实现：从主 registry 中复制白名单内的 ToolDefinition 到新 registry，
再为该 registry 建独立 ToolGateway —— 网关层（校验/权限/策略/超时）对子 agent 同样生效。
"""
import time

from app.config import Settings, get_settings
from app.llm.client import BaseLLMClient
from app.orchestrator.context import orchestration_depth
from app.orchestrator.models import AgentRunResult
from app.orchestrator.profiles import AgentProfile
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolRegistry
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span


class SubAgentExecutor:
    """执行单个子 agent（独立 ReAct 循环 + 过滤工具集）。"""

    def __init__(
        self,
        *,
        llm: BaseLLMClient,
        master_registry: ToolRegistry,
        settings: Settings | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm
        self.master_registry = master_registry
        self.recorder = recorder

    # ------------------------------------------------------------------
    def _filtered_registry(self, profile: AgentProfile) -> ToolRegistry:
        """按档案白名单过滤工具（None = 全部）。未知工具名静默跳过。

        delegate 特殊处理（多级编排核心）：
            - 它是编排层的"元能力"，不属于任何档案的工具清单，因此不受
              档案白名单限制 —— 任何子 agent 在非叶子层都可以再向下委派；
            - 可见性只由深度控制：depth < orchestrator_max_depth 保留，
              叶子层（depth >= max_depth）物理移除，模型根本不会产生
              嵌套调用。这是"递归有界"的第一道防线。
        """
        allow_delegate = orchestration_depth.get() < self.settings.orchestrator_max_depth
        registry = ToolRegistry()
        for tool in self.master_registry.all():
            if tool.name == "delegate":
                if allow_delegate:
                    registry.register(tool)
                continue
            if profile.allowed_tools is not None and tool.name not in profile.allowed_tools:
                continue
            registry.register(tool)
        return registry

    # ------------------------------------------------------------------
    async def execute(
        self,
        profile: AgentProfile,
        task: str,
        context: str = "",
    ) -> AgentRunResult:
        """运行一个子 agent。返回结构化结果，任何异常都被包装为 FAILED 结果。"""
        start = time.perf_counter()
        registry = self._filtered_registry(profile)
        gateway = ToolGateway(registry, settings=self.settings, recorder=self.recorder)

        user_content = task if not context else f"{task}\n\n【依赖步骤结果（仅作参考）】\n{context}"

        async def _execute_tool(name: str, args: dict):
            return await gateway.execute(name, args)

        # 消息序列：system（人设）+ user（委派任务）
        messages = [
            {"role": "system", "content": profile.system_prompt},
            {"role": "user", "content": user_content},
        ]

        result = AgentRunResult(agent=profile.name, task=task)
        try:
            # 子 agent 的 llm 调用也包上 llm_call span（嵌套在 agent.run 下）
            from app.agent.react_loop import run_react_loop

            llm_for_loop = _InstrumentedLLM(self.llm, self.recorder, self.settings.llm_model)

            async with trace_span(
                "agent.run",
                "agent",
                input={"agent": profile.name, "task": task, "tools": len(registry.all())},
                attributes={"agent_profile": profile.name, "orchestrated": True},
                recorder=self.recorder,
            ) as span:
                final_messages, answer, steps, tool_calls = await run_react_loop(
                    llm=llm_for_loop,
                    tools_schema=registry.schemas(),
                    messages=messages,
                    execute_tool=_execute_tool,
                    max_steps=profile.max_steps,
                )
                span.output = {
                    "answer": answer,
                    "steps": steps,
                    "tool_calls": len(tool_calls),
                    "tools_available": [t.name for t in registry.all()],
                }
            result.answer = answer or "(子 agent 未返回内容)"
            result.steps = steps
            result.tool_calls = [
                {"name": tc.name, "arguments": tc.arguments} for tc in tool_calls
            ]
            result.status = "SUCCEEDED"
        except Exception as exc:
            result.status = "FAILED"
            result.error = f"{type(exc).__name__}: {exc}"
            # 失败也保留部分成果：取消息序列中最后一条 assistant 文本，
            # 让下游子 agent 仍能利用已收集的资料（如超步数前检索到的摘要）
            for m in reversed(messages):
                if m.get("role") == "assistant" and m.get("content"):
                    result.answer = f"（部分成果）{m['content'][:2000]}"
                    break
        finally:
            result.duration_ms = round((time.perf_counter() - start) * 1000, 3)
        return result


class _InstrumentedLLM:
    """把子 agent 的 llm.chat 包上 llm_call span（对 ReAct 循环透明）。"""

    def __init__(self, llm: BaseLLMClient, recorder: TraceRecorder | None, model: str) -> None:
        self._llm = llm
        self._recorder = recorder
        self._model = model

    async def chat(self, messages, tools=None, **kwargs):
        from app.tracing.span import trace_span

        async with trace_span(
            "llm_call",
            "llm",
            input=messages,
            attributes={"model": self._model},
            recorder=self._recorder,
        ) as span:
            response = await self._llm.chat(messages, tools)
            span.attributes.update(
                model=response.model or self._model,
                finish_reason=response.finish_reason,
                total_tokens=response.usage.get("total_tokens", 0),
            )
            span.output = {
                "finish_reason": response.finish_reason,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
                "content_preview": (response.content or "")[:120],
            }
            return response
