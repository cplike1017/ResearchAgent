"""
AgentRuntime（第一~六阶段完整版）。

职责：把"用户输入"组装成一次 Agent 回合执行，并具备：
    - ReAct 循环（react_loop.run_react_loop）
    - 上下文构建（ContextBuilder）
    - 会话持久化（SessionRepository）：历史跨回合保留
    - 检查点（CheckpointRepository）：关键状态变化后保存，支持断点恢复
    - 工具治理（ToolGateway）：校验 / 权限 / 策略 / 超时
    - 链路追踪（TraceRecorder）：agent.run / llm_call / context_builder /
      checkpoint.save / checkpoint.load 等 Span

恢复语义：
    run() 内部分为两段：
        1) 准备状态（新建或加载 Session / Checkpoint）
        2) _run_with_state()：运行 ReAct 循环（run/resume 共用）
    resume() 从最新 Checkpoint 恢复状态后调用 _run_with_state() 继续，
    绝不重新从用户输入完整执行 —— 这是"真恢复"与"假恢复"的区别。
"""
import json
from uuid import uuid4

from app.agent.context_builder import ContextBuilder
from app.agent.models import AgentTurnResult, PlanStep
from app.agent.plan_loop import PlanExecutor
from app.agent.planner import Planner
from app.agent.react_loop import LoopHooks, run_react_loop
from app.agent.reflector import Reflector
from app.agent.state import AgentState
from app.checkpoint.models import CheckpointRecord
from app.config import Settings, get_settings
from app.errors import AgentError, CheckpointError
from app.llm.client import BaseLLMClient, ToolCallRequest, create_llm_client
from app.memory.store import MemoryStore
from app.tools.builtin import build_default_registry
from app.tools.gateway import ToolGateway
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolResult, UserContext
from app.tracing.context import current_trace_id
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span, trace_span_sync


class _InstrumentedLLM:
    """把 llm.chat 包上 llm_call Span 的适配器（对 ReAct 循环透明）。"""

    def __init__(self, chat_fn) -> None:
        self._chat = chat_fn

    async def chat(self, messages, tools=None, **kwargs):
        return await self._chat(messages, tools)


class AgentRuntime:
    """Agent 运行时（六阶段能力全部接入）。"""

    def __init__(
        self,
        *,
        llm: BaseLLMClient | None = None,
        registry: ToolRegistry | None = None,
        tool_gateway: ToolGateway | None = None,
        context_builder: ContextBuilder | None = None,
        session_repo=None,      # SQLiteSessionRepository | None
        checkpoint_repo=None,   # SQLiteCheckpointRepository | None
        recorder: TraceRecorder | None = None,  # None = 不追踪
        memory: MemoryStore | None = None,  # Stage 8 记忆层（None = 不启用）
        mcp_client=None,  # MCPClientManager | None（None = 不接入 MCP）
        skill_manager=None,  # SkillManager | None（None = 不启用技能）
        orchestrator=None,  # OrchestratorRunner | None（None = 不注册 delegate 工具）
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or create_llm_client(self.settings)
        self.registry = registry or build_default_registry()
        # Stage 12：编排器注入时，向 registry 注册 delegate 工具（主 agent 的委派入口）
        self.orchestrator = orchestrator
        if orchestrator is not None and self.settings.orchestrator_enabled:
            from app.orchestrator.tool import build_delegate_tool

            self.registry.register(build_delegate_tool(orchestrator), overwrite=True)
        self.recorder = recorder
        # Stage 5：工具执行统一走 Tool Gateway（校验 / 权限 / 策略 / 超时）
        self.tool_gateway = tool_gateway or ToolGateway(self.registry, settings=self.settings, recorder=recorder)
        self.context_builder = context_builder or ContextBuilder(self.settings, llm=self.llm, recorder=recorder)
        self.session_repo = session_repo
        self.checkpoint_repo = checkpoint_repo
        # Stage 8：记忆层（显式注入才启用；缺省不自动创建，保持向后兼容）
        self.memory = memory if (memory is not None and memory.enabled) else None
        # MCP：注入的 client（调用方负责 connect/close）；未注入则不接入
        self.mcp_client = mcp_client
        self._mcp_initialized = False
        # Skill：显式注入才启用（缺省 None = 不启用，保持向后兼容）
        self.skill_manager = skill_manager if (skill_manager is not None and skill_manager.enabled) else None
        # 每个 run/resume 调用内的临时状态
        self._state: AgentState | None = None
        self._persisted = 0
        self._last_checkpoint: CheckpointRecord | None = None
        self._extra_hooks: LoopHooks | None = None
        self._current_user: UserContext | None = None

    # ------------------------------------------------------------------
    # 工具执行入口：走 Tool Gateway（含 tool_gateway / tool.execute Span）
    # ------------------------------------------------------------------
    async def _execute_tool(self, name: str, args: dict) -> ToolResult:
        return await self.tool_gateway.execute(name, args, user=self._current_user)

    # ------------------------------------------------------------------
    # LLM 调用入口：包上 llm_call Span（记录 model / tokens / finish_reason）
    # ------------------------------------------------------------------
    async def _llm_chat(self, messages: list[dict], tools: list[dict] | None) -> None:
        async with trace_span(
            "llm_call",
            "llm",
            input=messages,
            attributes={"model": self.settings.llm_model},
            recorder=self.recorder,
        ) as span:
            response = await self.llm.chat(messages, tools)
            span.attributes.update(
                model=response.model or self.settings.llm_model,
                finish_reason=response.finish_reason,
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                total_tokens=response.usage.get("total_tokens", 0),
            )
            span.output = {
                "finish_reason": response.finish_reason,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls],
                "content_preview": (response.content or "")[:120],
            }
            return response

    # ------------------------------------------------------------------
    # 消息准备：从 Session 加载历史
    # ------------------------------------------------------------------
    async def _prepare_messages(self, message: str, session_id: str) -> list[dict]:
        messages: list[dict] = []
        if self.session_repo is not None:
            raw = self.session_repo.list_messages(session_id)
            # 库中原有条数（持久化计数基准：修复会增删消息，但库里还是原始条数）
            self._persisted = len(raw)
            # 防御：修复历史中不完整的 tool 配对 / 连续 user（如之前中断残留）
            messages = _repair_tool_pairing(raw)
        else:
            self._persisted = 0
        messages.append({"role": "user", "content": message})
        return messages

    # ------------------------------------------------------------------
    # 持久化 + 检查点
    # ------------------------------------------------------------------
    def _persist_and_checkpoint(self, state: AgentState, messages: list[dict], point: str) -> None:
        """把新消息写入 Session 并保存 Checkpoint（point 是保存原因标记）。

        防御：只持久化"tool 配对完整"的消息段——若 assistant(tool_calls) 声明了
        N 个调用但 tool 结果不足 N 条（如执行被取消中断），跳过持久化，
        避免把不完整序列写入历史（否则下次请求网关报 "No tool output found"）。
        """
        if self.session_repo is not None:
            new_messages = messages[self._persisted:]
            # 校验 tool 配对：从末尾回溯，找到最后一个不完整块并截断
            safe_new = _truncate_incomplete_tool_block(new_messages)
            for m in safe_new:
                self.session_repo.add_message(state.session_id, m)
            # 只推进到安全位置（未持久化的留在内存，等待补全）
            self._persisted = self._persisted + len(safe_new)
            self.session_repo.touch(state.session_id)
        if self.checkpoint_repo is not None:
            state.messages = messages
            with trace_span_sync(
                "checkpoint.save",
                "checkpoint",
                input={"point": point, "session_id": state.session_id},
                recorder=self.recorder,
            ) as span:
                self._last_checkpoint = self.checkpoint_repo.save(
                    session_id=state.session_id,
                    turn_id=state.turn_id,
                    step=state.step,
                    state=state.model_dump(mode="json"),
                )
                span.attributes.update(
                    checkpoint_id=self._last_checkpoint.checkpoint_id,
                    version=self._last_checkpoint.version,
                    session_id=state.session_id,
                    point=point,
                )

    # ------------------------------------------------------------------
    # 钩子（挂到 ReAct 循环的关键节点）
    # ------------------------------------------------------------------
    async def _hook_before_llm(self, step: int, messages: list[dict]) -> None:
        self._state.step = step
        self._persist_and_checkpoint(self._state, messages, "before_llm")
        if self._extra_hooks and self._extra_hooks.before_llm:
            await self._extra_hooks.before_llm(step, messages)

    async def _hook_after_decision(self, response, step: int) -> None:
        """LLM 决策后：记录待执行工具（PENDING_TOOL），保存检查点。"""
        state = self._state
        if response.tool_calls:
            state.status = "PENDING_TOOL"
            state.pending_tool_calls = [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls
            ]
        else:
            state.status = "RUNNING"
            state.pending_tool_calls = []
        self._persist_and_checkpoint(state, state.messages, "after_decision")
        if self._extra_hooks and self._extra_hooks.after_decision:
            await self._extra_hooks.after_decision(response, step)

    async def _hook_after_tool(self, tc, envelope: ToolResult, step: int) -> None:
        """工具执行后：状态回到 RUNNING，保存检查点。"""
        state = self._state
        state.status = "RUNNING"
        state.pending_tool_calls = []
        state.last_tool_result = envelope.model_dump(mode="json")
        self._persist_and_checkpoint(state, state.messages, "after_tool")
        if self._extra_hooks and self._extra_hooks.after_tool:
            await self._extra_hooks.after_tool(tc, envelope, step)

    async def _hook_before_final(self, response, step: int) -> None:
        """最终回答前：状态置 DONE，保存最后检查点。"""
        state = self._state
        state.status = "DONE"
        state.pending_tool_calls = []
        self._persist_and_checkpoint(state, state.messages, "before_final")
        if self._extra_hooks and self._extra_hooks.before_final:
            await self._extra_hooks.before_final(response, step)

    def _build_hooks(self) -> LoopHooks:
        return LoopHooks(
            before_llm=self._hook_before_llm,
            after_decision=self._hook_after_decision,
            after_tool=self._hook_after_tool,
            before_final=self._hook_before_final,
        )

    # ------------------------------------------------------------------
    # 核心执行段（run 与 resume 共用）
    # ------------------------------------------------------------------
    async def _run_with_state(self, state: AgentState) -> AgentTurnResult:
        """基于给定状态运行/继续（react 或 plan 模式）。"""
        self._state = state
        messages = state.messages

        # Stage 8：构建上下文前，用最后一条用户消息检索相关记忆
        # 分层：传入本会话 session_id → 全局记忆 + 本会话会话级记忆
        retrieved_docs: list[str] | None = None
        query = _last_user_text(messages)
        if self.memory is not None:
            if query:
                retrieved_docs = await self.memory.retrieve(query, session_id=state.session_id)

        # Skill：匹配用户输入对应的技能，指令注入上下文
        if self.skill_manager is not None and query:
            matched = await self.skill_manager.matched_skills(query)
            if matched:
                skill_blocks = [s.to_prompt_block() for s in matched]
                retrieved_docs = (retrieved_docs or []) + skill_blocks

        # 追踪开启时，把 llm.chat 替换为带 llm_call Span 的版本
        llm_for_loop = self.llm
        if self.recorder is not None and self.recorder.enabled:
            llm_for_loop = _InstrumentedLLM(self._llm_chat)

        # ---- Stage 9：plan 模式（计划 → 执行 → 反思）----
        if state.agent_mode == "plan":
            plan, answer, tool_calls, plan_revisions = await self._run_plan_mode(
                state, messages, llm_for_loop, retrieved_docs
            )
            final_messages = messages
            steps = sum(1 for _ in plan) or 0
            result = AgentTurnResult(
                session_id=state.session_id,
                turn_id=state.turn_id,
                answer=answer,
                steps=steps,
                tool_calls=tool_calls,
                messages=final_messages,
                checkpoint_id=self._last_checkpoint.checkpoint_id if self._last_checkpoint else None,
                trace_id=current_trace_id.get(),
                plan=plan,
                plan_revisions=plan_revisions,
            )
        else:
            # ---- react 模式（原行为）----
            final_messages, answer, steps, tool_calls = await run_react_loop(
                llm=llm_for_loop,
                tools_schema=self.registry.schemas(),
                messages=messages,
                execute_tool=self._execute_tool,
                max_steps=self.settings.max_agent_steps,
                context_builder=self.context_builder,
                hooks=self._build_hooks(),
                retrieved_docs=retrieved_docs,
            )
            result = AgentTurnResult(
                session_id=state.session_id,
                turn_id=state.turn_id,
                answer=answer,
                steps=steps,
                tool_calls=tool_calls,
                messages=final_messages,
                checkpoint_id=self._last_checkpoint.checkpoint_id if self._last_checkpoint else None,
                trace_id=current_trace_id.get(),
            )

        # Stage 8：回合结束后，把本轮信息提炼写入记忆（下一轮才能检索到）
        if self.memory is not None and self.settings.memory_auto_extract:
            await self.memory.remember(
                result.messages, session_id=state.session_id, turn_id=state.turn_id
            )

        return result

    # ------------------------------------------------------------------
    # Stage 9：plan 模式执行（计划 → 逐步骤 → 反思 → 必要时重规划）
    # ------------------------------------------------------------------
    async def _run_plan_mode(
        self,
        state: AgentState,
        messages: list[dict],
        llm_for_loop,
        retrieved_docs: list[str] | None,
    ) -> tuple[list[PlanStep], str, list[ToolCallRequest], int]:
        """计划模式主循环：Planner 出计划 → PlanExecutor 执行 → Reflector 反思。"""
        task = _last_user_text(messages) or ""
        tool_names = [t.name for t in self.registry.all()]
        planner = Planner(self.settings, llm=self.llm, tool_names=tool_names)
        reflector = Reflector(max_revisions=self.settings.max_plan_revisions)
        executor = PlanExecutor(
            llm=llm_for_loop,
            registry=self.registry,
            execute_tool=self._execute_tool,
            planner=planner,
            max_steps_per_step=max(2, self.settings.max_agent_steps // 2),
            context_builder=self.context_builder,
            hooks=self._build_hooks(),
        )

        all_tool_calls: list[ToolCallRequest] = []
        plan: list[PlanStep] = []
        revisions = 0
        current_task = task
        final_answer = ""

        # Stage 8 联动：规划前检索相关记忆，供 LLM 规划参考（分层：全局 + 本会话）
        memory_context: list[str] | None = None
        if self.memory is not None:
            memory_context = await self.memory.retrieve(task, session_id=state.session_id)

        while True:
            # 每次执行使用独立的回合消息快照（基于原始 messages 拷贝），
            # 重规划时从同一起点重放，避免消息累积导致协议非法
            turn_messages: list[dict] = list(messages)
            plan, answer, calls = await executor.execute(current_task, turn_messages, memory_context)
            all_tool_calls.extend(calls)
            final_answer = answer

            # 反思：是否需要重规划
            decision = reflector.reflect(current_task, plan, revisions_so_far=revisions)
            if not decision.need_replan:
                break
            revisions += 1
            current_task = decision.revised_task or current_task
            # 重规划前把失败的步骤标记为 SKIPPED，避免消息历史重复执行
            for s in plan:
                if s.status != "SUCCEEDED":
                    s.status = "SKIPPED"

        # 把最后一次执行产生的消息合并回全局（会话持久化 / result.messages）
        messages[:] = turn_messages

        state.plan = plan
        state.plan_revisions = revisions
        return plan, final_answer, all_tool_calls, revisions

    # ------------------------------------------------------------------
    # 主入口：正常执行（外层包 agent.run Span）
    # ------------------------------------------------------------------
    async def run(
        self,
        message: str,
        *,
        session_id: str | None = None,
        turn_id: str | None = None,
        user: UserContext | None = None,  # Stage 5 权限上下文
        extra_hooks: LoopHooks | None = None,  # 供 demo/测试注入（如模拟崩溃）
    ) -> AgentTurnResult:
        session_id = session_id or f"session_{uuid4().hex[:12]}"
        turn_id = turn_id or f"turn_{uuid4().hex[:12]}"

        # Stage 12：注入当前会话上下文（delegate 工具用它持久化编排结果）
        from app.orchestrator.context import current_session_id

        token_session = current_session_id.set(session_id)
        try:
            if self.recorder is None or not self.recorder.enabled:
                return await self._run_body(message, session_id, turn_id, user, extra_hooks)

            async with trace_span(
                "agent.run",
                "agent",
                input={"message": message, "session_id": session_id, "user_id": user.user_id if user else None},
                attributes={"session_id": session_id, "turn_id": turn_id},
                recorder=self.recorder,
            ) as span:
                result = await self._run_body(message, session_id, turn_id, user, extra_hooks)
                span.output = {
                    "answer": result.answer,
                    "steps": result.steps,
                    "tool_calls": len(result.tool_calls),
                }
                return result
        finally:
            current_session_id.reset(token_session)

    async def _run_body(self, message, session_id, turn_id, user, extra_hooks) -> AgentTurnResult:
        self._extra_hooks = extra_hooks
        self._last_checkpoint = None
        self._persisted = 0
        self._current_user = user

        # MCP：首次 run 时注册工具（幂等；调用方负责 connect，未连接则注册 0 个）
        if self.mcp_client is not None and not self._mcp_initialized:
            try:
                from app.mcp.bridge import MCPBridge

                # 若调用方已手动注册过 MCP 工具，跳过（bridge 自身也幂等）
                if not self.mcp_client.connections:
                    await self.mcp_client.connect_all()
                count = MCPBridge(self.mcp_client).register_all(self.registry)
                self._mcp_initialized = True
                if count:
                    print(f"[mcp] 已注册 {count} 个 MCP 工具到 registry", flush=True)
            except Exception as exc:
                print(f"[mcp] 初始化失败（不影响内置工具）: {exc}", flush=True)

        # 1) 确保会话存在
        if self.session_repo is not None and self.session_repo.get_session(session_id) is None:
            self.session_repo.create_session(session_id)

        # 2) 加载历史 + 追加用户消息
        messages = await self._prepare_messages(message, session_id)
        if self.session_repo is not None:
            # 用户消息立即持久化
            for m in messages[self._persisted:]:
                self.session_repo.add_message(session_id, m)
            self._persisted = len(messages)

        # 3) 构造状态并执行
        state = AgentState(
            session_id=session_id,
            turn_id=turn_id,
            step=0,
            status="RUNNING",
            messages=messages,
            agent_mode=self.settings.agent_mode,
        )
        return await self._run_with_state(state)

    # ------------------------------------------------------------------
    # 断点恢复入口（外层包 agent.run Span + checkpoint.load Span）
    # ------------------------------------------------------------------
    async def resume(
        self,
        session_id: str,
        *,
        turn_id: str | None = None,
        checkpoint_id: str | None = None,
        user: UserContext | None = None,
        extra_hooks: LoopHooks | None = None,
    ) -> AgentTurnResult:
        """从最新 Checkpoint 恢复执行（进程崩溃后继续，不重放用户输入）。"""
        if self.checkpoint_repo is None:
            raise AgentError("未配置 checkpoint_repo，无法恢复", code="NO_CHECKPOINT_REPO")

        if self.recorder is None or not self.recorder.enabled:
            return await self._resume_body(session_id, turn_id, checkpoint_id, user, extra_hooks)

        async with trace_span(
            "agent.run",
            "agent",
            input={"session_id": session_id, "resume": True},
            attributes={"session_id": session_id, "resume": True},
            recorder=self.recorder,
        ) as span:
            result = await self._resume_body(session_id, turn_id, checkpoint_id, user, extra_hooks)
            span.output = {"answer": result.answer, "steps": result.steps, "resumed": True}
            return result

    async def _resume_body(self, session_id, turn_id, checkpoint_id, user, extra_hooks) -> AgentTurnResult:
        self._extra_hooks = extra_hooks
        self._current_user = user

        with trace_span_sync(
            "checkpoint.load",
            "checkpoint",
            input={"session_id": session_id, "checkpoint_id": checkpoint_id},
            recorder=self.recorder,
        ) as span:
            checkpoint = (
                self.checkpoint_repo.load(checkpoint_id)
                if checkpoint_id
                else self.checkpoint_repo.load_latest(session_id)
            )
            if checkpoint is None:
                raise CheckpointError(f"会话 {session_id} 没有可恢复的检查点")
            self._last_checkpoint = checkpoint
            span.attributes.update(
                checkpoint_id=checkpoint.checkpoint_id,
                version=checkpoint.version,
                session_id=session_id,
            )

        # 从检查点反序列化状态 —— 状态来自 Checkpoint，而非用户输入
        state = AgentState(**checkpoint.state)
        state.turn_id = turn_id or state.turn_id
        messages = state.messages
        self._persisted = len(messages)  # 已持久化的消息数（恢复后新增的消息从这里开始写）

        # 恢复 PENDING_TOOL：重新执行待办工具调用
        resumed_calls: list[ToolCallRequest] = []
        if state.status == "PENDING_TOOL" and state.pending_tool_calls:
            pending = list(state.pending_tool_calls)
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": p["id"],
                            "type": "function",
                            "function": {"name": p["name"], "arguments": json.dumps(p["arguments"], ensure_ascii=False)},
                        }
                        for p in pending
                    ],
                }
            )
            for p in pending:
                # 恢复执行的工具调用同样计入回合结果（Eval 依赖）
                resumed_calls.append(ToolCallRequest(id=p["id"], name=p["name"], arguments=p["arguments"]))
                envelope = await self._execute_tool(p["name"], p["arguments"])
                messages.append(
                    {"role": "tool", "tool_call_id": p["id"], "name": p["name"], "content": envelope.to_json()}
                )
            state.status = "RUNNING"
            state.pending_tool_calls = []
            # 恢复后的新状态也保存一个检查点（版本继续递增）
            self._persist_and_checkpoint(state, messages, "resume")

        result = await self._run_with_state(state)
        if resumed_calls:
            result.tool_calls = resumed_calls + result.tool_calls
        return result


def _last_user_text(messages: list[dict]) -> str:
    """取消息序列中最后一条 user 消息的文本（记忆检索的查询来源）。"""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content") or ""
            return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return ""


def _truncate_incomplete_tool_block(messages: list[dict]) -> list[dict]:
    """截断不完整的 tool 配对块（防御持久化污染）。

    规则：从末尾向前找最近的 assistant(tool_calls) 消息，
    若其后跟随的 tool 消息数 < 声明的 tool_calls 数，则截断到该
    assistant 之前（不持久化不完整块）。返回安全可持久化的前缀。
    """
    safe = list(messages)
    # 从后往前扫描
    for i in range(len(safe) - 1, -1, -1):
        m = safe[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared = len(m.get("tool_calls") or [])
            # 数其后跟随的 tool 消息
            tool_count = 0
            for j in range(i + 1, len(safe)):
                if safe[j].get("role") == "tool":
                    tool_count += 1
                else:
                    break
            if tool_count < declared:
                # 不完整：截断到 assistant 之前
                return safe[:i]
            # 完整：保持（可能有更早的不完整块，继续向前检查）
            continue
    return safe


def _repair_tool_pairing(messages: list[dict]) -> list[dict]:
    """修复历史中不完整的 tool 配对（防御已污染数据）。

    处理三类非法序列：
      1. 孤儿 tool 消息：前面没有对应的 assistant(tool_calls) 声明；
      2. 悬空 assistant(tool_calls)：声明了 N 个调用但 tool 结果不足 N 条；
      3. 连续 user 消息：OpenAI 网关要求 user/assistant 交替，连续 user 会被拒绝。

    策略：
      - 对悬空的 assistant(tool_calls)：补充占位 tool 失败消息（配对完整）；
      - 对连续 user：合并为一条（保留最后一条，避免信息重复且序列合法）；
      - 对孤儿 tool：保留（网关对多余 tool 消息宽容，但会先修复悬空块）。
    """
    repaired: list[dict] = []
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        # 连续 user 合并（跳过中间无 assistant 的重复 user）
        if m.get("role") == "user":
            # 收集连续 user（跳过前面可能残留的孤立 user 之间的空档）
            j = i + 1
            while j < n and messages[j].get("role") == "user":
                j += 1
            if j > i + 1:
                # 合并：只保留最后一条 user（前面的"继续"等重复丢弃）
                repaired.append(messages[j - 1])
                i = j
                continue
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared = len(m.get("tool_calls") or [])
            # 收集后续连续 tool 消息
            j = i + 1
            tool_msgs = []
            while j < n and messages[j].get("role") == "tool":
                tool_msgs.append(messages[j])
                j += 1
            if len(tool_msgs) < declared:
                # 补充缺失的 tool 占位（失败信封），保证配对完整
                repaired.append(m)
                declared_ids = {tc.get("id") for tc in m.get("tool_calls", [])}
                present_ids = {tm.get("tool_call_id") for tm in tool_msgs}
                for tc in m.get("tool_calls", []):
                    if tc.get("id") not in present_ids:
                        repaired.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.get("id"),
                                "name": (tc.get("function") or {}).get("name", "unknown"),
                                "content": ToolResult.fail(
                                    (tc.get("function") or {}).get("name", "unknown"),
                                    AgentError("历史记录不完整：工具结果缺失", code="MISSING_TOOL_OUTPUT"),
                                ).to_json(),
                            }
                        )
                repaired.extend(tool_msgs)
                i = j
                continue
        repaired.append(m)
        i += 1
    return repaired
