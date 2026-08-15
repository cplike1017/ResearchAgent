"""
Context Builder（上下文构建器，第二阶段核心）。

解决什么问题：
    Stage 1 把全部 Session History 无脑发给模型。随着对话变长，token 成本
    线性上升、超长截断、噪音干扰决策。Stage 2 明确区分两件事：

        原始 Session History   !=   真正送入模型的 LLM Context

    模型实际看到的是：
        System Prompt（角色与规则）
      + Tool Schemas（工具定义，随 API tools 参数下发）
      + 历史摘要 Summary（历史超阈值时压缩生成）
      + 最近 N 条消息（滑动窗口 Sliding Window）
      + （预留）Retrieved Context（检索增强内容 / Memory）

    并在每轮 Tool Call 之后重新运行 —— 因为消息里新增了工具结果，
    模型必须"看到"它才能做下一步决策。
"""
import json
from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings
from app.llm.client import BaseLLMClient
from app.tracing.recorder import TraceRecorder
from app.tracing.span import trace_span


# ---------------------------------------------------------------------------
# 数据结构：一次上下文构建的结果
# ---------------------------------------------------------------------------
class ContextBuildResult(BaseModel):
    """Context Builder 的输出：真正送入模型的输入 + 统计信息。"""

    messages: list[dict] = Field(description="最终送入模型的 messages（system + 摘要 + 窗口内消息）")
    tools: list[dict] = Field(default_factory=list, description="随请求下发的工具 Schema 列表")
    total_history: int = Field(description="原始历史消息条数（Session History 全量）")
    selected: int = Field(description="实际选中的消息条数（不含 system / 摘要）")
    summary: str | None = Field(default=None, description="生成的摘要（未触发压缩则为 None）")
    estimated_tokens: int = Field(description="估算的 token 数")


# ---------------------------------------------------------------------------
# Token 估算（预留替换真实 tokenizer 的接口）
# ---------------------------------------------------------------------------
def estimate_text_tokens(text: str) -> int:
    """粗略估算：按 4 个字符 ≈ 1 token（英文经验值）。
    这是占位实现，真实场景可替换为 tiktoken 等 tokenizer，接口不变。"""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_messages_tokens(messages: list[dict]) -> int:
    """估算一组 messages 的 token 总数（含每条消息的结构开销近似）。"""
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += estimate_text_tokens(content)
        else:
            total += estimate_text_tokens(json.dumps(content or "", ensure_ascii=False))
        total += 4  # 每条消息的 role 等结构开销近似
    return total


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------
class ContextBuilder:
    """把 Session History 组装成真正送入模型的 LLM Context。"""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLMClient | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.llm = llm
        self.max_messages = self.settings.max_context_messages  # 滑动窗口大小 N
        self.threshold = self.settings.context_summary_threshold  # 压缩阈值
        self.strategy = self.settings.context_summary_strategy  # stub | llm | off
        self.recorder = recorder  # None = 不追踪（由上层显式注入）

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    async def build(
        self,
        session_history: list[dict],
        tools_schemas: list[dict] | None = None,
        *,
        state: Any | None = None,          # 预留：AgentState（Stage 3）
        retrieved_docs: list[str] | None = None,  # 预留：检索内容 / Memory
    ) -> ContextBuildResult:
        """构建最终模型输入。"""
        if self.recorder is None or not self.recorder.enabled:
            return await self._build_impl(session_history, tools_schemas, retrieved_docs)

        async with trace_span(
            "context_builder",
            "context_builder",
            input={"session_history": session_history},
            recorder=self.recorder,
        ) as span:
            built = await self._build_impl(session_history, tools_schemas, retrieved_docs)
            span.attributes.update(
                total_history=built.total_history,
                selected=built.selected,
                estimated_tokens=built.estimated_tokens,
                has_summary=built.summary is not None,
            )
            span.output = {
                "total_history": built.total_history,
                "selected": built.selected,
                "estimated_tokens": built.estimated_tokens,
            }
            return built

    async def _build_impl(
        self,
        session_history: list[dict],
        tools_schemas: list[dict] | None,
        retrieved_docs: list[str] | None,
    ) -> ContextBuildResult:
        total = len(session_history)

        # 1) 压缩决策：历史超过阈值 -> 生成 summary，窗口内消息照常保留
        summary: str | None = None
        if total > self.threshold and self.strategy != "off":
            summary = await self.compress_history(session_history)

        # 2) 滑动窗口：只取最近 N 条消息
        recent = session_history[-self.max_messages:]
        # 防御：窗口起点不能落在"孤儿 tool"上（OpenAI 要求 tool 必须跟在
        # assistant(tool_calls) 之后；窗口截断可能切掉 tool_calls 声明而留下 tool）
        recent = _repair_window_boundary(session_history, recent)

        # 3) 组装 System Prompt（角色 + 规则 + 工具清单 + 检索内容）
        system_prompt = self._build_system_prompt(tools_schemas, retrieved_docs)

        # 4) 拼装最终 messages：system -> 摘要 -> 窗口消息
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if summary:
            messages.append({"role": "system", "content": f"[历史摘要] {summary}"})
        messages.extend(recent)

        return ContextBuildResult(
            messages=messages,
            tools=tools_schemas or [],
            total_history=total,
            selected=len(recent),
            summary=summary,
            estimated_tokens=estimate_messages_tokens(messages),
        )

    # ------------------------------------------------------------------
    # 压缩（历史超阈值时调用）
    # ------------------------------------------------------------------
    async def compress_history(self, messages: list[dict]) -> str:
        """
        生成历史摘要。
        策略：
            stub —— 确定性规则摘要（不调用模型，测试 / 离线场景默认）
            llm  —— 调用 LLM 总结（需要真实模型）
            off  —— 不压缩（由 build 提前拦截）
        """
        if self.strategy == "llm" and self.llm is not None:
            return await self._llm_summary(messages)
        return self._stub_summary(messages)

    async def _llm_summary(self, messages: list[dict]) -> str:
        """调用 LLM 生成摘要（教学演示；真实场景可换更专业的摘要 Prompt）。"""
        prompt = (
            "请用不超过 3 句话概括下面这段对话的要点（保留关键事实、数值、结论）：\n"
            + json.dumps(messages, ensure_ascii=False)
        )
        response = await self.llm.chat([{"role": "user", "content": prompt}], tools=None)
        return response.content or ""

    def _stub_summary(self, messages: list[dict]) -> str:
        """确定性摘要：统计角色分布 + 工具使用 + 首尾抽样。
        不依赖模型，结果可复现，测试友好。"""
        n = len(messages)
        role_counts: dict[str, int] = {}
        tools_used: set[str] = set()
        for m in messages:
            role = m.get("role", "?")
            role_counts[role] = role_counts.get(role, 0) + 1
            if role == "tool":
                tools_used.add(m.get("name", "?"))

        parts = [f"历史共 {n} 条消息"]
        parts.append("，".join(f"{r} {c} 条" for r, c in sorted(role_counts.items())))
        if tools_used:
            parts.append(f"，使用过工具: {', '.join(sorted(tools_used))}")

        def _snip(content: Any) -> str:
            if isinstance(content, str):
                return content[:30]
            return json.dumps(content, ensure_ascii=False)[:30]

        head = _snip(messages[0].get("content")) if messages else ""
        tail = _snip(messages[-1].get("content")) if messages else ""
        parts.append(f"；最早消息: {head}...；最近消息: {tail}...")
        return "".join(parts)

    # ------------------------------------------------------------------
    # System Prompt 组装
    # ------------------------------------------------------------------
    def _build_system_prompt(
        self,
        tools_schemas: list[dict] | None,
        retrieved_docs: list[str] | None,
    ) -> str:
        lines = [
            "你是一个智能助手，可以调用工具完成任务。",
            "规则：",
            "1. 仅当需要外部信息或计算时才调用工具；",
            "2. 工具参数必须严格符合对应 JSON Schema；",
            "3. 拿到工具结果后，基于结果组织最终回答；",
            "4. 直接回答用户问题，不要解释内部机制。",
        ]
        if tools_schemas:
            names = ", ".join(t["function"]["name"] for t in tools_schemas)
            lines.append(f"可用工具: {names}")
        if retrieved_docs:
            lines.append("参考资料：" + " | ".join(retrieved_docs))
        return "\n".join(lines)


def _repair_window_boundary(full_history: list[dict], window: list[dict]) -> list[dict]:
    """修复滑动窗口截断导致的孤儿 tool 消息。

    窗口按条数截取时，可能把 assistant(tool_calls) 切掉而留下紧随的
    tool 消息（OpenAI 协议要求 tool 必须跟在带 tool_calls 的 assistant 之后）。
    策略：窗口开头若出现 tool 消息，向前扩展窗口包含其对应的
    assistant(tool_calls)；若开头是普通消息则保持。同时确保窗口内
    tool 配对完整（截断末尾的悬空 tool_calls 声明）。
    """
    if not window:
        return window

    # 1) 窗口开头是 tool -> 向前扩展找到其 assistant(tool_calls)
    start = len(full_history) - len(window)
    i = start
    while i < len(full_history) and full_history[i].get("role") == "tool":
        # 找到最近的 assistant(tool_calls)
        j = i - 1
        while j >= 0 and full_history[j].get("role") != "assistant":
            j -= 1
        if j >= 0 and full_history[j].get("role") == "assistant" and full_history[j].get("tool_calls"):
            i = j  # 从 assistant 开始
            break
        i += 1

    # 2) 重建窗口（向前扩展后保持最大条数内尽量完整）
    expanded = full_history[i:] if i < start else window
    # 若扩展后超出窗口上限太多，取最近 max 条但保证开头不是 tool
    if len(expanded) > len(window) + 8:  # 允许适度扩展
        expanded = window

    # 3) 截断末尾悬空的 assistant(tool_calls)（声明的 tool 未齐）
    #    从末尾向前找最后一个完整的配对块
    for idx in range(len(expanded) - 1, -1, -1):
        m = expanded[idx]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            declared = len(m.get("tool_calls") or [])
            tools_after = 0
            k = idx + 1
            while k < len(expanded) and expanded[k].get("role") == "tool":
                tools_after += 1
                k += 1
            if tools_after < declared:
                return expanded[:idx]  # 截断到该 assistant 之前
            break

    return expanded
