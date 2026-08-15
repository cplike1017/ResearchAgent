"""
最小 ReAct / Tool Loop（第一阶段核心）。

它回答一个基础问题：Agent 为什么需要循环？

因为模型一次输出只能做"一步决策"：
    - 要么直接给出 Final Answer；
    - 要么发起 Tool Call —— 此时工具结果必须重新进入 Messages，
      再交给模型做下一次决策，直到模型认为信息足够、给出最终回答。

本文件实现的是"循环本身"，不关心：
    - 上下文怎么裁剪（Stage 2 Context Builder）；
    - 状态怎么持久化（Stage 3 Session / Checkpoint）；
    - 工具怎么治理（Stage 5 Tool Gateway / Policy）；
    - 怎么观测（Stage 6 Tracing）。
这些能力通过参数 / 钩子（hooks）注入，保证各阶段增量叠加、不推翻重写。

消息协议遵循 OpenAI 风格：
    - 模型发起工具调用：追加一条 role=assistant 且带 tool_calls 的消息；
    - 工具执行结果：追加一条或多条 role=tool 的消息（带 tool_call_id 关联）。
"""
import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agent.models import AgentTurnResult
from app.errors import AgentError
from app.llm.client import BaseLLMClient, LLMResponse, ToolCallRequest
from app.tools.schemas import ToolResult


@dataclass
class LoopHooks:
    """ReAct 循环各关键节点的钩子（Stage 3 Checkpoint、Stage 6 Tracing 注入点）。"""

    # LLM 调用之前（此时可保存"LLM 决策前"检查点）
    before_llm: Callable[[int, list[dict]], Awaitable[None]] | None = None
    # LLM 返回决策之后（保存"LLM 决策后"检查点）
    after_decision: Callable[[LLMResponse, int], Awaitable[None]] | None = None
    # 每个工具执行完成之后（保存"工具执行后"检查点）
    after_tool: Callable[[ToolCallRequest, ToolResult, int], Awaitable[None]] | None = None
    # 即将返回最终回答之前（保存"Final Answer 前"检查点）
    before_final: Callable[[LLMResponse, int], Awaitable[None]] | None = None


async def run_react_loop(
    *,
    llm: BaseLLMClient,
    tools_schema: list[dict],
    messages: list[dict],
    execute_tool: Callable[[str, dict], Awaitable[ToolResult]],
    max_steps: int = 8,
    context_builder: Any | None = None,
    hooks: LoopHooks | None = None,
    retrieved_docs: list[str] | None = None,  # Stage 8 记忆层：检索到的参考资料
) -> tuple[list[dict], str, int, list[ToolCallRequest]]:
    """
    运行 ReAct 循环。

    :param llm:            统一 LLM 客户端
    :param tools_schema:   OpenAI 格式工具 Schema 列表
    :param messages:       可变的会话消息列表（函数内会追加 assistant/tool 消息）
    :param execute_tool:   工具执行器：async (name, args) -> ToolResult
    :param max_steps:      最大循环步数，防止死循环
    :param context_builder: 可选；非空时每轮循环都重新构建送入模型的上下文
    :param hooks:          可选钩子（检查点 / 追踪）
    :param retrieved_docs: 可选；记忆层检索到的参考资料，随每轮上下文注入
    :return: (最终messages, 最终回答, 步数, 全部工具调用)
    """
    steps = 0
    all_tool_calls: list[ToolCallRequest] = []

    while True:
        steps += 1
        if steps > max_steps:
            raise AgentError(f"超过最大循环步数 {max_steps}，已终止（疑似 Tool Loop 死循环）", code="MAX_STEPS_EXCEEDED")

        # ---------- 1) 构建本次送入模型的上下文 ----------
        # 注意：每轮循环都要重新构建 —— 因为上一轮的工具结果已经追加进 messages，
        # 模型必须"看到"新的工具结果才能做下一步决策。
        if hooks and hooks.before_llm:
            await hooks.before_llm(steps, messages)

        if context_builder is not None:
            built = await context_builder.build(
                messages, tools_schema, retrieved_docs=retrieved_docs
            )
            llm_messages: list[dict] = built.messages
            request_tools: list[dict] = built.tools or tools_schema
        else:
            llm_messages = messages
            request_tools = tools_schema

        # ---------- 2) LLM 决策 ----------
        response: LLMResponse = await llm.chat(llm_messages, request_tools)

        if hooks and hooks.after_decision:
            await hooks.after_decision(response, steps)

        # ---------- 3) 分支一：最终回答 ----------
        if response.is_final_answer:
            # 最终回答同样以 assistant 消息写回历史（Stage 3 持久化依赖完整的消息序列）
            messages.append({"role": "assistant", "content": response.content or ""})
            if hooks and hooks.before_final:
                await hooks.before_final(response, steps)
            return messages, response.content or "", steps, all_tool_calls

        # ---------- 4) 分支二：工具调用 ----------
        # 先把 assistant 的工具调用决策写进历史（OpenAI 协议要求工具结果必须跟在
        # 带 tool_calls 的 assistant 消息之后），再逐个执行工具并追加结果。
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                    }
                    for tc in response.tool_calls
                ],
            }
        )

        for tc in response.tool_calls:
            all_tool_calls.append(tc)
            try:
                envelope: ToolResult = await execute_tool(tc.name, tc.arguments)
            except asyncio.CancelledError:
                # 客户端断开/取消：补一条 tool 失败消息，保证 assistant(tool_calls)
                # 与 tool 消息配对完整（否则下次请求网关报 "No tool output found"）
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": ToolResult.fail(
                            tc.name, AgentError("执行被取消（客户端断开）", code="CANCELLED")
                        ).to_json(),
                    }
                )
                if hooks and hooks.after_tool:
                    # 用占位信封触发钩子（持久化这一条，保持序列一致）
                    await hooks.after_tool(
                        tc,
                        ToolResult.fail(
                            tc.name, AgentError("执行被取消（客户端断开）", code="CANCELLED")
                        ),
                        steps,
                    )
                raise
            # 工具结果以统一信封（JSON 字符串）重新进入 Messages —— 循环得以继续的关键
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "content": envelope.to_json(),
                }
            )
            if hooks and hooks.after_tool:
                await hooks.after_tool(tc, envelope, steps)

        # 回到循环顶部，把更新后的消息再交给模型
        continue
