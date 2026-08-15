"""
Stage 1 Demo：最小 ReAct / Tool Loop。

运行：python -m demos.stage1_demo

展示目标：完整看到 Tool Loop 中 Messages 的每一步演变：
    初始 messages
      ↓
    第一次 LLM Response（Tool Call JSON）
      ↓
    Tool Result
      ↓
    更新后的 messages
      ↓
    第二次 LLM Request
      ↓
    Final Answer

无 API Key 时使用内置 Stub 模型；配置了 LLM_BASE_URL / LLM_API_KEY 时自动走真实接口。
"""
import asyncio
import json

from app.agent.react_loop import run_react_loop
from app.config import get_settings
from app.llm.client import create_llm_client
from app.tools.builtin import build_default_registry

SEPARATOR = "=" * 64


def _dump(label: str, obj) -> None:
    """带标签打印任意对象（dict/list 转 JSON）。"""
    print(f"\n--- {label} ---")
    if isinstance(obj, (dict, list)):
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    else:
        print(obj)


async def run_case(llm, registry, user_text: str) -> None:
    """驱动一次完整的 ReAct 循环并打印每一步。"""
    print(f"\n{SEPARATOR}\n用户输入: {user_text}\n{SEPARATOR}")

    # 1) 初始 messages：只有用户消息
    messages: list[dict] = [{"role": "user", "content": user_text}]
    _dump("初始 messages", messages)

    # 2) 包装工具执行：打印 Tool Call 与 Tool Result
    async def execute_tool(name: str, args: dict):
        print(f"\n>>> 执行 Tool Call: {name} {json.dumps(args, ensure_ascii=False)}")
        result = await registry.execute(name, args)
        print(f">>> Tool Result（统一信封）:")
        print(json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
        return result

    # 3) 包装 LLM：打印每次真正送入模型的 Request（循环需要对象，因此包一层适配器）
    class PrintingLLM:
        """打印每次请求/响应的 LLM 适配器。"""

        async def chat(self, send_messages, tools, **kwargs):
            _dump("LLM Request（送入模型的 messages）", send_messages)
            response = await llm.chat(send_messages, tools)
            print("\n>>> LLM Response:")
            print(json.dumps(response.model_dump(exclude_none=True), ensure_ascii=False, indent=2))
            return response

    printing_llm = PrintingLLM()

    # 4) 运行循环本体（runtime.run 内部就是调用这个函数）
    final_messages, answer, steps, tool_calls = await run_react_loop(
        llm=printing_llm,
        tools_schema=registry.schemas(),
        messages=messages,
        execute_tool=execute_tool,
        max_steps=get_settings().max_agent_steps,
    )

    # 5) 打印最终结果
    _dump("更新后的最终 messages", final_messages)
    print(f"\n>>> Final Answer（第 {steps} 步）: {answer}")
    print(f">>> 本回合工具调用次数: {len(tool_calls)}")


async def main() -> None:
    settings = get_settings()
    llm = create_llm_client(settings)
    registry = build_default_registry()

    print(f"LLM Provider: {settings.llm_provider_resolved} | Model: {settings.llm_model}")
    print(f"可用工具: {[t.name for t in registry.all()]}")

    # 第一阶段验收的三个输入
    await run_case(llm, registry, "你好")
    await run_case(llm, registry, "计算 123 * 456")
    await run_case(llm, registry, "查询北京天气")


if __name__ == "__main__":
    asyncio.run(main())
