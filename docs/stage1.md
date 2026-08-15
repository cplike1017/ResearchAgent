# Stage 1：最小 ReAct / Tool Loop

> 阶段目标：亲手实现 Agent 最核心的执行机制 —— **循环**。
> 不借助任何 Agent Framework，一个 `while True` 讲清楚 Agent 为什么需要循环。

---

## 解决什么问题

一个"能调用工具"的 Agent 最朴素的问题：**模型一次只能做一步决策**，但任务往往需要多步（查天气 → 计算 → 汇总）。因此需要一个循环：

```text
User
 ↓
Messages
 ↓
LLM
 ↓
Decision
 ├─ Final Answer ────────── 结束
 └─ Tool Call
       ↓
     Tool（执行）
       ↓
 Tool Result（重新进入 Messages）
       ↓
     LLM（再决策）──────── 回到循环顶部
```

Stage 1 只做这一件事：**把循环本身实现出来并让它可运行、可观察**。

## 上一阶段有什么缺陷

无（这是第一阶段）。但先明确一个"从第一天就该有的缺陷意识"：

- 全部历史无脑发给模型（Stage 2 解决）；
- 状态不持久化，进程一停全丢（Stage 3 解决）；
- 请求直接在 HTTP 进程内同步执行（Stage 4 解决）；
- 工具裸调用，无校验 / 权限 / 超时（Stage 5 解决）；
- 无法追踪与评测（Stage 6 解决）。

## 本阶段新增什么组件

| 组件 | 文件 | 职责 |
|---|---|---|
| 统一 LLM Client | `app/llm/client.py` | 抽象 `chat(messages, tools) -> LLMResponse`；真实 OpenAI-compatible 实现 + 确定性 Stub |
| Tool Registry | `app/tools/registry.py` | 工具登记、JSON Schema 生成、直接执行入口 |
| 安全计算器 | `app/tools/builtin/calculator.py` | 基于 `ast` 白名单的算术求值（禁止 `eval`/`exec`/`shell`） |
| Stub 天气 | `app/tools/builtin/weather.py` | 假数据天气工具，避免外部 API 干扰教学 |
| ReAct 循环 | `app/agent/react_loop.py` | `run_react_loop()`：真正的 `while True` 决策循环 |
| Agent 运行时 | `app/agent/runtime.py` | `AgentRuntime.run()`：组装消息、跑循环、汇总结果 |
| 统一错误模型 | `app/errors.py` | `AgentError` 异常树，Trace / Job / API 共用 |
| 统一配置 | `app/config.py` | 全部环境变量注入，禁止硬编码 |

## 数据如何流动

```text
用户输入 "计算 123 * 456"
  → messages = [{"role": "user", "content": "计算 123 * 456"}]
  → LLM 决策: tool_calls=[calculator(expression="123 * 456")]
  → messages += assistant(tool_calls)          # 模型决策写入历史
  → 执行工具 → ToolResult 信封 {"success": true, "data": 56088.0, ...}
  → messages += tool(tool_call_id, content=信封JSON)   # 工具结果重新进入 Messages
  → LLM 再决策: 看到工具结果 → Final Answer
  → messages += assistant("计算结果：56088.0")
```

关键点：**Tool Result 以消息的形式重新进入 Messages**，模型下一次决策时才能"看到"它。这是 Tool Loop 能继续的前提。

## 核心数据结构

```python
# LLM 决策结果（统一表示）
class LLMResponse:
    content: str | None          # 最终回答文本
    tool_calls: list[ToolCallRequest]  # 本次要调用的工具
    finish_reason: str           # stop | tool_calls | ...
    usage: dict                  # token 统计

class ToolCallRequest:
    id: str                      # 工具调用 ID（tool_call_id 关联用）
    name: str
    arguments: dict              # 已解析为 dict 的参数

# 工具执行结果信封（统一写回 Messages 的格式）
class ToolResult:
    success: bool
    tool_name: str
    data: Any
    error: ToolErrorInfo | None
    metadata: dict               # duration_ms 等
```

## 关键代码

```python
# app/agent/react_loop.py —— 核心循环（完整实现见文件）
async def run_react_loop(*, llm, tools_schema, messages, execute_tool, max_steps, ...):
    steps = 0
    while True:
        steps += 1
        if steps > max_steps:
            raise AgentError("超过最大循环步数", code="MAX_STEPS_EXCEEDED")

        response = await llm.chat(messages, tools_schema)   # 模型决策

        if response.is_final_answer:                         # 分支一：结束
            return messages, response.content, steps, tool_calls

        # 分支二：工具调用
        messages.append({"role": "assistant", "tool_calls": [...]})  # 决策写入历史
        for tc in response.tool_calls:
            envelope = await execute_tool(tc.name, tc.arguments)     # 执行工具
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": envelope.to_json()})        # 结果重新进入 Messages
        # continue → 回到循环顶部，模型看到工具结果后再决策
```

安全计算器（禁止 `eval`，用 `ast` 白名单手工求值）：

```python
_ALLOWED_NODES = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
                  ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
                  ast.Mod, ast.Pow, ast.UAdd, ast.USub)

def safe_evaluate(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):            # 白名单校验：函数调用/属性/变量全拒绝
        if not isinstance(node, _ALLOWED_NODES):
            raise ToolExecutionError(...)
    return _eval_node(tree.body)           # 手工递归求值
```

## 输入示例

```text
你好
计算 123 * 456
查询北京天气
```

## 输出示例

```text
>>> LLM Response:
{
  "tool_calls": [{"id": "call_stub_calc", "name": "calculator",
                  "arguments": {"expression": "123 * 456"}}],
  "finish_reason": "tool_calls"
}

>>> Tool Result（统一信封）:
{"success": true, "tool_name": "calculator", "data": 56088.0, ...}

>>> Final Answer（第 2 步）: 计算结果：56088.0。
```

## 如何运行

```bash
# 1) 安装依赖
pip install -r requirements.txt

# 2) （可选）配置真实模型；不配置则用 Stub
# 复制 .env.example 为 .env 并填写 LLM_BASE_URL / LLM_API_KEY / LLM_MODEL

# 3) 运行 Demo
python -m demos.stage1_demo
```

## 如何测试

```bash
pytest tests/test_react_loop.py -v
```

覆盖：你好不调用工具、计算器调用、天气调用、Messages 演变序列、多工具调用、
死循环防护、安全计算器（危险输入拒绝 / 除零）、Schema 校验、未知城市错误。

## 常见错误

| 错误 | 原因 | 修复 |
|---|---|---|
| `MAX_STEPS_EXCEEDED` | 模型一直返回 tool_calls（参数不对 / 工具太少 / 模型失控） | 检查工具 Schema 与 LLM 决策；调大 `MAX_AGENT_STEPS` |
| 工具结果不生效 | 工具结果没有写回 Messages | 必须追加 `role=tool` 消息，且带 `tool_call_id` 关联 |
| 真实接口 400 | arguments 是 JSON 字符串未解析 / 缺 assistant tool_calls 消息 | 按 OpenAI 协议补齐消息序列 |
| `LLMError` | Base URL / Key 配置错误 | 检查 `.env` 与 `LLM_BASE_URL` 格式（`http://host/v1`） |
| 计算器报"不支持的语法" | 表达式含函数/变量 | 计算器只支持数字与 `+ - * / // % ** ()` |

## 面试如何表达

> "我实现了一个最小 ReAct 循环：`while True` 里调用 LLM，若返回 `tool_calls` 就执行工具并把结果以 `role=tool` 消息写回历史，再让模型继续决策；若返回最终回答就结束。我额外做了三个工程化处理：一是统一 LLM Client 抽象，核心运行时不绑定厂商 SDK；二是统一错误模型，工具失败以结构化信封返回而不是抛裸异常；三是安全计算器，用 `ast` 白名单手工求值，规避 `eval` 注入风险。循环还带 `max_steps` 死循环防护。"

---

下一阶段：Stage 2 Context Builder —— 不再把全部历史无脑发给模型。
