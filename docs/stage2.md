# Stage 2：Context Builder

> 阶段目标：**不要再把全部 Session History 无脑发送给模型**。
> 明确区分"原始 Session History"与"真正送入模型的 LLM Context"。

---

## 解决什么问题

Stage 1 的 ReAct 循环把 `messages` 原封不动发给模型。随着对话变长，问题越来越严重：

1. **Token 成本线性上升**：历史越长，每次调用越贵；
2. **超长截断**：超过模型上下文窗口后，最早的（往往是最重要的）消息被截掉；
3. **噪音干扰**：几十轮前的无关消息会稀释模型对当前任务的注意力。

Stage 2 引入 Context Builder（上下文构建器）：**模型实际看到什么，由它决定**。

```text
System Prompt（角色与规则）
+ Tool Schemas（工具定义）
+ Summary（历史超阈值时生成的摘要）
+ Recent Messages（滑动窗口：最近 N 条）
+ （预留）Retrieved Context（检索 / Memory）
```

## 上一阶段有什么缺陷

Stage 1 只有 `messages -> llm.chat(messages, tools)`，没有任何中间层：

- 模型输入 = 会话历史，二者是同一个东西；
- 没有裁剪、没有摘要、没有 token 估算；
- 每轮循环都重复发送全部历史，浪费且低效。

## 本阶段新增什么组件

| 组件 | 文件 | 职责 |
|---|---|---|
| Context Builder | `app/agent/context_builder.py` | `build()`：组装 System + 摘要 + 窗口消息，输出 ContextBuildResult |
| 滑动窗口 | `context_builder.py` | 只取最近 `max_context_messages` 条消息 |
| 历史压缩 | `context_builder.py` | 超过阈值时生成 summary（stub 确定性 / LLM 总结 / 关闭） |
| Token 估算 | `context_builder.py` | `estimate_messages_tokens()`：4 字符 ≈ 1 token（预留替换接口） |

## 数据如何流动

```text
Session History（原始，可能 40 条）
  ↓ ContextBuilder.build()
  1) 超阈值？→ compress_history() → summary
  2) 取最近 N 条 → recent
  3) 组装 System Prompt + 工具清单
  4) messages = [system] + ([摘要] system) + recent
  ↓
LLM Context（真正送入模型，条数有界）
```

关键点：**每一轮 Tool Call 之后 Context Builder 都会重新运行** ——
因为消息里新增了工具结果，模型必须"看到"它才能做下一步决策。
（回答面试题："Context Builder 在每轮 Tool Call 后是否重新运行？→ 是"）

## 核心数据结构

```python
class ContextBuildResult(BaseModel):
    messages: list[dict]        # 最终送入模型的 messages（system + 摘要 + 窗口）
    tools: list[dict]           # 随请求下发的工具 Schema
    total_history: int          # 原始历史条数（Session History 全量）
    selected: int               # 实际选中条数（窗口内，不含 system/摘要）
    summary: str | None         # 摘要（未触发压缩则为 None）
    estimated_tokens: int       # 估算 token 数

class ContextBuilder:
    max_messages: int           # 滑动窗口大小 N（MAX_CONTEXT_MESSAGES）
    threshold: int              # 压缩阈值（CONTEXT_SUMMARY_THRESHOLD）
    strategy: str               # stub | llm | off（CONTEXT_SUMMARY_STRATEGY）

    async def build(session_history, tools_schemas, *, state=None, retrieved_docs=None) -> ContextBuildResult
    async def compress_history(messages) -> str
    def estimate_text_tokens(text) -> int
```

## 关键代码

```python
async def build(self, session_history, tools_schemas=None, *, state=None, retrieved_docs=None):
    total = len(session_history)

    # 1) 压缩决策：历史超阈值 -> 生成 summary
    summary = None
    if total > self.threshold and self.strategy != "off":
        summary = await self.compress_history(session_history)

    # 2) 滑动窗口：只取最近 N 条
    recent = session_history[-self.max_messages:]

    # 3) 组装 System Prompt
    system_prompt = self._build_system_prompt(tools_schemas, retrieved_docs)

    # 4) 最终 messages：system -> 摘要 -> 窗口
    messages = [{"role": "system", "content": system_prompt}]
    if summary:
        messages.append({"role": "system", "content": f"[历史摘要] {summary}"})
    messages.extend(recent)
    return ContextBuildResult(messages=messages, total_history=total,
                              selected=len(recent), summary=summary,
                              estimated_tokens=estimate_messages_tokens(messages))
```

## 输入示例

12 轮真实对话累积的 40 条历史消息（user 12 / assistant 20 / tool 8）。

## 输出示例

```text
① 完整历史消息数量（Session History）: 40 条
② Context Builder 实际选中数量（窗口内）: 10 条
③ Summary: 历史共 40 条消息assistant 20 条，tool 8 条，user 12 条，
            使用过工具: calculator, get_weather；最早消息: 你好...；最近消息: 成都天气：晴，24°C，空气优。...
④ 估算 token 数: 206
⑤ 最终发送给 LLM 的 messages（共 12 条）:
   [0] role=system    content='你是一个智能助手，可以调用工具完成任务。...'
   [1] role=system    content='[历史摘要] 历史共 40 条消息...'
   [2..11] 窗口内最近 10 条消息
```

## 如何运行

```bash
python -m demos.stage2_demo
```

## 如何测试

```bash
pytest tests/test_context_builder.py -v
```

覆盖：窗口只选最近 N 条、窗口顺序保持、超阈值触发压缩、低于阈值不压缩、
strategy=off 关闭压缩、确定性摘要（相同输入相同输出）、
**历史从 10 条涨到 200 条时上下文条数保持有界**（核心验收）、
token 估算随内容增长、**每轮 Tool Call 后 Context Builder 重新运行**（模型能看到工具结果）、
system prompt 存在。

## 常见错误

| 错误 | 原因 | 修复 |
|---|---|---|
| 上下文无限增长 | 忘记裁剪，直接发全部历史 | 滑动窗口 `history[-N:]` |
| 摘要每轮都不一样 | 用 LLM 摘要且未关闭 | 测试用 `stub` 策略；生产可加缓存 |
| 窗口切断了工具结果 | 窗口太小 | 保证窗口至少覆盖最近一整个回合（assistant+tool+assistant） |
| token 估算不准 | 字符数 ≠ token | 这是占位实现；生产替换为 tiktoken 等真实 tokenizer（接口不变） |
| 把压缩当成裁剪 | 压缩后丢失了近期消息 | 正确做法是"摘要 + 最近 N 条"两者都要 |

## 面试如何表达

> "Stage 2 我实现了 Context Builder，核心是区分 Session History 和 LLM Context 两个概念。
> 构建流程：超阈值时先生成摘要（stub 确定性或 LLM 总结），再取滑动窗口内最近 N 条消息，
> 加上 System Prompt 组成最终模型输入，并输出 token 估算。两个关键设计：一是窗口保证
> LLM Context 条数有界 —— 历史无限增长但上下文不会；二是它在每一轮 Tool Call 之后都会
> 重新运行，因为模型必须看到新的工具结果才能做下一步决策。token 估算我预留了接口，
> 生产环境可以无缝替换成真实 tokenizer。"

---

下一阶段：Stage 3 Session + Checkpoint —— 执行状态如何保存恢复。
