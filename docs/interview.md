# ReAgent 面试指南：10 个高频追问与参考答案

> 用法：先自己用 30 秒口头回答，再对照本文校准。每个答案都对应项目里的
> 真实代码位置 —— 面试时说出"代码在哪、踩过什么坑"远比背概念可信。

---

## Q1. 为什么不用 LangGraph / CrewAI，而要自己从零实现？

**一句话**：实习里我用 LangGraph 做业务落地，正是那段经历让我想搞清楚框架
内部到底发生了什么，所以这个项目不用任何框架，把所有机制手写一遍。

**三层展开**：
1. **业务层**：实习中基于 LangGraph 构建多轮工作流时，遇到"状态恢复是否
   真的恢复"、"工具结果如何回到消息流"、"跨进程 Trace 为何断裂"这类问题，
   框架隐藏了细节，排查成本很高；
2. **协议层**：手写之后才暴露真实坑——OpenAI 网关要求 `tool` 消息必须跟在
   带 `tool_calls` 的 assistant 消息之后（否则 400 `No tool output found`）；
   滑动窗口截断可能把 tool 消息截成孤儿；这些是框架替你兜住但你不知道的事；
3. **能力层**：手写意味着 ReAct 循环、Checkpoint、工具网关、多 Agent 编排
   每一步都可解释、可修改、可教学。面试官问"LangGraph 的 checkpointer
   内部怎么存状态"，我能从自己实现的 Session/Checkpoint 讲起。

**代码位置**：`app/agent/react_loop.py`（循环本身）、`app/checkpoint/repository.py`、
`app/orchestrator/runner.py`。

**加分句**：用框架时我知道该在哪个节点挂 hook；不用框架时我知道 hook 内部
在做什么、什么时候框架的抽象会漏。

---

## Q2. 滑动窗口 vs 摘要压缩，你怎么取舍？为什么两个都要？

**一句话**：滑动窗口保证"最近上下文不丢"，摘要压缩保证"早期关键信息不丢"，
两者解决不同方向的遗忘。

**展开**：
- **滑动窗口**（`max_context_messages`）：只保留最近 N 条消息，控制 token 成本
  与延迟。代价是窗口前的信息完全丢失；
- **摘要压缩**（`context_summary_threshold`）：历史超过阈值时，把窗口外的
  旧消息压缩成一段摘要（stub 规则摘要 / LLM 总结两种策略），塞回上下文头部；
- **组合**：窗口负责"保近"，摘要负责"保远"，中间层可以都丢。

**踩过的坑**（这是面试亮点）：滑动窗口截断可能把一条 `assistant(tool_calls)`
留下、把它的 `tool` 结果切掉，导致请求 400。我实现了
`_repair_window_boundary`：向前扩展窗口直到不落在 tool 块中间；若仍不完整则
丢弃悬空的 tool_calls 声明。这个坑只有手写协议层才会遇到。

**代码位置**：`app/agent/context_builder.py`、`tests/test_window_boundary.py`。

---

## Q3. MCP 为什么选 stdio 传输？和 SSE/HTTP 怎么选？

**一句话**：stdio 适合本地可信子进程（npx 起的本地 server），SSE 适合远程服务；
选型取决于 server 跑在哪、谁负责生命周期。

**展开**：
- **stdio**：MCP SDK 用 `stdio_client` 拉起子进程，通过 stdin/stdout 走
  JSON-RPC。优势是本地零网络开销、进程生命周期跟随父进程；代价是子进程
  崩溃要自己处理、异步清理有坑；
- **SSE**：远程 MCP server（URL 连接），适合共享服务，但要处理鉴权、断线重连；
- **本项目**：filesystem / fetch / think / github 四个 server 都是本地 npx
  子进程，所以统一 stdio；`MCPClientManager` 管理连接池，bridge 负责把 MCP
  工具名规范化为 OpenAI 函数名（`demo.echo` → `demo_echo`）。

**踩过的坑**：MCP SDK 1.27 要求 `stdio_client.__aexit__` 与 enter 在同一个
asyncio 任务里执行；我最初用 `asyncio.shield` 包关闭操作，导致"取消作用域
跨任务"报错、关闭阶段刷屏。去掉 shield、直接 await 后干净关闭——这正是
"框架隐藏的细节"的又一例证。

**代码位置**：`app/mcp/client.py`、`app/mcp/bridge.py`。

---

## Q4. 并行子 agent 的上下文为什么不会互相污染？

**一句话**：每个子 agent 的 messages 是独立对象，工具注册表按档案过滤后
各持一份；同时 asyncio 任务有独立的 ContextVar 副本。

**展开**：
1. **消息隔离**：`SubAgentExecutor.execute` 为每个子 agent 新建
   `[system, user]` 消息列表，只注入自己的任务与依赖结果，看不到主 agent
   历史，也看不到兄弟 agent 的中间过程；
2. **工具隔离**：从主 registry 按档案白名单复制出独立 `ToolRegistry` +
   独立 `ToolGateway`（researcher 摸不到 run_code，analyst 摸不到 web_search）；
3. **上下文变量隔离**：Trace 的 `current_trace_id / current_span_id` 用
   ContextVar 实现，asyncio 创建子任务时会复制当前上下文——并行子 agent
   各自的 span 正确嵌套在 orchestrator.run 下，互不覆盖；
4. **结果聚合**：runner 按计划下标收集 `results[i]`，依赖步骤只拿到
   指定下标的结果（`_build_context`）。

**代码位置**：`app/orchestrator/executor.py`、`app/orchestrator/runner.py`。

---

## Q5. 多 Agent 编排的深度上限怎么防递归？

**一句话**：三层防御——深度 ContextVar 追踪、叶子层物理移除 delegate 工具、
handler 兜底拒绝。

**展开**：
1. **追踪**：`orchestration_depth` ContextVar，`runner.run()` 进入 +1、
   退出恢复；ContextVar 是任务局部的，并行子 agent 各自嵌套互不干扰；
2. **可见性控制**（第一道防线）：子 agent 能否看到 delegate 工具由深度决定——
   `depth < ORCHESTRATOR_MAX_DEPTH` 保留 delegate（可再委派），叶子层物理移除，
   模型根本不会产生嵌套调用（不是"提示它别用"，是它看不到）；
3. **handler 兜底**（第二道防线）：即使可见性被绕过，delegate 的 handler
   在深度超限时直接返回 FAILED，拒绝再委派；
4. **语义**：delegate 是编排层的"元能力"，不属于任何档案的工具白名单，
   所以 researcher 也能再委派，但永远受深度上限约束。

**代码位置**：`app/orchestrator/context.py`、`app/orchestrator/executor.py`、
`app/orchestrator/tool.py`。

---

## Q6. 记忆层（RAG）检索失败会怎样？为什么不直接抛异常？

**一句话**：记忆是"可选增强"，不是"单点故障"——检索失败降级为"本轮无记忆"，
绝不中断 Agent 回合。

**展开**：真实场景里 embedding API 会超时、会断连。最初实现里
`memory.retrieve` 失败会让整个 chat 500，我在真实部署中踩到后改成：
`retrieve` / `remember` 内部捕获异常返回空列表，Trace 里仍能看到
`memory.retrieve [ERROR]` span（可观测性不丢），但主流程照常执行。
这体现一个工程原则：**增强性依赖必须 fail-open，而不是 fail-closed**。

**代码位置**：`app/memory/store.py`。

---

## Q7. run_code 沙箱怎么保证安全？为什么不用容器？

**一句话**：AST 白名单 + 模块拦截 + 超时，三层防住"教学演示级"风险；
容器是更强的隔离但成本高，作为可选部署层。

**展开**：
1. **AST 白名单**：解析代码为 AST，只允许白名单内的节点类型
   （表达式、赋值、for/if/函数定义等），`import os` / `eval` / `exec` /
   `__import__` 直接拒绝——在代码执行前就拦截，而非运行时拦截；
2. **模块拦截**：`__import__` 被替换为 `_safe_import` 包装，只放行
   math/statistics/json/csv/re/collections 等白名单模块；
3. **超时**：网关统一 `asyncio.wait_for` 超时（30s），防死循环；
4. **边界声明**：这是"不可信代码"的弱隔离，生产环境应叠加容器/子进程隔离；
   项目的定位是教学与可信场景，沙箱是纵深防御的第一层。

**踩过的坑**：最初 `__import__` 被误放进拦截集合，导致合法代码报
`ImportError: __import__ not found`——白名单与黑名单的边界测试很重要。

**代码位置**：`app/tools/builtin/code_exec.py`、`tests/test_tool_gateway.py`。

---

## Q8. 你遇到过 "No tool output found for function call" 400 错误吗？怎么修？

**一句话**：遇到过，这是消息协议配对问题——`tool` 消息必须紧跟对应的
`assistant(tool_calls)`，任何一环断裂网关就 400。

**三类根因与修复**：
1. **执行中断**：ReAct 循环中工具执行被取消（CancelledError），assistant
   声明了 N 个调用但 tool 结果不足 N 条 → 循环捕获取消后补一条
   tool 失败占位消息，保持配对完整；
2. **滑动窗口截断**：窗口把 tool 结果切掉、留下 assistant 声明 →
   `_repair_window_boundary` 向前扩展窗口；
3. **连续 user 消息**：OpenAI 网关要求 user/assistant 交替，连续 user 会报
   `No tool call found for function call output` → `_repair_tool_pairing`
   合并连续 user 消息。

**代码位置**：`app/agent/runtime.py`（`_repair_tool_pairing`）、
`app/agent/context_builder.py`（`_repair_window_boundary`）、
`app/agent/react_loop.py`（取消占位）、`tests/test_tool_pairing.py`。

---

## Q9. Checkpoint 断点恢复和 Session 持久化有什么区别？恢复是"真恢复"吗？

**一句话**：Session 存"对话历史"，Checkpoint 存"执行状态"；恢复从
Checkpoint 反序列化状态机继续跑，而不是重放用户输入。

**展开**：
- **Session**（messages 表）：用户与 agent 的完整消息历史，用于下一轮对话
  的上下文构建；
- **Checkpoint**（checkpoints 表，版本递增）：`AgentState` 的快照——
  status、step、pending_tool_calls、plan 等执行中间态；
- **恢复语义**：`resume()` 从最新 Checkpoint 载入状态，如果状态是
  `PENDING_TOOL`（LLM 已决策但工具未执行完），恢复时重新执行待办工具调用
  再继续循环——绝不重新跑用户输入。这是"真恢复"与"假恢复"（重新执行一遍）
  的区别；
- **教学点**：恢复重放工具调用意味着工具可能被执行两次（幂等性话题），
  这是设计取舍，面试可以展开。

**代码位置**：`app/checkpoint/repository.py`、`app/agent/runtime.py`
（`resume` / `_resume_body`）。

---

## Q10. 240 个测试怎么组织的？离线怎么保证？

**一句话**：pytest + 三层隔离——Stub LLM、临时 SQLite/Trace 文件、
fakeredis，所有测试不依赖网络与真实模型。

**展开**：
1. **Stub LLM**：`StubLLMClient` 是确定性规则假模型（"计算 X"→调 calculator、
   "查询城市天气"→调 get_weather、有 tool 结果→收尾），让 ReAct 循环的
   行为完全可预期；
2. **隔离配置**：conftest 强制 `environment=test` + `llm_provider=stub` +
   临时数据库/trace 文件，防止 `.env` 的真实 Key 和真实文件泄漏进测试
   （踩过：`.env` 的 AGENT_MODE=plan 泄漏导致测试行为漂移）；
3. **分层覆盖**：单测（planner 解析/网关校验/窗口修复）→ 集成
   （runtime 端到端、编排并行/嵌套）→ 回归（eval runner）；
4. **规模**：240 passed，覆盖消息协议配对、窗口边界、记忆降级、
   编排深度限制等"只有踩过坑才会写"的测试。

**代码位置**：`tests/conftest.py`、`tests/test_orchestrator.py` 等 22 个测试文件。

---

## 附：追问反客为主

面试官问完你后，可以自然带出："您想不想看我把'调研+分析+成稿'这个任务
委派给三个子 agent 时，Trace 树长什么样？"——项目有 Web UI 可以直接演示，
这是简历文字给不了的冲击力。
