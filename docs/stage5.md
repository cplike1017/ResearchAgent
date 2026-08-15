# Stage 5：Tool Gateway + Policy

> 阶段目标：Agent 不再 `tool_registry[name](args)` 裸调用工具，而是统一走 **Tool Gateway（工具网关）**，
> 由它负责校验、权限、策略、超时、结果包装 —— 工具的"治理"集中收口。

---

## 解决什么问题

Stage 1~4 里，工具调用是"注册表查一下 → 直接执行"。问题：

1. **参数错误没人拦**：`{"city": 123}` 会一路传进 handler 才报错；
2. **没有权限控制**：任何调用方都能调用任何工具；
3. **没有策略**：高风险操作（删库、发邮件）没有任何规则约束；
4. **没有超时**：一个卡死的工具能永远阻塞 Worker 线程；
5. **错误格式不统一**：有的抛异常、有的返回奇怪结构。

Stage 5 引入统一治理链：

```text
Agent → Tool Call → Tool Gateway
  ├─ 1. Schema Validation（参数校验）
  ├─ 2. Permission（权限）
  ├─ 3. Policy（策略）
  ├─ 4. Timeout（超时）
  ├─ 5. Tool Execute（执行 + 瞬时错误重试）
  ├─ 6. Result Validation（返回值校验）
  → Tool Result Envelope（统一信封）→ Agent
```

## 上一阶段有什么缺陷

Stage 4 的 `registry.execute()` 只做了 pydantic 校验 + 错误包装，
没有权限 / 策略 / 超时。而且执行逻辑散落在注册表里，
"哪些工具能用、谁能用、怎么防卡死"这些问题都没有答案。

## 本阶段新增什么组件

| 组件 | 文件 | 职责 |
|---|---|---|
| Policy Engine | `app/tools/policy.py` | `PolicyDecision`（ALLOW/DENY/REQUIRE_CONFIRMATION）+ 规则（黑名单 / 风险等级） |
| Tool Gateway | `app/tools/gateway.py` | 统一执行链：校验→权限→策略→超时→执行→结果校验 |
| PermissionChecker | `app/tools/gateway.py` | `required_permission` 与用户权限比对 |
| UserContext | `app/tools/schemas.py` | 调用方身份（user_id / roles / permissions） |
| 瞬时重试 | `errors.py` | `ToolExecutionError(transient=True)` 触发 Gateway 重试 |
| 端到端权限 | `queue` + `api` | Job 携带 user 上下文，Worker 传给 Runtime |

## 数据如何流动

```text
POST /api/chat {message, user_id, roles, permissions}
  → Job.user = UserContext(...)
  → Redis → Worker
  → runtime.run(message, user=UserContext)
  → ReAct 循环发起 Tool Call
  → gateway.execute(name, args, user)
      ├─ registry.get(name)            不存在 -> ToolError
      ├─ input_model(**args)           失败 -> ToolValidationError
      ├─ permission_checker.check()    失败 -> ToolPermissionError
      ├─ policy_engine.evaluate()      DENY/REQUIRE_CONFIRMATION -> ToolPolicyError
      ├─ asyncio.wait_for(invoke)      超时 -> ToolTimeoutError
      │    └─ transient 错误 -> 重试 max_tool_retries 次
      ├─ output_model 校验（可选）      失败 -> ToolExecutionError
      └─ ToolResult 信封 → 写回 Messages
```

## 核心数据结构

```python
class UserContext(BaseModel):
    user_id: str
    roles: list[str]
    permissions: list[str]

class PolicyDecision(BaseModel):
    decision: str    # ALLOW | DENY | REQUIRE_CONFIRMATION
    reason: str      # 原因（写入 Trace / Job / API，可解释）
    policy_name: str # 命中的策略名

class ToolDefinition:
    name: str
    description: str
    input_model: type[BaseModel]      # 参数模型（生成 JSON Schema + 校验）
    handler: Callable                 # 工具业务逻辑
    timeout_seconds: float = 10.0     # 超时（Gateway 统一执行）
    risk_level: str = "low"           # low | medium | high（Policy 使用）
    required_permission: str | None   # 权限（Permission 使用）
    output_model: type[BaseModel] | None  # 返回值校验（可选）

# 统一信封（成功/失败都结构化）
class ToolResult:
    success: bool
    tool_name: str
    data: Any
    error: ToolErrorInfo | None   # {type, message, code}
    metadata: dict                # duration_ms / args / retries / user_id
```

## 关键代码

```python
# gateway.py：统一执行链
async def execute(self, name, args, user=None) -> ToolResult:
    tool = self.registry.get(name)                     # 0. 存在性
    validated = tool.input_model(**args)               # 1. Schema 校验
    self.permission_checker.check(tool, user)          # 2. 权限
    decision = self.policy_engine.evaluate(tool, user) # 3. 策略
    if decision.decision != "ALLOW":                   #    拒绝
        return ToolResult.fail(name, ToolPolicyError(decision.reason), ...)
    data = await asyncio.wait_for(                     # 4. 超时 + 执行
        tool.invoke(validated.model_dump()),
        timeout=tool.timeout_seconds,
    )
    return ToolResult.ok(name, data, ...)              # 5. 统一信封

# policy.py：最小规则引擎
def evaluate(self, tool, user) -> PolicyDecision:
    if tool.name in self.denied_tools:                 # 黑名单
        return PolicyDecision("DENY", "被黑名单禁止", "deny_list")
    if tool.risk_level in self.require_confirmation_risks:
        return PolicyDecision("REQUIRE_CONFIRMATION", "需要人工确认", "risk_level_confirmation")
    return PolicyDecision("ALLOW", "默认放行", "default_allow")
```

## 输入示例

```text
get_weather {"city": "北京"}            # 正常
get_weather {"city": 123}               # Schema 错误
admin_tool   {"message": "删库"}        # 普通用户 -> 权限拒绝
danger_tool  {"target": "users"}        # 高风险 -> 策略拒绝
slow_tool    {"delay": 1.0}             # 超时（0.2s 限制）
explode_tool {}                         # 内部异常
flaky_tool   {}                         # 瞬时错误 -> 自动重试
```

## 输出示例

```json
{"success": false, "tool_name": "get_weather", "data": null,
 "error": {"type": "ToolValidationError", "message": "参数校验失败: ...", "code": "ToolValidationError"},
 "metadata": {"args": {"city": 123}, "user_id": "user1", "duration_ms": 0, "retries": 0}}
```

## 如何运行

```bash
python -m demos.stage5_demo
pytest tests/test_tool_gateway.py -v
```

## 如何测试

覆盖：正常 / Schema 错误（类型错、缺字段）/ 未知工具 / 权限拒绝与放行 /
策略（高风险、黑名单）/ 超时 / 内部异常包装 / 非瞬时错误不重试 / 瞬时错误重试 /
结果校验失败 / 运行时端到端权限（无权限→回答含"缺少权限"，有权限→正常）。

## 常见错误

| 错误 | 原因 | 修复 |
|---|---|---|
| 校验在 Tool 内部做 | 把 pydantic 校验写在 handler 里 | 校验必须放在 Gateway 入口 |
| 超时没生效 | 用同步 time.sleep 且没走 wait_for | 统一 `asyncio.wait_for(tool.invoke(...))` |
| 权限和策略混为一谈 | 都写成 if 判断 | 权限=身份维度（required_permission），策略=规则维度（risk/黑名单） |
| transient 错误无限重试 | 重试逻辑没有上限 | 用 `max_tool_retries` 封顶 |
| 裸异常冒泡 | handler 直接 raise 且没捕获 | Gateway 统一捕获包装成信封 |

## 面试如何表达

> "Stage 5 我实现了 Tool Gateway，把工具治理集中收口：Schema 校验（pydantic）、
> 权限（required_permission 与用户上下文比对）、策略（PolicyDecision：黑名单、风险等级
> 确认）、超时（asyncio.wait_for 统一计时）、瞬时错误重试（transient 标记 + 次数上限）、
> 结果校验，所有结果统一返回 ToolResult 信封。我会强调两个概念区分：Permission 回答
> '这个人能不能用'，Policy 回答 '这个调用该不该放行'；以及超时为什么必须由 Gateway
> 统一负责 —— 不能让某个工具永远阻塞 Worker。"

---

下一阶段：Stage 6 Tracing + Evaluation —— 为什么失败、修改是否变好。
