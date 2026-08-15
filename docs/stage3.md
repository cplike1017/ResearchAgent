# Stage 3：Session + Checkpoint

> 阶段目标：Agent 不再是一次性函数调用，而具有 **Session（会话）、State（状态）、Checkpoint（检查点）**，
> 进程崩溃后能从断点继续执行。

---

## 解决什么问题

Stage 2 之前，Agent 是"输入一句话 → 输出一句话"的无状态函数：

- 对话历史每次都要重新传给模型（Stage 2 解决了裁剪，但没解决**存储**）；
- 进程一旦重启，所有状态全部丢失；
- 执行到一半崩溃（如工具调用前），只能从头重跑 —— 浪费且可能重复副作用。

Stage 3 引入三层持久化：

```text
Session    业务数据：会话是谁、聊过什么（消息历史）
AgentState 执行状态：这一步进行到哪（step / status / 待执行工具）
Checkpoint 执行快照：某个瞬间 AgentState 的完整拷贝（用于恢复）
```

## 上一阶段有什么缺陷

Stage 2 的 Context Builder 只是"裁剪"，它隐含假设历史已经存在某个地方 —— 但 Stage 2
没有任何存储：每次 `runtime.run()` 都是全新开始，多轮对话历史靠调用方自己拼。

另外，Stage 1/2 的循环一旦开始就不能中断：工具执行前崩溃 = 全部重来。

## 本阶段新增什么组件

| 组件 | 文件 | 职责 |
|---|---|---|
| Session 模型 | `app/session/models.py` | `SessionRecord` / `MessageRecord` |
| Session 仓库 | `app/session/repository.py` | SQLite：建会话、追加/查询消息 |
| Checkpoint 模型 | `app/checkpoint/models.py` | `CheckpointRecord`（含 version） |
| Checkpoint 仓库 | `app/checkpoint/repository.py` | SQLite：追加写版本、加载最新 |
| AgentState | `app/agent/state.py` | 状态机：RUNNING / PENDING_TOOL / DONE |
| Runtime 升级 | `app/agent/runtime.py` | 会话持久化 + 四个检查点钩子 + `resume()` |

## 数据如何流动

```text
run("查询北京天气", session_id)
  ├─ Session 不存在？创建
  ├─ 加载历史消息（list_messages）
  ├─ 追加用户消息并立即持久化
  ├─ 进入 ReAct 循环，四个关键节点保存 Checkpoint：
  │     before_llm    （LLM 决策前）
  │     after_decision（LLM 决策后，若为工具调用则 status=PENDING_TOOL）
  │     after_tool     （工具执行后）
  │     before_final   （最终回答前）
  └─ 返回 AgentTurnResult

resume(session_id)          ← 进程崩溃后调用
  ├─ load_latest(session_id) 加载最高版本 Checkpoint
  ├─ 反序列化 AgentState（状态来自 Checkpoint，不是用户输入）
  ├─ 若 status == PENDING_TOOL：补 assistant 消息 + 重新执行待办工具
  └─ 进入同一个 ReAct 循环继续（_run_with_state）
```

## 核心数据结构

```python
class SessionRecord(BaseModel):
    session_id: str; created_at: str; updated_at: str
    status: SessionStatus   # ACTIVE | CLOSED

class MessageRecord(BaseModel):
    message_id: str; session_id: str; role: str
    content: str            # 完整消息 dict 的 JSON（可含 tool_calls）
    created_at: str

class CheckpointRecord(BaseModel):
    checkpoint_id: str; session_id: str; turn_id: str
    step: int; version: int   # 同一会话 1,2,3... 递增
    state: dict               # AgentState 快照
    created_at: str

class AgentState(BaseModel):
    session_id: str; turn_id: str; step: int
    status: str               # RUNNING | PENDING_TOOL | DONE | FAILED
    messages: list[dict]
    pending_tool_calls: list[dict]   # 待执行工具
    last_tool_result: dict | None
```

## 关键代码

```python
# Checkpoint 追加写版本：旧 Checkpoint 永远不可能覆盖新状态
def save(self, *, session_id, turn_id, step, state) -> CheckpointRecord:
    record = CheckpointRecord(
        checkpoint_id=f"ckpt_{uuid4().hex[:12]}",
        version=self._next_version(session_id),   # MAX(version)+1
        ...
    )
    self._conn.execute("INSERT INTO checkpoints ...")   # 只插入，不 UPDATE

# 恢复：状态从 Checkpoint 来
async def resume(self, session_id, checkpoint_id=None):
    checkpoint = self.checkpoint_repo.load(checkpoint_id) \
                 or self.checkpoint_repo.load_latest(session_id)
    state = AgentState(**checkpoint.state)          # ← 反序列化 Checkpoint
    if state.status == "PENDING_TOOL":
        # 补 assistant 消息 + 重新执行待办工具（可能重复执行 —— 教学点）
        ...
    return await self._run_with_state(state)        # 继续循环
```

## 输入示例

```text
第一次运行：查询北京天气   （LLM 决策后崩溃）
第二次运行：resume()       （从 Checkpoint 继续）
```

## 输出示例

```text
--- 崩溃点检查点 ---
checkpoint_id : ckpt_ad6450b812b7
version       : 2
恢复前 state  :
{
  "status": "PENDING_TOOL",
  "messages": [{"role": "user", "content": "查询北京天气"}],
  "pending_tool_calls": [{"name": "get_weather", "arguments": {"city": "北京"}}]
}

--- 恢复后 ---
最新 version       : 6
Final Answer       : 北京天气：晴，25°C，微风。
Checkpoint 版本演进: [1, 2, 3, 4, 5, 6]
```

## 如何运行

```bash
python -m demos.stage3_demo
```

## 如何测试

```bash
pytest tests/test_session_checkpoint.py -v
```

覆盖：Session 增查、消息往返（含 tool_calls 结构）、状态更新、版本递增、
旧 Checkpoint 不覆盖新状态、运行时自动落库、多轮历史累积、崩溃恢复（PENDING_TOOL /
RUNNING 两种）、无检查点报错、Session 与 Checkpoint 的数据差异。

## 常见错误

| 错误 | 原因 | 修复 |
|---|---|---|
| 恢复后用户消息出现两次 | resume 时把用户输入又执行了一遍（假恢复） | 恢复必须反序列化 Checkpoint，不重新调用 run() |
| 恢复后工具执行了两次 | PENDING_TOOL 恢复重新执行待办工具（真实语义） | 要么接受（文档化），要么在 Checkpoint 里记录"已执行"状态 |
| 版本不变 | save 用了 UPDATE 覆盖 | 改为 INSERT 追加写，version 用 MAX+1 |
| 多线程访问 SQLite 报错 | 未加锁 / check_same_thread | 仓库内加 threading.Lock + check_same_thread=False |

## 面试如何表达

> "Stage 3 我把 Agent 从无状态函数升级为有状态实体：Session 存业务数据（消息历史），
> AgentState 表示执行状态机（RUNNING / PENDING_TOOL / DONE），Checkpoint 是状态机在某
> 一瞬间的快照，用 SQLite 追加写保存并带单调递增版本号。恢复时反序列化最新 Checkpoint
> 直接继续循环，而不是重放用户输入 —— 关键设计是版本号防止旧检查点覆盖新状态，以及
> PENDING_TOOL 状态下恢复会重新执行待办工具，这正好暴露了 Retry 导致工具重复执行的
> 经典问题。"

---

下一阶段：Stage 4 Redis Queue + Worker —— 并发请求怎么处理。
