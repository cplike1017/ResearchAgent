# Stage 6：Tracing + Evaluation

> 阶段目标：回答两个问题 —— **一次请求为什么失败能追踪**，**一次修改到底有没有变好能评测**。
> 自研最小 Tracing（不用 Langfuse / OpenTelemetry / Phoenix），自研规则评测（不依赖 LLM-as-a-Judge）。

---

## 解决什么问题

系统越来越复杂（Gateway → Redis → Worker → Agent → Tool），出问题时空口难查：

- **Trace（链路追踪）**：一次请求跨多个进程、多个组件，用 Log 无法关联。
  用 `trace_id` 串起全部 Span，回答"这次请求到底经过了什么、哪一步慢、哪一步错了"。
- **Eval（评测）**：改一个 Prompt / Tool Description，凭感觉"好像变好了"不靠谱。
  用固定数据集 + 可复现指标回答"到底变好还是变差"。

**Trace 和 Log 的区别（面试点）**：Log 是分散的、无关联的文本；
Trace 是结构化的、按父子关系组织的调用记录，天然支持聚合与树形展示。

## 上一阶段有什么缺陷

Stage 5 之前，没有任何观测手段：

- 失败只能看 Job 的 error 字段，不知道失败发生在哪一步、耗时多少；
- 跨进程（Gateway 入队 → Worker 执行）无法关联成同一条链路；
- "改了系统好不好"没有任何量化依据。

## 本阶段新增什么组件

| 组件 | 文件 | 职责 |
|---|---|---|
| Span 模型 | `app/tracing/models.py` | trace_id / span_id / parent_span_id / 耗时 / 状态 / 错误 |
| ContextVar 上下文 | `app/tracing/context.py` | current_trace_id / current_span_id，跨函数隐式传播 |
| Trace Recorder | `app/tracing/recorder.py` | JSONL 存储、脱敏、按 trace_id 加载、重建调用树 |
| trace_span | `app/tracing/span.py` | `with trace_span(...)` 创建 Span（异步 + 同步两版） |
| 全链路埋点 | runtime / gateway / queue / worker / api | agent.run / llm_call / tool_gateway / redis.enqueue / worker.process / gateway.request / checkpoint.* |
| Trace API | `app/api/routes.py` | `GET /api/traces/{trace_id}` 重建调用树 |
| Eval 数据集 | `evals/datasets/basic_agent.jsonl` | 30 个用例，8 类分布 |
| Evaluators | `evals/evaluators.py` | 规则评测 + Trace 指标 + P50/P95 |
| Runner | `evals/runner.py` | 执行数据集、输出指标、保存 Eval Run |
| Regression | `evals/report.py` | Baseline vs Candidate，Improved/Regressed 标注 |

## 数据如何流动

```text
HTTP POST /api/chat
  └─ trace_span("gateway.request")  ← 根 Span（新建 trace_id）
       ├─ trace_span("redis.enqueue")
       └─ Job.trace_context = {trace_id, parent_span_id}   ← 跨进程传播载体
Redis Queue
  └─ Worker: set_trace_context(job.trace_context)          ← 恢复上下文
       └─ trace_span("worker.process")
            └─ trace_span("agent.run")
                 ├─ context_builder
                 ├─ llm_call（model/tokens/finish_reason）
                 ├─ tool_gateway → tool.execute
                 ├─ llm_call
                 └─ checkpoint.save
JSONL（一行一个 Span）→ GET /api/traces/{trace_id} 重建调用树
```

## 核心数据结构

```python
class Span(BaseModel):
    trace_id: str           # 一次请求全局唯一（跨进程不变）
    span_id: str            # 本 Span 唯一
    parent_span_id: str | None   # 父 Span；None = 根
    name: str               # 如 "llm_call"
    span_type: str          # gateway | queue | worker | agent | llm | tool | checkpoint | eval
    start_time / end_time / duration_ms
    status: SpanStatus      # OK | ERROR
    input / output          # 按配置脱敏 / 省略
    attributes: dict        # model / tokens / tool_name / success ...
    error: dict | None      # {type, message, code}
```

## 关键代码

```python
# 1) trace_span：with 语法创建 Span
@contextlib.asynccontextmanager
async def trace_span(name, span_type, *, recorder=None):
    span = recorder.start_span(name, span_type)      # 读 current_trace_id / current_span_id
    token_trace = current_trace_id.set(span.trace_id)
    token_span = current_span_id.set(span.span_id)
    try:
        yield span                                   # with 体内的代码在这里执行
        recorder.end_span(span, output=span.output)  # 正常结束
    except Exception as exc:
        recorder.end_span(span, error=exc)           # 异常 -> ERROR
        raise
    finally:
        current_trace_id.reset(token_trace)          # ContextVar 恢复（token/reset）
        current_span_id.reset(token_span)

# 2) 跨进程传播：Gateway 写入
job.trace_context = get_trace_context()   # {"trace_id": T, "parent_span_id": X}
#    Worker 恢复
set_trace_context(job.trace_context["trace_id"], job.trace_context["parent_span_id"])
```

## 输入示例

```bash
curl -X POST http://localhost:8000/api/chat -H "Content-Type: application/json" \
  -d '{"message": "查询北京天气"}'
# → {"request_id": "req_xxx", "job_id": "job_xxx", "session_id": "session_xxx", "status": "QUEUED"}

curl http://localhost:8000/api/jobs/job_xxx
# → {"status": "SUCCEEDED", "result": {"answer": "北京天气：晴，25°C，微风。", "trace_id": "trace_xxx"}}

curl http://localhost:8000/api/traces/trace_xxx
```

## 输出示例

```text
TRACE trace_xxx
└── gateway.request  [gateway] 50ms
    ├── redis.enqueue  [queue] 2ms
    └── worker.process  [worker] 48ms
        └── agent.run  [agent] 47ms
            ├── context_builder  [context_builder] 1ms
            ├── llm_call  [llm] 10ms
            ├── tool_gateway  [tool_gateway] 5ms
            │   └── tool.execute  [tool] 4ms
            ├── llm_call  [llm] 12ms
            └── checkpoint.save  [checkpoint] 2ms
```

超时示例（Trace 中可见 ERROR）：

```text
└── tool.execute  [tool] ERROR
    └── error: ToolTimeoutError: 工具 slow_tool 执行超过 0.2s
```

## 如何运行

```bash
# 1) 单机 Trace 演示
python -m demos.stage6_demo

# 2) Docker 全链路（HTTP -> Redis -> Worker -> Agent -> Tool）
docker compose up --build -d
curl -X POST http://localhost:8000/api/chat -d '{"message": "查询北京天气"}'
# 轮询 job 直到 SUCCEEDED，取 result.trace_id
curl http://localhost:8000/api/traces/<trace_id>

# 3) 评测 + 回归
python -m evals.runner
python -m evals.runner --compare evals/runs/<baseline>.json
```

## 如何测试

```bash
pytest tests/test_tracing.py tests/test_evals.py -v
```

覆盖：Span 嵌套与树重建、异常 Span（ERROR + 错误记录 + 继续抛出）、ContextVar 恢复、
脱敏（敏感键）、内容省略（TRACE_CAPTURE_CONTENT=false）、完整 Agent 链路 Span、
Checkpoint Span、跨进程 Trace 传播（Gateway→Redis→Worker→Agent 同一 trace_id）、
Trace API、数据集分布、P50/P95 正确性、工具选择/参数评测、Trace 指标、回归报告标注、
30 用例端到端 Runner。

## 常见错误

| 错误 | 原因 | 修复 |
|---|---|---|
| 跨进程 Trace 断掉 | 没把 trace_context 写入 Job / 没在 Worker 恢复 | `get_trace_context()` 写 Job；Worker `set_trace_context()` |
| 子 Span 挂错父节点 | 在 with 外读取 current_span_id | 父子关系由 ContextVar 自动维护，别手动传 |
| 异常 Span 没记录 | 忘记 except 分支 | trace_span 的 except 分支负责 end_span(ERROR) |
| 上下文泄漏 | 忘记 reset | finally 里 token/reset（trace_span 已内置） |
| 敏感内容落盘 | 直接存完整 Prompt | TRACE_CAPTURE_CONTENT=false + redact() |
| P95 off-by-one | 手写排序索引错误 | 用线性插值公式 rank=(n-1)*q/100 |
| 只评最终答案 | 只看 answer | 必须从 Trace 评轨迹（工具/LLM 次数、错误、策略违规） |

## 概念解释（面试重点）

### @contextmanager / yield / with / \_\_enter\_\_ / \_\_exit\_\_

- `with obj as x:` 等价于 `x = obj.__enter__(); try: ... finally: obj.__exit__()`；
- `@contextmanager` 把一个生成器函数变成上下文管理器：
  - `yield` **之前**的代码 = `__enter__`；
  - `yield` **之后**的代码 = `__exit__` 的一部分（正常路径）；
  - 若 with 体内抛异常，异常在 `yield` 处抛出，被 `try/except` 捕获，
    因此 `except Exception` 分支能记录 ERROR 并重新 `raise`；
  - `finally` 无论成功失败都会执行 —— 所以 ContextVar 的恢复放在这里。
- 这正是 `with trace_span(...)` 能"自动开始、自动结束、自动恢复上下文"的原理。

### ContextVar 为什么比手动传参好

手动传参要把 `trace_id/span_id` 塞进每一个函数签名（llm.chat、tool.execute、checkpoint.save…），
侵入性极强且容易漏。ContextVar 是"当前线程/任务的隐式全局状态"：
- 读取方（recorder.start_span）直接读 current_trace_id；
- asyncio 在 `await` 与 `create_task` 时自动复制上下文；
- 嵌套 with 通过 token/reset 精确恢复，天然支持并发安全。

### P95 是什么

把 N 次请求的耗时排序，P95 = 第 95 百分位的值：
**95% 的请求耗时 ≤ 这个值**。它比平均值更能反映"尾延迟"（长尾慢请求），
是 SLO（服务等级目标）常用的指标。实现用线性插值：

```python
rank = (n - 1) * q / 100      # 例：n=100, q=95 -> rank=94.05
lo, hi = int(rank), min(int(rank)+1, n-1)
return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (rank - lo)
```

### 为什么 Agent Eval 不能只看最终答案

最终答案对也可能走了一条烂轨迹（多调了 3 次无用的工具、触发策略违规、
P95 从 2s 涨到 8s）。所以区分：

- **Outcome Quality**：结果对不对（任务成功 / 工具选对 / 参数正确）；
- **Trajectory Quality**：过程好不好（工具与 LLM 调用次数 / 无效工具 / 错误重试 / 策略违规 / 延迟）。

### 为什么修改后必须做 Regression Evaluation

单看一个指标可能误判（延迟降了但准确率崩了）。回归报告同时对比全部指标并标注
Improved / Regressed / Unchanged，防止"修好一个、弄坏另一个"。

## 如何运行评测

```bash
# 第一次（Baseline）
python -m evals.runner --tag baseline

# 修改 Prompt / Tool Description 后（Candidate）
python -m evals.runner --tag candidate --compare evals/runs/<baseline 的 json>
```

## 面试如何表达

> "Stage 6 我自己实现了一套最小 Tracing：Span 模型（trace_id/span_id/parent_span_id）、
> ContextVar 维护当前上下文、`with trace_span(...)` 上下文管理器自动开始/结束/恢复，
> JSONL 落盘。跨进程传播是关键 —— Gateway 把 trace_context 写进 Job，Worker 消费时
> set_trace_context 恢复，于是 HTTP→Redis→Worker→Agent→Tool 全部串在同一条 trace_id 下。
> 评测侧我用 30 个用例的固定数据集做规则评测：从 Trace 里统计工具选择准确率、参数准确率、
> 平均 LLM 调用、P50/P95 延迟等，并实现 Baseline vs Candidate 的回归报告，每个指标标注
> Improved/Regressed —— 只评最终答案是不够的，轨迹质量（Trajectory）和结果质量（Outcome）
> 必须分开看。"

---

六阶段全部完成。下一步：README 总览 + 完整架构图 + 最终集成验证。
