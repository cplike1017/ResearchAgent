# Stage 4：Redis Queue + Worker

> 阶段目标：把"HTTP 请求直接执行 Agent"改成"HTTP Gateway → Redis Queue → Worker → Agent Runtime"，
> 理解异步执行与并发扩展。

---

## 解决什么问题

Stage 3 之前，Agent 执行是同步的：一个 HTTP 请求进来，进程阻塞着跑完整个 ReAct 循环才返回。

两个致命问题：

1. **并发能力差**：慢任务（多次 LLM 调用 + 工具调用，秒级~十秒级）会占满进程，请求互相排队；
2. **耦合**：API 进程一旦崩溃，正在执行的 Agent 直接丢失；API 也无法横向扩容而不重复执行。

Stage 4 引入**消息队列**解耦：

```text
Client
  ↓ POST /api/chat
FastAPI Gateway（只接单：校验 + 幂等 + 入队，立即返回 job_id）
  ↓ RPUSH
Redis List（agent:jobs:queue）
  ↓ BLPOP
Worker × N（独立进程：取 Job → 执行 Agent → 写结果）
  ↓
Agent Runtime
```

- 为什么能提升并发？—— Gateway 只做毫秒级入队；真正的执行放到任意多个 Worker 上并行；
  Redis BLPOP 天然把 Job 分发给不同 Worker。
- 为什么 Gateway 和 Worker 要拆开？—— 各自可以独立扩容、独立部署、独立故障；
  Worker 崩溃不阻塞 API，Job 还可以重试。

## 上一阶段有什么缺陷

Stage 3 的 `runtime.run()` 仍是同步阻塞调用。文档虽然介绍了 `resume()`，
但没有任何机制让"一个请求"在进程间流转 —— 并发只能靠手动开多个进程，
没有统一的任务形态（Job）、没有状态查询、没有失败重试。

## 本阶段新增什么组件

| 组件 | 文件 | 职责 |
|---|---|---|
| Job 模型 | `app/queue/models.py` | `Job` / `JobStatus`（QUEUED/RUNNING/SUCCEEDED/FAILED） |
| Redis 队列 | `app/queue/producer.py` | Redis List + Hash 自研队列：入队（幂等）/ 出队 / 状态更新 / 重入队 |
| 消费者 | `app/queue/consumer.py` | `process_job`：RUNNING → 执行 → 成功 / 重试 / 失败 |
| Worker | `app/worker/worker.py` | 独立进程主循环：BLPOP → 处理 → 循环（可优雅退出） |
| HTTP Gateway | `app/api/routes.py` | `POST /api/chat`、`GET /api/jobs/{id}`、`GET /health` |
| 应用入口 | `app/main.py` | FastAPI + lifespan 连接 Redis |
| Docker | `Dockerfile` / `docker-compose.yml` | api / redis / worker 三服务，支持 `--scale worker=N` |

## 数据如何流动

```text
POST /api/chat {message, session_id?, idempotency_key?}
  → 生成 request_id / job_id
  → 幂等检查：request_id 已存在？返回已有 Job
  → SET NX 占位 + 写 Job Hash + RPUSH 队列
  → 返回 {request_id, job_id, session_id, status: "QUEUED"}

Worker 循环：
  → BLPOP 队列（阻塞 1s）
  → update_status(RUNNING)
  → 加载 Session → AgentRuntime.run(message, session_id)
  → update_status(SUCCEEDED, result={answer, trace_id})
  → 异常：attempt+1 < max_attempts ? 重入队 : FAILED
```

## 核心数据结构

```python
class Job(BaseModel):
    job_id: str
    request_id: str            # 幂等键
    session_id: str
    input: dict                # {"message": "..."}
    attempt: int               # 已尝试次数
    created_at: str
    status: JobStatus          # QUEUED | RUNNING | SUCCEEDED | FAILED
    result: dict | None        # {"answer": ..., "trace_id": ...}
    error: dict | None         # {"type": ..., "message": ..., "code": ...}
    trace_context: dict        # Stage 6: {trace_id, parent_span_id}

# Redis 键布局
agent:jobs:queue            List   # 待消费 Job id 队列
agent:jobs:{job_id}         Hash   # Job 全量字段
agent:requests:{request_id} String # 幂等占位（SET NX + TTL）
```

## 关键代码

```python
# 幂等入队（producer.py）：相同 request_id 绝不重复执行
async def enqueue(self, job: Job) -> Job:
    existing = await self._redis.get(self._request_key(job.request_id))
    if existing is not None:
        return await self.get_job(existing)          # 已有 -> 返回原 Job
    acquired = await self._redis.set(
        self._request_key(job.request_id), job.job_id, nx=True, ex=self.job_ttl
    )
    if not acquired:
        return await self.get_job(await self._redis.get(...))  # 并发被抢占
    await self.save_job(job)
    await self._redis.rpush(self.queue_name, job.job_id)
    return job

# 重试（consumer.py）：避免无限重试
if fresh.attempt + 1 < queue.max_attempts:
    fresh.attempt += 1
    fresh.status = JobStatus.QUEUED
    await queue.save_job(fresh)
    await queue.requeue(fresh.job_id)
else:
    await queue.update_status(fresh.job_id, JobStatus.FAILED, error=error)
```

## 输入示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "查询北京天气", "idempotency_key": "demo-1"}'
```

## 输出示例

```json
{"request_id": "demo-1", "job_id": "job_abc123", "session_id": "session_xxx", "status": "QUEUED"}
```

```bash
curl http://localhost:8000/api/jobs/job_abc123
```

```json
{"job_id": "job_abc123", "status": "SUCCEEDED",
 "result": {"answer": "北京天气：晴，25°C，微风。", "session_id": "session_xxx", "trace_id": null}}
```

## 如何运行

```bash
# 1) 启动整套服务（api + redis + worker）
docker compose up --build

# 2) 扩容 Worker
docker compose up --scale worker=3

# 3) 本地只跑队列 Demo（需要 Redis）
docker compose up -d redis
python -m demos.stage4_demo
```

## 如何测试

```bash
pytest tests/test_queue_worker.py -v
```

覆盖：入队/读取往返、BLPOP 出队、超时返回 None、幂等（同 request_id 不重复）、
Worker 成功处理、状态流转、重试至上限后 FAILED、Worker 主循环端到端、
3 个并发消费者处理 10 个 Job 无重复消费、HTTP API（入队/查询/幂等/404/健康检查）。

## 常见错误

| 错误 | 原因 | 修复 |
|---|---|---|
| 幂等失效 | 忘了 SET NX，先查后写有竞态 | 用 `SET key val NX EX ttl` 原子占位 |
| 无限重试 | 重试逻辑没有上限 | `attempt+1 < max_attempts` 才重入队 |
| 工具重复执行 | 工具成功但回写前崩溃，重试导致再执行 | 幂等工具 / 记录已执行状态（教学点） |
| BLPOP 阻塞无法退出 | 无限阻塞 | 用 1s 超时循环 + shutdown_event |
| 队列积压 | Worker 太少 / 处理太慢 | `--scale worker=N` |
| 跨进程 SQLite 锁 | 多 Worker 同时写 | WAL 模式 + busy_timeout（已内置） |

## 面试如何表达

> "Stage 4 我把执行从 API 进程剥离开：API 只做参数校验、生成 request_id/job_id、
> 以 SET NX 实现幂等入队，然后立即返回；Worker 用 BLPOP 消费 Redis List，执行 Agent
> 后写回状态。支持 max_attempts 重试且不会无限重试，`docker compose up --scale worker=3`
> 就能水平扩展。这里我会强调两个工程点：一是幂等 —— request_id 相同绝不重复执行，
> 二是重试语义 —— Worker 崩溃重试可能导致工具重复执行，这是分布式系统的经典问题，
> 需要幂等工具或执行记录来兜底。"

---

下一阶段：Stage 5 Tool Gateway + Policy —— Tool 怎么统一治理。
