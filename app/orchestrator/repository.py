"""编排结果仓库（SQLite）：把每次 delegate 编排的完整结果持久化到会话。

为什么需要单独存编排结果？
    delegate 工具的结果会作为 tool 消息进入主 agent 会话（消息里已有一份），
    但那是给 LLM 看的文本；编排的结构化信息 —— 计划、每个子 agent 的
    答案/工具调用/耗时/状态、最终合成答案 —— 需要按会话可查询、可回放：

        GET /api/web/orchestrations?session_id=xxx   → 该会话的全部编排记录
        GET /api/web/orchestrations/{run_id}         → 单次编排详情

表结构（orchestrations）：
    run_id        编排运行 ID（每次 delegate 调用一条）
    session_id    所属会话（多级编排：嵌套子编排记录在同一个 session 下）
    parent_run_id 父编排 run_id（多级编排用；顶层为 NULL）
    depth         编排深度（1 = 顶层；2 = 子 agent 再委派）
    task          编排任务
    status        SUCCEEDED | PARTIAL | FAILED
    plan_json     编排计划（SubTask 列表）
    results_json  各子 agent 结果（AgentRunResult 列表）
    final_answer  最终合成答案
    duration_ms   总耗时
    trace_id      Trace ID（关联完整调用树）
    created_at    创建时间

实现沿用项目惯例：标准库 sqlite3 + threading.Lock，与 Session/Checkpoint
仓库同库不同表，零额外依赖。
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, Field

from app.orchestrator.models import AgentRunResult, OrchestrationPlan, OrchestrationResult


class OrchestrationRecord(BaseModel):
    """一次编排的持久化记录。"""

    run_id: str = Field(default_factory=lambda: f"run_{uuid4().hex[:12]}")
    session_id: str = ""
    parent_run_id: str | None = None
    depth: int = 1
    task: str = ""
    status: str = "SUCCEEDED"
    plan: OrchestrationPlan = Field(default_factory=OrchestrationPlan)
    agent_results: list[AgentRunResult] = Field(default_factory=list)
    final_answer: str = ""
    duration_ms: float = 0.0
    trace_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @classmethod
    def from_result(
        cls,
        result: OrchestrationResult,
        *,
        session_id: str = "",
        parent_run_id: str | None = None,
        depth: int = 1,
    ) -> "OrchestrationRecord":
        return cls(
            session_id=session_id,
            parent_run_id=parent_run_id,
            depth=depth,
            task=result.task,
            status=result.status,
            plan=result.plan,
            agent_results=result.agent_results,
            final_answer=result.final_answer,
            duration_ms=result.duration_ms,
            trace_id=result.trace_id,
        )


def _sqlite_path(database_url: str) -> str:
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError(f"仅支持 sqlite URL: {database_url}")
    return database_url[len(prefix):]


class SQLiteOrchestrationRepository:
    """基于 SQLite 的编排结果仓库（与 Session/Checkpoint 同库不同表）。"""

    def __init__(self, database_url: str) -> None:
        path = _sqlite_path(database_url)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS orchestrations (
                    run_id        TEXT PRIMARY KEY,
                    session_id    TEXT NOT NULL DEFAULT '',
                    parent_run_id TEXT,
                    depth         INTEGER NOT NULL DEFAULT 1,
                    task          TEXT NOT NULL DEFAULT '',
                    status        TEXT NOT NULL DEFAULT 'SUCCEEDED',
                    plan_json     TEXT NOT NULL DEFAULT '{}',
                    results_json  TEXT NOT NULL DEFAULT '[]',
                    final_answer  TEXT NOT NULL DEFAULT '',
                    duration_ms   REAL NOT NULL DEFAULT 0,
                    trace_id      TEXT,
                    created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_orch_session ON orchestrations(session_id, created_at);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    def save(self, record: OrchestrationRecord) -> OrchestrationRecord:
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO orchestrations
                    (run_id, session_id, parent_run_id, depth, task, status,
                     plan_json, results_json, final_answer, duration_ms, trace_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.session_id,
                    record.parent_run_id,
                    record.depth,
                    record.task,
                    record.status,
                    record.plan.model_dump_json(),
                    json.dumps([r.model_dump() for r in record.agent_results], ensure_ascii=False),
                    record.final_answer,
                    record.duration_ms,
                    record.trace_id,
                    record.created_at,
                ),
            )
            self._conn.commit()
        return record

    # ------------------------------------------------------------------
    def list_by_session(self, session_id: str, *, limit: int = 50) -> list[OrchestrationRecord]:
        rows = self._query(
            "SELECT * FROM orchestrations WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        )
        return [_row_to_record(r) for r in rows]

    def list_recent(self, *, limit: int = 20) -> list[OrchestrationRecord]:
        rows = self._query(
            "SELECT * FROM orchestrations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [_row_to_record(r) for r in rows]

    def get(self, run_id: str) -> OrchestrationRecord | None:
        rows = self._query("SELECT * FROM orchestrations WHERE run_id = ?", (run_id,))
        return _row_to_record(rows[0]) if rows else None

    def list_children(self, parent_run_id: str) -> list[OrchestrationRecord]:
        rows = self._query(
            "SELECT * FROM orchestrations WHERE parent_run_id = ? ORDER BY created_at ASC",
            (parent_run_id,),
        )
        return [_row_to_record(r) for r in rows]

    def delete_by_session(self, session_id: str) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM orchestrations WHERE session_id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount

    # ------------------------------------------------------------------
    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_record(row: sqlite3.Row) -> OrchestrationRecord:
    try:
        plan = OrchestrationPlan.model_validate_json(row["plan_json"] or "{}")
    except Exception:
        plan = OrchestrationPlan()
    try:
        results = [AgentRunResult.model_validate(r) for r in json.loads(row["results_json"] or "[]")]
    except Exception:
        results = []
    return OrchestrationRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        parent_run_id=row["parent_run_id"],
        depth=row["depth"],
        task=row["task"],
        status=row["status"],
        plan=plan,
        agent_results=results,
        final_answer=row["final_answer"],
        duration_ms=row["duration_ms"],
        trace_id=row["trace_id"],
        created_at=row["created_at"],
    )
