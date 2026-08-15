"""
SQLite Checkpoint 仓库。

版本设计（最小实现）：
    - 同一 session 内 version 从 1 开始递增；
    - 保存采用"追加写"：每次 save 插入新版本行，绝不覆盖旧行；
      因此旧 Checkpoint 永远不可能覆盖新状态；
    - load_latest 取最高版本，天然是"最新状态"。

恢复语义（配合 AgentState.status）：
    - RUNNING       -> 直接从 messages 继续循环（LLM 将看到已有工具结果）
    - PENDING_TOOL  -> 需要重新执行 pending_tool_calls（工具可能重复执行）
    - DONE          -> 已结束，无需继续
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from uuid import uuid4

from app.checkpoint.models import CheckpointRecord
from app.session.repository import sqlite_path_from_url


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteCheckpointRepository:
    """基于 SQLite 的 Checkpoint 仓库。"""

    def __init__(self, database_url: str) -> None:
        path = sqlite_path_from_url(database_url)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # WAL 模式 + busy_timeout：多 Worker 进程并发写同一个 SQLite 时减少锁冲突
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id    TEXT NOT NULL,
                    turn_id       TEXT NOT NULL,
                    step          INTEGER NOT NULL,
                    version       INTEGER NOT NULL,
                    state         TEXT NOT NULL,
                    created_at    TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_checkpoints_session
                    ON checkpoints(session_id, version DESC);
                """
            )
            self._conn.commit()

    def _next_version(self, session_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM checkpoints WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return int(row["v"])

    def save(self, *, session_id: str, turn_id: str, step: int, state: dict) -> CheckpointRecord:
        """保存一个检查点，自动分配下一个版本号（追加写，不覆盖旧版本）。"""
        record = CheckpointRecord(
            checkpoint_id=f"ckpt_{uuid4().hex[:12]}",
            session_id=session_id,
            turn_id=turn_id,
            step=step,
            version=self._next_version(session_id),
            state=state,
            created_at=utc_now(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO checkpoints (checkpoint_id, session_id, turn_id, step, version, state, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.checkpoint_id,
                    session_id,
                    turn_id,
                    step,
                    record.version,
                    json.dumps(record.state, ensure_ascii=False),
                    record.created_at,
                ),
            )
            self._conn.commit()
        return record

    def load(self, checkpoint_id: str) -> CheckpointRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE checkpoint_id = ?", (checkpoint_id,)
            ).fetchone()
        return self._to_record(row)

    def load_latest(self, session_id: str) -> CheckpointRecord | None:
        """加载某会话的最新检查点（最高版本）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? ORDER BY version DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return self._to_record(row)

    def load_at_version(self, session_id: str, version: int) -> CheckpointRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM checkpoints WHERE session_id = ? AND version = ?",
                (session_id, version),
            ).fetchone()
        return self._to_record(row)

    def versions(self, session_id: str) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT version FROM checkpoints WHERE session_id = ? ORDER BY version ASC",
                (session_id,),
            ).fetchall()
        return [int(r["version"]) for r in rows]

    def _to_record(self, row) -> CheckpointRecord | None:
        if row is None:
            return None
        data = dict(row)
        data["state"] = json.loads(data["state"])
        return CheckpointRecord(**data)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
