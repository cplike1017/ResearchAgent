"""
SQLite Session 仓库。

职责：
    - 保存 / 查询 Session（元信息）
    - 追加 / 查询 Message（完整消息 JSON）

设计：
    - 使用标准库 sqlite3，不引入 ORM（教学清晰、零依赖）；
    - 连接带 threading.Lock，兼容多线程 Worker；
    - 消息用自增 seq 保证顺序；
    - 方法为同步实现（SQLite 本地 IO 极快），由异步调用方直接调用。
"""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from uuid import uuid4

from app.errors import AgentError
from app.session.models import MessageRecord, SessionRecord, SessionStatus


def sqlite_path_from_url(database_url: str) -> str:
    """把 sqlite:///... URL 解析为文件路径。"""
    prefix = "sqlite:///"
    if not database_url.startswith(prefix):
        raise AgentError(f"不支持的数据库 URL（仅支持 sqlite）: {database_url}")
    return database_url[len(prefix):]


def utc_now() -> str:
    """当前 UTC 时间的 ISO 字符串。"""
    return datetime.now(timezone.utc).isoformat()


class SQLiteSessionRepository:
    """基于 SQLite 的 Session / Message 仓库。"""

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

    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status     TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    seq        INTEGER NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session
                    ON messages(session_id, seq);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Session 操作
    # ------------------------------------------------------------------
    def create_session(self, session_id: str | None = None) -> SessionRecord:
        """创建会话；已存在则直接返回。"""
        session_id = session_id or f"session_{uuid4().hex[:12]}"
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is not None:
                return SessionRecord(**dict(row))
            now = utc_now()
            self._conn.execute(
                "INSERT INTO sessions (session_id, created_at, updated_at, status) VALUES (?, ?, ?, ?)",
                (session_id, now, now, SessionStatus.ACTIVE.value),
            )
            self._conn.commit()
        return SessionRecord(session_id=session_id, created_at=now, updated_at=now, status=SessionStatus.ACTIVE)

    def get_session(self, session_id: str) -> SessionRecord | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return SessionRecord(**dict(row)) if row is not None else None

    def update_status(self, session_id: str, status: SessionStatus | str) -> None:
        value = status.value if isinstance(status, SessionStatus) else status
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET status = ?, updated_at = ? WHERE session_id = ?",
                (value, utc_now(), session_id),
            )
            self._conn.commit()

    def touch(self, session_id: str) -> None:
        """更新 updated_at（会话活跃时间）。"""
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (utc_now(), session_id),
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # Message 操作
    # ------------------------------------------------------------------
    def add_message(self, session_id: str, message: dict) -> MessageRecord:
        """追加一条消息（message 为 OpenAI 风格 dict，整体 JSON 存储）。"""
        message_id = f"msg_{uuid4().hex[:12]}"
        record = MessageRecord(
            message_id=message_id,
            session_id=session_id,
            role=message.get("role", "unknown"),
            content=json.dumps(message, ensure_ascii=False),
            created_at=utc_now(),
        )
        with self._lock:
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 AS s FROM messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()["s"]
            self._conn.execute(
                "INSERT INTO messages (message_id, session_id, seq, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (record.message_id, session_id, seq, record.role, record.content, record.created_at),
            )
            self._conn.commit()
        return record

    def list_messages(self, session_id: str, limit: int | None = None) -> list[dict]:
        """按顺序返回会话的全部消息 dict（limit 可只取最近 N 条）。"""
        sql = "SELECT content FROM messages WHERE session_id = ? ORDER BY seq ASC"
        params: tuple = (session_id,)
        if limit is not None:
            # 取最近 limit 条：先按 seq DESC 截断再反转
            sql = (
                "SELECT content FROM (SELECT content FROM messages WHERE session_id = ? "
                "ORDER BY seq DESC LIMIT ?) ORDER BY seq ASC"
            )
            params = (session_id, limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [json.loads(r["content"]) for r in rows]

    def count_messages(self, session_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
        return int(row["c"])

    def close(self) -> None:
        with self._lock:
            self._conn.close()
