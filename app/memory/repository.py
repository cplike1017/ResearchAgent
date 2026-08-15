"""
SQLite + sqlite-vec 向量记忆仓库。

设计（与 session/repository.py 同思路：标准库 + 零 ORM）：
    - 复用 SQLite 文件（DATABASE_URL 解析同一函数），不引入新服务；
    - sqlite-vec 提供 vector(N) 列类型与 vec_distance_cosine 等距离函数；
    - 存储：memory_id / text / embedding(vector) / metadata / created_at；
    - 检索：vec_distance_cosine 排序取 Top-K，附时间衰减权重。

时间衰减（recency decay）：
    score_final = cosine_sim * decay，decay = exp(-age_hours / half_life_hours)
    最近记忆权重高，防止陈年旧事霸占上下文。
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import sqlite_vec

from app.memory.models import MemoryRecord
from app.session.repository import sqlite_path_from_url


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _now_epoch() -> float:
    return time.time()


class SQLiteVecMemoryRepository:
    """基于 SQLite + sqlite-vec 的语义记忆仓库。"""

    def __init__(self, database_url: str, dim: int = 1024) -> None:
        path = sqlite_path_from_url(database_url)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._dim = dim
        self._lock = threading.Lock()
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id         TEXT PRIMARY KEY,
                    text              TEXT NOT NULL,
                    embedding         vector({self._dim}),
                    memory_type       TEXT NOT NULL DEFAULT 'fact',
                    source_session_id TEXT NOT NULL DEFAULT '',
                    source_turn_id    TEXT NOT NULL DEFAULT '',
                    created_at        TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_memories_type
                    ON memories(memory_type);
                CREATE INDEX IF NOT EXISTS idx_memories_created
                    ON memories(created_at);
                """
            )
            self._conn.commit()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        embedding: list[float],
        *,
        memory_type: str = "fact",
        source_session_id: str = "",
        source_turn_id: str = "",
        memory_id: str | None = None,
    ) -> MemoryRecord:
        """写入一条记忆（带向量）。"""
        record = MemoryRecord(
            memory_id=memory_id or f"mem_{uuid4().hex[:12]}",
            text=text,
            embedding=embedding,
            memory_type=memory_type,
            source_session_id=source_session_id,
            source_turn_id=source_turn_id,
            created_at=utc_now(),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO memories (memory_id, text, embedding, memory_type, source_session_id, source_turn_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.memory_id,
                    record.text,
                    sqlite_vec.serialize_float32(embedding),
                    record.memory_type,
                    record.source_session_id,
                    record.source_turn_id,
                    record.created_at,
                ),
            )
            self._conn.commit()
        return record

    def add_batch(self, records: list[dict]) -> list[MemoryRecord]:
        """批量写入（一次回合可能提炼多条事实）。"""
        out = []
        with self._lock:
            for r in records:
                rec = MemoryRecord(
                    memory_id=r.get("memory_id") or f"mem_{uuid4().hex[:12]}",
                    text=r["text"],
                    embedding=r["embedding"],
                    memory_type=r.get("memory_type", "fact"),
                    source_session_id=r.get("source_session_id", ""),
                    source_turn_id=r.get("source_turn_id", ""),
                    created_at=utc_now(),
                )
                self._conn.execute(
                    "INSERT INTO memories (memory_id, text, embedding, memory_type, source_session_id, source_turn_id, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        rec.memory_id,
                        rec.text,
                        sqlite_vec.serialize_float32(rec.embedding),
                        rec.memory_type,
                        rec.source_session_id,
                        rec.source_turn_id,
                        rec.created_at,
                    ),
                )
                out.append(rec)
            self._conn.commit()
        return out

    # ------------------------------------------------------------------
    # 检索：Top-K 余弦相似度 + 时间衰减
    # ------------------------------------------------------------------
    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        half_life_hours: float = 24.0,
        memory_type: str | None = None,
    ) -> list[MemoryRecord]:
        """
        检索最相似的记忆。

        评分 = 余弦相似度 × 时间衰减（decay = exp(-age_hours / half_life_hours)）。
        sqlite-vec 的 vec_distance_cosine 返回 0(最相似)~2(最不相似)，
        转相似度：sim = 1 - distance。
        """
        q = sqlite_vec.serialize_float32(query_embedding)
        where = ""
        params: list[Any] = [q, top_k]
        if memory_type:
            where = "WHERE memory_type = ?"
            params = [q, memory_type, top_k]
        rows = self._conn.execute(
            f"""
            SELECT memory_id, text, memory_type, source_session_id, source_turn_id, created_at,
                   vec_distance_cosine(embedding, ?) AS distance
            FROM memories
            {where}
            ORDER BY distance ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

        now = _now_epoch()
        results: list[MemoryRecord] = []
        for row in rows:
            sim = 1.0 - float(row["distance"])
            if sim < min_score:
                continue
            age_hours = 0.0
            try:
                created = datetime.fromisoformat(row["created_at"])
                age_hours = max(0.0, (now - created.timestamp()) / 3600.0)
            except (ValueError, TypeError):
                pass
            decay = 2.0 ** (-age_hours / half_life_hours) if half_life_hours > 0 else 1.0
            results.append(
                MemoryRecord(
                    memory_id=row["memory_id"],
                    text=row["text"],
                    memory_type=row["memory_type"],
                    source_session_id=row["source_session_id"],
                    source_turn_id=row["source_turn_id"],
                    created_at=row["created_at"],
                    score=round(sim * decay, 4),
                ).to_search_result()
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ------------------------------------------------------------------
    # 管理
    # ------------------------------------------------------------------
    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM memories").fetchone()
        return int(row["c"])

    def list_all(self, limit: int = 100) -> list[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT memory_id, text, memory_type, source_session_id, source_turn_id, created_at "
                "FROM memories ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            MemoryRecord(
                memory_id=r["memory_id"],
                text=r["text"],
                memory_type=r["memory_type"],
                source_session_id=r["source_session_id"],
                source_turn_id=r["source_turn_id"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def delete(self, memory_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
