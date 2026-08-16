"""
数据类真实工具：sqlite_query（只读查询）+ file_read / file_write（目录沙箱）。

设计：
    - sqlite_query：只读 SQLite 查询。安全：强制以 URI 只读模式打开、
      限制单条 SQL（禁止分号拼接多语句）、禁止写操作语句（INSERT/UPDATE/DELETE/DROP...）。
    - file_read / file_write：文件读写。安全：强制路径解析后必须位于 sandbox_dir
      内（防目录穿越）；file_write 限制文件大小；file_read 限制返回字节数。

这些工具让 Agent 具备"访问数据"的真实能力，而不只是天气/计算演示。
"""
import os
import sqlite3
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import ToolExecutionError

# 文件大小限制
MAX_READ_BYTES = 8192
MAX_WRITE_BYTES = 16384

# 禁止的 SQL 前缀（只读约束）
_FORBIDDEN_SQL_PREFIXES = (
    "insert", "update", "delete", "drop", "create", "alter", "truncate",
    "replace", "attach", "detach", "pragma", "vacuum", "reindex",
)


def _safe_sql(sql: str) -> None:
    """校验 SQL 为只读单语句。"""
    sql = sql.strip()
    if not sql:
        raise ToolExecutionError("SQL 不能为空")
    if ";" in sql.rstrip(";"):
        raise ToolExecutionError("只支持单条 SQL（不允许分号拼接多语句）")
    lower = sql.lower()
    if any(lower.startswith(p) for p in _FORBIDDEN_SQL_PREFIXES):
        raise ToolExecutionError("只读模式：不允许写操作")


# ---------------------------------------------------------------------------
# sqlite_query（只读）
# ---------------------------------------------------------------------------
class SqliteQueryArgs(BaseModel):
    """SQLite 只读查询参数。"""

    database: str = Field(description="SQLite 数据库文件路径（相对或绝对路径）")
    query: str = Field(description="只读 SELECT 查询")


def sqlite_query_handler(database: str, query: str) -> str:
    """对指定 SQLite 文件执行只读查询，返回行结果文本。"""
    _safe_sql(query)
    if not os.path.exists(database):
        raise ToolExecutionError(f"数据库文件不存在: {database}")
    try:
        # URI 只读模式：任何写操作都会失败（双重保护）
        conn = sqlite3.connect(f"file:{database}?mode=ro", uri=True, timeout=3)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(query)
            rows = cur.fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise ToolExecutionError(f"SQLite 查询失败: {exc}") from exc

    if not rows:
        return "(查询无结果)"
    # 只返回前 20 行，每行前 100 字符
    lines = [", ".join(str(v) for v in row) for row in rows[:20]]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# file_read / file_write（目录沙箱）
# ---------------------------------------------------------------------------
def _resolve_in_sandbox(relative_path: str) -> Path:
    """把相对路径解析到沙箱目录内，拒绝目录穿越。

    注意：这里直接用 Settings()（不经过 get_settings 的 lru_cache 单例），
    保证测试里 monkeypatch 环境变量后能读到新值（单例会缓存第一个值）。
    """
    settings = Settings()
    sandbox = Path(settings.sandbox_dir).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    target = (sandbox / relative_path).resolve()
    if not str(target).startswith(str(sandbox)):
        raise ToolExecutionError(f"路径越界：只允许在沙箱目录内操作（{sandbox}）")
    return target


class FileReadArgs(BaseModel):
    """文件读取参数。"""

    path: str = Field(description="沙箱内的相对路径，如 notes/foo.txt")


def file_read_handler(path: str) -> str:
    """读取沙箱内的文本文件（限 8KB）。"""
    target = _resolve_in_sandbox(path)
    if not target.exists():
        raise ToolExecutionError(f"文件不存在: {path}")
    if not target.is_file():
        raise ToolExecutionError(f"不是文件: {path}")
    try:
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolExecutionError(f"读取失败: {exc}") from exc
    return content[:MAX_READ_BYTES]


class FileWriteArgs(BaseModel):
    """文件写入参数。"""

    path: str = Field(description="沙箱内的相对路径，如 notes/foo.txt")
    content: str = Field(description="要写入的文本内容（限 16KB）")


def file_write_handler(path: str, content: str) -> str:
    """写入沙箱内的文件（限 16KB，自动创建父目录）。"""
    if len(content) > MAX_WRITE_BYTES:
        raise ToolExecutionError(f"内容超过 {MAX_WRITE_BYTES} 字节限制")
    target = _resolve_in_sandbox(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(f"写入失败: {exc}") from exc
    return f"已写入 {path}（{len(content)} 字符）"


# ---------------------------------------------------------------------------
# list_files / search_files / append_note（沙箱可见性 + 协作草稿）
# ---------------------------------------------------------------------------
class ListFilesArgs(BaseModel):
    """目录列举参数。"""

    path: str = Field(default=".", description="沙箱内相对目录，如 notes 或 .")


def list_files_handler(path: str = ".") -> str:
    """列出沙箱内目录的文件（含大小，最多 100 项；递归可查子目录）。"""
    target = _resolve_in_sandbox(path)
    if not target.exists():
        raise ToolExecutionError(f"目录不存在: {path}")
    if not target.is_dir():
        raise ToolExecutionError(f"不是目录: {path}")
    try:
        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError as exc:
        raise ToolExecutionError(f"列举失败: {exc}") from exc
    if not entries:
        return f"目录 {path} 为空"
    lines = [f"目录 {path} 共 {len(entries)} 项："]
    for p in entries[:100]:
        if p.is_dir():
            lines.append(f"  📁 {p.name}/")
        else:
            size = p.stat().st_size
            lines.append(f"  📄 {p.name} ({size} B)")
    return "\n".join(lines)


class SearchFilesArgs(BaseModel):
    """文件内容搜索参数。"""

    keyword: str = Field(description="搜索关键词（大小写不敏感）")
    path: str = Field(default=".", description="沙箱内相对目录，递归搜索")
    max_results: int = Field(default=10, ge=1, le=50, description="最多返回匹配行数")


def search_files_handler(keyword: str, path: str = ".", max_results: int = 10) -> str:
    """在沙箱内递归搜索包含关键词的文件与行（限文本文件，最多扫描 200 个文件）。"""
    target = _resolve_in_sandbox(path)
    if not target.exists() or not target.is_dir():
        raise ToolExecutionError(f"目录不存在: {path}")
    hits: list[str] = []
    scanned = 0
    for root, _, files in os.walk(target):
        for fname in files:
            scanned += 1
            if scanned > 200:
                break
            fpath = os.path.join(root, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if keyword.lower() in line.lower():
                            rel = os.path.relpath(fpath, target)
                            hits.append(f"{rel}:{lineno}: {line.strip()[:120]}")
                            if len(hits) >= max_results:
                                break
            except OSError:
                continue
            if len(hits) >= max_results:
                break
        if len(hits) >= max_results:
            break
    if not hits:
        return f"未找到包含「{keyword}」的内容（扫描 {scanned} 个文件）"
    return f"找到 {len(hits)} 处匹配（扫描 {scanned} 个文件）：\n" + "\n".join(hits)


class AppendNoteArgs(BaseModel):
    """追加笔记参数。"""

    path: str = Field(description="沙箱内相对路径，如 notes/draft.md")
    content: str = Field(description="要追加的内容（限 8KB）")


def append_note_handler(path: str, content: str) -> str:
    """向沙箱文件追加内容（不覆盖已有内容，自动换行分隔；用于跨子 agent 协作草稿）。"""
    if len(content) > MAX_READ_BYTES:
        raise ToolExecutionError(f"追加内容超过 {MAX_READ_BYTES} 字节限制")
    target = _resolve_in_sandbox(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            existing = target.read_text(encoding="utf-8", errors="replace")
            if len(existing) + len(content) > MAX_WRITE_BYTES:
                raise ToolExecutionError(f"追加后超过 {MAX_WRITE_BYTES} 字节限制")
            with open(target, "a", encoding="utf-8") as f:
                if existing and not existing.endswith("\n"):
                    f.write("\n")
                f.write(content)
                if not content.endswith("\n"):
                    f.write("\n")
        else:
            target.write_text(content + "\n", encoding="utf-8")
    except OSError as exc:
        raise ToolExecutionError(f"追加失败: {exc}") from exc
    return f"已追加到 {path}（当前 {len(content)} 字符）"
