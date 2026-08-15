"""
Trace Recorder（JSONL 存储）。

第一版用 JSONL（一行一个 Span）足够：
    traces/traces.jsonl

职责：
    - start_span：创建 Span（生成 trace_id / span_id / 时间戳）
    - end_span：补全 duration_ms / status / error 并写入文件
    - load_trace：按 trace_id 读取全部 Span（供 /api/traces 重建调用树）
    - 脱敏（Redaction）：敏感键替换；TRACE_CAPTURE_CONTENT=false 时省略完整内容

安全要求：Trace 默认不保存 API Key / Authorization / Cookie / Password / Token。
"""
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.config import Settings, get_settings
from app.errors import error_to_dict
from app.tracing.context import current_span_id, current_trace_id
from app.tracing.models import Span, SpanStatus

# 敏感字段键名（命中即脱敏为 [REDACTED]）
_SENSITIVE_KEYS = {
    "api_key", "apikey", "authorization", "cookie", "password", "token",
    "secret", "access_token", "refresh_token", "private_key",
}

# 内容省略时保留的预览长度
_PREVIEW_LEN = 120


def redact(value, depth: int = 0) -> Any:
    """递归脱敏：把敏感键的值替换为 [REDACTED]。"""
    if depth > 6:
        return "[DEPTH_LIMIT]"
    if isinstance(value, dict):
        return {
            (k if not _is_sensitive(k) else f"{k}_redacted"): (
                "[REDACTED]" if _is_sensitive(k) else redact(v, depth + 1)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v, depth + 1) for v in value]
    return value


def _is_sensitive(key: str) -> bool:
    k = str(key).lower()
    return any(s in k for s in _SENSITIVE_KEYS)


def summarize_content(value: Any, max_len: int = _PREVIEW_LEN) -> dict:
    """把完整内容压缩为省略摘要（TRACE_CAPTURE_CONTENT=false 时使用）。"""
    if isinstance(value, str):
        preview = value[:max_len]
        return {"omitted": True, "preview": preview}
    if isinstance(value, list):
        # 消息列表：只统计角色分布 + 首条预览
        roles = [m.get("role", "?") for m in value if isinstance(m, dict)]
        return {"omitted": True, "message_count": len(roles), "roles": roles}
    return {"omitted": True}


class TraceRecorder:
    """JSONL Trace 记录器（进程内线程安全；跨进程用 O_APPEND 追加）。"""

    def __init__(
        self,
        trace_file: str,
        *,
        enabled: bool = True,
        capture_content: bool = False,
    ) -> None:
        self.trace_file = trace_file
        self.enabled = enabled
        self.capture_content = capture_content
        self._lock = threading.Lock()
        if enabled and trace_file:
            parent = os.path.dirname(trace_file)
            if parent:
                os.makedirs(parent, exist_ok=True)

    # ------------------------------------------------------------------
    # 创建 / 结束 Span
    # ------------------------------------------------------------------
    def start_span(
        self,
        name: str,
        span_type: str,
        *,
        input: Any = None,
        attributes: dict | None = None,
    ) -> Span:
        """创建一个 Span。trace_id 继承当前上下文，没有则新建。"""
        trace_id = current_trace_id.get() or f"trace_{uuid4().hex[:12]}"
        return Span(
            trace_id=trace_id,
            span_id=f"span_{uuid4().hex[:12]}",
            parent_span_id=current_span_id.get(),
            name=name,
            span_type=span_type,
            start_time=_now_iso(),
            input=self._sanitize_input(input, span_type),
            attributes=dict(attributes or {}),
        )

    def end_span(
        self,
        span: Span,
        *,
        output: Any = None,
        error: BaseException | None = None,
        status: SpanStatus | None = None,
    ) -> Span:
        """结束一个 Span：补全耗时与状态，写入 JSONL。"""
        span.end_time = _now_iso()
        span.duration_ms = _duration_ms(span.start_time, span.end_time)
        if output is not None:
            span.output = redact(output)
        if error is not None:
            span.status = SpanStatus.ERROR
            span.error = error_to_dict(error)
        elif status is not None:
            span.status = status
        if self.enabled and self.trace_file:
            self._write(span)
        return span

    # ------------------------------------------------------------------
    # 写入 / 读取
    # ------------------------------------------------------------------
    def _write(self, span: Span) -> None:
        line = span.to_json_line()
        with self._lock:
            with open(self.trace_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def load_trace(self, trace_id: str) -> list[Span]:
        """读取某 trace_id 的全部 Span（按文件顺序）。"""
        spans: list[Span] = []
        if not self.trace_file or not os.path.exists(self.trace_file):
            return spans
        with self._lock:
            with open(self.trace_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        span = Span.model_validate_json(line)
                    except Exception:
                        continue
                    if span.trace_id == trace_id:
                        spans.append(span)
        return spans

    def build_tree(self, trace_id: str) -> dict:
        """按 parent_span_id 重建调用树（供 /api/traces 与 demo 展示）。"""
        spans = self.load_trace(trace_id)
        if not spans:
            return {"trace_id": trace_id, "spans": []}
        by_id = {s.span_id: s for s in spans}
        roots = [s for s in spans if s.parent_span_id is None or s.parent_span_id not in by_id]

        def node(span: Span) -> dict:
            children = [
                node(s) for s in spans if s.parent_span_id == span.span_id
            ]
            children.sort(key=lambda c: c["start_time"])
            return {
                "span_id": span.span_id,
                "name": span.name,
                "span_type": span.span_type,
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "error": span.error,
                "attributes": span.attributes,
                "start_time": span.start_time,
                "children": children,
            }

        return {
            "trace_id": trace_id,
            "spans": [node(r) for r in roots],
        }

    # ------------------------------------------------------------------
    # 内容策略：脱敏 / 省略
    # ------------------------------------------------------------------
    def _sanitize_input(self, value: Any, span_type: str) -> Any:
        if value is None:
            return None
        if not self.capture_content and span_type in ("llm", "context_builder"):
            # 默认不保存完整 Prompt / 上下文内容
            return summarize_content(value)
        return redact(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _duration_ms(start_iso: str, end_iso: str) -> float:
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        return round((end - start).total_seconds() * 1000, 3)
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# 进程级默认 Recorder（由配置驱动，可被测试/组件注入覆盖）
# ---------------------------------------------------------------------------
_default_recorder: TraceRecorder | None = None


def get_default_recorder(settings: Settings | None = None) -> TraceRecorder:
    """获取进程级默认 Recorder（懒初始化）。"""
    global _default_recorder
    if _default_recorder is None:
        settings = settings or get_settings()
        _default_recorder = TraceRecorder(
            settings.trace_file,
            enabled=settings.trace_enabled,
            capture_content=settings.trace_capture_content,
        )
    return _default_recorder


def reset_default_recorder() -> None:
    """重置默认 Recorder（测试隔离用）。"""
    global _default_recorder
    _default_recorder = None
