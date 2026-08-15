"""Trace 数据模型：Span。"""
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SpanStatus(str, Enum):
    """Span 状态。"""

    OK = "OK"
    ERROR = "ERROR"


class Span(BaseModel):
    """一条链路记录（Span）。

    核心字段（面试点）：
        trace_id        —— 一次请求/一次 Agent 回合的全局唯一 ID（跨进程不变）
        span_id         —— 本 Span 的唯一 ID
        parent_span_id  —— 父 Span 的 ID（None 表示根 Span）；靠它重建调用树

    其余字段描述"这一段发生了什么"：
        name / span_type —— 名称与类型（gateway / queue / worker / agent / llm / tool / checkpoint / eval）
        start_time / end_time / duration_ms —— 时间与耗时
        status / error   —— 成功或失败（含结构化错误）
        input / output   —— 入参出参（按配置脱敏 / 省略）
        attributes       —— 附加指标（model / tokens / tool_name 等）
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    span_type: str = "generic"
    start_time: str = ""
    end_time: str | None = None
    duration_ms: float | None = None
    status: SpanStatus = SpanStatus.OK
    input: Any = None
    output: Any = None
    attributes: dict = Field(default_factory=dict)
    error: dict | None = Field(default=None, description="结构化错误 {type, message, code}")

    def to_json_line(self) -> str:
        """序列化为 JSONL 的一行。"""
        return self.model_dump_json()
