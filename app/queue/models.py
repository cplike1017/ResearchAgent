"""Job 数据模型。"""
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job 生命周期状态。"""

    QUEUED = "QUEUED"          # 已入队，等待 Worker 消费
    RUNNING = "RUNNING"        # 正在被 Worker 处理
    SUCCEEDED = "SUCCEEDED"    # 成功
    FAILED = "FAILED"          # 失败（已用尽重试次数）


class Job(BaseModel):
    """一个异步任务（Redis Job）。

    trace_context 在第六阶段被填充：{trace_id, parent_span_id}，
    用于把 HTTP -> Redis -> Worker -> Agent 串成同一条 Trace。
    """

    job_id: str
    request_id: str = Field(description="客户端请求 ID，用于幂等去重")
    session_id: str = ""
    input: dict = Field(default_factory=dict, description="任务输入，如 {'message': '...'}")
    user: dict = Field(default_factory=dict, description="调用方身份上下文（Stage 5 权限）")
    attempt: int = Field(default=0, description="已尝试次数（从 0 开始）")
    created_at: str = ""
    status: JobStatus = JobStatus.QUEUED
    result: dict | None = Field(default=None, description="成功结果，如 {'answer': ..., 'trace_id': ...}")
    error: dict | None = Field(default=None, description="失败的结构化错误 {type, message, code}")
    trace_context: dict = Field(default_factory=dict, description="Trace 传播上下文（Stage 6）")
