"""
统一错误模型（Error Model）。

设计目标：让 Trace（链路追踪）、Job Status（任务状态）、API Response（接口返回）
三处能够使用同一套异常类型互相映射，避免每一层各自定义一套错误。

异常层级：
    AgentError（基类）
    ├── LLMError            模型调用失败（网络 / 认证 / 格式）
    ├── ContextBuildError   上下文构建失败
    ├── CheckpointError     检查点保存 / 加载失败
    ├── QueueError          队列（Redis）操作失败
    └── ToolError           工具执行失败（基类）
        ├── ToolValidationError   参数 Schema 校验失败（Gateway 层拦截）
        ├── ToolPermissionError   权限不足（Gateway 层拦截）
        ├── ToolPolicyError       Policy 拒绝（Gateway 层拦截）
        ├── ToolTimeoutError      工具执行超时（Gateway 层统一处理）
        └── ToolExecutionError    工具内部抛出的业务异常
"""


class AgentError(Exception):
    """所有 Agent 相关异常的基类。"""

    def __init__(self, message: str = "", *, code: str = ""):
        super().__init__(message)
        self.message = message or self.__class__.__name__
        self.code = code or self.__class__.__name__

    def to_dict(self) -> dict:
        """转换为统一的结构化错误信息（用于 Tool Result Envelope / API 响应）。"""
        return {"type": self.__class__.__name__, "message": self.message, "code": self.code}


class LLMError(AgentError):
    """LLM 调用失败：网络错误、认证失败、响应格式非法等。"""


class ContextBuildError(AgentError):
    """Context Builder 构建模型输入失败。"""


class CheckpointError(AgentError):
    """Checkpoint 保存 / 加载失败。"""


class QueueError(AgentError):
    """Redis 队列操作失败（入队 / 出队 / 状态更新）。"""


class ToolError(AgentError):
    """工具执行失败基类。"""


class ToolValidationError(ToolError):
    """工具参数未通过 Schema 校验（由 Tool Gateway 拦截，不进入 Tool 内部）。"""


class ToolPermissionError(ToolError):
    """当前身份缺少调用该工具所需的权限（由 Tool Gateway 拦截）。"""


class ToolPolicyError(ToolError):
    """Policy Engine 拒绝了该工具调用（由 Tool Gateway 拦截）。"""


class ToolTimeoutError(ToolError):
    """工具执行超过 timeout_seconds（由 Tool Gateway 统一计时并中断）。"""


class ToolExecutionError(ToolError):
    """工具内部抛出的业务异常（如计算器遇到除零、天气库查无此城市）。

    :param transient: 是否为瞬时错误（如外部服务抖动）。
        Gateway 对 transient=True 的错误会按 max_tool_retries 重试；
        对确定性错误（参数/数据问题）不重试，避免无意义重放。
    """

    def __init__(self, message: str = "", *, transient: bool = False, code: str = ""):
        super().__init__(message, code=code)
        self.transient = transient


def error_to_dict(exc: BaseException) -> dict:
    """把任意异常安全地转换为 {type, message, code} 结构，避免敏感堆栈外泄。"""
    if isinstance(exc, AgentError):
        return exc.to_dict()
    return {"type": exc.__class__.__name__, "message": str(exc), "code": "UNKNOWN"}
