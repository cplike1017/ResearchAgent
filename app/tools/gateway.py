"""
Tool Gateway（工具网关，第五阶段核心）。

为什么需要 Gateway？—— 把"工具治理"从 Agent 循环里剥离出来，统一收口：

    Agent
      ↓ Tool Call
    Tool Gateway
      ├─ 1. Schema Validation（参数校验：错误参数在入口拦截）
      ├─ 2. Permission（权限校验：这个人能不能用）
      ├─ 3. Policy（策略评估：这个调用该不该放行）
      ├─ 4. Timeout（超时：不能让某个工具永远阻塞 Worker）
      ├─ 5. Tool Execute（执行，含瞬时错误重试）
      ├─ 6. Result Validation（返回值校验）
      ↓
    Tool Result Envelope（统一信封，成功/失败都结构化）
      ↓
    Agent

好处：Tool 自己只写业务逻辑；校验、权限、策略、超时、错误包装全部由
Gateway 统一负责，任何一层失败都返回结构化信封，绝不抛裸异常给循环。
"""
import asyncio
import time
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.errors import (
    ToolError,
    ToolExecutionError,
    ToolPermissionError,
    ToolPolicyError,
    ToolTimeoutError,
    ToolValidationError,
)
from app.tools.policy import PolicyEngine
from app.tools.registry import ToolRegistry
from app.tools.schemas import ToolResult, UserContext
from app.tracing.recorder import TraceRecorder, redact
from app.tracing.span import trace_span


class PermissionChecker:
    """最小权限校验：工具声明 required_permission，调用方必须持有。"""

    def check(self, tool, user: UserContext | None) -> None:
        if tool.required_permission is None:
            return
        permissions = (user.permissions if user is not None else []) or []
        if tool.required_permission not in permissions:
            raise ToolPermissionError(
                f"缺少权限「{tool.required_permission}」才能调用 {tool.name}（当前权限: {permissions}）"
            )


class ToolGateway:
    """统一工具执行入口：校验 -> 权限 -> 策略 -> 超时 -> 执行 -> 结果校验。"""

    def __init__(
        self,
        registry: ToolRegistry,
        policy_engine: PolicyEngine | None = None,
        permission_checker: PermissionChecker | None = None,
        settings: Settings | None = None,
        recorder: TraceRecorder | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings or Settings()
        # PolicyEngine 默认读取 self.settings 的风险配置（如 high 需确认）
        self.policy_engine = policy_engine or PolicyEngine(settings=self.settings)
        self.permission_checker = permission_checker or PermissionChecker()
        self.max_retries = self.settings.max_tool_retries
        self.recorder = recorder  # None = 不追踪

    # ------------------------------------------------------------------
    async def execute(
        self,
        name: str,
        args: dict,
        user: UserContext | None = None,
    ) -> ToolResult:
        """执行一次工具调用，返回统一信封（永不抛异常，除非 Gateway 自身故障）。"""
        if self.recorder is None or not self.recorder.enabled:
            return await self._execute_impl(name, args, user)

        async with trace_span(
            "tool_gateway",
            "tool_gateway",
            input={"tool_name": name, "arguments": redact(args), "user_id": user.user_id if user else None},
            attributes={"tool_name": name},
            recorder=self.recorder,
        ) as span:
            result = await self._execute_impl(name, args, user)
            span.output = {
                "success": result.success,
                "error_type": result.error.type if result.error else None,
            }
            return result

    async def _execute_impl(
        self,
        name: str,
        args: dict,
        user: UserContext | None,
    ) -> ToolResult:
        start = time.perf_counter()
        metadata: dict = {
            "args": args,
            "user_id": user.user_id if user else None,
        }

        # ---- 0) 工具是否存在 ----
        try:
            tool = self.registry.get(name)
        except ToolError as exc:
            return ToolResult.fail(name, exc, metadata=self._meta(start, metadata))

        # ---- 1) Schema Validation：错误参数在入口拦截 ----
        try:
            validated = tool.input_model(**args)
        except ValidationError as exc:
            err = ToolValidationError(f"参数校验失败: {exc.errors()}")
            return ToolResult.fail(name, err, metadata=self._meta(start, metadata))

        # ---- 2) Permission：身份维度 ----
        try:
            self.permission_checker.check(tool, user)
        except ToolPermissionError as exc:
            return ToolResult.fail(name, exc, metadata=self._meta(start, metadata))

        # ---- 3) Policy：规则维度 ----
        decision = self.policy_engine.evaluate(tool, user)
        if decision.decision == "DENY":
            err = ToolPolicyError(f"{decision.reason}（policy={decision.policy_name}）")
            return ToolResult.fail(name, err, metadata=self._meta(start, metadata))
        if decision.decision == "REQUIRE_CONFIRMATION":
            # 教学简化：没有确认通道时按拒绝处理，并保留策略名供审计
            err = ToolPolicyError(f"{decision.reason}（当前无确认通道，按拒绝处理）")
            return ToolResult.fail(name, err, metadata=self._meta(start, metadata))

        # ---- 4) Execute + Timeout + 瞬时错误重试 ----
        validated_args = validated.model_dump()
        attempt = 0
        while True:
            span_kwargs = {
                "input": {"tool_name": name, "arguments": redact(validated_args)},
                "attributes": {"tool_name": name, "attempt": attempt},
                "recorder": self.recorder,
            }
            try:
                if self.recorder is None or not self.recorder.enabled:
                    data = await asyncio.wait_for(
                        tool.invoke(validated_args), timeout=tool.timeout_seconds
                    )
                else:
                    async with trace_span("tool.execute", "tool", **span_kwargs) as tspan:
                        try:
                            data = await asyncio.wait_for(
                                tool.invoke(validated_args), timeout=tool.timeout_seconds
                            )
                            tspan.attributes.update(success=True, duration_ms=tspan.duration_ms)
                        except asyncio.TimeoutError as exc:
                            # 在 Span 层面就转换为 ToolTimeoutError，让 Trace 直接可读
                            tspan.attributes.update(success=False, error_type="ToolTimeoutError")
                            raise ToolTimeoutError(f"工具 {name} 执行超过 {tool.timeout_seconds}s") from exc
                        except ToolExecutionError as exc:
                            tspan.attributes.update(
                                success=False, error_type="ToolExecutionError", transient=exc.transient
                            )
                            raise
                        except Exception as exc:
                            tspan.attributes.update(success=False, error_type=type(exc).__name__)
                            raise
                break
            except ToolTimeoutError as exc:
                return ToolResult.fail(name, exc, metadata=self._meta(start, metadata))
            except asyncio.TimeoutError:
                err = ToolTimeoutError(f"工具 {name} 执行超过 {tool.timeout_seconds}s")
                return ToolResult.fail(name, err, metadata=self._meta(start, metadata))
            except ToolExecutionError as exc:
                if exc.transient and attempt < self.max_retries:
                    attempt += 1
                    continue  # 瞬时错误重试（最多 max_tool_retries 次）
                return ToolResult.fail(name, exc, metadata=self._meta(start, metadata))
            except Exception as exc:  # 未知异常也包装成结构化失败
                err = ToolExecutionError(f"工具内部异常: {exc}")
                return ToolResult.fail(name, err, metadata=self._meta(start, metadata))

        # ---- 5) Result Validation：返回值校验（可选） ----
        if tool.output_model is not None:
            try:
                data = tool.output_model.model_validate(data).model_dump()
            except ValidationError as exc:
                err = ToolExecutionError(f"工具返回值不符合声明模型: {exc.errors()}")
                return ToolResult.fail(name, err, metadata=self._meta(start, metadata))

        return ToolResult.ok(name, data, metadata=self._meta(start, metadata, retries=attempt))

    @staticmethod
    def _meta(start: float, base: dict, retries: int = 0) -> dict:
        m = dict(base)
        m["duration_ms"] = int((time.perf_counter() - start) * 1000)
        m["retries"] = retries
        return m
