"""
Stage 5 Demo：Tool Gateway + Policy。

运行：python -m demos.stage5_demo

展示 Tool Gateway 的统一治理链：
    Schema Validation -> Permission -> Policy -> Timeout -> Execute -> Result Validation

分别演示：正常 / Schema 错误 / 权限拒绝 / 策略拒绝 / 超时 / 内部异常 / 瞬时重试。
每种情况都返回统一的结构化信封（ToolResult）。
"""
import asyncio
import json
import time

from pydantic import BaseModel, Field

from app.errors import ToolExecutionError
from app.tools.builtin import build_default_registry
from app.tools.gateway import ToolGateway
from app.tools.policy import PolicyEngine
from app.tools.registry import ToolDefinition
from app.tools.schemas import UserContext

SEPARATOR = "=" * 64


# ---- 演示用附加工具 ----
class AdminArgs(BaseModel):
    message: str = Field(description="消息")


class DangerArgs(BaseModel):
    target: str = Field(description="目标")


class SlowArgs(BaseModel):
    delay: float = Field(description="延迟秒数")


class EmptyArgs(BaseModel):
    pass


def build_demo_registry():
    reg = build_default_registry()
    reg.register(
        ToolDefinition(
            name="admin_tool",
            description="需要 admin 权限",
            input_model=AdminArgs,
            handler=lambda message: f"admin 执行: {message}",
            required_permission="admin",
        )
    )
    reg.register(
        ToolDefinition(
            name="danger_tool",
            description="高风险工具（删除数据库）",
            input_model=DangerArgs,
            handler=lambda target: f"已删除 {target}",
            risk_level="high",
        )
    )
    reg.register(
        ToolDefinition(
            name="slow_tool",
            description="慢工具",
            input_model=SlowArgs,
            handler=lambda delay: (time.sleep(delay) or "slow done"),
            timeout_seconds=0.2,
        )
    )
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] <= 2:
            raise ToolExecutionError("外部服务瞬时抖动", transient=True)
        return "flaky ok"

    reg.register(ToolDefinition(name="flaky_tool", description="瞬时抖动工具", input_model=EmptyArgs, handler=flaky))
    reg.register(
        ToolDefinition(
            name="explode_tool",
            description="内部异常工具",
            input_model=EmptyArgs,
            handler=lambda: (_ for _ in ()).throw(ValueError("数据库连接失败")),
        )
    )
    return reg


def show(title: str, result) -> None:
    """打印一个演示用例的结构化结果。"""
    print(f"\n--- {title} ---")
    print(json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False, indent=2))


async def main() -> None:
    registry = build_demo_registry()
    # 构造 Gateway：默认 PolicyEngine（黑名单空、high 风险需确认）
    gateway = ToolGateway(registry, policy_engine=PolicyEngine(), settings=None)

    admin = UserContext(user_id="admin", roles=["admin"], permissions=["admin"])
    normal = UserContext(user_id="user1", roles=["user"], permissions=["user"])

    print(SEPARATOR)
    print("Tool Gateway 统一治理链演示")
    print(SEPARATOR)

    # 1) 正常调用
    show("1. 正常 Tool（get_weather 北京）", await gateway.execute("get_weather", {"city": "北京"}, user=normal))
    # 2) Schema 错误
    show("2. Schema 错误（city=123）", await gateway.execute("get_weather", {"city": 123}, user=normal))
    # 3) 权限拒绝
    show("3. Permission Denied（普通用户调 admin_tool）", await gateway.execute("admin_tool", {"message": "删库"}, user=normal))
    # 3b) 权限通过
    show("3b. Permission 通过（admin 调 admin_tool）", await gateway.execute("admin_tool", {"message": "发布"}, user=admin))
    # 4) 策略拒绝
    show("4. Policy Denied（高风险工具需确认，当前无确认通道）", await gateway.execute("danger_tool", {"target": "users"}, user=admin))
    # 5) 超时
    show("5. Timeout（slow_tool 睡 1s，超时 0.2s）", await gateway.execute("slow_tool", {"delay": 1.0}, user=normal))
    # 6) 内部异常
    show("6. 工具内部异常（explode_tool）", await gateway.execute("explode_tool", {}, user=normal))
    # 7) 瞬时错误自动重试
    show("7. 瞬时错误自动重试（flaky_tool，前 2 次失败）", await gateway.execute("flaky_tool", {}, user=normal))

    print("\n" + SEPARATOR)
    print("结论：所有结果都是统一信封（success/tool_name/data/error/metadata），")
    print("校验/权限/策略/超时/错误包装全部由 Gateway 统一负责，工具只写业务逻辑。")


if __name__ == "__main__":
    asyncio.run(main())
