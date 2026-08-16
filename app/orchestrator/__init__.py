"""多 Agent 编排包（Stage 12）：Manager/Worker 编排模式。

组件：
    profiles    —— 子 Agent 档案定义（researcher/analyst/writer/generalist）
    registry    —— 档案注册表（内置 + 动态注册，JSON 持久化）
    planner     —— 编排规划器（LLM 分工，坏输出降级单步）
    executor    —— 子 Agent 执行器（独立 ReAct + 过滤工具集）
    runner      —— 编排运行器（依赖图 + 并行 + 合成 + 结果持久化）
    repository  —— 编排结果仓库（SQLite，按会话查询/回放）
    tool        —— delegate 工具（主 agent 的委派入口）
"""
from app.orchestrator.models import AgentRunResult, OrchestrationPlan, OrchestrationResult
from app.orchestrator.planner import OrchestratorPlanner
from app.orchestrator.profiles import AgentProfile, BUILTIN_PROFILES, get_profile
from app.orchestrator.registry import ProfileRegistry, ProfileRegistryError
from app.orchestrator.repository import OrchestrationRecord, SQLiteOrchestrationRepository
from app.orchestrator.runner import OrchestratorRunner
from app.orchestrator.tool import build_delegate_tool

__all__ = [
    "AgentProfile",
    "AgentRunResult",
    "BUILTIN_PROFILES",
    "OrchestrationPlan",
    "OrchestrationRecord",
    "OrchestrationResult",
    "OrchestratorPlanner",
    "OrchestratorRunner",
    "ProfileRegistry",
    "ProfileRegistryError",
    "SQLiteOrchestrationRepository",
    "build_delegate_tool",
    "get_profile",
]
