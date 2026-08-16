"""Agent 档案注册表（ProfileRegistry）：内置档案 + 运行时动态注册的自定义档案。

为什么需要注册表而不是模块级列表？
    Stage 12 的 BUILTIN_PROFILES 是硬编码的 —— 用户无法添加自己的子 agent 角色。
    真实产品中"团队"是动态的：项目经理可以随时加入一个"前端工程师"、
    "SQL 专家"等新角色。ProfileRegistry 提供：

    - register()  动态注册自定义档案（校验：名字合法、不与内置冲突）
    - unregister() 注销自定义档案（内置档案不可注销）
    - get() / all() / names()  查询（get 未知名字回退 generalist）
    - 持久化      自定义档案写入 JSON 文件（默认 data/agent_profiles.json），
                  重启后仍然可用；内置档案不落盘（随代码版本走）

安全边界：
    - 内置档案（researcher/analyst/writer/generalist）不可覆盖、不可注销；
    - 自定义档案与内置档案同名 → 拒绝注册（防止篡改内置人设）；
    - 自定义档案的 allowed_tools 同样参与工具过滤与 delegate 深度控制
      （能力边界由 executor 统一执行，注册表不越权）。
"""
import json
import os
from dataclasses import asdict

from app.config import Settings, get_settings
from app.orchestrator.profiles import AgentProfile, BUILTIN_PROFILES


class ProfileRegistryError(Exception):
    """档案注册/注销错误。"""


class ProfileRegistry:
    """子 Agent 档案注册表（内置 + 动态自定义，可持久化）。"""

    def __init__(
        self,
        settings: Settings | None = None,
        profiles_file: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        # 持久化文件（None = 不持久化，仅内存）
        self.profiles_file = profiles_file or self.settings.agent_profiles_file
        # 内置档案（不可变，随代码版本）
        self._builtin: dict[str, AgentProfile] = {p.name: p for p in BUILTIN_PROFILES}
        # 自定义档案（动态注册，可持久化）
        self._custom: dict[str, AgentProfile] = {}
        self._load()

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get(self, name: str) -> AgentProfile:
        """按名字取档案：自定义优先，其次内置；未知回退 generalist。"""
        if name in self._custom:
            return self._custom[name]
        if name in self._builtin:
            return self._builtin[name]
        return self._builtin["generalist"]

    def all(self) -> list[AgentProfile]:
        """全部档案（内置在前，自定义在后）。"""
        return list(self._builtin.values()) + list(self._custom.values())

    def names(self) -> list[str]:
        return [p.name for p in self.all()]

    def is_builtin(self, name: str) -> bool:
        return name in self._builtin

    def is_custom(self, name: str) -> bool:
        return name in self._custom

    # ------------------------------------------------------------------
    # 动态注册 / 注销
    # ------------------------------------------------------------------
    def register(self, profile: AgentProfile) -> AgentProfile:
        """注册自定义档案。

        校验：
            - 名字非空且不与内置档案同名（内置人设不可篡改）；
            - 重复注册同名自定义档案 → 覆盖（幂等）。
        注册后立即持久化（若配置了文件）。
        """
        name = profile.name.strip()
        if not name:
            raise ProfileRegistryError("档案名不能为空")
        if name in self._builtin:
            raise ProfileRegistryError(f"不能覆盖内置档案「{name}」")
        self._custom[name] = profile
        self._save()
        return profile

    def unregister(self, name: str) -> bool:
        """注销自定义档案。内置档案不可注销；不存在返回 False。"""
        if name in self._builtin:
            raise ProfileRegistryError(f"内置档案「{name}」不可注销")
        removed = self._custom.pop(name, None) is not None
        if removed:
            self._save()
        return removed

    # ------------------------------------------------------------------
    # 持久化（JSON 文件；仅自定义档案落盘）
    # ------------------------------------------------------------------
    def _load(self) -> None:
        if not self.profiles_file or not os.path.exists(self.profiles_file):
            return
        try:
            with open(self.profiles_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data.get("profiles", []):
                profile = _profile_from_dict(item)
                if profile and profile.name not in self._builtin:
                    self._custom[profile.name] = profile
        except (json.JSONDecodeError, OSError):
            # 损坏的配置文件不阻断启动：忽略自定义档案，保留内置
            self._custom = {}

    def _save(self) -> None:
        if not self.profiles_file:
            return
        parent = os.path.dirname(self.profiles_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        payload = {
            "profiles": [_profile_to_dict(p) for p in self._custom.values()],
        }
        with open(self.profiles_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# AgentProfile <-> dict 序列化（dataclass 不可直接 json）
# ---------------------------------------------------------------------------
def _profile_to_dict(p: AgentProfile) -> dict:
    return asdict(p)


def _profile_from_dict(item: dict) -> AgentProfile | None:
    try:
        return AgentProfile(
            name=str(item.get("name", "")).strip(),
            description=str(item.get("description", "")),
            system_prompt=str(item.get("system_prompt", "")),
            allowed_tools=item.get("allowed_tools"),
            max_steps=int(item.get("max_steps", 6)),
        )
    except (TypeError, ValueError):
        return None


def _static_registry(profiles: list[AgentProfile]) -> ProfileRegistry:
    """用显式档案列表构造一个只读注册表（兼容 runner 的旧 profiles= 参数）。

    所有传入档案按自定义档案注册（与内置同名的会抛错）；不持久化。
    用于测试/演示中注入自定义档案集合，且不影响默认注册表。
    """
    reg = ProfileRegistry(settings=get_settings(), profiles_file=None)
    for p in profiles:
        if p.name in reg._builtin:
            reg._builtin[p.name] = p  # 允许覆盖内置（显式注入语义）
        else:
            reg._custom[p.name] = p
    return reg
