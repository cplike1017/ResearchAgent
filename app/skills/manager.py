"""
SkillManager：技能匹配与注入。

职责：
    1. 启动时加载 skills/ 目录的全部技能（loader.load_skills）；
    2. 根据用户输入匹配技能（触发词规则匹配 + 可选 LLM 语义匹配）；
    3. 把匹配到的技能指令返回，供 Context Builder 注入 system prompt。

匹配策略：
    - 触发词匹配（默认，确定性）：用户输入命中 skill.triggers 任一关键词即匹配；
    - llm 匹配（可选）：调用 LLM 判断输入最匹配哪个技能（真实语义理解）。

注入：
    - matched_skills() 返回匹配到的 Skill 列表；
    - AgentRuntime 在构建上下文时调用，指令进入 system prompt。
"""
from pathlib import Path

from app.config import Settings
from app.llm.client import BaseLLMClient
from app.skills.loader import load_skills
from app.skills.models import Skill


class SkillManager:
    """技能管理器。"""

    def __init__(
        self,
        skills_dir: Path | str | None = None,
        settings: Settings | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.llm = llm  # 供 llm 匹配策略
        self.skills_dir = Path(skills_dir or self.settings.skills_dir)
        self.skills: list[Skill] = load_skills(self.skills_dir)
        self.strategy = self.settings.skill_match_strategy  # trigger | llm

    @property
    def enabled(self) -> bool:
        return self.settings.skills_enabled and bool(self.skills)

    # ------------------------------------------------------------------
    def match(self, user_input: str) -> list[Skill]:
        """匹配用户输入对应的技能（触发词规则）。"""
        if not self.enabled or not user_input:
            return []
        text = user_input.lower()
        hits: list[tuple[int, Skill]] = []
        for skill in self.skills:
            for trig in skill.triggers:
                if trig.lower() in text:
                    hits.append((len(trig), skill))
                    break
        hits.sort(key=lambda x: x[0], reverse=True)  # 最长触发词优先
        return [s for _, s in hits]

    async def match_llm(self, user_input: str) -> list[Skill]:
        """LLM 语义匹配（strategy=llm 时使用）。"""
        if not self.enabled or not user_input:
            return []
        if self.strategy != "llm" or self.llm is None:
            return self.match(user_input)
        if not self.skills:
            return []

        listing = "\n".join(f"- {s.name}: {s.description}" for s in self.skills)
        prompt = (
            f"用户输入：{user_input}\n\n可用技能：\n{listing}\n\n"
            "判断该输入最匹配哪个技能（可多选，最多 2 个）。"
            "只输出技能名，每行一个；不匹配任何技能则输出 NONE。"
        )
        try:
            response = await self.llm.chat([{"role": "user", "content": prompt}], tools=None)
        except Exception:
            return self.match(user_input)
        names = {line.strip() for line in (response.content or "").splitlines() if line.strip()}
        if "NONE" in names:
            names.discard("NONE")
        by_name = {s.name: s for s in self.skills}
        return [by_name[n] for n in names if n in by_name]

    async def matched_skills(self, user_input: str) -> list[Skill]:
        """按配置策略匹配技能。"""
        if self.strategy == "llm" and self.llm is not None:
            return await self.match_llm(user_input)
        return self.match(user_input)

    def all_skills(self) -> list[Skill]:
        return self.skills

    def reload(self) -> None:
        """重新加载技能（热更新）。"""
        self.skills = load_skills(self.skills_dir)
