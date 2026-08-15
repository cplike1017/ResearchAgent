"""
Skill 加载器：从 skills/ 目录加载 SKILL.md 文件。

SKILL.md 格式（与 DSH skill 同款）：
    ---
    name: code_review
    description: 审查代码质量，发现 bug 与改进点
    triggers: [code review, 代码审查, review code, 审查代码]
    version: 1.0
    ---
    技能指令正文（markdown）...

加载规则：
    - skills/ 下每个子目录一个技能，必须含 SKILL.md；
    - frontmatter（--- 包裹的 YAML）解析元数据；正文为 instructions；
    - 无 frontmatter 时用目录名作为 name，全文作为 instructions。
"""
import re
from pathlib import Path
from typing import Any

from app.skills.models import Skill


def parse_skill_file(path: Path) -> Skill | None:
    """解析单个 SKILL.md。"""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None

    name = path.parent.name
    description = ""
    triggers: list[str] = []
    version = "1.0"
    instructions = raw

    # 解析 frontmatter（--- ... ---）
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", raw, re.S)
    if m:
        fm_text, instructions = m.group(1), m.group(2).strip()
        fm = _parse_frontmatter(fm_text)
        name = fm.get("name", name)
        description = fm.get("description", "")
        triggers = fm.get("triggers", [])
        version = str(fm.get("version", "1.0"))

    if not instructions.strip():
        return None

    return Skill(
        name=str(name),
        description=description,
        triggers=[str(t) for t in triggers],
        instructions=instructions.strip(),
        version=version,
        source=str(path),
    )


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """极简 YAML 解析：只支持键值、列表（- item）。"""
    result: dict[str, Any] = {}
    current_key: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            # 列表项（可能带缩进）
            if current_key:
                result.setdefault(current_key, []).append(stripped[2:].strip().strip('"').strip("'"))
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            current_key = key.strip()
            value = value.strip()
            if value:
                result[current_key] = value.strip('"').strip("'")
            else:
                result[current_key] = []
    return result


def load_skills(skills_dir: Path | str) -> list[Skill]:
    """加载 skills/ 下全部技能。"""
    skills_dir = Path(skills_dir)
    if not skills_dir.exists():
        return []
    skills: list[Skill] = []
    for sub in sorted(skills_dir.iterdir()):
        if sub.is_dir():
            skill_file = sub / "SKILL.md"
            if skill_file.exists():
                skill = parse_skill_file(skill_file)
                if skill:
                    skills.append(skill)
        elif sub.name == "SKILL.md":
            # 平铺的单个 SKILL.md（目录名 = skills 根）
            skill = parse_skill_file(sub)
            if skill:
                skills.append(skill)
    return skills
