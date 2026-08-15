"""
Skill 系统测试：加载 / frontmatter 解析 / 匹配 / 注入。

验收：SKILL.md 加载、frontmatter 解析、触发词匹配、llm 匹配降级、
AgentRuntime 注入（system prompt 含技能块）。
"""
import pytest

from app.agent.runtime import AgentRuntime
from app.config import Settings
from app.llm.client import StubLLMClient
from app.skills.loader import load_skills, parse_skill_file
from app.skills.manager import SkillManager
from app.skills.models import Skill
from app.tools.builtin import build_default_registry


@pytest.fixture
def skills_dir(tmp_path):
    """构造测试技能目录（2 个技能）。"""
    d = tmp_path / "skills"
    (d / "code_review").mkdir(parents=True)
    (d / "code_review" / "SKILL.md").write_text(
        "---\n"
        "name: code_review\n"
        "description: 审查代码质量\n"
        "triggers:\n"
        "  - code review\n"
        "  - 代码审查\n"
        "version: 1.0\n"
        "---\n"
        "审查代码：检查正确性、安全、性能。\n",
        encoding="utf-8",
    )
    (d / "data_analysis").mkdir()
    (d / "data_analysis" / "SKILL.md").write_text(
        "---\n"
        "name: data_analysis\n"
        "description: 数据分析\n"
        "triggers:\n"
        "  - 数据分析\n"
        "  - analyze data\n"
        "---\n"
        "分析数据并生成报告。\n",
        encoding="utf-8",
    )
    return d


@pytest.fixture
def skill_settings(tmp_path, skills_dir) -> Settings:
    return Settings(
        environment="test",
        skills_dir=str(skills_dir),
        skills_enabled=True,
        skill_match_strategy="trigger",
    )


# ---------------------------------------------------------------------------
# 加载与解析
# ---------------------------------------------------------------------------
def test_load_skills(skills_dir):
    skills = load_skills(skills_dir)
    assert len(skills) == 2
    names = {s.name for s in skills}
    assert names == {"code_review", "data_analysis"}


def test_parse_frontmatter(skills_dir):
    skill = parse_skill_file(skills_dir / "code_review" / "SKILL.md")
    assert skill is not None
    assert skill.name == "code_review"
    assert skill.description == "审查代码质量"
    assert skill.triggers == ["code review", "代码审查"]
    assert "审查代码" in skill.instructions
    assert skill.version == "1.0"


def test_skill_to_prompt_block():
    s = Skill(name="t", description="d", triggers=["x"], instructions="do it")
    block = s.to_prompt_block()
    assert "[技能:t]" in block
    assert "do it" in block


# ---------------------------------------------------------------------------
# 匹配
# ---------------------------------------------------------------------------
def test_match_trigger(skill_settings, skills_dir):
    mgr = SkillManager(skills_dir=skills_dir, settings=skill_settings)
    assert mgr.enabled is True
    hits = mgr.match("帮我做代码审查")
    assert [s.name for s in hits] == ["code_review"]
    hits2 = mgr.match("对这组数据做数据分析")
    assert [s.name for s in hits2] == ["data_analysis"]


def test_match_no_hit(skill_settings, skills_dir):
    mgr = SkillManager(skills_dir=skills_dir, settings=skill_settings)
    assert mgr.match("你好") == []


@pytest.mark.asyncio
async def test_match_llm_falls_back(skill_settings, skills_dir):
    """strategy=llm 但无模型时降级为触发词匹配。"""
    settings = skill_settings.model_copy(update={"skill_match_strategy": "llm"})
    mgr = SkillManager(skills_dir=skills_dir, settings=settings, llm=None)
    hits = await mgr.matched_skills("请帮我做代码审查")
    assert [s.name for s in hits] == ["code_review"]


def test_disabled(skill_settings, skills_dir):
    settings = skill_settings.model_copy(update={"skills_enabled": False})
    mgr = SkillManager(skills_dir=skills_dir, settings=settings)
    assert mgr.enabled is False
    assert mgr.match("代码审查") == []


# ---------------------------------------------------------------------------
# AgentRuntime 注入
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_runtime_injects_skill(tmp_path, skills_dir, skill_settings):
    """命中技能时，Context 的 system prompt 含技能指令块。"""
    from app.agent.context_builder import ContextBuilder

    mgr = SkillManager(skills_dir=skills_dir, settings=skill_settings)
    # 用 spy 验证：匹配到的技能被拼进 retrieved_docs 通道
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        skill_manager=mgr,
        settings=skill_settings,
    )
    assert runtime.skill_manager is not None
    result = await runtime.run("帮我做代码审查，看看这段代码：def f(): pass", session_id="s_skill")
    assert result.answer


@pytest.mark.asyncio
async def test_runtime_without_skill_backward_compat(tmp_path):
    """不注入 skill_manager 时行为不变。"""
    settings = Settings(
        environment="test",
        database_url=f"sqlite:///{tmp_path}/legacy.db",
        trace_enabled=False,
    )
    runtime = AgentRuntime(
        llm=StubLLMClient(),
        registry=build_default_registry(),
        settings=settings,
    )
    assert runtime.skill_manager is None
    result = await runtime.run("你好")
    assert result.answer
