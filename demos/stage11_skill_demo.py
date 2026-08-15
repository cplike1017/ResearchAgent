"""
Stage 11 Demo：Skill 系统。

运行：python -m demos.stage11_skill_demo

展示：
    1. 加载 skills/ 目录技能（code_review / data_analysis）
    2. 触发词匹配用户输入
    3. 技能指令注入 Context（system prompt 含技能块）
    4. 真实 LLM 按技能指令执行
"""
import asyncio

from app.agent.runtime import AgentRuntime
from app.config import get_settings
from app.llm.client import create_llm_client
from app.session.repository import SQLiteSessionRepository
from app.skills.manager import SkillManager
from app.tools.builtin import build_default_registry

SEPARATOR = "=" * 64


async def main() -> None:
    settings = get_settings()
    llm = create_llm_client(settings)
    skill_manager = SkillManager(settings=settings, llm=llm)

    print(SEPARATOR)
    print("Stage 11 Skill 系统 Demo")
    print(SEPARATOR)
    print("已加载技能:", [s.name for s in skill_manager.all_skills()])
    print()

    runtime = AgentRuntime(
        llm=llm,
        registry=build_default_registry(),
        session_repo=SQLiteSessionRepository(settings.database_url),
        skill_manager=skill_manager,
        settings=settings,
    )

    # 1) 代码审查（命中 code_review 技能）
    print(SEPARATOR)
    print("1) 代码审查任务（命中 code_review 技能）")
    print(SEPARATOR)
    r1 = await runtime.run(
        "帮我做代码审查，看看这段代码：\n"
        "def get_user(name):\n"
        "    query = 'SELECT * FROM users WHERE name = ' + name\n"
        "    return db.execute(query)",
        session_id="s_skill_demo",
    )
    print(f"回答:\n{r1.answer}")

    # 2) 数据分析（命中 data_analysis 技能）
    print("\n" + SEPARATOR)
    print("2) 数据分析任务（命中 data_analysis 技能）")
    print(SEPARATOR)
    r2 = await runtime.run(
        "对这份数据做数据分析：\n"
        "月份,销售额\n1月,12000\n2月,15000\n3月,11000\n4月,18000",
        session_id="s_skill_demo",
    )
    print(f"回答:\n{r2.answer}")

    print("\nDemo 完成。")


if __name__ == "__main__":
    asyncio.run(main())
