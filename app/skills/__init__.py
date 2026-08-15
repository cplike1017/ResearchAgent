"""Skill 系统包：可加载的专业技能指令集。

用法：
    from app.skills.manager import SkillManager

    manager = SkillManager(settings)
    skills = await manager.matched_skills(user_input)  # 匹配
    # 把 skills 的指令注入 Context Builder 的 system prompt
"""
