"""Skill 数据模型。"""
from pydantic import BaseModel, Field


class Skill(BaseModel):
    """一个可加载的技能定义。

    - name: 技能唯一名（如 code_review）
    - description: 一句话描述（用于 LLM 匹配）
    - triggers: 触发关键词列表（规则匹配）
    - instructions: 技能指令正文（注入 system prompt，指导 Agent 如何执行）
    - version: 技能版本
    """

    name: str
    description: str = ""
    triggers: list[str] = Field(default_factory=list)
    instructions: str = ""
    version: str = "1.0"
    source: str = Field(default="", description="来源文件路径")

    def to_prompt_block(self) -> str:
        """转成注入 system prompt 的文本块。"""
        return (
            f"[技能:{self.name}]\n"
            f"说明：{self.description}\n"
            f"执行要求：\n{self.instructions}"
        )
