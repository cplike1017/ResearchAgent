"""
reagent 应用包。

从零实现的教学型 ReAgent，分六阶段演进：
    Stage 1  ReAct / Tool Loop      —— Agent 怎么执行
    Stage 2  Context Builder        —— 模型实际看到什么
    Stage 3  Session / Checkpoint   —— 执行状态如何保存恢复
    Stage 4  Redis Queue / Worker   —— 并发请求怎么处理
    Stage 5  Tool Gateway / Policy  —— Tool 怎么统一治理
    Stage 6  Tracing / Evaluation   —— 为什么失败、修改是否变好

核心原则：不使用任何 Agent Framework（LangGraph / CrewAI / AutoGen 等），
所有底层机制（循环、上下文构建、会话、队列、网关、链路追踪）自己实现。
"""

__version__ = "0.6.0"
