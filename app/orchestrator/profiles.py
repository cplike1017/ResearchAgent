"""子 Agent 档案（AgentProfile）：多 Agent 编排的"角色卡"。

每个 profile 定义：
    - name            档案名（planner 用它指派任务）
    - description     给 planner 看的职责说明（LLM 决定"这一步该谁干"）
    - system_prompt   子 agent 的系统提示（人设 + 工作规范）
    - allowed_tools   工具白名单（None = 全部工具）；子 agent 只能看到白名单内的工具
    - max_steps       子 agent ReAct 循环步数上限

设计要点：子 agent 的"专业"来自两个隔离维度——
    1. 人设隔离：不同的 system_prompt（研究员 vs 分析师 vs 写手）
    2. 能力隔离：不同的工具白名单（研究员摸不到 run_code，分析师摸不到 web_search）
这比只换 prompt 更接近真实团队分工：工具权限本身就是边界。
"""
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 内置档案的工具白名单（与 app/tools/builtin 中的工具名对应）
# MCP 工具按前缀通配放行（fs_* 文件系统 / fetch_* 网页抓取 / github_* 代码仓库）
# ---------------------------------------------------------------------------
_RESEARCHER_TOOLS = [
    "web_search", "http_get", "http_get_json", "extract_web",
    "read_pdf", "read_excel", "get_time", "get_date",
    "arxiv_search", "list_files", "search_files",
    "fetch_*",  # MCP fetch server：网页抓取
    "github_*",  # MCP github server：查论文仓库/代码
]
_ANALYST_TOOLS = [
    "calculator", "run_code", "analyze_data", "sqlite_query",
    "http_get_json", "file_read", "file_write", "get_time", "get_date",
    "arxiv_search", "list_files", "search_files", "read_pdf", "read_excel",
    "fs_*",  # MCP filesystem server：读写沙箱文件
    "fetch_*",  # MCP fetch server：抓取数据源
]
_WRITER_TOOLS = [
    "file_write", "send_email", "get_time", "get_date", "calculator",
    "file_read", "list_files",
    "fs_*",  # MCP filesystem server：读写沙箱文件
]

# 工具使用纪律（统一注入所有子 agent 人设）：防止盲目重复调用
_TOOL_DISCIPLINE = (
    "工具使用纪律：\n"
    "1. 每个工具描述里都写明了「何时用 / 何时不用 / 失败怎么办」，先读描述再决定；\n"
    "2. 工具失败后：先分析原因（参数/网络/权限），换替代工具或调整参数，"
    "同一调用不要原样重试超过 1 次；\n"
    "3. 同一步内不要并行调用同一个工具多次（如 4 个 web_search 查相近关键词）——"
    "先用一次拿结果，根据结果决定下一步；\n"
    "4. 检索时先用 web_search 或 arxiv_search 找到来源，再决定是否深读（extract_web）；"
    "不要对所有搜索结果都逐个抓取全文；\n"
    "5. 已有足够信息时立即收尾给出结论，不要为了「用工具」而继续调用。"
)

_RESEARCHER_PROMPT = (
    "你是一名资深研究员（Research Agent），专注于信息检索与资料收集。\n"
    "工作规范：\n"
    "1. 使用 web_search / arxiv_search / http_get / extract_web 检索并阅读资料；\n"
    "2. 论文 / Excel 等本地文件用 read_pdf / read_excel 读取；\n"
    "3. 必须给出结论时附带来源（标题/URL/页码）；\n"
    "4. 输出结构化摘要：核心事实、关键数字、来源列表。\n"
    "只做研究，不做分析计算，也不撰写最终报告。\n"
    f"{_TOOL_DISCIPLINE}"
)

_ANALYST_PROMPT = (
    "你是一名数据分析师（Data Analyst），专注于数据计算与统计分析。\n"
    "工作规范：\n"
    "1. 优先用 run_code 写 Python 代码处理数据（沙箱内可用 math/statistics/json/csv）；\n"
    "2. 数据文件用 file_read / analyze_data / sqlite_query 读取；\n"
    "3. 输出数值结论：指标、对比、趋势、异常，保留有效位数；\n"
    "4. 无法得到确定数字时明确说明，不编造数据。\n"
    "只做分析，不做资料检索，也不撰写最终报告。\n"
    f"{_TOOL_DISCIPLINE}"
)

_WRITER_PROMPT = (
    "你是一名报告写手（Report Writer），负责把资料与结论组织成最终交付物。\n"
    "工作规范：\n"
    "1. 基于给定的资料撰写条理清晰的报告（Markdown 格式）；\n"
    "2. 结构建议：概述 / 分论点 / 数据支撑 / 结论与建议；\n"
    "3. 需要落盘时用 file_write 保存，需要发送时用 send_email；\n"
    "4. 不编造资料中没有的事实。\n"
    "你只负责整合与表达，不负责检索或计算。\n"
    f"{_TOOL_DISCIPLINE}"
)

_GENERALIST_PROMPT = (
    "你是一名全能助手（Generalist Agent），负责处理未细分到专业档案的任务。\n"
    "可以使用全部工具完成任务，注意选择最合适的工具组合。\n"
    f"{_TOOL_DISCIPLINE}"
)


@dataclass
class AgentProfile:
    """子 Agent 档案。"""

    name: str
    description: str
    system_prompt: str
    allowed_tools: list[str] | None = field(default=None)  # None = 全部工具
    max_steps: int = 6


# ---------------------------------------------------------------------------
# 内置档案注册表
# ---------------------------------------------------------------------------
BUILTIN_PROFILES: list[AgentProfile] = [
    AgentProfile(
        name="researcher",
        description="信息检索与资料收集：联网搜索、抓取网页、读取 PDF/Excel 等资料文件，输出带来源的摘要",
        system_prompt=_RESEARCHER_PROMPT,
        allowed_tools=_RESEARCHER_TOOLS,
        max_steps=10,
    ),
    AgentProfile(
        name="analyst",
        description="数据分析与计算：用 Python 代码/统计工具分析数据文件，输出数值结论与趋势",
        system_prompt=_ANALYST_PROMPT,
        allowed_tools=_ANALYST_TOOLS,
        max_steps=10,
    ),
    AgentProfile(
        name="writer",
        description="报告撰写与交付：把资料整理为 Markdown 报告，可保存文件或发送邮件",
        system_prompt=_WRITER_PROMPT,
        allowed_tools=_WRITER_TOOLS,
        max_steps=4,
    ),
    AgentProfile(
        name="generalist",
        description="全能助手：可处理任意任务（未细分到专业档案时的兜底）",
        system_prompt=_GENERALIST_PROMPT,
        allowed_tools=None,  # 全部工具
        max_steps=8,
    ),
]


def get_profile(name: str) -> AgentProfile:
    """按名字取档案；未知名字回退到 generalist（防御 planner 乱输出）。"""
    for p in BUILTIN_PROFILES:
        if p.name == name:
            return p
    return get_profile("generalist")


def profile_names() -> list[str]:
    return [p.name for p in BUILTIN_PROFILES]
