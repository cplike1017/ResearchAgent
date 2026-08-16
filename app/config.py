"""
全局配置（基于 pydantic-settings）。

规则：
1. 所有可配置项都从这里读取环境变量，代码中禁止硬编码 API Key / 地址。
2. 字段名与环境变量名大小写不敏感一一对应（如 database_url <-> DATABASE_URL）。
3. 测试中可以直接构造 Settings(**kwargs) 覆盖，不影响全局。
"""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """agent-runtime 全部配置项。"""

    # ---------- 应用 ----------
    app_name: str = "agent-runtime"
    # 运行环境：dev | test | prod（暂用于日志与默认值区分）
    environment: str = "dev"

    # ---------- LLM ----------
    # provider: auto | openai | stub
    #   auto   = 有 base_url + api_key 时用 openai，否则用 stub
    #   openai = 强制真实 OpenAI-compatible 接口
    #   stub   = 内置确定性假模型（离线教学 / 测试 / 无 Key 演示）
    llm_provider: str = "auto"
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_timeout_seconds: float = 60.0
    # LLM 调用瞬时错误重试次数（网络断连 / 5xx 重试；4xx 不重试）
    llm_max_retries: int = 2
    # LLM 重试间隔（秒，指数退避基数）
    llm_retry_backoff: float = 1.0

    # ---------- Embedding / Memory（Stage 8 记忆层）----------
    # provider: auto | openai | stub
    #   openai = OpenAI-compatible /embeddings 接口（如阿里云百炼 qwen3.7-text-embedding）
    #   stub   = 确定性伪向量（离线教学 / 测试，不调外部 API）
    embedding_provider: str = "auto"
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_timeout_seconds: float = 30.0
    embedding_dim: int = 1024
    # 记忆层开关：False 时 AgentRuntime 完全不接记忆
    memory_enabled: bool = False
    # 检索时的 Top-K 召回条数
    memory_top_k: int = 5
    # 检索最小相似度阈值（低于该值的记忆不注入上下文，防噪音）
    memory_min_score: float = 0.35
    # 回合结束后是否自动提炼事实写入记忆
    memory_auto_extract: bool = True
    # 事实提炼策略: stub(确定性) | llm(模型提炼) | off(关闭)
    memory_extract_strategy: str = "stub"
    # 重排序策略: stub(规则重排) | llm(模型重排) | off(跳过，等价朴素 Top-K)
    memory_rerank_strategy: str = "stub"
    # 粗召回倍数：先召回 top_k * 该倍数 的候选，再重排取最终 top_k
    memory_rerank_candidate_multiplier: int = 3
    # 规则重排时关键词重叠的权重（0~1；语义分数权重为 1 - 该值）
    memory_rerank_keyword_weight: float = 0.3

    # ---------- Redis 队列 ----------
    redis_url: str = "redis://localhost:6379/0"
    queue_name: str = "agent:jobs:queue"
    job_key_prefix: str = "agent:jobs:"
    request_key_prefix: str = "agent:requests:"

    # ---------- SQLite ----------
    database_url: str = "sqlite:///./data/agent.db"

    # ---------- Agent 运行时 ----------
    # Context Builder 滑动窗口条数
    max_context_messages: int = 10
    # 历史超过该条数触发压缩
    context_summary_threshold: int = 16
    # 压缩策略: stub(确定性摘要) | llm(模型总结) | off(关闭)
    context_summary_strategy: str = "stub"
    # 单轮最大循环步数
    max_agent_steps: int = 8
    # 工具调用最大重试次数（Gateway 层）
    max_tool_retries: int = 2
    # 工具默认超时（秒）
    tool_default_timeout_seconds: float = 10.0

    # ---------- Stage 9 规划层 ----------
    # 默认执行模式: react(单层 ReAct，兼容旧行为) | plan(计划→执行→反思)
    agent_mode: str = "react"
    # 规划策略: stub(规则分解) | llm(模型计划) | off(不规划，等同 react)
    planner_strategy: str = "stub"
    # 反思触发的最大重规划次数（防死循环）
    max_plan_revisions: int = 2
    # 计划的步骤数上限（防止拆出几百个步骤）
    max_plan_steps: int = 6

    # ---------- Job ----------
    max_attempts: int = 3
    job_ttl_seconds: int = 86400

    # ---------- 真实工具（Stage 10 工具生态）----------
    # Tavily Web Search API Key（RAG 网络检索 / web_search 工具共用）
    tavily_api_key: str = ""
    tavily_timeout_seconds: float = 15.0
    # HTTP 抓取工具默认超时
    http_tool_timeout_seconds: float = 10.0
    # 天气 API（Open-Meteo，免费无需 key；配置后优先于 Stub 数据）
    # 城市 -> 经纬度映射内置（_STUB 城市），真实数据走 Open-Meteo 免费接口
    weather_use_real_api: bool = True
    weather_api_timeout_seconds: float = 10.0
    # QWeather（和风天气）：配置后优先于 Open-Meteo
    qweather_host: str = ""
    qweather_api_key: str = ""
    qweather_timeout_seconds: float = 10.0
    # 文件工具沙箱根目录（file_read/file_write 只允许在此目录内操作）
    sandbox_dir: str = "./data/sandbox"

    # ---------- Policy（工具策略）----------
    # 需要人工确认的风险等级（默认 high；当前无确认通道，命中即拒绝）
    # 用户决定：暂设空（全部放行），后续引入 LLM 危险度评估后恢复
    policy_require_confirmation_risks: str = ""  # 逗号分隔，如 "high,medium"
    # 工具默认风险等级（MCP 外部工具注册时的默认值）
    tool_default_risk_level: str = "low"
    # 邮件工具（QQ SMTP）：SMTP_USER 为发件人邮箱，SMTP_PASSWORD 为授权码
    smtp_host: str = "smtp.qq.com"
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""

    # ---------- MCP（Model Context Protocol 接入）----------
    # MCP_SERVERS: JSON 数组，每个元素描述一个 MCP Server：
    #   [{"name": "filesystem", "transport": "stdio", "command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
    #    {"name": "remote", "transport": "sse", "url": "https://example.com/mcp"}]
    # transport: stdio(本地子进程) | sse(远程 HTTP)
    mcp_servers: str = "[]"
    # 连接 MCP Server 的超时（秒）：首次 npx 拉包较慢，给足时间
    mcp_connect_timeout_seconds: float = 60.0
    # 工具调用转发到 MCP Server 的超时（秒）
    mcp_tool_timeout_seconds: float = 30.0
    # 启动时是否自动连接并注册 MCP 工具（false 时手动调用 mcp bridge 注册）
    mcp_auto_register: bool = True
    # GitHub Token（github MCP server 子进程注入用）
    github_token: str = ""

    # ---------- Stage 12 多 Agent 编排 ----------
    # 编排总开关（False 时 runtime 不注册 delegate 工具）
    orchestrator_enabled: bool = True
    # 编排规划策略: llm(模型分工) | stub(单步 generalist 兜底)
    orchestrator_planner_strategy: str = "llm"
    # 编排计划步骤数上限（防模型拆出几十步）
    orchestrator_max_steps: int = 5
    # 并行执行子 agent 的数量上限
    orchestrator_max_parallel: int = 3

    # ---------- Skill 系统 ----------
    # 技能目录（skills/ 下每个子目录一个技能，含 SKILL.md）
    skills_dir: str = "./skills"
    # 技能总开关
    skills_enabled: bool = True
    # 匹配策略: trigger(触发词规则) | llm(模型语义匹配)
    skill_match_strategy: str = "trigger"

    # ---------- Tracing ----------
    trace_enabled: bool = True
    trace_capture_content: bool = False
    trace_file: str = "traces/traces.jsonl"

    # ---------- Eval ----------
    eval_run_dir: str = "evals/runs"
    # 版本号（可在发版时修改，用于 Eval Run 元数据）
    agent_version: str = "0.6.0"
    prompt_version: str = "v1"
    tool_schema_version: str = "v1"
    dataset_version: str = "v1"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def llm_provider_resolved(self) -> str:
        """解析 auto：根据是否配置了真实接口自动选择 provider。"""
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.llm_base_url and self.llm_api_key:
            return "openai"
        return "stub"

    @property
    def embedding_provider_resolved(self) -> str:
        """解析 embedding 的 auto：有 base_url + api_key 时用 openai，否则 stub。"""
        if self.embedding_provider != "auto":
            return self.embedding_provider
        if self.embedding_base_url and self.embedding_api_key:
            return "openai"
        return "stub"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局单例配置（lru_cache 保证进程内只解析一次 .env）。"""
    return Settings()
