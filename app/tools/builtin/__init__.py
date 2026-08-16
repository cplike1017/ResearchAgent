"""内置工具包：calculator / get_weather + web / data / code 类真实工具 + send_email。

描述设计规范（工具描述引导）：
    每个工具 description 遵循「何时用 / 何时不用 / 失败怎么办」三层结构，
    明确工具间的决策边界，减少模型盲目重复调用：
    1. 何时用 —— 适用场景与输入特征；
    2. 何时不用 —— 明确指出更合适的替代工具（避免反复尝试错误工具）；
    3. 失败怎么办 —— 工具失败时的处理建议（换工具/换参数，不无限重试）。
"""
from app.tools.builtin.analyze import AnalyzeDataArgs, analyze_data_handler
from app.tools.builtin.calculator import calculator_handler, CalculatorArgs
from app.tools.builtin.code_exec import RunCodeArgs, run_code_handler
from app.tools.builtin.data import (
    AppendNoteArgs,
    FileReadArgs,
    FileWriteArgs,
    ListFilesArgs,
    SearchFilesArgs,
    SqliteQueryArgs,
    append_note_handler,
    file_read_handler,
    file_write_handler,
    list_files_handler,
    search_files_handler,
    sqlite_query_handler,
)
from app.tools.builtin.documents import (
    ExtractWebArgs,
    ReadExcelArgs,
    ReadPdfArgs,
    extract_web_handler,
    read_excel_handler,
    read_pdf_handler,
)
from app.tools.builtin.mail import SendEmailArgs, send_email_handler
from app.tools.builtin.research import ArxivSearchArgs, arxiv_search_handler
from app.tools.builtin.weather import weather_handler, WeatherArgs
from app.tools.builtin.web import (
    DateArgs,
    HttpGetArgs,
    HttpGetJsonArgs,
    TimeArgs,
    WebSearchArgs,
    get_date_handler,
    get_time_handler,
    http_get_handler,
    http_get_json_handler,
    web_search_handler,
)
from app.tools.registry import ToolDefinition, ToolRegistry

# 描述中的统一引导后缀（提示模型工具失败时的正确行为，避免盲目重复）
_GUIDANCE = (
    "注意：若工具调用失败或返回异常，先分析原因再决定下一步——"
    "不要原样重试同一参数超过 1 次；换用替代工具或调整参数。"
)

_WEB_SEARCH_DESC = (
    "联网搜索，返回相关网页标题/链接/摘要。需要 TAVILY_API_KEY。"
    "何时用：需要获取最新信息、查找资料/论文/新闻时首选；"
    "何时不用：已有明确 URL 时直接用 http_get/extract_web 抓取，"
    "需要检索学术论文用 arxiv_search；"
    "失败怎么办：关键词太泛会返回无关结果，换更精确的关键词（加引号/年份/会议名）。"
)

_HTTP_GET_DESC = (
    "抓取指定网址的网页文本内容（限 8KB，原始 HTML 或纯文本）。"
    "何时用：已有明确 URL、需要网页原始内容或轻量抓取时；"
    "何时不用：需要正文提炼（去导航）用 extract_web；"
    "需要 JSON 数据用 http_get_json；需要全文阅读论文页用 arxiv_search 获取摘要。"
    "失败怎么办：URL 失效/超时是常态——换 arxiv 或 Google 缓存链接，"
    "或改用 extract_web 重试；不要对同一 URL 反复重试超过 2 次。"
)

_HTTP_GET_JSON_DESC = (
    "抓取返回 JSON 的 REST API 并结构化输出（限 32KB）。"
    "何时用：对接公开 API（如 arXiv export API、GitHub API）获取数据；"
    "何时不用：网页正文用 http_get/extract_web；"
    "失败怎么办：多数 API 有速率限制（429）或需要鉴权——检查参数与 URL 编码，"
    "换用 arxiv_search 等专用工具代替手拼 API；同一端点失败 2 次即放弃。"
)

_EXTRACT_WEB_DESC = (
    "抓取网页并提取正文为 markdown 风格文本（去导航/脚本，限 16KB）。"
    "何时用：深入阅读某网页全文、提取文章正文时首选；"
    "何时不用：只想要标题/摘要用 web_search 结果即可；"
    "失败怎么办：部分站点反爬（403）或超时——换搜索引擎缓存或换来源网站，"
    "不要对同一 URL 连续重试超过 2 次。"
)

_ARXIV_DESC = (
    "检索 arXiv 学术论文（免费 API，无需 Key），返回标题/作者/年份/摘要/链接。"
    "何时用：调研论文、验证文献是否存在、查找某主题的学术工作；"
    "何时不用：非学术内容用 web_search；"
    "失败怎么办：网络不可达时改用 web_search 检索论文标题；"
    "查询结果为空时去掉生僻词、用更通用的术语重试。"
)

_READ_PDF_DESC = (
    "提取沙箱内 PDF 文件的文本内容（指定页或全部，限 8KB）。"
    "何时用：读取本地论文/文档 PDF；"
    "何时不用：Excel/CSV 用 read_excel，网页用 extract_web；"
    "失败怎么办：扫描版 PDF 无文本层会返回空——改用 web_search 找文字版或摘要。"
)

_READ_EXCEL_DESC = (
    "读取沙箱内 Excel（.xlsx/.xlsm）或 CSV 为 markdown 表格（限 200 行）。"
    "何时用：查看本地表格数据；"
    "何时不用：需要统计分析用 analyze_data，需要自定义计算用 run_code；"
    "失败怎么办：先 list_files 确认文件存在与扩展名，再读取。"
)

_ANALYZE_DATA_DESC = (
    "分析沙箱内 CSV/JSON：均值/中位数/分位数/标准差、按列分组、趋势检测，返回结构化摘要。"
    "何时用：快速了解数据分布、列统计、趋势时首选；"
    "何时不用：复杂计算/自定义逻辑用 run_code；"
    "失败怎么办：列名/路径错误时先 file_read 或 list_files 确认文件结构。"
)

_RUN_CODE_DESC = (
    "在受限沙箱中执行 Python 代码（数据分析/计算/文件处理）。"
    "白名单模块：math/statistics/json/csv/re/collections。超时 10s，禁止 os/subprocess/网络。"
    "何时用：analyze_data 覆盖不了的复杂计算、多步数据处理、批量文件操作；"
    "何时不用：简单统计用 analyze_data，简单计算用 calculator；"
    "失败怎么办：代码报错时读取错误信息修正代码重试（最多 2 次），"
    "不要盲目重跑相同代码；需要网络/系统能力时换用其他工具。"
)

_CALCULATOR_DESC = (
    "计算数学表达式，支持 + - * / // % ** 和括号。例如：123 * 456。"
    "何时用：单次简单算术；何时不用：多步或带逻辑的计算用 run_code。"
)

_FILE_READ_DESC = (
    "读取沙箱内文本文件（限 8KB，相对路径）。"
    "何时用：查看本地资料/代码/报告；"
    "何时不用：不确定文件是否存在先 list_files，查找内容用 search_files，"
    "PDF/Excel 用 read_pdf/read_excel；"
    "失败怎么办：文件不存在时 list_files 确认实际路径。"
)

_FILE_WRITE_DESC = (
    "写入沙箱内文本文件（限 16KB，自动建目录，覆盖写）。"
    "何时用：产出最终交付物（报告/代码）；"
    "何时不用：积累草稿用 append_note（追加不覆盖）；"
    "失败怎么办：内容超限时拆分或压缩。"
)

_APPEND_NOTE_DESC = (
    "向沙箱文件追加内容（不覆盖已有内容，自动换行分隔）。"
    "何时用：跨子 Agent 协作积累草稿/阶段成果、记录发现；"
    "何时不用：交付最终成品用 file_write。"
)

_LIST_FILES_DESC = (
    "列出沙箱目录内文件与子目录（含大小）。"
    "何时用：任务开始时先探查沙箱有什么、或确认文件是否存在；"
    "何时不用：需要文件内容用 file_read/read_pdf/read_excel。"
)

_SEARCH_FILES_DESC = (
    "在沙箱目录内递归搜索含关键词的文件内容（返回匹配行）。"
    "何时用：在已有资料/报告中查找特定主题、复用之前成果；"
    "何时不用：找文件本身用 list_files。"
)

_SQLITE_DESC = (
    "对指定 SQLite 数据库文件执行只读 SELECT 查询（禁止写操作）。"
    "何时用：数据在 .db/.sqlite 文件中时；"
    "何时不用：CSV/Excel 用 analyze_data/read_excel；"
    "失败怎么办：先确认文件路径与表名（可用 sqlite_query 查 sqlite_master）。"
)

_WEATHER_DESC = (
    "查询指定中国城市当前天气（北京/上海/广州/深圳/杭州/成都/武汉/西安/南京/重庆）。"
    "何时用：用户询问天气时；失败怎么办：城市不支持时告知支持列表。"
)

_SEND_EMAIL_DESC = (
    "发送文本邮件到指定邮箱（需配置 SMTP）。"
    "何时用：用户明确要求发邮件时；何时不用：未要求发送时只生成内容；"
    "失败怎么办：SMTP 配置错误时告知用户检查 SMTP_USER/SMTP_PASSWORD。"
)

_TIME_DESC = "获取当前 UTC 时间（YYYY-MM-DD HH:MM:SS）。何时用：需要时间戳/截止时间判断时。"
_DATE_DESC = "获取当前 UTC 日期（YYYY-MM-DD）。何时用：需要当前日期判断时。"


def build_default_registry() -> ToolRegistry:
    """构建包含全部内置工具的默认注册表（描述含决策引导）。"""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calculator",
            description=_CALCULATOR_DESC,
            input_model=CalculatorArgs,
            handler=calculator_handler,
            timeout_seconds=3.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="get_weather",
            description=_WEATHER_DESC,
            input_model=WeatherArgs,
            handler=weather_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="web_search",
            description=_WEB_SEARCH_DESC,
            input_model=WebSearchArgs,
            handler=web_search_handler,
            timeout_seconds=20.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="http_get",
            description=_HTTP_GET_DESC,
            input_model=HttpGetArgs,
            handler=http_get_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="get_time",
            description=_TIME_DESC,
            input_model=TimeArgs,
            handler=get_time_handler,
            timeout_seconds=3.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="get_date",
            description=_DATE_DESC,
            input_model=DateArgs,
            handler=get_date_handler,
            timeout_seconds=3.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="sqlite_query",
            description=_SQLITE_DESC,
            input_model=SqliteQueryArgs,
            handler=sqlite_query_handler,
            timeout_seconds=10.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="file_read",
            description=_FILE_READ_DESC,
            input_model=FileReadArgs,
            handler=file_read_handler,
            timeout_seconds=5.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="file_write",
            description=_FILE_WRITE_DESC,
            input_model=FileWriteArgs,
            handler=file_write_handler,
            timeout_seconds=5.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="send_email",
            description=_SEND_EMAIL_DESC,
            input_model=SendEmailArgs,
            handler=send_email_handler,
            timeout_seconds=20.0,
            risk_level="medium",
        )
    )
    registry.register(
        ToolDefinition(
            name="analyze_data",
            description=_ANALYZE_DATA_DESC,
            input_model=AnalyzeDataArgs,
            handler=analyze_data_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="run_code",
            description=_RUN_CODE_DESC,
            input_model=RunCodeArgs,
            handler=run_code_handler,
            timeout_seconds=30.0,
            risk_level="medium",
        )
    )
    registry.register(
        ToolDefinition(
            name="http_get_json",
            description=_HTTP_GET_JSON_DESC,
            input_model=HttpGetJsonArgs,
            handler=http_get_json_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="read_pdf",
            description=_READ_PDF_DESC,
            input_model=ReadPdfArgs,
            handler=read_pdf_handler,
            timeout_seconds=20.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="read_excel",
            description=_READ_EXCEL_DESC,
            input_model=ReadExcelArgs,
            handler=read_excel_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="extract_web",
            description=_EXTRACT_WEB_DESC,
            input_model=ExtractWebArgs,
            handler=extract_web_handler,
            timeout_seconds=20.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="arxiv_search",
            description=_ARXIV_DESC,
            input_model=ArxivSearchArgs,
            handler=arxiv_search_handler,
            timeout_seconds=20.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="list_files",
            description=_LIST_FILES_DESC,
            input_model=ListFilesArgs,
            handler=list_files_handler,
            timeout_seconds=5.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="search_files",
            description=_SEARCH_FILES_DESC,
            input_model=SearchFilesArgs,
            handler=search_files_handler,
            timeout_seconds=10.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="append_note",
            description=_APPEND_NOTE_DESC,
            input_model=AppendNoteArgs,
            handler=append_note_handler,
            timeout_seconds=5.0,
            risk_level="low",
        )
    )
    return registry
