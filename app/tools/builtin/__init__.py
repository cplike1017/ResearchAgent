"""内置工具包：calculator / get_weather + web / data / code 类真实工具 + send_email。"""
from app.tools.builtin.analyze import AnalyzeDataArgs, analyze_data_handler
from app.tools.builtin.calculator import calculator_handler, CalculatorArgs
from app.tools.builtin.code_exec import RunCodeArgs, run_code_handler
from app.tools.builtin.data import (
    FileReadArgs,
    FileWriteArgs,
    SqliteQueryArgs,
    file_read_handler,
    file_write_handler,
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


def build_default_registry() -> ToolRegistry:
    """构建包含全部内置工具的默认注册表。"""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="calculator",
            description="计算数学表达式，支持 + - * / // % ** 和括号。例如：123 * 456",
            input_model=CalculatorArgs,
            handler=calculator_handler,
            timeout_seconds=3.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="get_weather",
            description="查询指定中国城市的当前天气（真实 API，支持：北京 上海 广州 深圳 杭州 成都 武汉 西安 南京 重庆）",
            input_model=WeatherArgs,
            handler=weather_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="web_search",
            description="联网搜索，返回相关网页标题/链接/摘要。需要 TAVILY_API_KEY。",
            input_model=WebSearchArgs,
            handler=web_search_handler,
            timeout_seconds=20.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="http_get",
            description="抓取指定网址的网页文本内容（限 8KB）。用于读取在线资料。",
            input_model=HttpGetArgs,
            handler=http_get_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="get_time",
            description="获取当前 UTC 时间，格式 YYYY-MM-DD HH:MM:SS",
            input_model=TimeArgs,
            handler=get_time_handler,
            timeout_seconds=3.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="get_date",
            description="获取当前 UTC 日期，格式 YYYY-MM-DD",
            input_model=DateArgs,
            handler=get_date_handler,
            timeout_seconds=3.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="sqlite_query",
            description="对指定的 SQLite 数据库文件执行只读 SELECT 查询，返回结果行。",
            input_model=SqliteQueryArgs,
            handler=sqlite_query_handler,
            timeout_seconds=10.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="file_read",
            description="读取沙箱目录内的文本文件（限 8KB）。路径为沙箱内相对路径。",
            input_model=FileReadArgs,
            handler=file_read_handler,
            timeout_seconds=5.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="file_write",
            description="写入沙箱目录内的文本文件（限 16KB，自动创建父目录）。路径为沙箱内相对路径。",
            input_model=FileWriteArgs,
            handler=file_write_handler,
            timeout_seconds=5.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="send_email",
            description="发送一封文本邮件到指定邮箱。需要配置 SMTP（QQ 邮箱）。",
            input_model=SendEmailArgs,
            handler=send_email_handler,
            timeout_seconds=20.0,
            risk_level="medium",
        )
    )
    registry.register(
        ToolDefinition(
            name="analyze_data",
            description="分析沙箱内的 CSV/JSON 数据文件：统计指标（均值/中位数/分位数/标准差）、按列分组、趋势检测。返回结构化统计摘要。",
            input_model=AnalyzeDataArgs,
            handler=analyze_data_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="run_code",
            description="在受限沙箱中执行 Python 代码（数据分析/计算/文件处理）。白名单模块：math/statistics/json/csv/re/collections 等。超时 10s，禁止 os/subprocess/网络。",
            input_model=RunCodeArgs,
            handler=run_code_handler,
            timeout_seconds=30.0,
            risk_level="medium",
        )
    )
    registry.register(
        ToolDefinition(
            name="http_get_json",
            description="抓取返回 JSON 的 API 并结构化格式化输出（限 32KB）。用于对接 REST API 获取数据。",
            input_model=HttpGetJsonArgs,
            handler=http_get_json_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="read_pdf",
            description="提取沙箱内 PDF 文件的文本内容（支持指定页或全部，限 8KB）。用于读取论文/文档。",
            input_model=ReadPdfArgs,
            handler=read_pdf_handler,
            timeout_seconds=20.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="read_excel",
            description="读取沙箱内的 Excel（.xlsx/.xlsm）或 CSV 文件为 markdown 表格（支持指定 sheet，限 200 行）。",
            input_model=ReadExcelArgs,
            handler=read_excel_handler,
            timeout_seconds=15.0,
            risk_level="low",
        )
    )
    registry.register(
        ToolDefinition(
            name="extract_web",
            description="抓取网页并提取正文为 markdown 风格文本（去导航/脚本，限 16KB）。用于深入阅读网页内容。",
            input_model=ExtractWebArgs,
            handler=extract_web_handler,
            timeout_seconds=20.0,
            risk_level="low",
        )
    )
    return registry
