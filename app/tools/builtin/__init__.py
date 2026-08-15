"""内置工具包：calculator / get_weather + web_search / http_get / get_time / get_date + sqlite_query / file 读写 + send_email。"""
from app.tools.builtin.calculator import calculator_handler, CalculatorArgs
from app.tools.builtin.data import (
    FileReadArgs,
    FileWriteArgs,
    SqliteQueryArgs,
    file_read_handler,
    file_write_handler,
    sqlite_query_handler,
)
from app.tools.builtin.mail import SendEmailArgs, send_email_handler
from app.tools.builtin.weather import weather_handler, WeatherArgs
from app.tools.builtin.web import (
    DateArgs,
    HttpGetArgs,
    TimeArgs,
    WebSearchArgs,
    get_date_handler,
    get_time_handler,
    http_get_handler,
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
    return registry
