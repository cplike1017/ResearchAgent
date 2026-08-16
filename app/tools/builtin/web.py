"""
真实工具包：web_search / http_get / get_time / get_date。

区别于 calculator/weather 的 Stub 定位：
    - web_search：调用 Tavily API 真实检索网络（需 TAVILY_API_KEY）
    - http_get：真实 HTTP 抓取（httpx），带超时与大小限制
    - get_time / get_date：本地时间，零依赖，给 Agent 提供时间感（无状态模型的刚需）

安全设计：
    - http_get 限制响应大小（MAX_BYTES）与超时，禁止重定向跟踪（防 SSRF 重定向滥用）
    - web_search 失败（无 Key/网络）返回结构化失败而非抛异常（Gateway 兜底）
"""
from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.errors import ToolExecutionError

# 响应体上限：8KB（教学场景足够，防内存滥用）
MAX_BYTES = 8192
# 只允许 http/https
_ALLOWED_SCHEMES = ("http://", "https://")


# ---------------------------------------------------------------------------
# web_search（Tavily）
# ---------------------------------------------------------------------------
class WebSearchArgs(BaseModel):
    """搜索参数。"""

    query: str = Field(description="搜索关键词（中文英文均可）")
    max_results: int = Field(default=3, ge=1, le=5, description="返回结果条数（1-5）")


def _tavily_search(query: str, max_results: int, settings: Settings) -> str:
    """调用 Tavily API 搜索，返回格式化结果文本。"""
    if not settings.tavily_api_key:
        raise ToolExecutionError(
            "TAVILY_API_KEY 未配置，无法使用 web_search。请在 .env 配置 TAVILY_API_KEY"
        )
    try:
        resp = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": settings.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=settings.tavily_timeout_seconds,
        )
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"搜索请求失败: {exc}", transient=True) from exc
    if resp.status_code != 200:
        raise ToolExecutionError(f"搜索接口返回 {resp.status_code}: {resp.text[:200]}")

    try:
        results = resp.json().get("results", [])
    except ValueError as exc:
        raise ToolExecutionError(f"搜索响应格式非法: {exc}") from exc

    if not results:
        return "未找到相关结果。"

    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("url", "")
        content = (r.get("content") or "")[:200]
        lines.append(f"{i}. {title}\n   {url}\n   {content}")
    return "\n".join(lines)


def web_search_handler(query: str, max_results: int = 3) -> str:
    """Tavily 网络搜索工具处理器。"""
    return _tavily_search(query, max_results, get_settings())


# ---------------------------------------------------------------------------
# http_get
# ---------------------------------------------------------------------------
class HttpGetArgs(BaseModel):
    """HTTP 抓取参数。"""

    url: str = Field(description="要抓取的 http/https 网址")


def http_get_handler(url: str) -> str:
    """抓取网页正文（限大小、限超时、不跟随重定向）。"""
    url = url.strip()
    if not url.startswith(_ALLOWED_SCHEMES):
        raise ToolExecutionError("只支持 http/https 协议")
    try:
        with httpx.Client(
            timeout=Settings().http_tool_timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "agent-runtime/0.1 (educational)"},
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"HTTP 请求失败: {exc}", transient=True) from exc
    if resp.status_code != 200:
        raise ToolExecutionError(f"HTTP {resp.status_code}")

    text = resp.text[:MAX_BYTES]
    # 简易去标签（保留可读文本）
    import re

    text = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return "(页面无可读文本)"
    return text[:MAX_BYTES]


# ---------------------------------------------------------------------------
# http_get_json（结构化 API 抓取）
# ---------------------------------------------------------------------------
class HttpGetJsonArgs(BaseModel):
    """JSON API 抓取参数。"""

    url: str = Field(description="返回 JSON 的 API 地址（http/https）")


def http_get_json_handler(url: str) -> str:
    """抓取 JSON API 并格式化返回（限 32KB）。"""
    url = url.strip()
    if not url.startswith(_ALLOWED_SCHEMES):
        raise ToolExecutionError("只支持 http/https 协议")
    try:
        with httpx.Client(
            timeout=Settings().http_tool_timeout_seconds,
            follow_redirects=False,
            headers={"User-Agent": "agent-runtime/0.1 (educational)"},
        ) as client:
            resp = client.get(url)
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"HTTP 请求失败: {exc}", transient=True) from exc
    if resp.status_code != 200:
        raise ToolExecutionError(f"HTTP {resp.status_code}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise ToolExecutionError(f"响应不是合法 JSON: {exc}") from exc

    # 截断并格式化
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if len(text) > 32768:
        text = text[:32768] + "\n...（截断）"
    return text


# ---------------------------------------------------------------------------
# get_time / get_date（本地时间，零依赖）
# ---------------------------------------------------------------------------
class TimeArgs(BaseModel):
    """时间查询参数（无参）。"""

    pass


def get_time_handler() -> str:
    """当前 UTC 时间。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class DateArgs(BaseModel):
    """日期查询参数（无参）。"""

    pass


def get_date_handler() -> str:
    """当前日期。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")
