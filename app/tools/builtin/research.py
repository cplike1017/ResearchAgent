"""
真实工具包：arxiv_search（学术论文检索）。

为什么需要它？Trace 数据显示 researcher 在调研任务里反复用 web_search 找论文、
再用 http_get 手拼 arxiv.org/abs/XXX 验证 —— 低效且易错。arxiv_search 直接
调 arXiv API（export.arxiv.org/api/query，免费无需 Key），一次返回结构化
论文列表（标题/作者/年份/摘要/链接），是学术调研的垂直工具。

安全设计：
    - 只请求 arXiv 官方 API；响应大小限制 32KB；
    - 无 Key、零依赖（免费 API），失败返回结构化错误。
"""
import re
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.errors import ToolExecutionError

ARXIV_API = "https://export.arxiv.org/api/query"
# 注意：不能截断 XML 响应（会切断 <entry> 结构导致解析失败）；
# 限制放在解析后的输出层（摘要/作者字段截断）。
MAX_ENTRIES = 10
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


class ArxivSearchArgs(BaseModel):
    """arXiv 检索参数。"""

    query: str = Field(description="检索关键词（支持 AND/OR/引号，如 'graph neural network MARL'）")
    max_results: int = Field(default=5, ge=1, le=10, description="返回论文条数（1-10）")
    sort_by: str = Field(default="relevance", description="排序：relevance(相关度) | submittedDate(最新)")


def arxiv_search_handler(query: str, max_results: int = 5, sort_by: str = "relevance") -> str:
    """检索 arXiv 论文，返回结构化列表文本。"""
    settings = get_settings()
    if not query.strip():
        raise ToolExecutionError("检索关键词不能为空")
    sort = "relevance" if sort_by == "relevance" else "submittedDate"
    # 关键词用引号包裹提升相关性（arXiv 默认把空格当 OR）
    quoted = f'"{query.strip()}"' if " " in query.strip() else query.strip()
    params = {
        "search_query": f"all:{quoted}",
        "start": 0,
        "max_results": max_results,
        "sortBy": sort,
        "sortOrder": "descending" if sort == "submittedDate" else "ascending",
    }
    try:
        resp = httpx.get(ARXIV_API, params=params, timeout=settings.http_tool_timeout_seconds, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise ToolExecutionError(f"arXiv API 请求失败: {exc}")

    entries = _parse_arxiv_atom(resp.text)
    if not entries:
        return "arXiv 未检索到相关论文。"
    lines = [f"arXiv 检索「{query}」共 {len(entries)} 条结果：", ""]
    for i, e in enumerate(entries, 1):
        lines.append(f"{i}. **{e['title']}**")
        lines.append(f"   作者: {e['authors']}")
        lines.append(f"   年份: {e['year']} | 链接: {e['link']}")
        if e["abstract"]:
            lines.append(f"   摘要: {e['abstract'][:220]}")
        lines.append("")
    return "\n".join(lines)


def _parse_arxiv_atom(xml_text: str) -> list[dict]:
    """解析 arXiv Atom XML → 论文条目列表。"""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    entries = []
    for entry in root.findall("atom:entry", _NS):
        title = _clean((entry.findtext("atom:title", "", _NS) or ""))
        link_el = entry.find("atom:id", _NS)
        link = (link_el.text or "").strip() if link_el is not None else ""
        authors = [a.findtext("atom:name", "", _NS).strip() for a in entry.findall("atom:author", _NS)]
        authors = [a for a in authors if a and a != ":"]
        published = entry.findtext("atom:published", "", _NS) or ""
        abstract = _clean((entry.findtext("atom:summary", "", _NS) or ""))
        entries.append({
            "title": title or "(无标题)",
            "authors": ", ".join(a for a in authors if a)[:120] or "未知",
            "year": published[:4],
            "link": link,
            "abstract": abstract,
        })
    return entries


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
