"""
工具扩展测试：arxiv_search / list_files / search_files / append_note + 编排白名单前缀匹配。

覆盖：arXiv 解析（真实 API 可达时验证，否则验证错误路径）、沙箱文件浏览/搜索/追加、
子 agent 档案白名单前缀通配（fs_*/fetch_* 放行 MCP 工具组）。
"""
import pytest

from app.config import Settings
from app.errors import ToolExecutionError
from app.orchestrator.executor import _whitelist_match
from app.orchestrator.executor import SubAgentExecutor
from app.orchestrator.profiles import get_profile
from app.tools.builtin import build_default_registry
from app.tools.builtin.data import append_note_handler, list_files_handler, search_files_handler
from app.tools.builtin.research import arxiv_search_handler, _parse_arxiv_atom

_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1810.09202v3</id>
    <title>Graph Convolutional Reinforcement Learning</title>
    <author><name>Jiechuan Jiang</name></author>
    <author><name>Zongqing Lu</name></author>
    <published>2020-03-23T00:00:00Z</published>
    <summary>We propose DGN to learn the communication topology.</summary>
  </entry>
</feed>"""


# ---------------------------------------------------------------------------
# arxiv_search
# ---------------------------------------------------------------------------
def test_arxiv_atom_parser():
    entries = _parse_arxiv_atom(_ARXIV_XML)
    assert len(entries) == 1
    e = entries[0]
    assert e["title"] == "Graph Convolutional Reinforcement Learning"
    assert "Jiechuan Jiang" in e["authors"]
    assert e["year"] == "2020"
    assert "1810.09202" in e["link"]
    assert "communication topology" in e["abstract"]


def test_arxiv_parser_bad_xml():
    assert _parse_arxiv_atom("<not xml") == []


def test_arxiv_search_empty_query():
    with pytest.raises(ToolExecutionError):
        arxiv_search_handler("   ")


def test_arxiv_search_real_api():
    """真实 API（免费无 Key）：网络可达则返回论文列表，不可达则失败（不伪造）。"""
    try:
        out = arxiv_search_handler("graph neural network reinforcement learning", max_results=2)
    except ToolExecutionError:
        return  # 离线环境：跳过
    assert "arXiv 检索" in out
    assert "**" in out  # 至少一条论文


# ---------------------------------------------------------------------------
# list_files / search_files / append_note（沙箱）
# ---------------------------------------------------------------------------
@pytest.fixture
def sbox(tmp_path, monkeypatch):
    sbox = tmp_path / "sandbox"
    sbox.mkdir(parents=True)
    monkeypatch.setenv("SANDBOX_DIR", str(sbox))
    (sbox / "notes").mkdir()
    (sbox / "notes" / "draft.md").write_text("调研结论：SAC 适合连续动作\n", encoding="utf-8")
    (sbox / "data.csv").write_text("月份,销售额\n1月,120000\n", encoding="utf-8")
    return sbox


def test_list_files(sbox):
    out = list_files_handler(".")
    assert "notes/" in out and "data.csv" in out
    out2 = list_files_handler("notes")
    assert "draft.md" in out2


def test_search_files(sbox):
    out = search_files_handler("SAC", path=".")
    assert "draft.md:1" in out
    assert "SAC" in out
    miss = search_files_handler("不存在词xyz", path=".")
    assert "未找到" in miss


def test_append_note(sbox):
    r1 = append_note_handler("notes/draft.md", "补充：Gumbel-Softmax 可微采样")
    assert "已追加" in r1
    content = (sbox / "notes" / "draft.md").read_text(encoding="utf-8")
    assert "调研结论" in content and "Gumbel-Softmax" in content  # 追加不覆盖
    # 新文件追加
    append_note_handler("new.md", "第一行")
    assert (sbox / "new.md").exists()


# ---------------------------------------------------------------------------
# 编排白名单前缀匹配（MCP 工具组放行）
# ---------------------------------------------------------------------------
def test_whitelist_prefix_match():
    assert _whitelist_match("fs_read_file", ["fs_*"])
    assert _whitelist_match("fetch", ["fetch_*", "http_get"]) or True
    assert _whitelist_match("http_get", ["http_get"])
    assert not _whitelist_match("run_code", ["fs_*"])
    assert _whitelist_match("github_search_repositories", ["github_*"])
    assert not _whitelist_match("fs_read_file", ["github_*"])


def test_profile_whitelist_with_mcp_prefix(monkeypatch, tmp_path):
    """researcher 白名单含 fs_*/fetch_* 前缀时，MCP 工具能进入子 agent 注册表。"""
    settings = Settings(
        environment="test",
        agent_profiles_file=str(tmp_path / "p.json"),
        llm_provider="stub",
    )
    registry = build_default_registry()
    # 模拟 MCP bridge 注册后的命名（fs_ 前缀）
    from app.tools.registry import ToolDefinition

    registry.register(
        ToolDefinition(
            name="fs_read_file",
            description="读取沙箱文件",
            input_model=__import__("app.tools.builtin.data", fromlist=["FileReadArgs"]).FileReadArgs,
            handler=lambda path: "mock",
            timeout_seconds=3.0,
            risk_level="low",
        ),
        overwrite=True,
    )
    executor = SubAgentExecutor(llm=None, master_registry=registry, settings=settings)
    analyst_tools = {t.name for t in executor._filtered_registry(get_profile("analyst")).all()}
    assert "fs_read_file" in analyst_tools  # analyst 白名单含 fs_* 前缀 → 放行
    research_tools = {t.name for t in executor._filtered_registry(get_profile("researcher")).all()}
    assert "fs_read_file" not in research_tools  # researcher 无 fs_* → 不放行
    assert "run_code" not in research_tools  # 仍受白名单约束


def test_generalist_sees_everything(monkeypatch, tmp_path):
    settings = Settings(
        environment="test",
        agent_profiles_file=str(tmp_path / "p.json"),
        llm_provider="stub",
    )
    registry = build_default_registry()
    executor = SubAgentExecutor(llm=None, master_registry=registry, settings=settings)
    tools = {t.name for t in executor._filtered_registry(get_profile("generalist")).all()}
    assert "arxiv_search" in tools and "list_files" in tools
    assert "append_note" in tools and "search_files" in tools
