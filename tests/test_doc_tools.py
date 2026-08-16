"""
文档工具测试：read_pdf / read_excel / extract_web。

覆盖：PDF 文本提取、Excel 表格渲染、网页正文提取、路径越界防护。
"""
import pathlib

import pytest

from app.config import Settings
from app.errors import ToolExecutionError
from app.tools.builtin.documents import (
    _render_table,
    extract_web_handler,
    read_excel_handler,
    read_pdf_handler,
)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    sbox = tmp_path / "sandbox"
    sbox.mkdir(parents=True)
    monkeypatch.setenv("SANDBOX_DIR", str(sbox))
    # 测试 PDF
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "PDF Test Content Line1", fontsize=12)
    page.insert_text((72, 100), "Second line of PDF", fontsize=12)
    doc.save(str(sbox / "test.pdf"))
    doc.close()
    # 测试 Excel
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "数据"
    ws.append(["月份", "值"])
    ws.append(["1月", 100])
    ws.append(["2月", 200])
    wb.save(str(sbox / "test.xlsx"))
    return sbox


def test_read_pdf(sandbox):
    out = read_pdf_handler("test.pdf")
    assert "PDF Test Content Line1" in out
    assert "共 1 页" in out


def test_read_pdf_page_range(sandbox):
    with pytest.raises(ToolExecutionError):
        read_pdf_handler("test.pdf", page=5)  # 越界


def test_read_pdf_not_pdf(sandbox):
    (sandbox / "a.txt").write_text("hi", encoding="utf-8")
    with pytest.raises(ToolExecutionError):
        read_pdf_handler("a.txt")


def test_read_excel(sandbox):
    out = read_excel_handler("test.xlsx")
    assert "| 月份 | 值 |" in out
    assert "100" in out
    assert "数据" in out  # sheet 名


def test_read_excel_sheet_choice(sandbox):
    out = read_excel_handler("test.xlsx", sheet="数据")
    assert "100" in out


def test_read_excel_csv(sandbox):
    (sandbox / "d.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    out = read_excel_handler("d.csv")
    assert "| a | b |" in out
    assert "3" in out


def test_path_traversal_blocked(sandbox):
    with pytest.raises(ToolExecutionError):
        read_pdf_handler("../../etc/passwd")
    with pytest.raises(ToolExecutionError):
        read_excel_handler("../evil.xlsx")


def test_extract_web_rejects_non_http():
    with pytest.raises(ToolExecutionError):
        extract_web_handler("file:///etc/passwd")


def test_render_table_empty():
    assert "无数据" in _render_table([], "h")


def test_render_table_normal():
    out = _render_table([["a", "b"], ["1", "2"]], "header")
    assert "| a | b |" in out
    assert "| 1 | 2 |" in out
