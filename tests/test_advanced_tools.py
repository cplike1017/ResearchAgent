"""
新工具测试：analyze_data / run_code / http_get_json。

覆盖：数据分析统计、代码沙箱（合法执行 / 危险拦截 / 异常处理）、JSON 抓取。
"""
import pathlib

import pytest

from app.config import Settings
from app.errors import ToolExecutionError
from app.tools.builtin.analyze import analyze_data_handler, _load_data, _numeric_cols, _stats
from app.tools.builtin.code_exec import run_code_handler
from app.tools.builtin.web import http_get_json_handler


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """测试沙箱目录 + 测试数据。"""
    sbox = tmp_path / "sandbox"
    sbox.mkdir(parents=True)
    monkeypatch.setenv("SANDBOX_DIR", str(sbox))
    csv = "月份,销售额,成本,利润\n1月,120000,85000,35000\n2月,135000,90000,45000\n3月,110000,82000,28000\n"
    (sbox / "data.csv").write_text(csv, encoding="utf-8")
    return sbox


# ---------------------------------------------------------------------------
# analyze_data
# ---------------------------------------------------------------------------
def test_analyze_stats(sandbox):
    out = analyze_data_handler("data.csv")
    assert "销售额" in out
    assert "mean=121666.6667" in out  # (120000+135000+110000)/3
    assert "max=135000.0" in out
    assert "趋势" in out


def test_analyze_group_by(sandbox):
    out = analyze_data_handler("data.csv", numeric_columns=["利润"], group_by="月份")
    assert "按 [月份] 分组" in out
    assert "1月" in out


def test_analyze_missing_file(sandbox):
    with pytest.raises(ToolExecutionError):
        analyze_data_handler("nope.csv")


def test_stats_helper():
    st = _stats([1.0, 2.0, 3.0, 4.0])
    assert st["mean"] == 2.5
    assert st["median"] == 2.5
    assert st["min"] == 1.0
    assert st["max"] == 4.0


def test_numeric_cols_detection(sandbox):
    rows = _load_data(sandbox / "data.csv")
    cols = _numeric_cols(rows)
    assert "销售额" in cols
    assert "月份" not in cols  # 中文月份非数值


# ---------------------------------------------------------------------------
# run_code（沙箱）
# ---------------------------------------------------------------------------
def test_run_code_ok():
    out = run_code_handler('print("hello")\nprint(sum([1,2,3]))')
    assert "hello" in out
    assert "6" in out


def test_run_code_blocked_import():
    with pytest.raises(ToolExecutionError):
        run_code_handler("import os\nprint(os.listdir('.'))")


def test_run_code_blocked_eval():
    with pytest.raises(ToolExecutionError):
        run_code_handler("print(eval('1+1'))")


def test_run_code_exception_caught():
    """代码异常不崩溃，返回异常信息。"""
    out = run_code_handler("x = 1/0")
    assert "ZeroDivisionError" in out


def test_run_code_blocked_subprocess():
    with pytest.raises(ToolExecutionError):
        run_code_handler("import subprocess\nsubprocess.run(['ls'])")


# ---------------------------------------------------------------------------
# http_get_json
# ---------------------------------------------------------------------------
def test_http_get_json_rejects_non_http():
    with pytest.raises(ToolExecutionError):
        http_get_json_handler("file:///etc/passwd")
