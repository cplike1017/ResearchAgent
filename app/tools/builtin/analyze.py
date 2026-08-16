"""
数据分析工具：analyze_data。

读取沙箱内的 CSV/JSON 数据，执行统计聚合，返回结构化结果。
解决：之前 Agent 分析数据只能靠模型硬算（低效易错），
现在用 pandas-like 统计（自研实现，不引入重依赖）。

能力：
    - 支持 CSV / JSON 文件
    - 统计指标：count / mean / median / min / max / std / p25 / p75 / p90
    - 按列聚合 + 可选 groupby
    - 趋势检测：数值列按序比较首尾/整体斜率
"""
import csv
import json
import math
import statistics
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import ToolExecutionError


class AnalyzeDataArgs(BaseModel):
    """数据分析参数。"""

    path: str = Field(description="沙箱内的数据文件相对路径（CSV 或 JSON）")
    numeric_columns: list[str] | None = Field(default=None, description="要分析的数值列名（缺省自动检测）")
    group_by: str | None = Field(default=None, description="分组列名（可选，按该列分组统计）")


def _resolve_path(relative_path: str) -> Path:
    settings = Settings()
    sandbox = Path(settings.sandbox_dir).resolve()
    target = (sandbox / relative_path).resolve()
    if not str(target).startswith(str(sandbox)):
        raise ToolExecutionError(f"路径越界：只允许沙箱目录内（{sandbox}）")
    if not target.exists():
        raise ToolExecutionError(f"文件不存在: {relative_path}")
    return target


def _load_data(path: Path) -> list[dict]:
    """加载 CSV 或 JSON 文件为 dict 列表。"""
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                return data
            if isinstance(data, dict):
                # 可能是 {"data": [...]} 或单行
                for v in data.values():
                    if isinstance(v, list) and all(isinstance(d, dict) for d in v):
                        return v
                return [data]
            raise ToolExecutionError("JSON 必须是对象数组")
        except json.JSONDecodeError as exc:
            raise ToolExecutionError(f"JSON 解析失败: {exc}")
    if suffix == ".csv":
        try:
            with open(path, encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader if any(v.strip() for v in r.values() if v)]
            if not rows:
                raise ToolExecutionError("CSV 无数据行")
            return rows
        except csv.Error as exc:
            raise ToolExecutionError(f"CSV 解析失败: {exc}")
    raise ToolExecutionError(f"不支持的文件类型: {suffix}（支持 .csv / .json）")


def _to_float(v) -> float | None:
    try:
        f = float(str(v).replace(",", "").replace("%", "").strip())
        return f
    except (ValueError, TypeError):
        return None


def _numeric_cols(rows: list[dict]) -> list[str]:
    """自动检测数值列。"""
    if not rows:
        return []
    cols = []
    for col in rows[0].keys():
        vals = [_to_float(r.get(col)) for r in rows[:20]]
        numeric = sum(1 for v in vals if v is not None)
        if numeric >= max(3, len(vals) // 2):
            cols.append(col)
    return cols


def _stats(vals: list[float]) -> dict:
    vals = sorted(v for v in vals if v is not None)
    if not vals:
        return {"count": 0}
    n = len(vals)
    result = {
        "count": n,
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "mean": round(statistics.fmean(vals), 4),
        "median": round(statistics.median(vals), 4),
    }
    if n >= 2:
        result["std"] = round(statistics.stdev(vals), 4)
    for q, name in ((0.25, "p25"), (0.75, "p75"), (0.9, "p90")):
        idx = min(n - 1, int(n * q))
        result[name] = round(vals[idx], 4)
    return result


def analyze_data_handler(path: str, numeric_columns: list[str] | None = None, group_by: str | None = None) -> str:
    """分析数据文件，返回统计摘要。"""
    target = _resolve_path(path)
    rows = _load_data(target)
    if not rows:
        raise ToolExecutionError("数据为空")

    # 确定数值列
    cols = numeric_columns or _numeric_cols(rows)
    if not cols:
        # 无数值列时返回结构概览
        return _overview(rows, group_by)

    lines = [f"数据文件: {path}", f"行数: {len(rows)}", f"列: {list(rows[0].keys())}"]

    if group_by and group_by in rows[0]:
        # 分组统计
        groups: dict[str, list[dict]] = {}
        for r in rows:
            g = str(r.get(group_by, "未知"))
            groups.setdefault(g, []).append(r)
        lines.append(f"\n按 [{group_by}] 分组（{len(groups)} 组）:")
        for g, g_rows in sorted(groups.items()):
            lines.append(f"\n■ 组 {g}（{len(g_rows)} 行）:")
            for col in cols:
                vals = [_to_float(r.get(col)) for r in g_rows]
                st = _stats(vals)
                if st["count"]:
                    lines.append(f"  {col}: mean={st['mean']} median={st['median']} min={st['min']} max={st['max']}")
    else:
        # 全量统计
        lines.append("\n统计指标:")
        for col in cols:
            vals = [_to_float(r.get(col)) for r in rows]
            st = _stats(vals)
            if st["count"]:
                lines.append(f"  {col}: count={st['count']} mean={st['mean']} median={st['median']} "
                             f"min={st['min']} max={st['max']} std={st.get('std','-')} p90={st.get('p90','-')}")

    # 趋势检测（第一个数值列）
    if cols:
        col = cols[0]
        vals = [_to_float(r.get(col)) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 3:
            first, last = vals[0], vals[-1]
            pct = ((last - first) / abs(first) * 100) if first else 0
            trend = "上升" if pct > 3 else "下降" if pct < -3 else "平稳"
            lines.append(f"\n趋势（{col} 首→尾）: {trend}（{pct:+.1f}%）")

    return "\n".join(lines)


def _overview(rows: list[dict], group_by: str | None) -> str:
    """无数值列时的结构概览。"""
    lines = [f"数据行数: {len(rows)}", f"列: {list(rows[0].keys())}"]
    lines.append("（未检测到数值列，仅展示结构）")
    for r in rows[:5]:
        lines.append(f"  {json.dumps(r, ensure_ascii=False)[:120]}")
    return "\n".join(lines)
