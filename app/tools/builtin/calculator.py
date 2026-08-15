"""
安全计算器工具。

安全要求：禁止 eval() / exec() / shell=True。
实现方式：先用 ast.parse 把表达式解析为 AST，再人工遍历求值；
任何不在白名单里的节点（函数调用、属性访问、变量名等）直接拒绝。
"""
import ast
import operator
from typing import Any

from pydantic import BaseModel, Field

from app.errors import ToolExecutionError


class CalculatorArgs(BaseModel):
    """计算器参数：一个数学表达式字符串。"""

    expression: str = Field(description="数学表达式，仅支持数字、+ - * / // % ** 与括号")


# 允许的二元运算符白名单
_BINOPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# 允许的一元运算符白名单
_UNARYOPS: dict[type, Any] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# 允许出现的 AST 节点类型（白名单校验用）
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


def _check_node_types(tree: ast.AST) -> None:
    """校验 AST 中只存在白名单节点，杜绝任何函数调用 / 变量 / 属性访问。"""
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ToolExecutionError(f"表达式中包含不支持的语法: {type(node).__name__}")


def _eval_node(node: ast.AST) -> float:
    """手工递归求值白名单 AST。"""
    if isinstance(node, ast.Constant):
        # 只允许数字常量
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return float(node.value)
        raise ToolExecutionError("表达式只支持数字常量")
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ToolExecutionError(f"不支持的运算符: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op in (operator.truediv, operator.floordiv, operator.mod) and right == 0:
            raise ToolExecutionError("除数为 0")
        return float(op(left, right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ToolExecutionError(f"不支持的一元运算符: {type(node.op).__name__}")
        return float(op(_eval_node(node.operand)))
    raise ToolExecutionError(f"不支持的表达式节点: {type(node).__name__}")


def safe_evaluate(expression: str) -> float:
    """
    安全求值：AST 白名单 + 手工求值，绝不使用 eval/exec。

    >>> safe_evaluate("123 * 456")
    56088.0
    """
    if not expression or not expression.strip():
        raise ToolExecutionError("表达式为空")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolExecutionError(f"表达式语法错误: {exc}") from exc
    _check_node_types(tree)
    result = _eval_node(tree.body)
    # 防御极端数值（溢出 / NaN）
    try:
        if result != result or result in (float("inf"), float("-inf")):
            raise ToolExecutionError("计算结果溢出")
    except TypeError:
        pass
    return result


def calculator_handler(expression: str) -> float:
    """工具处理器：安全计算并返回数值。"""
    return safe_evaluate(expression)
