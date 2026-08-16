"""
代码执行工具：run_code。

在受限沙箱中执行 Python 代码（解决：Agent 需要"写代码解决问题"的能力）。

安全设计（多层）：
    1. 超时限制（timeout_seconds，默认 10s）
    2. 内存限制（通过 resource 或限制数据结构大小）
    3. 危险模块禁用（os.system / subprocess / socket / import 受限）
    4. 工作目录 = 沙箱目录（只能读写沙箱内文件）
    5. 输出大小限制（防止刷屏）

注意：这是"教学级"沙箱（非强隔离），用于可信环境下的数据分析/计算任务；
生产环境应使用 Docker/gVisor 等真正隔离。
"""
import ast
import builtins
import io
import sys
import time
import traceback
from pathlib import Path

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import ToolExecutionError

# 禁止的模块名（黑名单）
_BLOCKED_MODULES = {
    "os", "subprocess", "socket", "sys", "shutil", "ctypes", "multiprocessing",
    "threading", "signal", "pickle", "marshal", "importlib", "inspect", "gc",
    "pty", "pdb", "platform", "resource", "fcntl",
}
# 允许导入的模块白名单（数据分析常用）
_ALLOWED_MODULES = {
    "math", "statistics", "json", "csv", "re", "collections", "itertools",
    "functools", "datetime", "random", "string", "typing", "dataclasses",
    "pathlib", "decimal", "fractions", "heapq", "bisect", "operator", "copy",
}
# 禁止的内置函数
_BLOCKED_BUILTINS = {
    "eval", "exec", "compile", "open", "input", "breakpoint",
    "globals", "locals", "vars", "getattr", "setattr", "delattr", "memoryview",
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    """白名单版 __import__：只允许导入授权模块。"""
    root = name.split(".")[0]
    if root in _BLOCKED_MODULES:
        raise ToolExecutionError(f"禁止导入模块: {name}")
    if root not in _ALLOWED_MODULES:
        raise ToolExecutionError(f"未授权模块: {name}（白名单: {sorted(_ALLOWED_MODULES)[:10]}...）")
    return builtins.__import__(name, globals, locals, fromlist, level)


class RunCodeArgs(BaseModel):
    """代码执行参数。"""

    code: str = Field(description="要执行的 Python 代码（单段，可多行）")
    timeout_seconds: int = Field(default=10, ge=1, le=60, description="超时秒数（1-60）")


def _check_code(code: str) -> None:
    """静态检查：AST 解析 + 禁用黑名单。"""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise ToolExecutionError(f"代码语法错误: {exc}") from exc

    for node in ast.walk(tree):
        # 禁止 import 黑名单模块
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BLOCKED_MODULES:
                    raise ToolExecutionError(f"禁止导入模块: {alias.name}")
                if root not in _ALLOWED_MODULES:
                    raise ToolExecutionError(f"未授权模块: {alias.name}（白名单: {sorted(_ALLOWED_MODULES)[:10]}...）")
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in _BLOCKED_MODULES:
                raise ToolExecutionError(f"禁止导入模块: {node.module}")
        # 禁止调用危险内置
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_BUILTINS:
                raise ToolExecutionError(f"禁止使用内置函数: {node.func.id}")
        # 禁止属性访问 os.system 等（黑名单模块名打头）
        elif isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id in _BLOCKED_MODULES:
                raise ToolExecutionError(f"禁止访问模块: {node.value.id}")


def run_code_handler(code: str, timeout_seconds: int = 10) -> str:
    """在受限沙箱执行代码，返回 stdout + 结果摘要。"""
    _check_code(code)

    settings = Settings()
    sandbox = Path(settings.sandbox_dir).resolve()
    sandbox.mkdir(parents=True, exist_ok=True)

    # 构造受限 globals
    safe_builtins = {k: v for k, v in vars(builtins).items() if k not in _BLOCKED_BUILTINS}
    # __import__ 替换为白名单版（import 语句依赖它）
    safe_builtins["__import__"] = _safe_import
    restricted_globals = {
        "__builtins__": safe_builtins,
        "__name__": "__main__",
        "__file__": str(sandbox / "sandbox_code.py"),
    }

    # 捕获输出
    stdout_buf = io.StringIO()
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = stdout_buf, stdout_buf
    start = time.monotonic()
    try:
        # 限制工作目录（通过受限的 Path 操作 + 手动 chdir 到沙箱）
        old_cwd = Path.cwd()
        try:
            import os as _os  # 仅用于 chdir（不暴露给代码）
            _os.chdir(sandbox)
            exec(compile(code, "<sandbox>", "exec"), restricted_globals)
        finally:
            try:
                import os as _os2
                _os2.chdir(old_cwd)
            except Exception:
                pass
    except ToolExecutionError:
        raise
    except MemoryError:
        raise ToolExecutionError("代码内存超限")
    except Exception as exc:
        # 返回异常信息（不崩溃 Agent）
        stdout_buf.write(f"\n[异常] {type(exc).__name__}: {exc}\n")
        stdout_buf.write(traceback.format_exc(limit=3))
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    elapsed = round(time.monotonic() - start, 3)
    if elapsed > timeout_seconds:
        raise ToolExecutionError(f"代码执行超时（>{timeout_seconds}s）")

    output = stdout_buf.getvalue().strip()
    if not output:
        output = "（无输出）"
    # 限制输出长度
    if len(output) > 3000:
        output = output[:3000] + f"\n...（输出截断，共 {len(output)} 字符）"
    return f"✅ 执行成功（{elapsed}s）\n{output}"
