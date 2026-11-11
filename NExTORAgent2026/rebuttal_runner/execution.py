"""Subprocess execution helpers for generated rebuttal programs."""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from .scoring import AnswerParseError, parse_answer


_PYTHON_BLOCK = re.compile(r"```python[ \t]*\r?\n(.*?)```", re.IGNORECASE | re.DOTALL)
_SOLVER_STATUS_PREFIX = "SolverStatus=JSON"
_HELPER_NAMES = {
    "__rebuttal_optimize__",
    "__rebuttal_optimize_calls",
    "__rebuttal_solver_status",
    "__rebuttal_emit_solver_status",
    "__rebuttal_json",
    "__rebuttal_math",
    "__rebuttal_atexit",
}


class _InvalidProgram(ValueError):
    pass


def parse_solver_answer(execution: dict[str, Any], answer_spec: dict[str, Any] | None = None
                        ) -> tuple[Any | None, dict[str, str] | None]:
    """Parse FinalAnswer only after a bounded Gurobi run recorded a solution."""
    if execution.get("status") != "executed" or execution.get("returncode") != 0:
        return None, {"code": "execution_failed", "message": "program did not execute successfully"}
    solver = execution.get("solver_status")
    if execution.get("solver_limits_applied") is not True or not isinstance(solver, dict):
        return None, {"code": "solver_not_verified",
                      "message": "bounded Gurobi optimize status was not recorded; non-Gurobi execution or missing optimize call"}
    sol_count = solver.get("sol_count")
    if isinstance(sol_count, bool) or not isinstance(sol_count, int) or sol_count <= 0:
        return None, {"code": "no_solver_solution",
                      "message": f"solver recorded no incumbent solution (sol_count={sol_count!r})"}
    try:
        answer = parse_answer(execution.get("stdout", ""))
    except AnswerParseError as exc:
        code = "missing_final_answer" if "missing standalone FinalAnswer" in str(exc) else "invalid_final_answer"
        return None, {"code": code, "message": str(exc), "path": exc.path}
    if answer_spec:
        kind = answer_spec.get("kind")
        fields = answer_spec.get("fields")
        valid = True
        if kind == "scalar":
            valid = isinstance(answer, (int, float)) and not isinstance(answer, bool)
        elif kind == "numeric_list":
            valid = isinstance(answer, list) and (not isinstance(fields, list) or len(answer) == len(fields))
        elif kind == "named_numeric":
            valid = isinstance(answer, dict) and (not isinstance(fields, list) or set(answer) == set(fields))
        if not valid:
            return None, {"code": "invalid_answer_shape",
                          "message": f"FinalAnswer JSON shape does not match answer_spec kind={kind!r} and fields={fields!r}",
                          "path": "$"}
    return answer, None


def _parse_source(source: str) -> ast.Module:
    try:
        return ast.parse(source, filename="generated.py", mode="exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise _InvalidProgram(f"Python syntax error: {exc}") from None


def extract_code(text: str) -> str:
    """Return one Python fenced block, or the input if it is parseable bare code."""
    if not isinstance(text, str):
        raise TypeError("model response must be a string")
    blocks = _PYTHON_BLOCK.findall(text)
    if len(blocks) > 1:
        raise ValueError("multiple python code blocks are ambiguous")
    source = blocks[0].strip("\r\n") if blocks else text
    try:
        ast.parse(source, filename="generated.py", mode="exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError(f"response does not contain parseable Python code: {exc}") from None
    return source


class _OptimizeCallCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)
        if node.id in {"eval", "exec", "getattr", "__import__"}:
            raise _InvalidProgram(
                f"dynamic {node.id} reference prevents reliable optimize rewriting"
            )

    def visit_arg(self, node: ast.arg) -> None:
        self.names.add(node.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_alias(self, node: ast.alias) -> None:
        self.names.add(node.asname or node.name.split(".")[0])
        imported_name = node.name.rsplit(".", 1)[-1]
        if imported_name in {"eval", "exec", "getattr", "__import__", "methodcaller", "attrgetter"}:
            raise _InvalidProgram(
                f"dynamic {imported_name} import prevents reliable optimize rewriting"
            )

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.names.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.names.update(node.names)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name:
            self.names.add(node.name)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest:
            self.names.add(node.rest)
        self.generic_visit(node)

    def visit_Await(self, node: ast.Await) -> None:
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "optimize"
        ):
            raise _InvalidProgram("awaited .optimize() cannot be reliably rewritten")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == "optimize":
                self.calls.append(node)
                self.visit(node.func.value)
                for arg in node.args:
                    self.visit(arg)
                for keyword in node.keywords:
                    self.visit(keyword.value)
                return
            if node.func.attr.lower().startswith("optimize"):
                raise _InvalidProgram(
                    f"unsupported solver method {node.func.attr!r}; only direct .optimize(...) can be limited"
                )
            if node.func.attr in {"eval", "exec"}:
                raise _InvalidProgram(f"dynamic {node.func.attr}() prevents reliable optimize rewriting")
        if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
            raise _InvalidProgram(f"dynamic {node.func.id}() prevents reliable optimize rewriting")
        if isinstance(node.func, ast.Name) and node.func.id == "getattr":
            if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
                raise _InvalidProgram("dynamic getattr() prevents reliable optimize rewriting")
            if node.args[1].value == "optimize":
                raise _InvalidProgram("getattr(..., 'optimize') prevents reliable optimize rewriting")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in {"__getattribute__", "__getattr__", "methodcaller", "attrgetter"}:
            raise _InvalidProgram(
                f"dynamic {node.attr} reference prevents reliable optimize rewriting"
            )
        if node.attr == "optimize":
            raise _InvalidProgram(
                "a .optimize reference not called directly cannot be reliably limited"
            )
        if node.attr.lower().startswith("optimize"):
            raise _InvalidProgram(
                f"unsupported solver method {node.attr!r}; only direct .optimize(...) can be limited"
            )
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if node.value == "optimize":
            raise _InvalidProgram(
                "dynamic optimize lookup prevents reliable optimize rewriting"
            )


def _rewrite_optimize_calls(
    source: str,
    tree: ast.Module,
    *,
    solver_time_limit: float,
    solver_threads: int,
) -> tuple[str, bool]:
    collector = _OptimizeCallCollector()
    collector.visit(tree)
    conflicts = collector.names.intersection(_HELPER_NAMES)
    if conflicts:
        raise _InvalidProgram(
            "generated code conflicts with injected helper name(s): "
            + ", ".join(sorted(conflicts))
        )
    if not collector.calls:
        return source, False

    source_bytes = source.encode("utf-8")
    lines = source_bytes.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    replacements: list[tuple[int, int, bytes]] = []
    for node in collector.calls:
        assert isinstance(node.func, ast.Attribute)
        helper_call = ast.Call(
            func=ast.Name(id="__rebuttal_optimize__", ctx=ast.Load()),
            args=[node.func.value, *node.args],
            keywords=node.keywords,
        )
        replacement = ast.unparse(helper_call).encode("utf-8")
        start = line_offsets[node.lineno - 1] + node.col_offset
        end = line_offsets[node.end_lineno - 1] + node.end_col_offset
        replacements.append((start, end, replacement))

    rewritten = source_bytes
    for start, end, replacement in sorted(replacements, reverse=True):
        rewritten = rewritten[:start] + replacement + rewritten[end:]

    prelude = f'''import atexit as __rebuttal_atexit
import json as __rebuttal_json
import math as __rebuttal_math

__rebuttal_solver_status = None
__rebuttal_optimize_calls = []

def __rebuttal_optimize__(model, *args, **kwargs):
    global __rebuttal_solver_status
    model.Params.TimeLimit = {solver_time_limit!r}
    model.Params.Threads = {solver_threads!r}
    result = None
    optimize_completed = False
    try:
        result = model.optimize(*args, **kwargs)
        optimize_completed = True
    finally:
        runtime = None
        if optimize_completed:
            try:
                candidate = float(model.Runtime)
                if __rebuttal_math.isfinite(candidate) and candidate >= 0:
                    runtime = candidate
            except Exception:
                pass
        try:
            status = model.Status
        except Exception:
            status = None
        try:
            sol_count = model.SolCount
        except Exception:
            sol_count = None
        __rebuttal_optimize_calls.append({{
            "runtime_seconds": runtime,
            "status": status,
            "sol_count": sol_count,
        }})
        known_runtime = [call["runtime_seconds"] for call in __rebuttal_optimize_calls
                         if call["runtime_seconds"] is not None]
        missing_runtime_count = len(__rebuttal_optimize_calls) - len(known_runtime)
        __rebuttal_solver_status = {{
            "status": status,
            "sol_count": sol_count,
            "optimize_calls": __rebuttal_optimize_calls,
            "optimize_call_count": len(__rebuttal_optimize_calls),
            "known_solver_runtime_seconds": sum(known_runtime),
            "solver_runtime_seconds": sum(known_runtime) if not missing_runtime_count else None,
            "missing_runtime_count": missing_runtime_count,
        }}
    return result

def __rebuttal_emit_solver_status():
    if __rebuttal_solver_status is not None:
        print("SolverStatus=JSON" + __rebuttal_json.dumps(__rebuttal_solver_status))

__rebuttal_atexit.register(__rebuttal_emit_solver_status)

'''
    insertion_line = 0
    body_index = 0
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        insertion_line = body[0].end_lineno or body[0].lineno
        body_index = 1
    while body_index < len(body):
        statement = body[body_index]
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            insertion_line = statement.end_lineno or statement.lineno
            body_index += 1
            continue
        break
    insert_at = line_offsets[insertion_line]
    head, tail = rewritten[:insert_at], rewritten[insert_at:]
    separator = b"" if not head or head.endswith((b"\n", b"\r")) else b"\n"
    return (head + separator + prelude.encode("utf-8") + tail).decode("utf-8"), True


def _child_environment() -> dict[str, str]:
    env = os.environ.copy()
    for name in list(env):
        if re.search(r"(?i)(api[_-]?(key|token)|secret)", name):
            env.pop(name, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _result(
    *,
    status: str,
    returncode: int | None,
    elapsed_seconds: float,
    stdout: str,
    stderr: str,
    solver_status: dict[str, Any] | None,
    solver_limits_applied: bool,
    message: str | None = None,
) -> dict[str, Any]:
    result = {
        "status": status,
        "returncode": returncode,
        "elapsed_seconds": elapsed_seconds,
        "stdout": stdout,
        "stderr": stderr,
        "solver_status": solver_status,
        "solver_limits_applied": solver_limits_applied,
    }
    if message is not None:
        result["message"] = message
    return result


async def execute_code(
    code: str,
    directory: Path,
    python_executable: str,
    timeout_seconds: float,
    solver_time_limit: float,
    solver_threads: int = 1,
) -> dict[str, Any]:
    """Save and run extracted Python code in a bounded subprocess.

    `code` is source text, normally returned by `extract_code`. The original
    source is kept in generated.py; executed.py contains the optimize wrapper
    when direct solver calls can be rewritten reliably.
    """
    workdir = Path(directory)
    workdir.mkdir(parents=True, exist_ok=True)
    generated_path = workdir / "generated.py"
    executed_path = workdir / "executed.py"
    stdout_path = workdir / "stdout.txt"
    stderr_path = workdir / "stderr.txt"

    if not isinstance(code, str):
        code = str(code)
    generated_path.write_text(code, encoding="utf-8", newline="\n")
    (workdir / "stdout.txt").write_text("", encoding="utf-8")
    (workdir / "stderr.txt").write_text("", encoding="utf-8")
    started = time.monotonic()

    try:
        timeout_seconds = float(timeout_seconds)
        solver_time_limit = float(solver_time_limit)
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise _InvalidProgram("timeout_seconds must be finite and greater than zero")
        if not math.isfinite(solver_time_limit) or solver_time_limit <= 0:
            raise _InvalidProgram("solver_time_limit must be finite and greater than zero")
        if isinstance(solver_threads, bool) or not isinstance(solver_threads, int) or solver_threads < 1:
            raise _InvalidProgram("solver_threads must be a positive integer")

        source = code.removeprefix("\ufeff")
        tree = _parse_source(source)
        executed, limits_applied = _rewrite_optimize_calls(
            source,
            tree,
            solver_time_limit=solver_time_limit,
            solver_threads=solver_threads,
        )
        try:
            ast.parse(executed, filename="executed.py", mode="exec")
        except SyntaxError as exc:
            raise _InvalidProgram(f"rewritten program syntax error: {exc}") from None
        executed_path.write_text(executed, encoding="utf-8", newline="\n")
    except _InvalidProgram as exc:
        executed_path.write_text(code, encoding="utf-8", newline="\n")
        elapsed = time.monotonic() - started
        message = str(exc)
        stderr_path.write_text(message + "\n", encoding="utf-8", newline="\n")
        return _result(
            status="syntax_error" if message.startswith("Python syntax error:") else "invalid_program",
            returncode=None,
            elapsed_seconds=elapsed,
            stdout="",
            stderr=message + "\n",
            solver_status=None,
            solver_limits_applied=False,
            message=message,
        )

    try:
        process = await asyncio.create_subprocess_exec(
            python_executable,
            str(executed_path),
            cwd=str(workdir),
            env=_child_environment(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        elapsed = time.monotonic() - started
        message = f"could not start Python subprocess: {exc}"
        stderr_path.write_text(message + "\n", encoding="utf-8", newline="\n")
        return _result(
            status="invalid_program",
            returncode=None,
            elapsed_seconds=elapsed,
            stdout="",
            stderr=message + "\n",
            solver_status=None,
            solver_limits_applied=False,
            message=message,
        )

    timed_out = False
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(), timeout=timeout_seconds
        )
    except asyncio.TimeoutError:
        timed_out = True
        process.kill()
        stdout_bytes, stderr_bytes = await process.communicate()

    elapsed = time.monotonic() - started
    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    stdout_path.write_text(stdout, encoding="utf-8", newline="\n")
    stderr_path.write_text(stderr, encoding="utf-8", newline="\n")

    solver_status = None
    marker_lines = [
        line[len(_SOLVER_STATUS_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(_SOLVER_STATUS_PREFIX)
    ]
    if marker_lines:
        if len(marker_lines) == 1:
            try:
                parsed = json.loads(marker_lines[0])
                if isinstance(parsed, dict) and {"status", "sol_count"}.issubset(parsed):
                    solver_status = parsed
            except json.JSONDecodeError:
                pass

    if timed_out:
        status = "timeout"
    elif process.returncode != 0:
        status = "invalid_program"
    elif limits_applied and (len(marker_lines) != 1 or solver_status is None):
        status = "invalid_program"
        message = "solver status marker missing or ambiguous after optimize call"
        stderr = (stderr + message + "\n")
        stderr_path.write_text(stderr, encoding="utf-8", newline="\n")
        return _result(
            status=status,
            returncode=process.returncode,
            elapsed_seconds=elapsed,
            stdout=stdout,
            stderr=stderr,
            solver_status=None,
            solver_limits_applied=True,
            message=message,
        )
    else:
        status = "executed"

    return _result(
        status=status,
        returncode=process.returncode,
        elapsed_seconds=elapsed,
        stdout=stdout,
        stderr=stderr,
        solver_status=solver_status,
        solver_limits_applied=limits_applied,
    )
