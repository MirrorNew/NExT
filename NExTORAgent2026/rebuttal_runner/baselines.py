"""Adapters for the original CoE, OptiMUS, and OR-LLM-Agent sources.

The legacy projects are deliberately loaded from source ASTs at call time. This
avoids their module-level API clients, CLI parsing, and credentials while leaving
the selected algorithm functions and prompt templates unchanged.
"""
from __future__ import annotations

import ast
import asyncio
from contextlib import contextmanager
import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import threading
from typing import Any, Callable

from .execution import execute_code, extract_code
from .models import CallBudgetExceeded, LoggedModel, ModelCallError
from .scoring import AnswerParseError, parse_answer


_ROOT_NAMES = {
    "coe": "Chain-of-Experts-main",
    "optimus": "OptiMUS-main",
    "or_llm_agent": "or_llm_agent-main",
}
class BaselineUnavailable(RuntimeError):
    """The requested source snapshot or required dependency is unavailable."""


class RepairBudgetExceeded(RuntimeError):
    """The baseline attempted another repair after the shared repair limit."""


class _CallRecorder:
    def __init__(self, model: LoggedModel, case_id: str, run_id: str, maximum: int,
                 output_dir: Path):
        self.model = model
        self.case_id = case_id
        self.run_id = run_id
        self.maximum = maximum
        self.output_dir = output_dir / "model_calls"
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.exhaustion_error: str | None = None

    async def complete(self, messages: list[dict[str, str]], stage: str) -> str:
        with self._lock:
            if len(self.calls) >= self.maximum:
                self.exhaustion_error = f"model-call budget exhausted ({self.maximum})"
                raise CallBudgetExceeded(self.exhaustion_error)
            ordinal = len(self.calls) + 1
            marker = {"ordinal": ordinal, "stage": stage, "status": "started",
                      "messages": [dict(m) for m in messages]}
            self.calls.append(marker)
        call_dir = self.output_dir / f"call_{ordinal:04d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        marker["call_dir"] = str(Path("model_calls") / f"call_{ordinal:04d}")
        _write_json(call_dir / "prompt.json", messages)
        _write_json(call_dir / "call.json", marker)
        try:
            result = await self.model.complete(
                messages, stage=stage, case_id=self.case_id, run_id=self.run_id
            )
            marker.update(status="completed", call_id=result["call_id"],
                          returned_model=result.get("returned_model"),
                          usage=result.get("usage"),
                          elapsed_seconds=result.get("elapsed_seconds"),
                          response=result.get("content"))
            _write_json(call_dir / "response.json", {
                "call_id": result.get("call_id"), "returned_model": result.get("returned_model"),
                "usage": result.get("usage"), "elapsed_seconds": result.get("elapsed_seconds"),
                "content": result.get("content")})
            _write_json(call_dir / "call.json", marker)
            return result["content"]
        except ModelCallError as exc:
            marker.update(status="failed", call_id=exc.record.get("call_id"),
                          error=str(exc))
            _write_json(call_dir / "call.json", marker)
            raise
        except Exception as exc:
            marker.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            _write_json(call_dir / "call.json", marker)
            raise

    def complete_from_thread(self, loop: asyncio.AbstractEventLoop,
                             messages: list[dict[str, str]], stage: str) -> str:
        future = asyncio.run_coroutine_threadsafe(self.complete(messages, stage), loop)
        return future.result()

    @property
    def call_ids(self) -> list[str]:
        return [c["call_id"] for c in self.calls if c.get("call_id")]


def _resolve_root(name: str, source_root: str | Path) -> Path:
    root = Path(source_root).resolve()
    expected = _ROOT_NAMES[name]
    if root.name.casefold() == expected.casefold():
        candidate = root
    else:
        candidate = root / expected
    if not candidate.is_dir():
        raise BaselineUnavailable(f"source snapshot not found: {candidate}")
    return candidate


def source_fingerprint(name: str, source_root: str | Path) -> dict[str, Any]:
    """Identify a local baseline snapshot without importing or executing it."""
    root = _resolve_root(name, source_root)
    if name == "coe":
        paths = [root / p for p in ("main.py", "comment.py", "comment_pool.py",
                                    "conductor.py", "reducer.py")]
        paths.extend(sorted((root / "experts").glob("*.py")))
    elif name == "optimus":
        paths = [root / p for p in ("parameters.py", "objective.py", "constraint.py",
                                    "constraint_model.py", "objective_model.py", "target_code.py",
                                    "generate_code.py", "execute_code.py", "utils.py")]
    else:
        paths = [root / "or_llm_eval_async.py", root / "utils.py"]
    records = []
    for path in sorted(p for p in paths if p.is_file()):
        rel = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append((rel, digest))
    joined = "\n".join(f"{p}:{h}" for p, h in records).encode("utf-8")
    return {"root": str(root), "git_commit": None,
            "sha256": hashlib.sha256(joined).hexdigest(),
            "python_files": len(records)}


def baseline_availability(name: str, source_root: str | Path) -> dict[str, Any]:
    """Read-only source/dependency preflight; does not import baseline code."""
    try:
        root = _resolve_root(name, source_root)
        required = {
            "coe": ("main.py", "comment.py", "comment_pool.py", "conductor.py", "reducer.py",
                    "experts/base_expert.py", "experts/modeling_expert.py",
                    "experts/programming_expert.py", "experts/modeling_knowledge_supplement_expert.py",
                    "experts/parameter_extractor.py", "experts/code_reviewer.py",
                    "experts/programming_example_provider.py", "experts/terminology_interpreter.py",
                    "experts/lp_file_generator.py"),
            "optimus": ("parameters.py", "objective.py", "constraint.py", "constraint_model.py",
                        "objective_model.py", "target_code.py", "generate_code.py", "execute_code.py", "utils.py"),
            "or_llm_agent": ("or_llm_eval_async.py", "utils.py"),
        }[name]
        missing = [item for item in required if not (root / item).is_file()]
        package_names = {"coe": ("langchain", "langchain-core"),
                         "optimus": ("numpy", "gurobipy"),
                         "or_llm_agent": ()}[name]
        dependencies = {}
        for package in package_names:
            try:
                dependencies[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                dependencies[package] = None
        missing_dependencies = [p for p, version in dependencies.items() if version is None]
        return {"available": not missing and not missing_dependencies,
                "root": str(root), "missing": missing,
                "missing_dependencies": missing_dependencies,
                "dependencies": dependencies,
                "source": source_fingerprint(name, root) if not missing else None}
    except (KeyError, BaselineUnavailable) as exc:
        return {"available": False, "error": str(exc), "missing": []}


class _DropEmbeddedProviderArgs(ast.NodeTransformer):
    """Remove legacy hardcoded endpoint/key expressions before compiling a class."""
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "ChatOpenAI":
            node.keywords = [kw for kw in node.keywords if kw.arg not in {"api_key", "base_url"}]
        return node


def _compile_defs(path: Path, names: set[str], namespace: dict[str, Any], *,
                  literal_names: set[str] | None = None,
                  drop_embedded_provider_args: bool = False) -> None:
    """Compile only selected definitions and literal prompt/constants."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            body.append(node)
        elif literal_names and isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id in literal_names for t in targets):
                body.append(node)
    selected = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    if drop_embedded_provider_args:
        selected = ast.fix_missing_locations(_DropEmbeddedProviderArgs().visit(selected))
    exec(compile(selected, str(path), "exec"), namespace)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def _public_inputs(case: dict[str, Any]) -> tuple[str, dict[str, Any] | None, dict[str, Any]]:
    question = case.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("case_public.question must be non-empty text")
    answer_spec = case.get("answer_spec")
    if not isinstance(answer_spec, dict):
        raise ValueError("case_public.answer_spec must be an object")
    math_input = case.get("math_input")
    if math_input is not None and not isinstance(math_input, dict):
        raise ValueError("case_public.math_input must be an object or None")
    return question, math_input, answer_spec


def _messages_case(question: str, math_input: dict[str, Any] | None,
                   level: str, ned_advice: Any) -> str:
    if level == "end_to_end":
        return question
    pieces = [question]
    if math_input is not None:
        pieces.append("\n\nFrozen mathematical model (authoritative for this coding run):\n" +
                      json.dumps(math_input, ensure_ascii=False))
    if ned_advice is not None:
        pieces.append("\n\nStructural advice supplied for this run may guide the implementation. "
                      "It does not change the mathematical model:\n" +
                      json.dumps(ned_advice, ensure_ascii=False))
    return "".join(pieces)


def _empty_scaffold(answer_spec: dict[str, Any]) -> str:
    # Runtime import avoids a module cycle when the runner imports this adapter.
    from .runner import CODE_SYSTEM
    return """```python
import gurobipy as gp
from gurobipy import GRB

def solve():
    model = gp.Model()
    # Implement the optimization problem described above.
    model.optimize()
    if model.SolCount:
        print(f\"FinalAnswer=【{model.ObjVal}】\")

if __name__ == '__main__':
    solve()
```

Shared solver and output requirements:
""" + CODE_SYSTEM + "\nAnswer format for this run:\n" + json.dumps(answer_spec, ensure_ascii=False)


def _fragment_code_contract(answer_spec: dict[str, Any]) -> str:
    return ("Use Gurobi's supported nonlinear modeling API. For NLExpr comparisons such as "
            "NLExpr >= constant, introduce legal resultant-variable equalities, bounds, and "
            "linear relations; give resultants appropriate bounds, including negative bounds "
            "when required. Preserve the original domains and relations. The generated full "
            "program reports the requested optimized scalar objective value. Answer format "
            "for this run: " + json.dumps(answer_spec, ensure_ascii=False))


@contextmanager
def _zero_legacy_callback():
    """Keep legacy callback scopes intact; authoritative usage is LoggedModel's ledger."""
    class Usage:
        prompt_tokens = completion_tokens = total_tokens = 0
    yield Usage()


def _coe_callable(root: Path, recorder: _CallRecorder, loop: asyncio.AbstractEventLoop,
                  case_id: str):
    try:
        from langchain import LLMChain, PromptTemplate
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
    except ImportError as exc:
        raise BaselineUnavailable(f"CoE requires its LangChain dependencies: {exc}") from None

    class LoggedChatModel(BaseChatModel):
        model_name: str
        temperature: float = 0
        max_tokens: int | None = None
        recorder: Any
        loop: Any
        case_id: str
        _stage: str = "baseline_coe_expert"

        @property
        def _llm_type(self) -> str:
            return "rebuttal-logged-chat-model"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {"model_name": self.model_name, "temperature": self.temperature}

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            converted = []
            for message in messages:
                if isinstance(message, SystemMessage):
                    role = "system"
                elif isinstance(message, HumanMessage):
                    role = "user"
                elif isinstance(message, AIMessage):
                    role = "assistant"
                else:
                    role = "user"
                converted.append({"role": role, "content": str(message.content)})
            stage = getattr(self, "_stage", "baseline_coe_expert")
            content = self.recorder.complete_from_thread(self.loop, converted, stage)
            generation = ChatGeneration(message=AIMessage(content=content))
            return ChatResult(generations=[generation])

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            converted = []
            for message in messages:
                role = "system" if isinstance(message, SystemMessage) else (
                    "assistant" if isinstance(message, AIMessage) else "user")
                converted.append({"role": role, "content": str(message.content)})
            content = await self.recorder.complete(converted, "baseline_coe_expert")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def chat_factory(model_name: str, temperature: float = 0, **kwargs):
        return LoggedChatModel(model_name=model_name, temperature=temperature,
                               recorder=recorder, loop=loop, case_id=case_id)

    base_ns = {"PromptTemplate": PromptTemplate, "LLMChain": LLMChain,
               "ChatOpenAI": chat_factory, "get_openai_callback": _zero_legacy_callback,
               "defaultdict": __import__("collections").defaultdict}
    package = root / "experts"
    _compile_defs(package / "base_expert.py", {"BaseExpert"}, base_ns,
                  drop_embedded_provider_args=True)
    expert_names = {
        "ModelingExpert": "modeling_expert.py",
        "ProgrammingExpert": "programming_expert.py",
        "ModelingKnowledgeSupplementExpert": "modeling_knowledge_supplement_expert.py",
        "ParameterExtractor": "parameter_extractor.py",
        "CodeReviewer": "code_reviewer.py",
        "ProgrammingExampleProvider": "programming_example_provider.py",
        "TerminologyInterpreter": "terminology_interpreter.py",
        "LPFileGenerator": "lp_file_generator.py",
    }
    expert_ns = dict(base_ns)
    expert_ns["BaseExpert"] = base_ns["BaseExpert"]
    for name, filename in expert_names.items():
        _compile_defs(package / filename, {name}, expert_ns)
    comment_ns: dict[str, Any] = {}
    _compile_defs(root / "comment.py", {"Comment"}, comment_ns)
    pool_ns = {"Comment": comment_ns["Comment"]}
    _compile_defs(root / "comment_pool.py", {"CommentPool"}, pool_ns)
    conductor_ns = dict(base_ns, BaseExpert=base_ns["BaseExpert"], random=__import__("random"))
    _compile_defs(root / "conductor.py", {"Conductor"}, conductor_ns)
    reducer_ns = dict(base_ns, BaseExpert=base_ns["BaseExpert"])
    _compile_defs(root / "reducer.py", {"Reducer"}, reducer_ns)

    class UnusedEvaluator:
        def __init__(self, *_args, **_kwargs):
            pass

    class NumpyShim:
        @staticmethod
        def ones(shape):
            rows, cols = shape
            return [[1 for _ in range(cols)] for _ in range(rows)]

    def extract_code_from_string(text):
        try:
            return extract_code(text)
        except (ValueError, TypeError):
            return text

    main_ns = {
        "os": __import__("os"), "json": json, "np": NumpyShim,
        "Comment": comment_ns["Comment"], "CommentPool": pool_ns["CommentPool"],
        "Conductor": conductor_ns["Conductor"], "Reducer": reducer_ns["Reducer"],
        "Evaluator": UnusedEvaluator,
        "extract_code_from_string": extract_code_from_string,
        **{key: expert_ns[key] for key in expert_names},
    }
    _compile_defs(root / "main.py", {"chain_of_experts"}, main_ns)
    return main_ns["chain_of_experts"], expert_ns["ProgrammingExpert"]


def _attempt_dir(base: Path, number: int) -> Path:
    path = base / f"attempt_{number:02d}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_execution(execution: dict[str, Any], answer_spec: dict[str, Any]) -> Any:
    if not _solver_execution_valid(execution):
        return None
    try:
        return parse_answer(execution.get("stdout", ""))
    except AnswerParseError:
        pass
    # OptiMUS's original code prints objective text without a FinalAnswer marker.
    fields = answer_spec.get("fields", [])
    if fields == ["objective_value"]:
        stdout = execution.get("stdout", "")
        matches = re.findall(r"(?im)^\s*(?:Optimal Objective Value|Optimal Revenue|Best objective|Optimal objective):\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", stdout)
        if len(matches) == 1:
            try:
                return float(matches[0])
            except ValueError:
                return None
    return None


def _solver_execution_valid(execution: dict[str, Any]) -> bool:
    solver = execution.get("solver_status") or {}
    return (execution.get("status") == "executed" and execution.get("returncode") == 0
            and execution.get("solver_limits_applied") is True
            and isinstance(solver.get("sol_count"), int) and solver.get("sol_count", 0) > 0)


async def _run_coe(root: Path, case: dict[str, Any], model: LoggedModel,
                   recorder: _CallRecorder, output: Path, *, level: str,
                   ned_advice: Any, python_executable: str, process_timeout: float,
                   solver_time_limit: float, solver_threads: int,
                   max_repairs: int) -> dict[str, Any]:
    question, math_input, answer_spec = _public_inputs(case)
    loop = asyncio.get_running_loop()
    chain, programming_expert = _coe_callable(root, recorder, loop, str(case["case_id"]))
    problem = {"description": question, "code_example": _empty_scaffold(answer_spec)}
    trace: dict[str, Any] = {"level": level,
                             "algorithm": "chain_of_experts" if level == "end_to_end" else "coe_programming_expert",
                             "stages": (["conductor_experts_reducer"] if level == "end_to_end"
                                        else ["programming_expert"]),
                             "max_collaborate_nums": 3 if level == "end_to_end" else 0,
                             "max_trials": 1 if level == "end_to_end" else 0,
                             "reflection_effective": False,
                             "repair_policy": "no_internal_repairs_reflection_disabled",
                             "fixed_math_input_injected": level != "end_to_end"}
    response = None
    status = "generated"
    code = ""
    execution = None
    try:
        if level == "end_to_end":
            response = await asyncio.to_thread(chain, problem, 3, model.config.model, False, 1)
        else:
            advice_text = "Authoritative frozen mathematical model:\n" + json.dumps(
                math_input, ensure_ascii=False)
            if ned_advice is not None:
                advice_text += "\nSupplied structural advice (does not change the model):\n" + json.dumps(
                    ned_advice, ensure_ascii=False)
            class FixedModelComment:
                def get_current_comment_text(self):
                    return advice_text
            expert = programming_expert(model.config.model)
            response = await asyncio.to_thread(expert.forward, problem, FixedModelComment())
        code = extract_code(str(response))
        execution = await execute_code(code, _attempt_dir(output, 0), python_executable,
                                       process_timeout, solver_time_limit, solver_threads)
        prediction = _parse_execution(execution, answer_spec)
        status = "completed" if prediction is not None else "execution_failed"
    except CallBudgetExceeded as exc:
        prediction, status = None, "call_budget_exhausted"
        trace["error"] = str(exc)
    except ModelCallError as exc:
        prediction, status = None, "model_call_failed"
        trace["error"] = str(exc)
    except Exception as exc:
        prediction, status = None, "baseline_failed"
        trace["error"] = f"{type(exc).__name__}: {exc}"
    return {"status": status, "prediction": prediction, "attempts": [
                {"attempt": 0, "response": response, "generated_code": code,
                 "execution": execution}], "execution": execution,
            "call_ids": recorder.call_ids, "baseline_trace": trace}


def _or_agent_functions(root: Path, namespace: dict[str, Any]) -> dict[str, Callable]:
    utility_path = root / "utils.py"
    utility_ns = {"re": re}
    _compile_defs(utility_path, {"is_number_string", "extract_best_objective"}, utility_ns)
    namespace["is_number_string"] = utility_ns["is_number_string"]
    namespace["extract_best_objective"] = utility_ns["extract_best_objective"]
    path = root / "or_llm_eval_async.py"
    _compile_defs(path, {"async_generate_or_code_solver", "async_or_llm_agent"}, namespace)
    return namespace


def _or_attempt_records(generated: list[dict[str, Any]], answer_spec: dict[str, Any]) -> list[dict[str, Any]]:
    code_items = [item for item in generated
                  if item.get("generated_code") is not None or item.get("execution") is not None]
    attempts = []
    for index, item in enumerate(code_items):
        execution = item.get("execution")
        prediction = _parse_execution(execution or {}, answer_spec)
        attempts.append({"attempt": index, "prompt": item.get("messages"),
                         "response": item.get("response"),
                         "generated_code": item.get("generated_code"),
                         "execution": execution,
                         "baseline_result": item.get("baseline_result"),
                         "prediction": prediction})
    return attempts


async def _run_or_agent(root: Path, case: dict[str, Any], recorder: _CallRecorder,
                        output: Path, *, level: str, ned_advice: Any,
                        python_executable: str, process_timeout: float,
                        solver_time_limit: float, solver_threads: int,
                        max_repairs: int) -> dict[str, Any]:
    question, math_input, answer_spec = _public_inputs(case)
    generated: list[dict[str, Any]] = []
    current_attempt = 0
    repair_count = 0
    failed_execution_pending = False
    injected_math_stage = False

    async def query(messages, model_name="o3-mini", temperature=0.2):
        nonlocal repair_count, failed_execution_pending, injected_math_stage
        if level != "end_to_end" and not injected_math_stage:
            injected_math_stage = True
            frozen = {"authoritative_math_input": math_input}
            if ned_advice is not None:
                frozen["supplied_ned_advice"] = ned_advice
            return json.dumps(frozen, ensure_ascii=False)
        prepared = [dict(m) for m in messages]
        if generated or level != "end_to_end":
            from .runner import CODE_SYSTEM
            prepared.insert(0, {"role": "system", "content": CODE_SYSTEM +
                                "\nAnswer format for this run:\n" +
                                json.dumps(answer_spec, ensure_ascii=False)})
        stage = "baseline_or_llm_agent"
        if failed_execution_pending:
            if repair_count >= max_repairs:
                raise RepairBudgetExceeded(f"repair budget exhausted ({max_repairs})")
            repair_count += 1
            failed_execution_pending = False
            stage = "baseline_or_llm_agent_repair"
        content = await recorder.complete(prepared, stage)
        generated.append({"messages": prepared, "response": content, "stage": stage})
        return content

    async def bounded_execute(text_content, timeout=None):
        nonlocal current_attempt, failed_execution_pending
        blocks = re.findall(r"```python\s*([\s\S]*?)```", text_content)
        if len(blocks) != 1:
            current_attempt += 1
            failed_execution_pending = True
            generated[-1]["execution"] = {"status": "invalid_output", "returncode": None,
                                           "stdout": "", "stderr": (
                                               "Expected exactly one Python code block; "
                                               f"found {len(blocks)}")}
            return False, generated[-1]["execution"]["stderr"]
        try:
            code = extract_code(blocks[0])
        except Exception as exc:
            current_attempt += 1
            failed_execution_pending = True
            generated[-1]["execution"] = {"status": "invalid_output", "returncode": None,
                                           "stdout": "", "stderr": f"code extraction failed: {exc}"}
            return False, generated[-1]["execution"]["stderr"]
        attempt_dir = _attempt_dir(output, current_attempt)
        current_attempt += 1
        result = await execute_code(code, attempt_dir, python_executable,
                                    process_timeout, solver_time_limit, solver_threads)
        generated[-1]["generated_code"] = code
        generated[-1]["execution"] = result
        prediction = _parse_execution(result, answer_spec)
        failed_execution_pending = prediction is None
        best_objective = functions.get("extract_best_objective", lambda _text: None)(
            result.get("stdout", "")) if _solver_execution_valid(result) else None
        generated[-1]["baseline_result"] = (str(best_objective)
                                             if best_objective is not None else
                                             str(prediction) if isinstance(prediction, (int, float)) else None)
        if not failed_execution_pending:
            return True, generated[-1]["baseline_result"] or "None"
        error = result.get("stderr", "") or result.get("stdout", "")
        if not _solver_execution_valid(result):
            error = (error + "\nNo bounded solver solution was recorded. "
                     "The code must call model.optimize() and report a solution.").strip()
        elif prediction is None:
            error = (error + "\nThe required FinalAnswer format is missing or invalid.").strip()
        return False, error

    namespace: dict[str, Any] = {"asyncio": asyncio, "copy": copy,
                                 "async_query_llm": query,
                                 "async_extract_and_execute_python_code": bounded_execute,
                                 "DEFAULT_SUBPROCESS_TIMEOUT": process_timeout}
    functions = _or_agent_functions(root, namespace)
    try:
        ok, result = await functions["async_or_llm_agent"](
            question,
            recorder.model.config.model, max_repairs + 1)
        # The original function returns only its score. Use the last generated program,
        # rather than accepting that score as a substitute for the runner's execution.
        attempts = []
        attempts = _or_attempt_records(generated, answer_spec)
        prediction = next((a["prediction"] for a in reversed(attempts)
                           if a.get("prediction") is not None), None)
        status = "completed" if prediction is not None else (
            "execution_failed" if generated else "no_generation")
        trace = {"algorithm": "async_or_llm_agent", "level": level,
                 "stages": (["math_model_generation", "code_generation",
                             "execution_feedback_repairs", "no_solution_recovery"]
                            if level == "end_to_end" else
                            ["fixed_math_input_injection", "code_generation",
                             "execution_feedback_repairs"]),
                 "math_input_authoritative": level != "end_to_end",
                 "baseline_internal_success": bool(ok),
                 "baseline_internal_result": result,
                 "repair_limit": max_repairs,
                 "internal_execute_calls": current_attempt}
        return {"status": status, "prediction": prediction, "attempts": attempts,
                "execution": attempts[-1].get("execution") if attempts else None,
                "call_ids": recorder.call_ids, "baseline_trace": trace}
    except CallBudgetExceeded as exc:
        attempts = _or_attempt_records(generated, answer_spec)
        return {"status": "call_budget_exhausted", "prediction": None,
                "attempts": attempts, "execution": attempts[-1].get("execution") if attempts else None,
                "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "async_or_llm_agent", "level": level,
                                   "error": str(exc), "internal_execute_calls": current_attempt}}
    except RepairBudgetExceeded as exc:
        attempts = _or_attempt_records(generated, answer_spec)
        return {"status": "repair_budget_exhausted", "prediction": None,
                "attempts": attempts, "execution": attempts[-1].get("execution") if attempts else None,
                "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "async_or_llm_agent", "level": level,
                                   "repair_limit": max_repairs, "repairs_used": repair_count,
                                   "error": str(exc), "internal_execute_calls": current_attempt}}
    except ModelCallError as exc:
        attempts = _or_attempt_records(generated, answer_spec)
        return {"status": "model_call_failed", "prediction": None, "attempts": attempts,
                "execution": attempts[-1].get("execution") if attempts else None,
                "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "async_or_llm_agent", "level": level,
                                   "error": str(exc)}}
    except Exception as exc:
        attempts = _or_attempt_records(generated, answer_spec)
        return {"status": "baseline_failed", "prediction": None, "attempts": attempts,
                "execution": attempts[-1].get("execution") if attempts else None,
                "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "async_or_llm_agent", "level": level,
                                   "error": f"{type(exc).__name__}: {exc}"}}


def _load_optimus(root: Path, get_response: Callable):
    """Build OptiMUS's staged functions from definitions without its client globals."""
    def utility_ns():
        ns: dict[str, Any] = {"os": __import__("os"), "json": json, "re": re,
                              "get_response": get_response}
        names = {"extract_json_from_end", "extract_json_from_end_backup", "extract_list_from_end",
                 "shape_string_to_list", "extract_equal_sign_closed"}
        _compile_defs(root / "utils.py", names, ns)
        return ns

    util = utility_ns()
    common = {"json": json, "re": re, "os": __import__("os"), "np": __import__("numpy"),
              **{k: v for k, v in util.items() if k in {
                  "extract_json_from_end", "extract_list_from_end", "extract_equal_sign_closed",
                  "shape_string_to_list"}}, "get_response": get_response}
    stages: dict[str, dict[str, Any]] = {}
    specs = {
        "parameters": ({"get_params", "extract_score"}, {"prompt_params", "prompt_params_q", "qs"}),
        "objective": ({"get_objective", "extract_objective"}, {"prompt_objective"}),
        "constraint": ({"get_constraints", "extract_score_constraint", "logic_check"}, {"prompt_constraints", "prompt_constraints_q", "prompt_constraints_redundant", "prompt_constraints_complete", "qs"}),
        "constraint_model": ({"get_constraint_formulations", "extract_formulation_from_end", "logic_check", "extract_score_constraint_model"}, {"qs"}),
        "objective_model": ({"get_objective_formulation"}, {"prompt_objective_model"}),
        "target_code": ({"get_codes", "extract_code_from_end"}, {"directions", "prompt_constraints_code", "prompt_objective_code", "qs"}),
        "generate_code": ({"generate_code", "get_var_code", "get_param_code"}, None),
        "execute_code": ({"execute_and_debug"}, {"debug_template"}),
    }
    for module_name, (functions, constants) in specs.items():
        ns = dict(common)
        if module_name in {"constraint_model", "objective_model", "target_code"}:
            ns["extract_json_from_end"] = util["extract_json_from_end"]
            ns["extract_list_from_end"] = util["extract_list_from_end"]
            ns["shape_string_to_list"] = util["shape_string_to_list"]
            ns["extract_equal_sign_closed"] = util["extract_equal_sign_closed"]
        # A small AST allowlist preserves all prompt constants that are plain strings/lists.
        constant_names = constants or set()
        _compile_defs(root / f"{module_name}.py", functions, ns,
                      literal_names=constant_names)
        stages[module_name] = ns
    return stages


def _optimus_state_from_math_input(question: str, math_input: dict[str, Any]):
    """Map supported frozen inputs to OptiMUS's codegen schema.

    Variables are scalar binary, integer, or continuous values, with bounds
    parsed only from explicit scalar inequalities, numeric intervals, or an
    explicit real-unbounded marker. The caller injects parsed bounds into the
    original OptiMUS addVar codegen only for coding runs. Parameters need
    identifier names and a Value; their type label is passed through.
    """
    params = {}
    for index, item in enumerate(math_input["Parameters_List"]):
        name = item["Name"]
        if not name.isidentifier() or "Value" not in item:
            raise BaselineUnavailable(
                f"OptiMUS coding adapter cannot represent parameter {name!r} at index {index}"
            )
        value = item["Value"]
        dims = []
        nested = value
        while isinstance(nested, list):
            dims.append(len(nested))
            nested = nested[0] if nested else None
        params[name] = {"shape": dims, "definition": item.get("Meaning", ""),
                        "type": str(item.get("Type", "float")).casefold(), "value": value}

    type_map = {"binary": "BINARY", "binary type": "BINARY", "bool": "BINARY", "boolean": "BINARY",
                "integer": "INTEGER", "integer type": "INTEGER", "int": "INTEGER", "intvar": "INTEGER",
                "continuous": "CONTINUOUS", "continuous type": "CONTINUOUS",
                "real": "CONTINUOUS", "real type": "CONTINUOUS", "float": "CONTINUOUS"}
    variables = {}
    variable_bounds = {}
    for index, item in enumerate(math_input["Variables_List"]):
        symbol = item["symbol"]
        vartype = type_map.get(str(item.get("Type", "")).strip().casefold())
        if not symbol.isidentifier() or vartype is None:
            raise BaselineUnavailable(
                f"OptiMUS coding adapter cannot represent variable {symbol!r} at index {index}"
            )
        range_keys = [key for key in item if str(key).strip().casefold() in {"range", "domain"}]
        range_values = [item[key] for key in range_keys]
        if len(range_values) > 1 and any(value != range_values[0] for value in range_values[1:]):
            raise BaselineUnavailable(f"OptiMUS coding adapter found conflicting range fields for {symbol!r}")
        range_text = range_values[0] if range_values else ""
        definition = str(item.get("Meaning", ""))
        if range_text:
            definition += ("; domain/range: " if definition else "domain/range: ") + str(range_text)
        bounds = _parse_scalar_variable_bounds(symbol, vartype, range_text)
        variable_bounds[symbol] = bounds
        # The runner's validated public schema does not carry array shape. Refuse
        # likely indexed variables instead of silently compiling them as scalars.
        if any(f"{symbol}[" in str(c["Mathematical_expressions"]) for c in math_input["Constraint_Table"]) or f"{symbol}[" in math_input["Objective"]["Mathematical_expressions"]:
            raise BaselineUnavailable(
                f"OptiMUS coding adapter requires scalar variables; indexed variable {symbol!r} needs explicit shape support"
            )
        variables[symbol] = {"shape": [], "type": vartype, "definition": definition}

    constraints = []
    for index, item in enumerate(math_input["Constraint_Table"]):
        expression = item["Mathematical_expressions"]
        name = item.get("name", f"constraint_{index}")
        constraints.append({"description": f"{name}: {expression}",
                            "formulation": expression, "code": None})
    objective_data = math_input["Objective"]
    objective_expression = objective_data["Mathematical_expressions"]
    objective_description = objective_data.get("Objective_sentence", objective_expression)
    objective = {"description": objective_description,
                 "formulation": objective_expression, "code": None}
    state = {"description": question, "parameters": params, "variables": variables,
             "constraints": constraints, "objective": objective}
    return params, variables, constraints, objective, state, variable_bounds


_SCALAR_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def _parse_scalar_variable_bounds(symbol: str, vartype: str, range_text: Any) -> tuple[float, float]:
    """Parse only unambiguous scalar numeric bounds; never infer from prose."""
    if vartype == "BINARY" and range_text in (None, ""):
        return 0.0, 1.0
    if not isinstance(range_text, str) or "\ufffd" in range_text:
        raise BaselineUnavailable(f"OptiMUS coding adapter cannot parse domain for {symbol!r}")
    text = re.sub(r"\s+", "", range_text).replace("−", "-")
    compact_symbol = re.escape(symbol)
    if vartype == "BINARY" and text.casefold() in {"binary", "{0,1}", f"{symbol}∈{{0,1}}".casefold()}:
        return 0.0, 1.0

    inf = r"(?:[+-]?(?:inf(?:inity)?|∞|unbounded))"
    endpoint = rf"(?:{_SCALAR_NUM}|{inf})"

    def number(raw: str) -> float:
        low = raw.casefold().replace("−", "-")
        if low in {"inf", "+inf", "infinity", "+infinity", "∞", "+∞", "unbounded"}:
            return math.inf
        if low in {"-inf", "-infinity", "-∞"}:
            return -math.inf
        try:
            value = float(raw)
        except ValueError as exc:
            raise BaselineUnavailable(f"Invalid scalar endpoint {raw!r} for {symbol!r}") from exc
        if not math.isfinite(value):
            raise BaselineUnavailable(f"Non-finite numeric endpoint {raw!r} for {symbol!r}")
        return value

    match = re.fullmatch(rf"([\[(])({endpoint}),({endpoint})([\])])", text)
    if match:
        open_mark, low_text, high_text, close_mark = match.groups()
        lb, ub = number(low_text), number(high_text)
        if (open_mark == "(" and math.isfinite(lb)) or (close_mark == ")" and math.isfinite(ub)):
            raise BaselineUnavailable(f"Open finite bound is unsupported for {symbol!r}: {range_text!r}")
    else:
        match = re.fullmatch(rf"{compact_symbol}=({_SCALAR_NUM})", text)
        if match:
            lb = ub = number(match.group(1))
        else:
            match = re.fullmatch(rf"({endpoint})(<=|≤){compact_symbol}(<=|≤)({endpoint})", text)
            if match and match.group(2) == match.group(3):
                lb, ub = number(match.group(1)), number(match.group(4))
            else:
                match = re.fullmatch(rf"{compact_symbol}(>=|≥)({endpoint})", text)
                if match:
                    lb, ub = number(match.group(2)), math.inf
                else:
                    match = re.fullmatch(rf"{compact_symbol}(<=|≤)({endpoint})", text)
                    if match:
                        lb, ub = -math.inf, number(match.group(2))
                    else:
                        match = re.fullmatch(rf"({endpoint})(<=|≤){compact_symbol}", text)
                        if match:
                            lb, ub = number(match.group(1)), math.inf
                        else:
                            match = re.fullmatch(rf"({endpoint})(>=|≥){compact_symbol}", text)
                            if match:
                                lb, ub = -math.inf, number(match.group(1))
                            elif (re.fullmatch(rf"{compact_symbol}(?:∈|in)(?:ℝ|[Rr])", text)
                                  or text.casefold() in {"unbounded", "realunbounded", "unboundedreal", "allreal", "(-∞,∞)", "(-∞,+∞)", "(-inf,inf)", "(-inf,+inf)", "(-infinity,infinity)", "(-infinity,+infinity)", "ℝ"}):
                                lb, ub = -math.inf, math.inf
                            else:
                                raise BaselineUnavailable(
                                    f"OptiMUS coding adapter cannot uniquely parse scalar domain {range_text!r} for {symbol!r}"
                                )
    if lb > ub or lb == math.inf or ub == -math.inf or (vartype == "BINARY" and (lb, ub) != (0.0, 1.0)):
        raise BaselineUnavailable(f"OptiMUS coding adapter cannot preserve domain {range_text!r} for {symbol!r}")
    return lb, ub


def baseline_coding_support(name: str, math_input: Any) -> dict[str, Any]:
    """Offline preflight for a baseline's frozen-math coding adapter.

    This performs no source loading, model calls, or solver calls. Other
    baselines rely on the runner's validated schema and require only that the
    frozen math input is present. OptiMUS additionally requires lossless
    mapping into its scalar code-generation schema.
    """
    aliases = {"or-llm-agent": "or_llm_agent", "or_llm": "or_llm_agent",
               "chain-of-experts": "coe", "chain_of_experts": "coe",
               "optimus": "optimus"}
    key = aliases.get(name.strip().casefold().replace(" ", "_"),
                      name.strip().casefold().replace(" ", "_"))
    if key not in _ROOT_NAMES:
        return {"supported": False, "reason": f"unknown baseline {name!r}"}
    if math_input is None:
        return {"supported": False, "reason": "frozen math_input is missing"}
    if not isinstance(math_input, dict):
        return {"supported": False, "reason": "math_input must be an object"}
    if key != "optimus":
        return {"supported": True}
    try:
        _optimus_state_from_math_input("", math_input)
    except (BaselineUnavailable, KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        return {"supported": False, "reason": str(exc)}
    return {"supported": True}


async def _run_optimus(root: Path, case: dict[str, Any], recorder: _CallRecorder,
                       output: Path, *, level: str, ned_advice: Any,
                       python_executable: str, process_timeout: float,
                       solver_time_limit: float, solver_threads: int,
                       max_repairs: int) -> dict[str, Any]:
    question, math_input, answer_spec = _public_inputs(case)
    loop = asyncio.get_running_loop()
    attempts: list[dict[str, Any]] = []
    execution = None
    state = None

    def sync_response(prompt: str, model: str | None = None) -> str:
        text = prompt
        if level != "end_to_end" and math_input is not None:
            text += "\n\nFrozen mathematical model (authoritative):\n" + json.dumps(math_input, ensure_ascii=False)
            if ned_advice is not None:
                text += "\n\nStructural advice (does not change the model):\n" + json.dumps(ned_advice, ensure_ascii=False)
        return recorder.complete_from_thread(loop, [{"role": "user", "content": text}],
                                             "baseline_optimus_stage")

    try:
        stages = _load_optimus(root, sync_response)
        desc = _messages_case(question, math_input, level, ned_advice)
        if level != "end_to_end":
            params, variables, constraints, objective, state, variable_bounds = _optimus_state_from_math_input(
                desc, math_input)
            original_get_var_code = stages["generate_code"]["get_var_code"]

            def bounded_get_var_code(symbol, shape, var_type, definition, solver="gurobipy"):
                code = original_get_var_code(symbol, shape, var_type, definition, solver)
                if shape != [] or symbol not in variable_bounds:
                    raise BaselineUnavailable(
                        f"OptiMUS coding adapter expected scalar bounds for {symbol!r}"
                    )
                lb, ub = variable_bounds[symbol]

                def bound_expr(value):
                    if value == math.inf:
                        return "GRB.INFINITY"
                    if value == -math.inf:
                        return "-GRB.INFINITY"
                    return repr(value)

                needle = "model.addVar("
                if code.count(needle) != 1:
                    raise BaselineUnavailable(
                        f"OptiMUS original get_var_code output is not a single addVar for {symbol!r}"
                    )
                return code.replace(needle,
                                    f"model.addVar(lb={bound_expr(lb)}, ub={bound_expr(ub)}, ", 1)

            stages["generate_code"]["get_var_code"] = bounded_get_var_code
            constraints, objective = await asyncio.to_thread(
                stages["target_code"]["get_codes"],
                desc + "\n\n" + _fragment_code_contract(answer_spec),
                params, variables, constraints, objective,
                recorder.model.config.model, True)
            state.update(constraints=constraints, objective=objective)
            code = await asyncio.to_thread(
                stages["generate_code"]["generate_code"], state, str(output), None)
            stages_used = ["code_generation"]
        else:
            stages_used = ["parameters", "objective", "constraints",
                           "constraint_formulations", "objective_formulation", "code_generation"]
            params = await asyncio.to_thread(stages["parameters"]["get_params"], desc, True)
            objective = await asyncio.to_thread(stages["objective"]["get_objective"], desc, params, recorder.model.config.model, True)
            constraints = await asyncio.to_thread(stages["constraint"]["get_constraints"], desc, params, recorder.model.config.model, True)
            constraints, variables = await asyncio.to_thread(
                stages["constraint_model"]["get_constraint_formulations"], desc, params,
                constraints, recorder.model.config.model, True)
            objective = await asyncio.to_thread(
                stages["objective_model"]["get_objective_formulation"], desc, params,
                variables, objective, recorder.model.config.model, True)
            constraints, objective = await asyncio.to_thread(
                stages["target_code"]["get_codes"], desc + "\n\n" +
                _fragment_code_contract(answer_spec), params, variables, constraints,
                objective, recorder.model.config.model, True)
            state = {"description": question, "parameters": params, "variables": variables,
                     "constraints": constraints, "objective": objective}
            code = await asyncio.to_thread(stages["generate_code"]["generate_code"], state, str(output), None)
        execution = None
        for attempt in range(max_repairs + 1):
            attempt_dir = _attempt_dir(output, attempt)
            _write_json(attempt_dir / "data.json", {"parameters": params})
            execution = await execute_code(code, attempt_dir, python_executable, process_timeout,
                                           solver_time_limit, solver_threads)
            prediction = _parse_execution(execution, answer_spec)
            attempts.append({"attempt": attempt, "generated_code": code,
                             "execution": execution, "prediction": prediction})
            if prediction is not None:
                break
            if attempt >= max_repairs:
                break
            prompt_template = stages["execute_code"].get("debug_template")
            if not isinstance(prompt_template, str):
                raise BaselineUnavailable("OptiMUS execute_code.py lacks a literal debug_template")
            prompt = prompt_template.format(description=desc, code=code,
                                            error=execution.get("stderr", "") or execution.get("stdout", ""))
            response = await recorder.complete([{"role": "user", "content": prompt}],
                                               "baseline_optimus_repair")
            try:
                code = extract_code(response)
            except ValueError:
                code = response
        prediction = _parse_execution(execution or {}, answer_spec)
        status = "completed" if prediction is not None else "execution_failed"
        return {"status": status, "prediction": prediction,
                "attempts": attempts, "execution": execution,
                "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "OptiMUS_staged_pipeline", "level": level,
                                   "stages": stages_used,
                                   "repair_limit": max_repairs,
                                   "math_input_authoritative": level != "end_to_end",
                                   "final_state": state}}
    except BaselineUnavailable as exc:
        return {"status": "unsupported_coding_input", "prediction": None,
                "attempts": attempts, "execution": execution, "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "OptiMUS_staged_pipeline", "level": level,
                                   "error": str(exc), "math_input_authoritative": True,
                                   "final_state": state}}
    except CallBudgetExceeded as exc:
        return {"status": "call_budget_exhausted", "prediction": None, "attempts": attempts,
                "execution": execution, "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "OptiMUS_staged_pipeline", "level": level,
                                   "error": str(exc), "final_state": state}}
    except ModelCallError as exc:
        return {"status": "model_call_failed", "prediction": None, "attempts": attempts,
                "execution": execution, "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "OptiMUS_staged_pipeline", "level": level,
                                   "error": str(exc), "final_state": state}}
    except Exception as exc:
        return {"status": "baseline_failed", "prediction": None, "attempts": attempts,
                "execution": execution, "call_ids": recorder.call_ids,
                "baseline_trace": {"algorithm": "OptiMUS_staged_pipeline", "level": level,
                                   "error": f"{type(exc).__name__}: {exc}", "final_state": state}}


async def run_baseline(name: str, case_public: dict[str, Any], model: LoggedModel,
                       output_dir: str | Path, run_id: str, *, level: str,
                       ned_advice: Any = None, source_root: str | Path,
                       python_executable: str, process_timeout: float,
                       solver_time_limit: float, solver_threads: int,
                       max_calls: int, max_repairs: int) -> dict[str, Any]:
    """Run one original baseline under the shared LoggedModel and executor.

    Supported levels are ``end_to_end`` and ``coding``. The latter uses the
    supplied frozen mathematical input as authoritative context. ``math`` is
    accepted as a compatibility spelling for ``coding``.
    """
    aliases = {"or-llm-agent": "or_llm_agent", "or_llm": "or_llm_agent",
               "coe": "coe", "optimus": "optimus"}
    key = name.strip().casefold().replace(" ", "_")
    key = aliases.get(key, key)
    if key not in _ROOT_NAMES:
        return {"status": "baseline_unavailable", "prediction": None, "attempts": [],
                "execution": None, "call_ids": [],
                "baseline_trace": {"error": f"unknown baseline {name!r}"}}
    if level not in {"end_to_end", "coding", "math"}:
        return {"status": "unsupported_level", "prediction": None, "attempts": [],
                "execution": None, "call_ids": [],
                "baseline_trace": {"baseline": key, "level": level}}
    if max_calls < 1 or max_repairs < 0 or solver_threads < 1:
        raise ValueError("max_calls/threads must be positive and max_repairs nonnegative")
    if level != "end_to_end" and case_public.get("math_input") is None:
        return {"status": "missing_math_input", "prediction": None, "attempts": [],
                "execution": None, "call_ids": [],
                "baseline_trace": {"baseline": key, "level": level}}
    try:
        root = _resolve_root(key, source_root)
        availability = baseline_availability(key, root)
        if not availability["available"]:
            raise BaselineUnavailable(str(availability))
    except (BaselineUnavailable, KeyError) as exc:
        return {"status": "baseline_unavailable", "prediction": None, "attempts": [],
                "execution": None, "call_ids": [],
                "baseline_trace": {"baseline": key, "level": level, "error": str(exc)}}
    case_id = str(case_public.get("case_id", ""))
    if not case_id:
        raise ValueError("case_public.case_id is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recorder = _CallRecorder(model, case_id, run_id, max_calls, output)
    try:
        if key == "coe":
            result = await _run_coe(root, case_public, model, recorder, output, level=level,
                                    ned_advice=ned_advice, python_executable=python_executable,
                                    process_timeout=process_timeout,
                                    solver_time_limit=solver_time_limit,
                                    solver_threads=solver_threads, max_repairs=max_repairs)
        elif key == "or_llm_agent":
            result = await _run_or_agent(root, case_public, recorder, output, level=level,
                                         ned_advice=ned_advice,
                                         python_executable=python_executable,
                                         process_timeout=process_timeout,
                                         solver_time_limit=solver_time_limit,
                                         solver_threads=solver_threads,
                                         max_repairs=max_repairs)
        else:
            result = await _run_optimus(root, case_public, recorder, output, level=level,
                                         ned_advice=ned_advice,
                                         python_executable=python_executable,
                                         process_timeout=process_timeout,
                                         solver_time_limit=solver_time_limit,
                                         solver_threads=solver_threads,
                                         max_repairs=max_repairs)
    except BaselineUnavailable as exc:
        result = {"status": "baseline_unavailable", "prediction": None, "attempts": [],
                  "execution": None, "baseline_trace": {"error": str(exc)}}
    if recorder.exhaustion_error and result.get("status") not in {
        "model_call_failed", "baseline_unavailable", "unsupported_level", "missing_math_input"
    }:
        result["status"] = "call_budget_exhausted"
        result["prediction"] = None
        result.setdefault("baseline_trace", {})["call_budget_error"] = recorder.exhaustion_error
    result["call_ids"] = sorted(set(result.get("call_ids", []) + recorder.call_ids))
    default_stages = {
        "coe": {
            "end_to_end": ["conductor_expert_selection", "expert_comment_generation", "reducer_code_generation"],
            "coding": ["programming_expert_code_generation"],
            "math": ["programming_expert_code_generation"],
        },
        "optimus": {
            "end_to_end": ["parameter_extraction", "objective_extraction", "constraint_extraction",
                           "constraint_formulation", "objective_formulation", "code_generation"],
            "coding": ["code_generation"], "math": ["code_generation"],
        },
        "or_llm_agent": {
            "end_to_end": ["math_model_generation", "code_generation", "execution_feedback_repairs", "no_solution_recovery"],
            "coding": ["fixed_math_input_injection", "code_generation", "execution_feedback_repairs"],
            "math": ["fixed_math_input_injection", "code_generation", "execution_feedback_repairs"],
        },
    }
    result.setdefault("baseline_trace", {}).setdefault("stages", default_stages[key][level])
    result["baseline_trace"].update({"baseline": key, "source": availability["source"],
                                     "level_label": "baseline_conditioned_coding" if level in {"coding", "math"} else "end_to_end",
                                     "max_calls": max_calls,
                                     "model_calls": recorder.calls})
    _write_json(output / "baseline_result.json", result)
    return result
