"""One paired execution path; historical result directories are never mutated."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from nedtree_4r_adapter import build_nedtree_4r_advice
from .data import public_case, validate_math_input
from .execution import execute_code, extract_code
from .models import LoggedModel, ModelCallError
from .scoring import AnswerParseError, parse_answer, score_answer


PROTOCOL_VERSION = "nedtree-paired-v1"
MATH_SCHEMA = (
    'Return one JSON object with Parameters_List [{Name,Meaning,Type,Value}], '
    'Variables_List [{symbol,Meaning,Type,Range}], Constraint_Table '
    '[{name,Mathematical_expressions}], Objective '
    '{Objective_sentence,Mathematical_expressions}, and Problem_Type. '
    'Keep symbol spelling and case consistent. Write explicit domains, objective '
    'direction and original outer relations; do not add auxiliary substitutions. '
    'Do not solve for or guess the answer. No markdown.'
)
CODE_SYSTEM = """You write complete Python programs using gurobipy.
Implement the supplied mathematical model, domains and requested outputs faithfully.
You may use Gurobi native nonlinear expressions, legal resultant variables, and
any auxiliary variables you find useful. NLExpr inequalities must be represented
through legal resultant-variable equalities and bounds/linear relations; both
sides may need resultants. Give resultants appropriate bounds, including negative
values where required. Never silently change a relation, relax a constraint,
replace a missing quantity, or import a reference answer. Keep original domains.
Use model = gp.Model(...), model.optimize(), and inspect model.SolCount before
reading a solution. The harness applies the stated solver limits.
Print exactly one final line FinalAnswer=【<JSON>】. Its JSON structure and names
must match answer_spec; output the asked quantity, not an arbitrary objective.
For kind='scalar', emit only a JSON number (for example 2.0), not an object.
When its fields are ['objective_value'], the evaluation requests the optimized
objective value even if the narrative also asks for a plan. Keep the model unchanged.
For kind='numeric_list', emit a JSON array in the requested order.
For kind='named_numeric', emit an object with exactly the requested field names.
Do not print a final answer without a solution. Return one Python fenced block.
"""


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def implementation_fingerprint() -> str:
    package = Path(__file__).resolve().parent
    files = sorted(package.glob("*.py")) + [
        package.parent / "NEDTree-4R.py", package.parent / "nedtree_4r_adapter.py"]
    return digest([(path.name, hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest())
                   for path in files])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.S)
        if not match:
            raise ValueError("Expected one JSON object or JSON code block")
        text = match.group(1)
    result = json.loads(text)
    if not isinstance(result, dict):
        raise ValueError("Mathematical input must be a JSON object")
    return result


async def extract_inputs(manifest: dict, model: LoggedModel, output: Path,
                         *, refine: bool = True) -> dict:
    """Produce explicit shared inputs once; keep upstream failures in the manifest."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an input manifest: {output}")
    if len({c["case_id"] for c in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("Duplicate case_id values would overwrite extraction outputs")
    result = json.loads(json.dumps(manifest))
    cache = output.parent / (output.stem + "_upstream")
    cache.mkdir(parents=True, exist_ok=True)
    for case in result["cases"]:
        case_id = case["case_id"]
        folder = cache / digest(case_id)[:16]
        if (folder / "started.json").exists():
            raise RuntimeError(f"Interrupted extraction exists for {case_id}; use a new output")
        write_json(folder / "started.json", {"case_id": case_id})
        calls = []
        run_id = "upstream:" + digest([manifest["dataset_version"], case_id])[:20]
        try:
            if "question_may_include_answer_material" in case.get("review_reasons", []):
                raise ValueError("Question may include reference material; resolve the review flag before model calls")
            prompt = [{"role": "system", "content":
                       "Extract the complete mathematical model from the problem. " + MATH_SCHEMA},
                      {"role": "user", "content": case["question"]}]
            write_json(folder / "extract_prompt.json", prompt)
            response = await model.complete(prompt, stage="extract", case_id=case_id, run_id=run_id)
            calls.append(response["call_id"])
            write_json(folder / "extract_response.json", response)
            math_input = validate_math_input(json_response(response["content"]))
            if refine:
                prompt = [{"role": "system", "content":
                           "Check and refine the extracted model against the original problem. " + MATH_SCHEMA},
                          {"role": "user", "content": json.dumps(
                              {"question": case["question"], "extracted_model": math_input},
                              ensure_ascii=False)}]
                write_json(folder / "model_prompt.json", prompt)
                response = await model.complete(prompt, stage="model", case_id=case_id, run_id=run_id)
                calls.append(response["call_id"])
                write_json(folder / "model_response.json", response)
                math_input = validate_math_input(json_response(response["content"]))
            case["math_input"] = math_input
            case["upstream"] = {"status": "complete", "call_ids": calls,
                                "input_origin": "automatic", "run_id": run_id,
                                "ledger_path": str(model.ledger_path.resolve()), "refine": refine}
        except Exception as exc:
            if isinstance(exc, ModelCallError):
                calls.append(exc.record["call_id"])
            case["math_input"] = None
            case["upstream"] = {"status": "failed", "call_ids": calls, "error": str(exc),
                                "input_origin": "automatic", "run_id": run_id,
                                "ledger_path": str(model.ledger_path.resolve()), "refine": refine}
        write_json(folder / "result.json", {"case_id": case_id, "math_input": case["math_input"],
                                            "upstream": case["upstream"]})
    result["input_generation"] = {"protocol": PROTOCOL_VERSION,
                                  "model": asdict(model.config), "refine": refine,
                                  "repeat_scope": "one shared automatic upstream per case"}
    write_json(output, result)
    return result


def coding_messages(case: dict, math_input: dict, trace: dict | None,
                    solver_time_limit: float, solver_threads: int) -> list[dict]:
    payload = public_case({**case, "math_input": math_input})
    payload["solver_limits"] = {"time_limit_seconds": solver_time_limit,
                                 "threads": solver_threads}
    if trace is not None:
        payload["ned_advice"] = trace
    return [{"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


async def run_arm(case: dict, math_input: dict | None, arm: str, framework: str,
                  model: LoggedModel, folder: Path, identity: dict, *, max_repairs: int,
                  python_executable: str, timeout_seconds: float,
                  solver_time_limit: float, solver_threads: int) -> dict:
    started = time.perf_counter()
    result = {**identity, "case_id": case["case_id"], "source_id": case["source_id"],
              "arm": arm, "framework": framework, "model": model.config.model,
              "input_hash": digest(math_input), "status": "started", "prediction": None,
              "score": {"correct": False, "reason": "not_completed"},
              "attempts": [], "call_ids": [],
              "repair_count": 0,
              "upstream_call_ids": case.get("upstream", {}).get("call_ids", []),
              "upstream": case.get("upstream"), "scoring": case["scoring"],
              "execution": None, "ned_trace": None}
    write_json(folder / "started.json", result)
    try:
        if math_input is None:
            result["status"] = "upstream_failed" if case.get("upstream") else "missing_math_input"
            result["score"]["reason"] = result["status"]
            return result
        if arm == "ned":
            # Unsupported candidates remain visible and use the unchanged original model.
            try:
                trace = build_nedtree_4r_advice(math_input)
            except Exception as exc:
                trace = {"status": "fallback", "error": str(exc), "candidates": [],
                         "coding_advice": "Decomposition failed. Implement the supplied original model."}
            result["ned_trace"] = trace
        messages = coding_messages(case, math_input, result["ned_trace"],
                                   solver_time_limit, solver_threads)
        for attempt in range(max_repairs + 1):
            result["repair_count"] = attempt
            attempt_dir = folder / f"attempt_{attempt:02d}"
            write_json(attempt_dir / "prompt.json", messages)
            response = await model.complete(messages, stage="code" if attempt == 0 else "repair_code",
                                            case_id=case["case_id"], run_id=identity["run_id"])
            result["call_ids"].append(response["call_id"])
            write_json(attempt_dir / "response.json", response)
            observation = {"attempt": attempt, "call_id": response["call_id"],
                           "prediction": None, "score": {"correct": False}, "execution": None}
            try:
                code = extract_code(response["content"])
                execution = await execute_code(code, attempt_dir, python_executable,
                                               timeout_seconds, solver_time_limit, solver_threads)
                observation["execution"] = execution
                if execution["status"] == "executed" and execution.get("returncode") == 0:
                    solver = execution.get("solver_status") or {}
                    if not execution.get("solver_limits_applied") or solver.get("sol_count", 0) < 1:
                        raise ValueError("No recorded solver solution; an answer alone is insufficient")
                    prediction = parse_answer(execution["stdout"])
                    observation["prediction"] = prediction
                    observation["score"] = score_answer(prediction, case["reference"], case["scoring"],
                                                        case_id=case["case_id"])
                    observation["status"] = "answer_obtained"
                else:
                    observation["status"] = "execution_failed"
            except (AnswerParseError, ValueError, SyntaxError) as exc:
                observation["status"] = "invalid_output"
                observation["error"] = str(exc)
            result["attempts"].append(observation)
            result.update({"prediction": observation["prediction"], "score": observation["score"],
                           "execution": observation["execution"], "status": observation["status"]})
            write_json(attempt_dir / "result.json", observation)
            # Reference correctness is never used as repair feedback or a stopping oracle.
            if observation["status"] == "answer_obtained" or attempt == max_repairs:
                break
            execution = observation.get("execution") or {}
            feedback = {"status": observation["status"], "error": observation.get("error"),
                        "stdout": execution.get("stdout", "")[-8000:],
                        "stderr": execution.get("stderr", "")[-8000:]}
            if framework == "ours":
                analysis_messages = [
                    {"role": "system", "content": "Diagnose the generated code using only the supplied model and execution feedback. Give concrete corrections; do not guess a reference answer."},
                    {"role": "user", "content": json.dumps(
                        {"input": public_case({**case, "math_input": math_input}),
                         "code": response["content"], "feedback": feedback}, ensure_ascii=False)}]
                write_json(attempt_dir / "repair_analysis_prompt.json", analysis_messages)
                analysis = await model.complete(analysis_messages, stage="repair_analysis",
                                                case_id=case["case_id"], run_id=identity["run_id"])
                result["call_ids"].append(analysis["call_id"])
                write_json(attempt_dir / "repair_analysis.json", analysis)
                feedback["analysis"] = analysis["content"]
            messages.extend([{"role": "assistant", "content": response["content"]},
                             {"role": "user", "content": "Correct the program from this execution feedback:\n" +
                              json.dumps(feedback, ensure_ascii=False)}])
        return result
    except asyncio.CancelledError:
        result["status"] = "interrupted"
        raise
    except ModelCallError as exc:
        result["call_ids"].append(exc.record["call_id"])
        result.update(status="model_call_failed", error=str(exc))
        return result
    except Exception as exc:
        result.update(status="runner_failed", error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(folder / "result.json", result)


async def run_paired(manifest: dict, model: LoggedModel, output: Path, *, framework: str = "ours",
                     repeats: int = 1, max_repairs: int = 0, concurrency: int = 1,
                     python_executable: str, timeout_seconds: float = 180,
                     solver_time_limit: float = 60, solver_threads: int = 1,
                     allow_unreviewed: bool = False, resume: bool = False) -> list[dict]:
    if framework not in {"ours", "standard"}:
        raise ValueError("Only the implemented ours/standard controlled paths are available")
    if min(repeats, concurrency, solver_threads) < 1 or max_repairs < 0:
        raise ValueError("repeats/concurrency/threads must be positive; repairs must be nonnegative")
    if min(timeout_seconds, solver_time_limit) <= 0:
        raise ValueError("Execution limits must be positive")
    cases = manifest["cases"]
    if not cases:
        raise ValueError("Manifest contains no cases")
    if len({c["case_id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case_id values would overwrite paired outputs")
    if not allow_unreviewed and any(c["review_status"] not in {"ready", "verified", "approved"} for c in cases):
        raise ValueError("Unreviewed cases remain; inspect the manifest or explicitly run an exploratory job")
    for case in cases:
        if "question_may_include_answer_material" in case.get("review_reasons", []):
            raise ValueError(f"Resolve possible answer material in question {case['case_id']} before model calls")
        # Validate the scoring/reference contract before any paid call.
        check = score_answer(case["reference"], case["reference"], case["scoring"],
                             case_id=case["case_id"])
        if not check["correct"]:
            raise ValueError(f"Invalid reference/scoring contract for {case['case_id']}: {check}")
    protocol = {"protocol_version": PROTOCOL_VERSION, "dataset": manifest["dataset"],
                "implementation_hash": implementation_fingerprint(),
                "dataset_version": manifest["dataset_version"], "manifest_hash": digest(manifest),
                "model_config": asdict(model.config), "framework": framework, "repeats": repeats,
                "max_repairs": max_repairs, "concurrency": concurrency,
                "python_executable": str(Path(python_executable).resolve()),
                "timeout_seconds": timeout_seconds, "solver_time_limit": solver_time_limit,
                "solver_threads": solver_threads, "allow_unreviewed": allow_unreviewed,
                "repeat_scope": "coding conditioned on the frozen mathematical input",
                "arms": ["native", "ned"], "case_ids": [c["case_id"] for c in cases]}
    # Compare the same JSON representation after a restart (e.g. tuples become lists).
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    protocol_hash = digest(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if not resume:
            raise FileExistsError("Output already exists; choose a new path or explicitly resume")
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Resume protocol differs from the recorded protocol")
    else:
        if output.exists() and any(output.iterdir()):
            # The model ledger is created lazily, so unrelated contents are never adopted.
            raise FileExistsError("Output directory is not empty")
        write_json(protocol_path, protocol)
        write_json(output / "manifest.json", manifest)
    semaphore = asyncio.Semaphore(concurrency)

    async def paired_case(case: dict, repeat: int) -> list[dict]:
        async with semaphore:
            prefix = digest([case["case_id"], repeat])[:20]
            pair_dir = output / "cases" / prefix
            math_error = None
            try:
                math_input = validate_math_input(case["math_input"]) if case["math_input"] is not None else None
            except ValueError as exc:
                math_input, math_error = None, str(exc)
            write_json(pair_dir / "shared_input.json", {"case_id": case["case_id"], "repeat": repeat,
                                                       "math_input": math_input, "error": math_error})
            records = []
            for arm in ("native", "ned"):
                folder = pair_dir / arm
                result_file = folder / "result.json"
                identity = {"dataset": manifest["dataset"], "dataset_version": manifest["dataset_version"],
                            "repeat": repeat, "run_id": f"{protocol_hash[:16]}:{prefix}:{arm}",
                            "protocol_hash": protocol_hash, "stage": "controlled_coding",
                            "exploratory": allow_unreviewed}
                if resume and result_file.exists():
                    existing = json.loads(result_file.read_text(encoding="utf-8"))
                    if existing.get("status") in {"started", "interrupted"}:
                        raise RuntimeError(f"Interrupted unit for {case['case_id']}/{arm}; inspect its ledger before rerunning")
                    if any(existing.get(k) != v for k, v in identity.items()) or existing.get("arm") != arm or existing.get("case_id") != case["case_id"]:
                        raise ValueError(f"Result identity mismatch for {case['case_id']}/{arm}")
                    records.append(existing)
                    continue
                if (folder / "started.json").exists():
                    raise RuntimeError(f"Interrupted arm {case['case_id']}/{arm}; use a new run directory")
                records.append(await run_arm(case, math_input, arm, framework, model, folder, identity,
                                             max_repairs=max_repairs, python_executable=python_executable,
                                             timeout_seconds=timeout_seconds,
                                             solver_time_limit=solver_time_limit, solver_threads=solver_threads))
            return records

    groups = await asyncio.gather(*(paired_case(c, r) for c in cases for r in range(repeats)))
    records = [record for group in groups for record in group]
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records
