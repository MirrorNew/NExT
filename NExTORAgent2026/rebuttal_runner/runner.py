"""One paired execution path; historical result directories are never mutated."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

from .data import public_case, validate_math_input
from .execution import execute_code, extract_code
from .models import BudgetedModel, CallBudgetExceeded, LoggedModel, ModelCallError
from .scoring import AnswerParseError, parse_answer, score_answer


PROTOCOL_VERSION = "nedtree-integrated-v2"
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
    files = sorted(package.glob("*.py")) + sorted((package.parent / "agents").glob("*.py"))
    return digest([(str(path.relative_to(package.parent)), hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in files])


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                         encoding="utf-8")
    temporary.replace(path)


def case_input(case: dict, math_input=None, math_model=None) -> dict:
    result = public_case({**case, "math_input": math_input})
    result["case_id"] = case["case_id"]
    if math_model is not None:
        result["math_model"] = math_model
    return result


def select_cases(manifest: dict) -> tuple[list[dict], list[dict]]:
    """Only explicit data exclusions leave the planned denominator."""
    included, excluded = [], []
    for case in manifest["cases"]:
        if case["review_status"].startswith("excluded_"):
            excluded.append({"case_id": case["case_id"], "source_id": case["source_id"],
                             "reason": case["review_status"]})
        else:
            included.append(case)
    return included, excluded


def validate_run(manifest, *, framework, level, arms, allow_unreviewed=False):
    if framework not in {"ours", "standard", "coe", "optimus", "or_llm_agent"}:
        raise ValueError("Unknown framework")
    if level not in {"coding", "math", "end_to_end"}:
        raise ValueError("level must be coding, math, or end_to_end")
    if not arms or len(arms) != len(set(arms)) or any(a not in {"native", "ned"} for a in arms):
        raise ValueError("arms must contain native and/or ned without duplicates")
    if level == "end_to_end" and framework != "ours" and tuple(arms) != ("native",):
        raise ValueError("External/standard end_to_end uses its original native workflow; use coding for Native/NED transfer")
    if len({c["case_id"] for c in manifest["cases"]}) != len(manifest["cases"]):
        raise ValueError("Duplicate case_id values would overwrite outputs")
    cases, excluded = select_cases(manifest)
    if not cases:
        raise ValueError("No included cases")
    if not allow_unreviewed and any(c["review_status"] not in {"ready", "verified", "approved"} for c in cases):
        raise ValueError("Unreviewed cases remain; inspect the manifest")
    for case in cases:
        if "question_may_include_answer_material" in case.get("review_reasons", []):
            raise ValueError(f"Resolve possible answer material in question {case['case_id']} before model calls")
        check = score_answer(case["reference"], case["reference"], case["scoring"],
                             case_id=case["case_id"])
        if not check["correct"]:
            raise ValueError(f"Invalid reference/scoring contract for {case['case_id']}: {check}")
        if level != "end_to_end" and "ned" in arms and case.get("auxiliary"):
            auxiliary = case["auxiliary"]
            normalized = validate_math_input(case["math_input"])
            if auxiliary.get("input_hash") != digest(normalized) or auxiliary.get("question_hash") != digest(case["question"]):
                raise ValueError(f"Cached Auxiliary input mismatch for {case['case_id']}")
    return cases, excluded


async def extract_inputs(manifest: dict, model: LoggedModel, output: Path,
                         *, refine: bool = True, max_calls: int = 64) -> dict:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_suffix(".extract.lock")
    try:
        handle = lock.open("x", encoding="utf-8")
    except FileExistsError:
        raise RuntimeError("Concurrent or interrupted extraction owns this output; inspect its lock and ledger") from None
    try:
        with handle:
            json.dump({"pid": os.getpid(), "output": str(output.resolve())}, handle)
        return await _extract_inputs(manifest, model, output, refine=refine, max_calls=max_calls)
    finally:
        lock.unlink()


async def _extract_inputs(manifest: dict, model: LoggedModel, output: Path,
                          *, refine: bool = True, max_calls: int = 64) -> dict:
    """Freeze the actual ours upstream and its single Auxiliary completion."""
    from .ours import prepare_auxiliary, prepare_ours_input
    if not refine:
        raise ValueError("The integrated upstream preserves ModelingAgent; --no-refine is no longer a pipeline mode")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an input manifest: {output}")
    cases, excluded = validate_run(manifest, framework="ours", level="end_to_end",
                                   arms=("native", "ned"))
    result = json.loads(json.dumps(manifest))
    result["cases"] = [c for c in result["cases"] if c["case_id"] in {v["case_id"] for v in cases}]
    result["excluded_cases"] = excluded + manifest.get("excluded_cases", [])
    for case in result["cases"]:
        case.update(math_input=None, upstream={"status": "not_started", "call_ids": []})
        case.pop("math_model", None)
        case.pop("auxiliary", None)
    partial_output = output.with_suffix(".partial.json")
    result["input_generation"] = {"protocol": PROTOCOL_VERSION, "status": "in_progress",
                                  "model": asdict(model.config),
                                  "upstream": "ours_parameter_factor_modeling",
                                  "auxiliary_calls_per_successful_input": 1,
                                  "repeat_scope": "coding repeats reuse this frozen input and auxiliary output"}
    cache = output.parent / (output.stem + "_upstream")
    for case in result["cases"]:
        case_id = case["case_id"]
        folder = cache / digest(case_id)[:16]
        folder.mkdir(parents=True, exist_ok=True)
        try:
            with (folder / "started.json").open("x", encoding="utf-8") as handle:
                json.dump({"case_id": case_id}, handle)
        except FileExistsError:
            raise RuntimeError(f"Concurrent, interrupted or prior extraction exists for {case_id}; use a new output") from None
        run_id = "upstream:" + digest([manifest["dataset_version"], case_id])[:20]
        scoped = BudgetedModel(model, max_calls)
        try:
            prepared = await prepare_ours_input(case_input(case), scoped, folder, run_id)
            case["math_input"] = validate_math_input(prepared["math_input"])
            case["math_model"] = prepared["math_model"]
            case["upstream"] = {"status": "complete", "call_ids": list(scoped.call_ids),
                                "stage_costs": prepared.get("stage_costs", []),
                                "input_origin": "automatic", "run_id": run_id,
                                "ledger_path": str(model.ledger_path.resolve())}
        except Exception as exc:
            case["math_input"] = None
            case["upstream"] = {"status": "failed", "call_ids": list(scoped.call_ids),
                                "error": str(exc), "input_origin": "automatic",
                                "ledger_path": str(model.ledger_path.resolve()), "run_id": run_id}
        # Preserve paid upstream work before attempting the Auxiliary call.
        write_json(folder / "upstream_result.json", {
            "case_id": case_id, "math_input": case["math_input"],
            "math_model": case.get("math_model"), "upstream": case["upstream"]})
        write_json(partial_output, result)
        if case["math_input"] is not None:
            auxiliary_model = BudgetedModel(model, 1)
            try:
                case["auxiliary"] = await prepare_auxiliary(
                    case_input(case, case["math_input"], case["math_model"]),
                    case["math_input"], auxiliary_model, folder, run_id + ":auxiliary")
            except Exception as exc:
                case["auxiliary"] = {"status": "fallback", "call_ids": list(auxiliary_model.call_ids),
                                     "tree": {"status": "fallback", "candidates": [], "error": str(exc)},
                                     "error": str(exc)}
            case["auxiliary"].update(input_hash=digest(case["math_input"]),
                                     question_hash=digest(case["question"]),
                                     ledger_path=str(model.ledger_path.resolve()))
        write_json(folder / "result.json", {"case_id": case_id, "math_input": case["math_input"],
                                            "math_model": case.get("math_model"),
                                            "upstream": case["upstream"], "auxiliary": case.get("auxiliary")})
        write_json(partial_output, result)
    result["input_generation"]["status"] = "complete"
    write_json(output, result)
    return result


def coding_messages(case, math_input, trace, solver_time_limit, solver_threads):
    payload = public_case({**case, "math_input": math_input})
    payload["solver_limits"] = {"time_limit_seconds": solver_time_limit, "threads": solver_threads}
    if trace is not None:
        payload["ned_advice"] = trace
    return [{"role": "system", "content": CODE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def public_auxiliary(auxiliary):
    """Only accepted construction advice is prompt material; raw failures remain audit data."""
    if auxiliary is None:
        return None
    tree = auxiliary.get("tree") or {}
    if tree.get("status") not in {"validated", "no_decomposition"}:
        return {"tree": {"status": "fallback", "candidates": []},
                "text_advice": "No valid auxiliary structure. Implement the original supplied model."}
    return {"tree": tree, "math_model_advice": tree.get("math_model_advice", [])}


async def run_arm(case, math_input, arm, framework, model, folder, identity, *,
                  max_repairs, python_executable, timeout_seconds, solver_time_limit,
                  solver_threads, auxiliary=None, math_model=None, source_root=None,
                  max_calls=64):
    started = time.perf_counter()
    auxiliary_ids = auxiliary.get("call_ids", []) if arm == "ned" and auxiliary else []
    coding_advice = public_auxiliary(auxiliary) if arm == "ned" else None
    result = {**identity, "case_id": case["case_id"], "source_id": case["source_id"],
              "arm": arm, "framework": framework, "model": model.config.model,
              "input_hash": digest({"question": case["question"], "math_input": math_input,
                                    "math_model": math_model, "answer_spec": case["answer_spec"]}),
              "status": "started", "prediction": None,
              "score": {"correct": False, "reason": "not_completed"},
              "attempts": [], "call_ids": list(auxiliary_ids), "repair_count": 0,
              "upstream_call_ids": (case.get("upstream") or {}).get("call_ids", []),
              "upstream": case.get("upstream"), "scoring": case["scoring"],
              "elapsed_scope": "arm execution after shared upstream and Auxiliary preparation; request costs are accounted by call IDs",
              "representation_call_ids": list(auxiliary_ids),
              "execution": None, "ned_trace": auxiliary.get("tree") if auxiliary and arm == "ned" else None}
    write_json(folder / "started.json", result)
    try:
        level = identity["level"]
        if math_input is None and (level != "end_to_end" or framework == "ours"):
            result["status"] = "upstream_failed" if case.get("upstream") else "missing_math_input"
            result["score"]["reason"] = result["status"]
            return result
        if (case.get("baseline_input_check") or {}).get("supported") is False:
            result.update(status="unsupported_coding_input", baseline_trace=case["baseline_input_check"])
            result["score"]["reason"] = result["status"]
            return result
        if framework in {"coe", "optimus", "or_llm_agent"}:
            from .baselines import run_baseline
            baseline_result = await run_baseline(
                framework, case_input(case, math_input), model, folder, identity["run_id"],
                level=level, ned_advice=coding_advice, source_root=source_root,
                python_executable=python_executable, process_timeout=timeout_seconds,
                solver_time_limit=solver_time_limit, solver_threads=solver_threads,
                max_calls=max_calls, max_repairs=max_repairs)
            for key in ("status", "prediction", "attempts", "execution", "baseline_trace", "error"):
                if key in baseline_result:
                    result[key] = baseline_result[key]
            result["repair_count"] = max(0, len(result["attempts"]) - 1)
            result["call_ids"].extend(baseline_result.get("call_ids", []))
            result["score"] = score_answer(result["prediction"], case["reference"], case["scoring"],
                                            case_id=case["case_id"]) if result["prediction"] is not None else {
                                                "correct": False, "reason": result["status"]}
            # Scoring happens after all baseline control flow; it is never feedback.
            for attempt in result["attempts"]:
                if attempt.get("prediction") is not None:
                    attempt["score"] = score_answer(attempt["prediction"], case["reference"], case["scoring"],
                                                     case_id=case["case_id"])
            return result
        messages = coding_messages(case, math_input, coding_advice, solver_time_limit, solver_threads)
        feedback = None
        for attempt in range(max_repairs + 1):
            result["repair_count"] = attempt
            attempt_dir = folder / f"attempt_{attempt:02d}"
            if framework == "ours":
                from .ours import generate_ours_code
                generated = await generate_ours_code(
                    case_input(case, math_input), math_input,
                    math_model or json.dumps(math_input, ensure_ascii=False),
                    coding_advice, model, attempt_dir,
                    identity["run_id"], feedback=feedback, code_system=CODE_SYSTEM)
                response = generated["response"]
                result["call_ids"].extend(generated["call_ids"])
            else:
                write_json(attempt_dir / "prompt.json", messages)
                response = await model.complete(messages, stage="code" if attempt == 0 else "repair_code",
                                                case_id=case["case_id"], run_id=identity["run_id"])
                result["call_ids"].append(response["call_id"])
            write_json(attempt_dir / "response.json", response)
            observation = {"attempt": attempt, "prediction": None,
                           "score": {"correct": False}, "execution": None}
            try:
                code = extract_code(response["content"])
                execution = await execute_code(code, attempt_dir, python_executable,
                                               timeout_seconds, solver_time_limit, solver_threads)
                observation["execution"] = execution
                if execution["status"] == "executed" and execution.get("returncode") == 0:
                    solver = execution.get("solver_status") or {}
                    if not execution.get("solver_limits_applied") or solver.get("sol_count", 0) < 1:
                        raise ValueError("No recorded solver solution; an answer alone is insufficient")
                    observation["prediction"] = parse_answer(execution["stdout"])
                    observation["score"] = score_answer(observation["prediction"], case["reference"],
                                                        case["scoring"], case_id=case["case_id"])
                    observation["status"] = "answer_obtained"
                else:
                    observation["status"] = "execution_failed"
            except (AnswerParseError, ValueError, SyntaxError) as exc:
                observation.update(status="invalid_output", error=str(exc))
            result["attempts"].append(observation)
            result.update({k: observation[k] for k in ("prediction", "score", "execution", "status")})
            write_json(attempt_dir / "result.json", observation)
            if observation["status"] == "answer_obtained" or attempt == max_repairs:
                break
            execution = observation.get("execution") or {}
            feedback = {"code": response["content"],
                        "error": json.dumps({"status": observation["status"], "error": observation.get("error"),
                                             "stdout": execution.get("stdout", "")[-8000:],
                                             "stderr": execution.get("stderr", "")[-8000:]}, ensure_ascii=False)}
            messages.extend([{"role": "assistant", "content": response["content"]},
                             {"role": "user", "content": "Correct the program from this execution feedback:\n" + feedback["error"]}])
        return result
    except asyncio.CancelledError:
        result["status"] = "interrupted"
        raise
    except CallBudgetExceeded as exc:
        result.update(status="call_budget_exhausted", error=str(exc))
        return result
    except ModelCallError as exc:
        result["call_ids"].append(exc.record["call_id"])
        result.update(status="model_call_failed", error=str(exc))
        return result
    except Exception as exc:
        result.update(status="runner_failed", error=f"{type(exc).__name__}: {exc}")
        return result
    finally:
        result["call_ids"] = list(dict.fromkeys(result["call_ids"] + getattr(model, "call_ids", [])))
        result["elapsed_seconds"] = time.perf_counter() - started
        write_json(folder / "result.json", result)


def _exclusive_run(function):
    @wraps(function)
    async def guarded(manifest, model, output, **kwargs):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        lock = output / ".run.lock"
        try:
            handle = lock.open("x", encoding="utf-8")
        except FileExistsError:
            raise RuntimeError("Output is locked by another or interrupted process; inspect .run.lock before reuse") from None
        try:
            with handle:
                json.dump({"pid": os.getpid(), "output": str(output.resolve())}, handle)
            return await function(manifest, model, output, **kwargs)
        finally:
            # Only remove this invocation's exclusively created lock.
            lock.unlink()
    return guarded


@_exclusive_run
async def run_paired(manifest, model, output, *, framework="ours", level="coding",
                     arms=("native", "ned"), repeats=1, max_repairs=0, concurrency=1,
                     python_executable, timeout_seconds=180, solver_time_limit=60,
                     solver_threads=1, allow_unreviewed=False, resume=False,
                     source_root=None, max_calls=64, max_upstream_calls=32):
    from .ours import prepare_auxiliary, prepare_ours_input
    if min(repeats, concurrency, solver_threads, max_calls, max_upstream_calls) < 1 or max_repairs < 0:
        raise ValueError("Repeat/concurrency/thread/call limits must be positive; repairs nonnegative")
    if min(timeout_seconds, solver_time_limit) <= 0:
        raise ValueError("Execution limits must be positive")
    cases, excluded = validate_run(manifest, framework=framework, level=level, arms=arms,
                                   allow_unreviewed=allow_unreviewed)
    baseline_source = None
    if framework in {"coe", "optimus", "or_llm_agent"}:
        from .baselines import baseline_availability, source_fingerprint
        if not source_root:
            raise ValueError("External baseline requires --baseline-root")
        availability = baseline_availability(framework, source_root)
        if not availability.get("available"):
            raise ValueError(f"Baseline source unavailable: {availability}")
        baseline_source = source_fingerprint(framework, source_root)
    protocol = {"protocol_version": PROTOCOL_VERSION, "dataset": manifest["dataset"],
                "dataset_version": manifest["dataset_version"], "manifest_hash": digest(manifest),
                "implementation_hash": implementation_fingerprint(), "baseline_source": baseline_source,
                "model_config": asdict(model.config), "framework": framework, "level": level,
                "arms": list(arms), "case_ids": [c["case_id"] for c in cases],
                "excluded_cases": excluded + manifest.get("excluded_cases", []),
                "repeats": repeats, "max_repairs": max_repairs, "max_calls_per_arm": max_calls,
                "max_calls_shared_upstream": max_upstream_calls, "concurrency": concurrency,
                "python_executable": str(Path(python_executable).resolve()),
                "timeout_seconds": timeout_seconds, "solver_time_limit": solver_time_limit,
                "solver_threads": solver_threads, "allow_unreviewed": allow_unreviewed,
                "repeat_scope": ("fresh automatic upstream per repeat; ours arms share that upstream"
                                 if level == "end_to_end" else "coding conditioned on frozen mathematical input"),
                "representation_source": "one AuxiliaryModelAgent completion; no second decomposition"}
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    protocol_hash = digest(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if not resume:
            raise FileExistsError("Output already exists; choose a new path or explicitly resume")
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Resume protocol differs from the recorded protocol")
    else:
        if any(path.name != ".run.lock" for path in output.iterdir()):
            raise FileExistsError("Output directory is not empty")
        write_json(protocol_path, protocol)
        write_json(output / "manifest.json", manifest)
    semaphore = asyncio.Semaphore(concurrency)

    async def paired_case(case, repeat):
        async with semaphore:
            prefix = digest([case["case_id"], repeat])[:20]
            pair_dir = output / "cases" / prefix
            shared_path = pair_dir / "shared_input.json"
            if shared_path.exists():
                shared = json.loads(shared_path.read_text(encoding="utf-8"))
            else:
                if (pair_dir / "upstream_started.json").exists():
                    raise RuntimeError(f"Interrupted shared upstream for {case['case_id']}; inspect its ledger")
                write_json(pair_dir / "upstream_started.json", {"case_id": case["case_id"], "repeat": repeat})
                shared = {"case_id": case["case_id"], "repeat": repeat, "math_input": None,
                          "math_model": None, "upstream": case.get("upstream")}
                if level == "end_to_end" and framework == "ours":
                    upstream_model = BudgetedModel(model, max_upstream_calls)
                    try:
                        prepared = await prepare_ours_input(
                            case_input(case), upstream_model, pair_dir / "upstream",
                            f"{protocol_hash[:16]}:{prefix}:upstream")
                        shared.update(math_input=validate_math_input(prepared["math_input"]),
                                      math_model=prepared["math_model"],
                                      upstream={"status": "complete", "call_ids": list(upstream_model.call_ids),
                                                "input_origin": "automatic",
                                                "stage_costs": prepared.get("stage_costs", [])})
                    except Exception as exc:
                        shared["upstream"] = {"status": "failed", "error": str(exc),
                                              "input_origin": "automatic", "call_ids": list(upstream_model.call_ids)}
                elif level != "end_to_end":
                    try:
                        shared["math_input"] = validate_math_input(case["math_input"]) if case["math_input"] is not None else None
                        shared["math_model"] = case.get("math_model")
                    except ValueError as exc:
                        shared["upstream"] = {**(case.get("upstream") or {}), "status": "failed", "error": str(exc)}
                else:
                    # Real external end-to-end starts from the question, not ours' cache.
                    shared["upstream"] = None
                write_json(shared_path, shared)
            effective = {**case, "upstream": shared["upstream"]}
            baseline_input_check = None
            if level != "end_to_end" and framework in {"coe", "optimus", "or_llm_agent"} and shared["math_input"] is not None:
                from .baselines import baseline_coding_support
                baseline_input_check = baseline_coding_support(framework, shared["math_input"])
                effective["baseline_input_check"] = baseline_input_check
            records = []
            for arm in arms:
                folder = pair_dir / arm
                identity = {"dataset": manifest["dataset"], "dataset_version": manifest["dataset_version"],
                            "repeat": repeat, "run_id": f"{protocol_hash[:16]}:{prefix}:{arm}",
                            "protocol_hash": protocol_hash, "stage": level, "level": level,
                            "exploratory": allow_unreviewed}
                result_file = folder / "result.json"
                if resume and result_file.exists():
                    existing = json.loads(result_file.read_text(encoding="utf-8"))
                    if existing.get("status") in {"started", "interrupted"}:
                        raise RuntimeError(f"Interrupted unit for {case['case_id']}/{arm}; inspect its ledger")
                    if any(existing.get(k) != v for k, v in identity.items()) or existing.get("arm") != arm or existing.get("case_id") != case["case_id"]:
                        raise ValueError(f"Result identity mismatch for {case['case_id']}/{arm}")
                    records.append(existing)
                    continue
                if (folder / "started.json").exists():
                    raise RuntimeError(f"Interrupted arm {case['case_id']}/{arm}; use a new run directory")
                arm_model = BudgetedModel(model, max_calls)
                auxiliary = None
                if arm == "ned" and shared["math_input"] is not None and (baseline_input_check or {}).get("supported", True):
                    auxiliary_path = pair_dir / "auxiliary_input.json"
                    if auxiliary_path.exists():
                        auxiliary = json.loads(auxiliary_path.read_text(encoding="utf-8"))
                    elif level != "end_to_end" and case.get("auxiliary"):
                        auxiliary = case["auxiliary"]
                        if auxiliary.get("input_hash") != digest(shared["math_input"]) or auxiliary.get("question_hash") != digest(case["question"]):
                            raise ValueError(f"Cached Auxiliary input mismatch for {case['case_id']}")
                        write_json(auxiliary_path, auxiliary)
                    else:
                        marker = pair_dir / "auxiliary_started.json"
                        if marker.exists():
                            raise RuntimeError(f"Interrupted auxiliary call for {case['case_id']}; inspect its ledger")
                        write_json(marker, {"case_id": case["case_id"]})
                        try:
                            auxiliary = await prepare_auxiliary(
                                case_input(case, shared["math_input"], shared["math_model"]),
                                shared["math_input"], arm_model, pair_dir, identity["run_id"] + ":auxiliary")
                        except Exception as exc:
                            auxiliary = {"status": "fallback", "call_ids": list(arm_model.call_ids),
                                         "tree": {"status": "fallback", "candidates": [], "error": str(exc)},
                                         "error": str(exc)}
                        write_json(auxiliary_path, auxiliary)
                records.append(await run_arm(
                    effective, shared["math_input"], arm, framework, arm_model, folder, identity,
                    max_repairs=max_repairs, python_executable=python_executable,
                    timeout_seconds=timeout_seconds, solver_time_limit=solver_time_limit,
                    solver_threads=solver_threads, auxiliary=auxiliary,
                    math_model=shared["math_model"], source_root=source_root, max_calls=max_calls))
            return records

    groups = await asyncio.gather(*(paired_case(c, r) for c in cases for r in range(repeats)))
    records = [record for group in groups for record in group]
    with (output / "results.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    return records
