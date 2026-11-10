"""Offline preparation and inspection of declared experiment boundaries."""
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import os
from pathlib import Path

from .data import validate_math_input
from .runner import digest, validate_run


def resolved_arms(level, framework, arms=None):
    if arms is not None:
        return tuple(arms)
    return ("native",) if level == "end_to_end" and framework != "ours" else ("native", "ned")


def plan_run(manifest, config, *, framework, level, arms=None, repeats=1, max_repairs=0,
             max_calls=64, max_upstream_calls=32, source_root=None,
             solver_time_limit=60, process_timeout=180, solver_threads=1):
    arms = resolved_arms(level, framework, arms)
    cases, excluded = validate_run(manifest, framework=framework, level=level, arms=arms)
    if min(repeats, max_calls, max_upstream_calls, solver_threads) < 1 or max_repairs < 0:
        raise ValueError("Invalid count or repair limit")
    if min(solver_time_limit, process_timeout) <= 0:
        raise ValueError("Solver/process limits must be positive")
    missing, invalid = [], []
    for case in cases:
        if level == "end_to_end":
            continue
        if case["math_input"] is None:
            missing.append(case["case_id"])
        else:
            try:
                validate_math_input(case["math_input"])
            except ValueError as exc:
                invalid.append({"case_id": case["case_id"], "reason": str(exc)})
    baseline = None
    unsupported = []
    if framework in {"coe", "optimus", "or_llm_agent"}:
        from .baselines import baseline_availability, baseline_coding_support
        baseline = baseline_availability(framework, source_root) if source_root else {
            "available": False, "reason": "--baseline-root required"}
        if level != "end_to_end":
            for case in cases:
                if case["math_input"] is not None and case["case_id"] not in {row["case_id"] for row in invalid}:
                    support = baseline_coding_support(framework, validate_math_input(case["math_input"]))
                    if not support["supported"]:
                        unsupported.append({"case_id": case["case_id"], **support})
    key_present = bool(os.getenv(config.api_key_env))
    issues = []
    if missing or invalid:
        issues.append("Frozen mathematical inputs missing or invalid; prepare/cache them before coding diagnostics")
    if not key_present:
        issues.append("Configured credential environment variable is not set in this process")
    if baseline is not None and not baseline.get("available"):
        issues.append("Baseline source is unavailable")
    if unsupported:
        issues.append("Some frozen inputs cannot be represented by the baseline coding adapter; retained as explicit failures")
    if level == "end_to_end" and framework == "ours" and max_upstream_calls < 5:
        issues.append("Ours upstream requires five calls; declared shared upstream cap is smaller")
    pairs = len(cases) * repeats
    return {
        "dataset": manifest["dataset"], "dataset_version": manifest["dataset_version"],
        "manifest_hash": digest(manifest), "level": level, "framework": framework,
        "arms": list(arms), "repeats": repeats, "included_cases": len(cases),
        "excluded_cases": excluded + manifest.get("excluded_cases", []),
        "planned_arm_records": pairs * len(arms), "missing_math_input": missing,
        "invalid_math_input": invalid, "unsupported_coding_inputs": unsupported,
        "model_config": asdict(config),
        "credential_present": key_present, "provider_checked_online": False,
        "baseline": baseline, "max_repairs_after_initial": max_repairs,
        "max_calls_per_arm": max_calls, "max_calls_shared_upstream": max_upstream_calls,
        "hard_call_ceiling": pairs * (max_calls * len(arms) + (
            max_upstream_calls if level == "end_to_end" and framework == "ours" else 0)),
        "call_ceiling_note": "A declared stop limit, not a prediction of token usage or monetary cost",
        "solver_limits": {"time_limit_seconds": solver_time_limit, "threads": solver_threads,
                          "process_timeout_seconds": process_timeout},
        "offline_configuration_ready": not issues, "issues": issues,
        "formal_experiments_run": False,
    }


def attach_cached_inputs(manifest, cached, *, cache_path):
    """Reuse only identical questions; never copy references, scores or review flags."""
    if cached["dataset"] != manifest["dataset"]:
        raise ValueError("Cache dataset differs from target dataset")
    by_id = {}
    for case in cached["cases"]:
        if case["source_id"] in by_id:
            raise ValueError("Cache has duplicate source_id values")
        by_id[case["source_id"]] = case
    result = deepcopy(manifest)
    statuses = Counter()
    for case in result["cases"]:
        source = by_id.get(case["source_id"])
        if case["review_status"].startswith("excluded_"):
            status = "excluded_from_run"
        elif source is None:
            status = "not_in_cache"
        elif source["question"] != case["question"]:
            status = "question_version_mismatch"
        elif source["math_input"] is None:
            status = "cache_input_missing"
        else:
            try:
                case["math_input"] = validate_math_input(source["math_input"])
            except ValueError:
                status = "cache_input_invalid"
            else:
                status = "attached"
                case["upstream"] = deepcopy(source.get("upstream") or {
                    "status": "complete", "input_origin": "historical_cache", "call_ids": []})
                # Replacing math must not retain advice/text for the target's old input.
                case.pop("math_model", None)
                case.pop("auxiliary", None)
                if isinstance(source.get("math_model"), str):
                    case["math_model"] = source["math_model"]
                auxiliary = source.get("auxiliary")
                if auxiliary and auxiliary.get("input_hash") == digest(case["math_input"]) and auxiliary.get("question_hash") == digest(case["question"]):
                    case["auxiliary"] = deepcopy(auxiliary)
        case["cached_input_alignment"] = {"status": status, "source": str(Path(cache_path).resolve()),
                                          "cache_hash": digest(cached),
                                          "semantic_correctness_independently_verified": False}
        statuses[status] += 1
    result["cached_input_alignment"] = {"counts": dict(statuses),
                                         "scope": "Frozen-input coding only; never a fresh end-to-end run"}
    return result
