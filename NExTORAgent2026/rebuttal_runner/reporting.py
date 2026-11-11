"""Aggregation, paired comparisons, and stored-prediction rescoring."""

from __future__ import annotations

import copy
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable

from .scoring import score_answer


_GROUP_FIELDS = ("dataset", "dataset_version", "model", "framework", "level", "arm", "rescore_threshold")
_PAIR_SCOPE_FIELDS = ("dataset", "dataset_version", "model", "framework", "level", "rescore_threshold")
_ARMS = {"native", "ned"}
_BOOTSTRAP_SEED = 0
_BOOTSTRAP_SAMPLES = 500


def _freeze(value: Any) -> str:
    """Create a stable grouping key without requiring IDs to be hashable."""
    import json

    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def _field_key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_freeze(record.get(field)) for field in fields)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _is_correct(record: dict[str, Any]) -> bool:
    score = record.get("score")
    return (
        "prediction" in record
        and record.get("prediction") is not None
        and isinstance(score, dict)
        and score.get("correct") is True
    )


def _status_value(record: dict[str, Any]) -> str:
    value = record.get("status")
    return "missing" if value is None else str(value)


def _call_ids(record: dict[str, Any]) -> list[str]:
    """Return each result's upstream and own call IDs once, preserving order."""
    result: list[str] = []
    seen: set[str] = set()
    for field in ("upstream_call_ids", "call_ids"):
        values = record.get(field, [])
        if not isinstance(values, list):
            continue
        for call_id in values:
            if isinstance(call_id, str) and call_id and call_id not in seen:
                seen.add(call_id)
                result.append(call_id)
    return result


def _event_index(call_events: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], set[str]]:
    indexed: dict[str, Any] = {}
    duplicates: set[str] = set()
    for event in call_events:
        if not isinstance(event, dict):
            continue
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            continue
        if call_id in indexed:
            duplicates.add(call_id)
        else:
            indexed[call_id] = event
    return indexed, duplicates


def _token_value(call_id: str, indexed: dict[str, Any], duplicates: set[str]) -> int | float | None:
    if call_id in duplicates or call_id not in indexed:
        return None
    event = indexed[call_id]
    usage = event.get("usage")
    has_nested = isinstance(usage, dict) and "total_tokens" in usage
    has_legacy = "total_tokens" in event
    nested_value = usage.get("total_tokens") if has_nested else None
    legacy_value = event.get("total_tokens") if has_legacy else None
    if has_nested and has_legacy and nested_value != legacy_value:
        return None
    value = nested_value if has_nested else legacy_value
    if not _is_number(value) or value < 0:
        return None
    return value


def summarize_ledger(events, records):
    """Account for the run's entire saved ledger, even before a trial is saved."""
    indexed, duplicates = _event_index(events)
    referenced = {call_id for record in records for call_id in _call_ids(record)}
    tokens = {call_id: _token_value(call_id, indexed, duplicates) for call_id in indexed}
    missing = sorted(call_id for call_id, value in tokens.items() if value is None)
    known = sum(value for value in tokens.values() if value is not None)
    return {"recorded_call_count": len(indexed), "known_total_tokens": known,
            "full_total_tokens": known if not missing else None,
            "missing_usage_call_ids": missing,
            "unattributed_call_ids": sorted(set(indexed) - referenced),
            "scope": "Current run ledger only, including recorded calls without a saved arm result; excludes external cached ledgers"}


def _usage_for_records(
    records: list[dict[str, Any]],
    indexed_events: dict[str, Any],
    duplicate_event_ids: set[str],
) -> dict[str, Any]:
    attributed_known: int | float = 0
    attributed_references = 0
    attributed_missing: list[dict[str, Any]] = []
    unique_ids: set[str] = set()

    for record_index, record in enumerate(records):
        record_ids = _call_ids(record)
        for call_id in record_ids:
            unique_ids.add(call_id)
            attributed_references += 1
            tokens = _token_value(call_id, indexed_events, duplicate_event_ids)
            if tokens is None:
                attributed_missing.append({"call_id": call_id, "record_index": record_index})
            else:
                attributed_known += tokens

    unique_known: int | float = 0
    unique_missing: list[str] = []
    for call_id in sorted(unique_ids):
        tokens = _token_value(call_id, indexed_events, duplicate_event_ids)
        if tokens is None:
            unique_missing.append(call_id)
        else:
            unique_known += tokens

    attributed_full = attributed_known if attributed_references > 0 and not attributed_missing else None
    unique_full = unique_known if unique_ids and not unique_missing else None
    return {
        "attributed": {
            "referenced_call_count": attributed_references,
            "known_call_count": attributed_references - len(attributed_missing),
            "known_total_tokens": attributed_known,
            "full_total_tokens": attributed_full,
            "missing": attributed_missing,
            "full_coverage": attributed_references > 0 and not attributed_missing,
        },
        "unique_execution": {
            "unique_call_count": len(unique_ids),
            "known_call_count": len(unique_ids) - len(unique_missing),
            "known_total_tokens": unique_known,
            "full_total_tokens": unique_full,
            "missing": unique_missing,
            "full_coverage": bool(unique_ids) and not unique_missing,
        },
    }


def _execution_for_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    known_total: int | float = 0
    missing_runs = 0
    solver_runtime_known_total = 0.0
    optimize_call_count = 0
    missing_runtime_count = 0
    missing_runtime_runs = 0
    for record in records:
        attempts = record.get("attempts")
        execution_rows = (
            [(a.get("execution"), a.get("status")) for a in attempts]
            if attempts
            else [(record.get("execution"), record.get("status"))]
        )
        missing = False
        for execution, run_status in execution_rows:
            elapsed = execution.get("elapsed_seconds") if isinstance(execution, dict) else None
            if _is_number(elapsed) and elapsed >= 0:
                known_total += elapsed
            else:
                missing = True
            if not isinstance(execution, dict):
                upstream = record.get("upstream")
                upstream_failed = (
                    run_status == "upstream_failed"
                    or (isinstance(upstream, dict) and upstream.get("status") == "failed")
                )
                if not upstream_failed:
                    # Missing execution on a successful or otherwise unknown record
                    # is not evidence that no solver ran.
                    missing_runtime_runs += 1
                continue
            solver = execution.get("solver_status")
            calls = solver.get("optimize_calls") if isinstance(solver, dict) else None
            solver_was_used = (
                execution.get("solver_limits_applied") is True
                or isinstance(solver, dict)
                or (solver is not None and solver != "")
            )
            if isinstance(calls, list):
                if not calls:
                    if solver_was_used:
                        missing_runtime_runs += 1
                    continue
                recorded_count = solver.get("optimize_call_count", len(calls))
                optimize_call_count += len(calls)
                if not (isinstance(recorded_count, int) and not isinstance(recorded_count, bool)
                        and recorded_count == len(calls)):
                    missing_runtime_runs += 1
                for call in calls:
                    runtime = call.get("runtime_seconds") if isinstance(call, dict) else None
                    if _is_number(runtime) and runtime >= 0:
                        solver_runtime_known_total += runtime
                    else:
                        missing_runtime_count += 1
            elif solver_was_used:
                # Legacy solver records have no per-optimize Runtime evidence.
                missing_runtime_runs += 1
        if missing:
            missing_runs += 1
    return {
        "known_total_seconds": known_total,
        "missing_runs": missing_runs,
        "full_total_seconds": known_total if records and missing_runs == 0 else None,
        "scope": "all recorded execution attempts; legacy records without attempts use their saved final execution",
        "solver_runtime": {
            "optimize_call_count": optimize_call_count,
            "known_total_seconds": solver_runtime_known_total,
            "missing_runtime_count": missing_runtime_count,
            "missing_runtime_runs": missing_runtime_runs,
            "full_total_seconds": (
                solver_runtime_known_total
                if records and missing_runtime_count == 0 and missing_runtime_runs == 0
                else None
            ),
            "scope": "sum of Gurobi Runtime for recorded optimize calls only; separate from subprocess elapsed time",
        },
    }


def _request_latency(records, indexed_events, duplicate_event_ids):
    attributed = 0.0
    unique = {}
    missing = []
    for record in records:
        for call_id in _call_ids(record):
            event = indexed_events.get(call_id) or {}
            elapsed = event.get("elapsed_seconds")
            if call_id in duplicate_event_ids or not _is_number(elapsed) or elapsed < 0:
                missing.append(call_id)
                continue
            attributed += elapsed
            unique[call_id] = elapsed
    return {"known_attributed_seconds": attributed,
            "full_attributed_seconds": attributed if not missing else None,
            "known_unique_seconds": sum(unique.values()),
            "missing_call_ids": sorted(set(missing)),
            "scope": "provider-request latency; shared upstream charged once per path, not batch wall time"}


def _group_summary(records: list[dict[str, Any]], indexed_events: dict[str, Any], duplicate_event_ids: set[str]) -> dict[str, Any]:
    first = records[0]
    correct_count = sum(_is_correct(record) for record in records)
    result = {field: first.get(field) for field in _GROUP_FIELDS}
    result.update({
        "question_count": len({_freeze(record.get("case_id")) for record in records}),
        "run_count": len(records),
        "correct_run_count": correct_count,
        "accuracy": correct_count / len(records) if records else None,
        "status_counts": dict(sorted(Counter(_status_value(record) for record in records).items())),
        "solver_status_counts": dict(sorted(Counter(
            str(record.get("execution", {}).get("solver_status", "missing"))
            if isinstance(record.get("execution"), dict)
            else "missing"
            for record in records
        ).items())),
        "execution": _execution_for_records(records),
        "usage": _usage_for_records(records, indexed_events, duplicate_event_ids),
        "model_request_latency": _request_latency(records, indexed_events, duplicate_event_ids),
        "ned_trace_summaries": [
            record.get("ned_trace", {}).get("summary")
            for record in records
            if isinstance(record.get("ned_trace"), dict) and "summary" in record.get("ned_trace", {})
        ],
    })
    return result


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _bootstrap_ci(case_differences: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str | None]:
    by_source: dict[str, list[float]] = defaultdict(list)
    for item in case_differences:
        source_id = item.get("source_id")
        if source_id is not None and str(source_id):
            by_source[_freeze(source_id)].append(float(item["difference"]))
    source_means = {
        source_id: sum(values) / len(values)
        for source_id, values in by_source.items()
        if values
    }
    if len(source_means) < 2:
        return None, "fewer_than_two_independent_sources"

    rng = random.Random(_BOOTSTRAP_SEED)
    source_values = list(source_means.values())
    sample_count = len(source_values)
    estimates: list[float] = []
    for _ in range(_BOOTSTRAP_SAMPLES):
        draw = [source_values[rng.randrange(sample_count)] for _ in range(sample_count)]
        estimates.append(sum(draw) / sample_count)
    return {
        "lower": _percentile(estimates, 0.025),
        "upper": _percentile(estimates, 0.975),
        "confidence_level": 0.95,
        "method": "source_cluster_percentile_bootstrap",
        "source_count": sample_count,
        "iterations": _BOOTSTRAP_SAMPLES,
        "seed": _BOOTSTRAP_SEED,
    }, None


def _comparison_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_identity: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"native": [], "ned": []})
    for record in records:
        arm = record.get("arm")
        if arm not in _ARMS:
            continue
        fields = (*_PAIR_SCOPE_FIELDS, "case_id", "repeat")
        by_identity[_field_key(record, fields)][arm].append(record)

    valid_by_scope: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    invalid_by_scope: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    unmatched_by_scope: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)

    for identity, sides in by_identity.items():
        # Scope occupies the first four positions in the match identity.
        scope = identity[:len(_PAIR_SCOPE_FIELDS)]
        native = sides["native"]
        ned = sides["ned"]
        base = {field: (native or ned)[0].get(field) for field in (*_PAIR_SCOPE_FIELDS, "case_id", "repeat")}
        if len(native) > 1 or len(ned) > 1:
            invalid_by_scope[scope].append({**base, "reason": "duplicate_arm_record"})
            if not native or not ned:
                present_arm = "native" if native else "ned"
                unmatched_by_scope[scope].append({**base, "present_arm": present_arm, "missing_arm": "ned" if present_arm == "native" else "native"})
            continue
        if not native or not ned:
            present_arm = "native" if native else "ned"
            unmatched_by_scope[scope].append({**base, "present_arm": present_arm, "missing_arm": "ned" if present_arm == "native" else "native"})
            continue

        native_record, ned_record = native[0], ned[0]
        native_hash = native_record.get("input_hash")
        ned_hash = ned_record.get("input_hash")
        if (not isinstance(native_hash, str) or not native_hash
                or not isinstance(ned_hash, str) or not ned_hash
                or native_hash != ned_hash):
            invalid_by_scope[scope].append({
                **base,
                "reason": "input_hash_missing" if not native_hash or not ned_hash else "input_hash_mismatch",
                "native_input_hash": native_hash,
                "ned_input_hash": ned_hash,
            })
            continue
        native_source = native_record.get("source_id")
        ned_source = ned_record.get("source_id")
        source_id = native_source if (
            native_source is not None
            and native_source != ""
            and _freeze(native_source) == _freeze(ned_source)
        ) else None
        valid_by_scope[scope].append({
            **base,
            "source_id": source_id,
            "native_correct": _is_correct(native_record),
            "ned_correct": _is_correct(ned_record),
        })

    scopes = set(valid_by_scope) | set(invalid_by_scope) | set(unmatched_by_scope)
    summaries: list[dict[str, Any]] = []
    for scope in sorted(scopes):
        valid_pairs = valid_by_scope[scope]
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in valid_pairs:
            by_case[_freeze(pair.get("case_id"))].append(pair)

        counts = Counter()
        case_differences: list[dict[str, Any]] = []
        for repeats in by_case.values():
            native_mean = sum(pair["native_correct"] for pair in repeats) / len(repeats)
            ned_mean = sum(pair["ned_correct"] for pair in repeats) / len(repeats)
            difference = ned_mean - native_mean
            if difference > 0:
                counts["ned_wins"] += 1
            elif difference < 0:
                counts["native_wins"] += 1
            elif native_mean == 1 and ned_mean == 1:
                counts["both_correct"] += 1
            elif native_mean == 0 and ned_mean == 0:
                counts["both_wrong"] += 1
            else:
                counts["tied_mixed"] += 1
            source_ids = {_freeze(pair.get("source_id")) for pair in repeats if pair.get("source_id") is not None}
            has_missing_source = any(pair.get("source_id") is None for pair in repeats)
            case_source = repeats[0].get("source_id") if len(source_ids) == 1 and not has_missing_source else None
            case_differences.append({
                "case_id": repeats[0].get("case_id"),
                "source_id": case_source,
                "native_accuracy": native_mean,
                "ned_accuracy": ned_mean,
                "difference": difference,
                "repeat_pair_count": len(repeats),
            })

        ci, ci_reason = _bootstrap_ci(case_differences)
        representative = (valid_pairs or invalid_by_scope[scope] or unmatched_by_scope[scope])[0]
        row = {field: representative.get(field) for field in _PAIR_SCOPE_FIELDS}
        row.update({
            "valid_pair_run_count": len(valid_pairs),
            "paired_case_count": len(by_case),
            "ned_wins": counts["ned_wins"],
            "native_wins": counts["native_wins"],
            "both_correct": counts["both_correct"],
            "both_wrong": counts["both_wrong"],
            "tied_mixed": counts["tied_mixed"],
            "invalid_pair_count": len(invalid_by_scope[scope]),
            "invalid_pairs": invalid_by_scope[scope],
            "unmatched_record_count": len(unmatched_by_scope[scope]),
            "unmatched_records": unmatched_by_scope[scope],
            "case_differences": case_differences,
            "ci_excluded_case_count": sum(item["source_id"] is None for item in case_differences),
            "ci_95": ci,
            "ci_reason": ci_reason,
        })
        summaries.append(row)
    return summaries


def _planning_summary(records: list[dict[str, Any]], planned: dict[str, Any]) -> dict[str, Any]:
    case_ids = planned.get("case_ids")
    repeats = planned.get("repeats")
    arms = planned.get("arms")
    if not isinstance(case_ids, list) or isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("planned requires case_ids:list and positive integer repeats")
    if len({_freeze(case_id) for case_id in case_ids}) != len(case_ids):
        raise ValueError("planned.case_ids must be unique")
    if not isinstance(arms, list) or not arms or any(arm not in _ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError("planned.arms must be unique values from ['native', 'ned']")
    expected_per_arm = len(case_ids) * repeats
    planned_case_keys = {_freeze(case_id) for case_id in case_ids}

    scopes: dict[tuple[str, ...], dict[str, Any]] = {}
    for record in records:
        scope_key = _field_key(record, _PAIR_SCOPE_FIELDS)
        scopes.setdefault(scope_key, {field: record.get(field) for field in _PAIR_SCOPE_FIELDS})
    if not scopes:
        scopes[tuple("null" for _ in _PAIR_SCOPE_FIELDS)] = {field: None for field in _PAIR_SCOPE_FIELDS}

    observed: dict[tuple[str, ...], dict[str, dict[str, set[tuple[str, str]]]]] = defaultdict(lambda: defaultdict(set))
    for record in records:
        arm = record.get("arm")
        case_key = _freeze(record.get("case_id"))
        if arm in arms and case_key in planned_case_keys and record.get("repeat") is not None:
            scope_key = _field_key(record, _PAIR_SCOPE_FIELDS)
            observed[scope_key][arm].add((case_key, _freeze(record.get("repeat"))))

    per_scope: list[dict[str, Any]] = []
    total_missing = 0
    complete = True
    for scope_key, scope_values in sorted(scopes.items()):
        observed_counts: dict[str, int] = {}
        missing_counts: dict[str, int] = {}
        for arm in arms:
            arm_count = len(observed[scope_key].get(arm, set()))
            observed_counts[arm] = arm_count
            missing_counts[arm] = max(0, expected_per_arm - arm_count)
            total_missing += missing_counts[arm]
            if missing_counts[arm] > 0 or arm_count > expected_per_arm:
                complete = False
        per_scope.append({
            "scope": scope_values,
            "planned_per_arm": expected_per_arm,
            "observed_unique_case_repeats_per_arm": observed_counts,
            "missing_records_per_arm": missing_counts,
        })
    return {
        "planned_per_arm": expected_per_arm,
        "arms": list(arms),
        "case_count": len(case_ids),
        "repeats_per_case": repeats,
        "missing_records": total_missing,
        "complete": complete,
        "per_scope": per_scope,
    }


def summarize_records(
    records: list[dict[str, Any]],
    call_events: list[dict[str, Any]] = (),
    planned: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize runs, usage, paired arms, and optional planned coverage.

    Excluded records are ignored. A planned case is counted present by unique
    ``(case_id, repeat)`` rows in each observed dataset/version/model/framework
    scope. For paired reporting, the required fields are matched exactly; input
    hashes must agree before a pair contributes to gains. Rescored rows are
    partitioned by ``rescore_threshold`` in groups, comparisons, and planning.
    """
    if not isinstance(records, list):
        raise TypeError("records must be a list")
    included: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"records[{index}] must be a dict")
        if record.get("excluded") is True:
            continue
        included.append(record)
    if not isinstance(call_events, (list, tuple)):
        raise TypeError("call_events must be a list or tuple")

    indexed_events, duplicate_event_ids = _event_index(call_events)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in included:
        groups[_field_key(record, _GROUP_FIELDS)].append(record)

    group_rows = [
        _group_summary(group_records, indexed_events, duplicate_event_ids)
        for _, group_records in sorted(groups.items())
    ]
    correct_count = sum(_is_correct(record) for record in included)
    result: dict[str, Any] = {
        "overall": {
            "question_count": len({
                _field_key(record, ("dataset", "dataset_version", "case_id"))
                for record in included
            }),
            "run_count": len(included),
            "correct_run_count": correct_count,
            "accuracy": correct_count / len(included) if included else None,
            "status_counts": dict(sorted(Counter(_status_value(record) for record in included).items())),
            "execution": _execution_for_records(included),
            "usage": _usage_for_records(included, indexed_events, duplicate_event_ids),
        },
        "groups": group_rows,
        "comparisons": _comparison_summaries(included),
        "planning": _planning_summary(included, planned) if planned is not None else None,
    }
    return result


def _manifest_references(manifest: dict[str, Any]) -> dict[tuple[str, str, str], Any]:
    if not isinstance(manifest, dict):
        raise TypeError("manifest must be a dict")
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must be 1")
    dataset = manifest.get("dataset")
    dataset_version = manifest.get("dataset_version")
    if not isinstance(dataset, str) or not dataset or not isinstance(dataset_version, str) or not dataset_version:
        raise ValueError("manifest requires non-empty dataset and dataset_version strings")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("manifest.cases must be a list")
    result: dict[tuple[str, str, str], Any] = {}
    for index, item in enumerate(cases):
        if isinstance(item, dict) and str(item.get("review_status", "")).startswith("excluded_"):
            continue
        if not isinstance(item, dict) or not all(key in item for key in ("case_id", "reference", "scoring")):
            raise ValueError(f"manifest.cases[{index}] requires case_id, reference, and scoring")
        scoring = item.get("scoring")
        if not isinstance(scoring, dict) or scoring.get("name") not in {
            "nextor_relative", "objective_relative", "nextor_spring_legacy",
        }:
            raise ValueError(f"manifest.cases[{index}].scoring.name must use a relative protocol")
        reference_check = score_answer(item["reference"], item["reference"], scoring,
                                       case_id=item["case_id"])
        if not reference_check["correct"]:
            raise ValueError(f"invalid reference/scoring contract for {item['case_id']}")
        key = tuple(_freeze(value) for value in (dataset, dataset_version, item["case_id"]))
        if key in result:
            raise ValueError(f"duplicate manifest case_id at index {index}")
        result[key] = (item["reference"], copy.deepcopy(scoring))
    return result


def rescore_records(
    records: Iterable[dict[str, Any]],
    manifest: dict[str, Any],
    thresholds: tuple[float, ...] = (0.01, 0.001, 0.0001),
    exclude_case_ids: Iterable[Any] = (),
) -> list[dict[str, Any]]:
    """Clone records and rescore stored predictions against versioned references.

    Manifest shape follows ``data.py``: top-level ``schema_version``,
    ``dataset``, ``dataset_version``, and ``cases``; each case supplies
    ``case_id``, ``reference``, and an explicit supported relative protocol.
    One clone is returned per input and threshold. Excluded case IDs are
    marked and left out of later summaries.
    """
    references = _manifest_references(manifest)
    threshold_values = tuple(thresholds)
    if not threshold_values:
        raise ValueError("thresholds must contain at least one value")
    # Reuse the scorer's protocol validation for finite/non-negative thresholds.
    for threshold in threshold_values:
        score_answer(1, 1, {"name": "nextor_relative", "tolerance": threshold})
    excluded = {_freeze(case_id) for case_id in exclude_case_ids}

    result: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"records[{index}] must be a dict")
        if not all(field in record for field in ("dataset", "dataset_version", "case_id")):
            raise ValueError(f"records[{index}] requires dataset, dataset_version, and case_id")
        is_excluded = _freeze(record["case_id"]) in excluded
        reference_key = tuple(_freeze(record[field]) for field in ("dataset", "dataset_version", "case_id"))
        if not is_excluded and reference_key not in references:
            raise ValueError(
                f"no manifest reference for dataset={record['dataset']!r}, "
                f"dataset_version={record['dataset_version']!r}, case_id={record['case_id']!r}"
            )

        for threshold in threshold_values:
            cloned = copy.deepcopy(record)
            cloned["rescore_threshold"] = threshold
            if is_excluded:
                cloned["excluded"] = True
            elif "prediction" not in record or record.get("prediction") is None:
                cloned["score"] = {
                    "correct": False,
                    "reason": "missing_prediction",
                    "errors": [{"path": "$", "message": "prediction is missing"}],
                }
            else:
                reference, scoring = references[reference_key]
                cloned["score"] = score_answer(
                    record["prediction"],
                    reference,
                    {**scoring, "tolerance": threshold},
                    case_id=record["case_id"],
                )
                cloned["scoring"] = {**scoring, "tolerance": threshold}
            result.append(cloned)
    return result


__all__ = ["summarize_records", "rescore_records"]
