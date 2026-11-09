"""Small, deterministic data preparation helpers for rebuttal experiments.

This module only reads JSON and builds prompt-safe case records. It never
executes source content or treats a reference value as prompt material.
"""

from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA_VERSION = 1
_CASE_FIELDS = {
    "case_id",
    "source_id",
    "question",
    "math_input",
    "reference",
    "answer_spec",
    "scoring",
    "review_status",
}
_MATH_FIELDS = {
    "Parameters_List",
    "Variables_List",
    "Constraint_Table",
    "Objective",
    "Problem_Type",
}


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _read_source(source: Any) -> tuple[Any, str, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        raw = path.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read UTF-8 JSON source {path}") from exc
        return data, str(path), hashlib.sha256(raw).hexdigest()

    data = copy.deepcopy(source)
    try:
        encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("source must be JSON-compatible") from exc
    return data, "<in-memory>", hashlib.sha256(encoded).hexdigest()


def _validate_case(case: Any, index: int) -> None:
    if not isinstance(case, dict):
        raise ValueError(f"cases[{index}] must be an object")
    missing = _CASE_FIELDS - case.keys()
    if missing:
        raise ValueError(f"cases[{index}] missing required fields: {', '.join(sorted(missing))}")
    _require_nonempty_string(case["case_id"], f"cases[{index}].case_id")
    _require_nonempty_string(case["source_id"], f"cases[{index}].source_id")
    if not isinstance(case["question"], str):
        raise ValueError(f"cases[{index}].question must be a string")
    if case["math_input"] is not None and not isinstance(case["math_input"], dict):
        raise ValueError(f"cases[{index}].math_input must be an object or null")
    if not isinstance(case["answer_spec"], dict):
        raise ValueError(f"cases[{index}].answer_spec must be an object")
    answer_spec = case["answer_spec"]
    if set(answer_spec) - {"kind", "fields", "order"}:
        raise ValueError(f"cases[{index}].answer_spec contains unsupported fields")
    if not isinstance(answer_spec.get("kind"), str):
        raise ValueError(f"cases[{index}].answer_spec.kind must be a string")
    for key in ("fields", "order"):
        if key in answer_spec and (
            not isinstance(answer_spec[key], list)
            or not all(isinstance(item, str) for item in answer_spec[key])
        ):
            raise ValueError(f"cases[{index}].answer_spec.{key} must be a list of strings")
    if not isinstance(case["scoring"], dict):
        raise ValueError(f"cases[{index}].scoring must be an object")
    _require_nonempty_string(case["review_status"], f"cases[{index}].review_status")
    if case["review_status"].lower() in {"ready", "verified", "approved"}:
        if not _valid_numeric_reference(case["reference"]):
            raise ValueError(f"cases[{index}].reference must contain finite numeric value(s) for runnable cases")


def _valid_numeric_reference(value: Any) -> bool:
    def finite_number(item: Any) -> bool:
        if isinstance(item, bool):
            return False
        if isinstance(item, int):
            return True
        return isinstance(item, float) and math.isfinite(item)

    if finite_number(value):
        return True
    if isinstance(value, list):
        return bool(value) and all(finite_number(item) for item in value)
    return isinstance(value, dict) and bool(value) and all(finite_number(item) for item in value.values())


def _numeric_string(value: Any) -> tuple[Any, bool]:
    """Convert only a complete finite numeric string; preserve other text."""
    if not isinstance(value, str):
        return value, _valid_numeric_reference(value)
    stripped = value.strip()
    if not stripped or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", stripped):
        return value, False
    try:
        decimal = Decimal(stripped)
    except InvalidOperation:
        return value, False
    if not decimal.is_finite():
        return value, False
    if decimal == decimal.to_integral_value():
        return int(decimal), True
    number = float(decimal)
    return (number, True) if math.isfinite(number) else (value, False)


def _convert_named_numeric(value: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    converted = {}
    for key, item in value.items():
        number, supported = _numeric_string(item)
        if not supported:
            return copy.deepcopy(value), False
        converted[str(key)] = number
    return converted, bool(converted)


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate a version-1 JSON manifest."""
    path = Path(path)
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load UTF-8 JSON manifest {path}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    _require_nonempty_string(manifest.get("dataset"), "dataset")
    _require_nonempty_string(manifest.get("dataset_version"), "dataset_version")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("cases must be a list")
    seen: set[str] = set()
    for index, case in enumerate(cases):
        _validate_case(case, index)
        case_id = case["case_id"]
        if case_id in seen:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
    return manifest


def _normalise_parameters(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Parameters_List must be a list")
    result = []
    for index, parameter in enumerate(value):
        if isinstance(parameter, str):
            name = parameter
            item = {"Name": name}
        elif isinstance(parameter, dict):
            name = parameter.get("Name", parameter.get("name"))
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Parameters_List[{index}] needs a Name")
            item = {"Name": name.strip()}
            for key in ("Type", "Value", "Unit", "Units", "Meaning", "meaning"):
                if key in parameter:
                    item[key] = copy.deepcopy(parameter[key])
        else:
            raise ValueError(f"Parameters_List[{index}] must be an object or string")
        result.append(item)
    return result


def _normalise_variables(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("Variables_List must be a non-empty list")
    result = []
    for index, variable in enumerate(value):
        if isinstance(variable, str):
            item = {"symbol": variable}
        elif isinstance(variable, dict):
            symbol = variable.get("symbol", variable.get("Symbol"))
            item = {"symbol": symbol}
            for key in ("Meaning", "Type", "Range", "Range ", "Domain", "domain"):
                if key in variable:
                    item[key] = copy.deepcopy(variable[key])
        else:
            raise ValueError(f"Variables_List[{index}] must be an object or string")
        symbol = item.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"Variables_List[{index}] needs a symbol")
        item["symbol"] = symbol.strip()
        result.append(item)
    return result


def _normalise_constraints(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("Constraint_Table must be a list")
    result = []
    for index, constraint in enumerate(value):
        if isinstance(constraint, dict):
            expression = constraint.get("Mathematical_expressions")
            if expression is None:
                expression = constraint.get("expression", constraint.get("Expression"))
            if expression is None:
                raise ValueError(f"Constraint_Table[{index}] needs an expression")
            item = {}
            name = constraint.get("name", constraint.get("Name"))
            if name is not None:
                item["name"] = copy.deepcopy(name)
            item["Mathematical_expressions"] = expression
        elif isinstance(constraint, (list, tuple)) and len(constraint) >= 2:
            item = {"name": str(constraint[0]), "Mathematical_expressions": constraint[1]}
        elif isinstance(constraint, str):
            item = {"Mathematical_expressions": constraint}
        else:
            raise ValueError(f"Constraint_Table[{index}] has unsupported structure")
        expression = item["Mathematical_expressions"]
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Constraint_Table[{index}] expression must be non-empty text")
        result.append(item)
    return result


def validate_math_input(entry: dict[str, Any]) -> dict[str, Any]:
    """Copy only NED adapter fields and normalize their supported structures."""
    if not isinstance(entry, dict):
        raise ValueError("math input entry must be an object")
    if "math_input" in entry:
        entry = entry["math_input"]
        if not isinstance(entry, dict):
            raise ValueError("math_input must be an object")

    raw_objective = entry.get("Objective")
    if not isinstance(raw_objective, dict):
        if isinstance(raw_objective, str):
            objective = {"Mathematical_expressions": raw_objective}
        else:
            raise ValueError("Objective must be an object or expression string")
    else:
        objective = copy.deepcopy(raw_objective)
    expression = objective.get("Mathematical_expressions")
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Objective.Mathematical_expressions must be non-empty text")
    objective = {"Mathematical_expressions": expression}
    for key in ("Objective_sentence", "Sense", "sense", "direction"):
        if isinstance(raw_objective, dict) and key in raw_objective:
            objective[key] = copy.deepcopy(raw_objective[key])

    problem_type = entry.get("Problem_Type")
    if not isinstance(problem_type, str) or not problem_type.strip():
        raise ValueError("Problem_Type must be non-empty text")

    return {
        "Parameters_List": _normalise_parameters(entry.get("Parameters_List", [])),
        "Variables_List": _normalise_variables(entry.get("Variables_List")),
        "Constraint_Table": _normalise_constraints(entry.get("Constraint_Table", [])),
        "Objective": objective,
        "Problem_Type": problem_type.strip(),
    }


def _records(data: Any) -> list[tuple[str, dict[str, Any]]]:
    if isinstance(data, list):
        pairs = [(str(index), item) for index, item in enumerate(data)]
    elif isinstance(data, dict):
        pairs = [(str(key), value) for key, value in data.items()]
    else:
        raise ValueError("source must be a JSON object or list")
    if any(not isinstance(item, dict) for _, item in pairs):
        raise ValueError("each source record must be an object")
    return pairs


def _reference(record: dict[str, Any], source_format: str) -> tuple[Any, dict[str, Any], Any]:
    if source_format == "optibench" and isinstance(record.get("results"), dict):
        raw = copy.deepcopy(record["results"])
        named, supported = _convert_named_numeric(raw)
        if supported:
            return named, {"label_status": "candidate", "kind": "named_numeric"}, raw
        return raw, {"label_status": "candidate", "kind": "unsupported_needs_review"}, raw

    for key in ("answer", "reference", "ground_truth", "gold"):
        if key in record:
            raw = copy.deepcopy(record[key])
            value = raw
            if isinstance(value, str):
                converted, supported = _numeric_string(value)
                if supported:
                    return converted, {"label_status": "candidate", "kind": "scalar"}, raw
                return value, {"label_status": "candidate", "kind": "scalar"}, raw
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if _valid_numeric_reference(value):
                    return value, {"label_status": "candidate", "kind": "scalar"}, raw
                return value, {"label_status": "candidate", "kind": "unsupported_needs_review"}, raw
            if isinstance(value, list) and _valid_numeric_reference(value):
                return value, {"label_status": "candidate", "kind": "numeric_list"}, raw
            return value, {"label_status": "candidate", "kind": "unsupported_needs_review"}, raw
    return None, {"label_status": "missing", "kind": "unsupported_needs_review"}, None


def _answer_spec(reference: Any) -> dict[str, Any]:
    if isinstance(reference, str) or (
        isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and _valid_numeric_reference(reference)
    ):
        return {"kind": "scalar", "fields": ["answer"], "order": ["answer"]}
    if isinstance(reference, dict) and _valid_numeric_reference(reference):
        fields = list(reference)
        return {"kind": "named_numeric", "fields": fields, "order": fields}
    if isinstance(reference, list) and _valid_numeric_reference(reference):
        return {"kind": "numeric_list", "fields": ["answer"], "order": ["answer"]}
    return {"kind": "needs_review", "fields": [], "order": []}


def _is_nonlinear(record: dict[str, Any]) -> bool | None:
    value = record.get("Problem_Type", record.get("problem_type", record.get("type")))
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if normalized.startswith("nonlinear") or normalized.startswith("non-linear") or normalized == "nlp":
        return True
    if normalized.startswith("linear") or normalized in {"lp", "milp", "ilp"}:
        return False
    return None


def prepare_manifest(
    source: Any,
    dataset: str,
    source_format: str = "nora",
    dataset_version: str | None = None,
    *,
    nonlinear_only: bool = False,
    scoring_name: str | None = None,
) -> dict[str, Any]:
    """Convert NORA or OptiBench JSON records without dropping review cases."""
    _require_nonempty_string(dataset, "dataset")
    if source_format not in {"nora", "optibench"}:
        raise ValueError("source_format must be 'nora' or 'optibench'")
    if nonlinear_only and source_format != "optibench":
        raise ValueError("nonlinear_only is supported only for optibench sources with explicit type labels")
    data, source_path, source_hash = _read_source(source)
    if scoring_name is not None:
        from .scoring import score_answer

        contract = {"name": scoring_name}
        try:
            score_answer(1, 1, contract)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported scoring_name: {scoring_name}") from exc
    cases = []
    for source_key, original in _records(data):
        record = copy.deepcopy(original)
        if nonlinear_only and _is_nonlinear(record) is False:
            continue
        source_id = str(record.get("index", source_key))
        case_id = f"{dataset}:{source_id}"
        question = record.get("question")
        reasons = []
        if not isinstance(question, str) or not question:
            question = ""
            reasons.append("missing_or_invalid_question")
        elif re.search(r"(?im)^\s*(?:Answer|Ground\s+truth|Reference)\s*:", question):
            reasons.append("question_may_include_answer_material")

        math_input = None
        if source_format == "nora":
            try:
                math_input = validate_math_input(record)
            except ValueError as exc:
                reasons.append(f"math_input_unavailable: {exc}")

        reference, reference_info, raw_reference = _reference(record, source_format)
        if scoring_name is not None:
            scoring = {"name": scoring_name}
        else:
            scoring = {"name": None}
            reasons.append("scoring_protocol_missing")
        if reference_info["kind"] == "unsupported_needs_review":
            reasons.append("reference_requires_review")
        answer_spec = _answer_spec(reference)

        label = record.get("label", record.get("ori", record.get("type", "candidate")))
        metadata = {
            key: copy.deepcopy(record[key])
            for key in ("model", "model_name", "unit", "units")
            if key in record and isinstance(record[key], (str, int, float, list, dict))
        }
        review_status = "needs_review"
        cases.append(
            {
                "case_id": case_id,
                "source_id": source_id,
                "question": question,
                "math_input": math_input,
                "reference": reference,
                "reference_source": raw_reference,
                "reference_status": reference_info["label_status"],
                "reference_kind": reference_info["kind"],
                "answer_spec": answer_spec,
                "scoring": scoring,
                "review_status": review_status,
                "review_reasons": reasons,
                "source": {
                    "path": source_path,
                    "sha256": source_hash,
                    "label": str(label),
                },
                "metadata": metadata,
            }
        )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": dataset,
        "dataset_version": dataset_version or f"sha256:{source_hash}",
        "cases": cases,
    }
    # Also enforce identity and required fields for in-memory prepared manifests.
    seen: set[str] = set()
    for index, case in enumerate(cases):
        _validate_case(case, index)
        if case["case_id"] in seen:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        seen.add(case["case_id"])
    return manifest


def public_case(case: dict[str, Any]) -> dict[str, Any]:
    """Return the explicit, reference-free subset safe for model prompts."""
    if not isinstance(case, dict):
        raise ValueError("case must be an object")
    math_input = case.get("math_input")
    if math_input is not None:
        math_input = validate_math_input(math_input)
    answer_spec = case.get("answer_spec")
    if not isinstance(answer_spec, dict):
        raise ValueError("answer_spec must be an object")
    return {
        "question": copy.deepcopy(case.get("question", "")),
        "math_input": copy.deepcopy(math_input),
        "answer_spec": {
            "kind": answer_spec.get("kind", "needs_review"),
            "fields": [item for item in answer_spec.get("fields", []) if isinstance(item, str)],
            "order": [item for item in answer_spec.get("order", []) if isinstance(item, str)],
        },
    }
