"""Strict parsing and scoring for rebuttal benchmark answers."""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any


class AnswerParseError(ValueError):
    """Raised when stdout does not contain exactly one valid final answer."""

    def __init__(self, message: str, path: str = "$") -> None:
        super().__init__(message)
        self.path = path


_ANSWER_LINE = re.compile(r"^\s*FinalAnswer=(.*?)\s*$")
_MARKER_PRESENT = re.compile(r"FinalAnswer\s*=")
_PROTOCOL_ALIASES = {
    "nextor": "nextor_relative",
    "optibench": "optibench_absolute",
    "optmath": "optmath_normalized",
    "exact": "exact",
}
_PROTOCOL_NAMES = {
    "nextor_relative",
    "objective_relative",
    "nextor_spring_legacy",
    "optibench_absolute",
    "optmath_normalized",
    "exact",
}
_DEFAULT_TOLERANCES = {
    "nextor_relative": Decimal("0.01"),
    "objective_relative": Decimal("0.01"),
    "nextor_spring_legacy": Decimal("0.01"),
    "optibench_absolute": Decimal("0.0001"),
    "optmath_normalized": Decimal("0.000001"),
}


def _valid_json_value(value: Any, path: str = "$", *, top_level: bool = True) -> None:
    """Require JSON answer data to be numeric and finite, with string dict keys."""
    if isinstance(value, bool):
        raise AnswerParseError(f"boolean is not a numeric answer at {path}", path)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise AnswerParseError(f"non-finite number at {path}", path)
        return
    if isinstance(value, list):
        if not value:
            raise AnswerParseError(f"empty list is not an answer at {path}", path)
        for index, item in enumerate(value):
            _valid_json_value(item, f"{path}[{index}]", top_level=False)
        return
    if isinstance(value, dict):
        if not value:
            raise AnswerParseError(f"empty object is not an answer at {path}", path)
        for key, item in value.items():
            if not isinstance(key, str):
                raise AnswerParseError(f"non-string object key at {path}", path)
            _valid_json_value(item, f"{path}[{json.dumps(key, ensure_ascii=False)}]", top_level=False)
        return
    allowed = "number, list, or dict" if top_level else "number or nested list/dict"
    raise AnswerParseError(f"expected {allowed} at {path}", path)


def _reject_json_constant(token: str) -> None:
    raise AnswerParseError(f"invalid non-finite JSON number {token}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise AnswerParseError(f"duplicate object key {key!r}")
        value[key] = item
    return value


def parse_answer(stdout: str) -> Any:
    """Parse exactly one standalone ``FinalAnswer=<JSON>`` line.

    The JSON payload may be wrapped in either ``【...】`` or ``<...>``.
    Only numbers, lists, and dictionaries containing the same numeric data
    are accepted; strings, null, booleans, and non-finite values are rejected.
    """
    if not isinstance(stdout, str):
        raise AnswerParseError("stdout must be text")

    marked_lines = [line for line in stdout.splitlines() if _MARKER_PRESENT.search(line)]
    if not marked_lines:
        raise AnswerParseError("missing standalone FinalAnswer line")
    if len(marked_lines) != 1:
        raise AnswerParseError("multiple FinalAnswer markers")

    match = _ANSWER_LINE.fullmatch(marked_lines[0])
    if match is None:
        raise AnswerParseError("FinalAnswer must be the only content on its line")
    payload = match.group(1).strip()
    if len(payload) >= 2 and payload[0] == "【" and payload[-1] == "】":
        payload = payload[1:-1].strip()
    elif len(payload) >= 2 and payload[0] == "<" and payload[-1] == ">":
        payload = payload[1:-1].strip()

    try:
        value = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except AnswerParseError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AnswerParseError(f"invalid JSON answer: {exc}") from exc

    _valid_json_value(value)
    return value


def _decimal(value: int | float | Decimal) -> Decimal:
    # Decimal(str(...)) gives stable, human-readable threshold comparisons for
    # JSON numbers and avoids binary rounding at strict tolerance boundaries.
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid numeric value {value!r}") from exc


def _error_number(value: Decimal) -> int | float | str:
    """Return a finite Python number where possible, otherwise a decimal string."""
    if value == value.to_integral_value():
        try:
            return int(value)
        except (OverflowError, ValueError):
            return str(value)
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return str(value)
    return converted if math.isfinite(converted) else str(value)


def _protocol_settings(protocol: dict[str, Any]) -> tuple[str, Decimal | None]:
    if not isinstance(protocol, dict):
        raise ValueError("protocol must be a dict containing 'name'")
    name = protocol.get("name")
    if not isinstance(name, str):
        raise ValueError("protocol['name'] must be a string")
    canonical_name = _PROTOCOL_ALIASES.get(name, name)
    if canonical_name not in _PROTOCOL_NAMES:
        raise ValueError(f"unknown scoring protocol {name!r}")
    if canonical_name == "exact":
        if "tolerance" in protocol:
            raise ValueError("exact protocol does not accept a tolerance")
        return canonical_name, None

    raw_tolerance = protocol.get("tolerance", _DEFAULT_TOLERANCES[canonical_name])
    if isinstance(raw_tolerance, bool) or not isinstance(raw_tolerance, (int, float, Decimal)):
        raise ValueError("protocol tolerance must be a finite non-negative number")
    if isinstance(raw_tolerance, float) and not math.isfinite(raw_tolerance):
        raise ValueError("protocol tolerance must be a finite non-negative number")
    if isinstance(raw_tolerance, Decimal) and not raw_tolerance.is_finite():
        raise ValueError("protocol tolerance must be a finite non-negative number")
    tolerance = _decimal(raw_tolerance)
    if tolerance < 0:
        raise ValueError("protocol tolerance must be non-negative")
    return canonical_name, tolerance


def _path_child(path: str, key: str) -> str:
    return f"{path}[{json.dumps(key, ensure_ascii=False)}]"


def _compare(
    prediction: Any,
    reference: Any,
    protocol_name: str,
    tolerance: Decimal | None,
    path: str = "$",
    zero_absolute_tolerance: Decimal = Decimal(0),
) -> list[dict[str, Any]]:
    """Return leaf/shape mismatches. Empty list means the complete value passed."""
    prediction_is_number = isinstance(prediction, (int, float)) and not isinstance(prediction, bool)
    reference_is_number = isinstance(reference, (int, float)) and not isinstance(reference, bool)

    if prediction_is_number and reference_is_number:
        pred_decimal = _decimal(prediction)
        ref_decimal = _decimal(reference)
        precision = max(
            50,
            len(pred_decimal.as_tuple().digits) + len(ref_decimal.as_tuple().digits) + 20,
            0 if tolerance is None else len(tolerance.as_tuple().digits) + len(ref_decimal.as_tuple().digits) + 20,
        )
        with localcontext() as context:
            context.prec = precision
            absolute = abs(pred_decimal - ref_decimal)
            relative = None if ref_decimal == 0 else absolute / abs(ref_decimal)

            if protocol_name == "exact":
                passed = prediction == reference
            elif protocol_name in {"nextor_relative", "objective_relative", "nextor_spring_legacy"}:
                passed = absolute <= zero_absolute_tolerance if ref_decimal == 0 else absolute <= tolerance * abs(ref_decimal)
            elif protocol_name == "optibench_absolute":
                passed = absolute < tolerance
            else:  # optmath_normalized
                passed = absolute < tolerance * (abs(ref_decimal) + Decimal(1))

        if passed:
            return []
        return [{
            "path": path,
            "kind": "numeric_mismatch",
            "prediction": prediction,
            "reference": reference,
            "absolute_error": _error_number(absolute),
            "relative_error": None if relative is None else _error_number(relative),
        }]

    if isinstance(prediction, list) and isinstance(reference, list):
        errors: list[dict[str, Any]] = []
        if len(prediction) != len(reference):
            errors.append({
                "path": path,
                "kind": "length_mismatch",
                "prediction_length": len(prediction),
                "reference_length": len(reference),
                "absolute_error": None,
                "relative_error": None,
            })
        for index, (pred_item, ref_item) in enumerate(zip(prediction, reference)):
            errors.extend(_compare(pred_item, ref_item, protocol_name, tolerance, f"{path}[{index}]"))
        return errors

    if isinstance(prediction, dict) and isinstance(reference, dict):
        errors = []
        pred_keys, ref_keys = set(prediction), set(reference)
        for key in sorted(pred_keys - ref_keys):
            errors.append({
                "path": _path_child(path, key),
                "kind": "unexpected_key",
                "absolute_error": None,
                "relative_error": None,
            })
        for key in sorted(ref_keys - pred_keys):
            errors.append({
                "path": _path_child(path, key),
                "kind": "missing_key",
                "absolute_error": None,
                "relative_error": None,
            })
        for key in sorted(pred_keys & ref_keys):
            errors.extend(_compare(prediction[key], reference[key], protocol_name, tolerance, _path_child(path, key)))
        return errors

    return [{
        "path": path,
        "kind": "shape_or_type_mismatch",
        "prediction_type": type(prediction).__name__,
        "reference_type": type(reference).__name__,
        "absolute_error": None,
        "relative_error": None,
    }]


def score_answer(
    prediction: Any, reference: Any, protocol: dict[str, Any], *, case_id: str | None = None,
) -> dict[str, Any]:
    """Score a numeric answer under one explicit protocol.

    Canonical names are ``nextor_relative`` (<= relative tolerance, exact at
    zero), ``optibench_absolute`` (< absolute tolerance),
    ``optmath_normalized`` (< abs-difference/(abs(reference)+1)), and ``exact``.
    Short aliases ``nextor``, ``optibench``, and ``optmath`` map to their
    corresponding canonical protocols; ``exact`` is its own name.

    The approved supplement uses ``objective_relative``: a scalar objective,
    relative error <= tolerance, and absolute error <= 1e-5 only at zero.
    ``nextor_spring_legacy`` permits either reference item for NEXTOR:11 only;
    it records the matched attribute instead of implying objective correctness.
    Historical protocols above keep their original behavior.
    """
    protocol_name, tolerance = _protocol_settings(protocol)
    if protocol_name == "nextor_spring_legacy":
        if case_id != "NEXTOR:11":
            raise ValueError("nextor_spring_legacy is restricted to NEXTOR:11")
        if (not isinstance(reference, list) or len(reference) != 2
                or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in reference)
                or reference[1] != 9.0):
            raise ValueError("spring reference must be [objective_value, 9.0]")
    try:
        _valid_json_value(prediction)
        _valid_json_value(reference)
    except AnswerParseError as exc:
        return {
            "correct": False,
            "reason": "invalid_answer_value",
            "errors": [{
                "path": exc.path,
                "message": str(exc),
                "absolute_error": None,
                "relative_error": None,
            }],
        }

    if protocol_name == "nextor_spring_legacy":
        predictions = prediction if isinstance(prediction, list) else [prediction]
        matches = []
        for pred_index, value in enumerate(predictions):
            if not isinstance(value, (int, float)):
                continue
            for ref_index, target in enumerate(reference):
                if not _compare(value, target, protocol_name, tolerance,
                                zero_absolute_tolerance=Decimal("0.00001")):
                    matches.append({
                        "prediction_index": pred_index,
                        "reference_index": ref_index,
                        "reference_kind": "objective_value" if ref_index == 0 else "effective_coils",
                        "prediction": value,
                        "reference": target,
                    })
        return {
            "correct": bool(matches),
            "reason": "spring_legacy_match" if matches else "spring_legacy_mismatch",
            "matched_references": matches,
            "errors": [] if matches else [{"path": "$", "kind": "no_allowed_reference_match"}],
        }

    if protocol_name == "objective_relative" and (
        not isinstance(prediction, (int, float)) or not isinstance(reference, (int, float))
    ):
        return {"correct": False, "reason": "scalar_objective_required",
                "errors": [{"path": "$", "kind": "shape_or_type_mismatch"}]}
    zero_tolerance = Decimal("0.00001") if protocol_name == "objective_relative" else Decimal(0)
    errors = _compare(prediction, reference, protocol_name, tolerance,
                      zero_absolute_tolerance=zero_tolerance)
    return {
        "correct": not errors,
        "reason": "match" if not errors else "mismatch",
        "errors": errors,
    }


__all__ = ["AnswerParseError", "parse_answer", "score_answer"]
