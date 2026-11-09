"""Deterministic, opt-in bridge from a NExTOR entry to NEDTree-4R advice."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re

import sympy


_MODULE_PATH = Path(__file__).with_name("NEDTree-4R.py")
_SPEC = spec_from_file_location("nextor_nedtree_4r", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import machinery guard
    raise ImportError(f"Cannot load NEDTree-4R from {_MODULE_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

TopDownNEDTree = _MODULE.TopDownNEDTree
NEDTreeError = _MODULE.NEDTreeError


DOMAIN_OBLIGATION_NOTE = (
    "Domain conditions are obligations for the CodingAgent to enforce in the "
    "generated model; this adapter does not add or prove them automatically."
)
_SYMBOL_PATTERN = re.compile(r"^[^\W\d]\w*$", re.UNICODE)
_RELATION_PATTERN = re.compile(r"<=|>=|==|(?<![<>!])=(?!=)|<|>")


def _declared_names(items, key):
    names = []
    for item in items or []:
        value = item.get(key) if isinstance(item, dict) else item
        if value is None:
            continue
        name = str(value).strip()
        if _SYMBOL_PATTERN.fullmatch(name) and name not in names:
            names.append(name)
    return names


def _normalise_expression(text):
    expression = str(text).strip().strip("$")
    replacements = {
        "≤": "<=",
        "≥": ">=",
        "·": "*",
        "×": "*",
        "−": "-",
        r"\left": "",
        r"\right": "",
    }
    for old, new in replacements.items():
        expression = expression.replace(old, new)
    return expression.strip().rstrip(".,;")


def _split_chained_nonstrict_relation(expression):
    matches = list(_RELATION_PATTERN.finditer(expression))
    if len(matches) != 2 or any(match.group() not in {"<=", ">="} for match in matches):
        return None
    first, second = matches
    left = expression[: first.start()].strip()
    middle = expression[first.end() : second.start()].strip()
    right = expression[second.end() :].strip()
    if not left or not middle or not right:
        return None
    return [
        f"{left} {first.group()} {middle}",
        f"{middle} {second.group()} {right}",
    ]


def _objective_candidate(objective):
    if isinstance(objective, dict):
        raw = objective.get("Mathematical_expressions")
    else:
        raw = objective
    if not raw or not str(raw).strip():
        return None

    notes = []
    expression = str(raw).strip()
    before_subject_to = re.split(
        r"\bsubject\s+to\b", expression, maxsplit=1, flags=re.IGNORECASE
    )
    if len(before_subject_to) > 1:
        expression = before_subject_to[0]
        notes.append("kept the objective segment before 'subject to'")

    expression = re.sub(
        r"^\s*(?:maximize|maximise|max|minimize|minimise|min)\b\s*",
        "",
        expression,
        flags=re.IGNORECASE,
    )
    where_parts = re.split(r",\s*where\b", expression, maxsplit=1, flags=re.IGNORECASE)
    if len(where_parts) > 1:
        expression = where_parts[0]
        notes.append("kept the objective expression before the explanatory 'where' clause")
    equality_parts = re.split(r"(?<![<>!])=(?!=)", expression)
    if len(equality_parts) > 1:
        expression = equality_parts[-1]
        notes.append("used the rightmost objective expression after '='")

    return {
        "source": "objective",
        "original": str(raw),
        "expression": _normalise_expression(expression),
        "normalization_notes": notes,
    }


def _constraint_candidates(table):
    candidates = []
    for index, row in enumerate(table or []):
        if isinstance(row, dict):
            label = row.get("name") or row.get("Name") or f"constraint_{index}"
            raw = (
                row.get("Mathematical_expressions")
                or row.get("expression")
                or row.get("Expression")
            )
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            label, raw = row[0], row[1]
        else:
            label, raw = f"constraint_{index}", row
        if not raw or not str(raw).strip():
            continue

        expression = str(raw).strip()
        notes = []
        if "⇔" in expression:
            expression = expression.split("⇔", 1)[0]
            notes.append("used the left representation before '⇔'")

        chained_parts = _split_chained_nonstrict_relation(expression)
        comma_parts = [part.strip() for part in expression.split(",")]
        if chained_parts:
            parts = chained_parts
            notes.append("split a chained non-strict bound into two constraints")
        elif len(comma_parts) > 1 and all(
            _RELATION_PATTERN.search(part) for part in comma_parts
        ):
            parts = comma_parts
            notes.append("split comma-separated constraints")
        else:
            parts = [expression]

        for part_index, part in enumerate(parts):
            suffix = f":{part_index}" if len(parts) > 1 else ""
            candidates.append(
                {
                    "source": f"constraint:{label}{suffix}",
                    "original": str(raw),
                    "expression": _normalise_expression(part),
                    "normalization_notes": list(notes),
                }
            )
    return candidates


def _fallback(reason):
    return (
        "Use the existing mathematical-model/native-Gurobi coding path for this "
        f"candidate; no NEDTree-4R decomposition was applied ({reason})."
    )


def _serialise_candidate(candidate, variables, parameters, candidate_index):
    item = dict(candidate)
    expression = item["expression"]
    if not expression:
        item.update(
            status="unsupported",
            error="EMPTY_EXPRESSION",
            fallback=_fallback("empty expression"),
        )
        return item

    try:
        result = TopDownNEDTree(parameters, variables).process(expression)
    except NEDTreeError as exc:
        # Declared grammar/domain failures fall back visibly. Unexpected
        # implementation errors are deliberately not swallowed by the adapter.
        item.update(
            status="unsupported",
            error=f"{type(exc).__name__}: {exc}",
            fallback=_fallback(str(exc)),
        )
        return item

    if result["relation"] in {">", "<"}:
        reason = "UNSUPPORTED_STRICT_RELATION: Gurobi does not support > or <"
        item.update(
            status="unsupported",
            error=reason,
            fallback=_fallback(reason),
        )
        return item

    # Each candidate is decomposed independently, so TopDownNEDTree starts its
    # temporary names from y_temp_1 each time. Namespace them before combining
    # all candidates into one CodingAgent context to prevent cross-candidate
    # collisions.
    namespace = f"ned_c{candidate_index}"
    rename_map = {
        symbol: sympy.Symbol(f"{namespace}_{symbol}")
        for symbol in result["definitions"]
    }
    definitions = [
        f"{rename_map[symbol]} = {definition.xreplace(rename_map)}"
        for symbol, definition in result["definitions"].items()
    ]
    if not definitions:
        item.update(
            status="linear_skipped",
            error=None,
            fallback=_fallback("the candidate is already linear"),
        )
        return item

    item.update(
        status="applied",
        error=None,
        fallback=None,
        auxiliary_namespace=namespace,
        linearized_form=str(result["linear_expr"].xreplace(rename_map)),
        relation=str(result["relation"]),
        new_definition_set=definitions,
        domain_conditions=list(result["domain_constraints"]),
        domain_conditions_note=DOMAIN_OBLIGATION_NOTE,
        validation=dict(result["validation"]),
    )
    return item


def _format_coding_advice(items, overall_status):
    lines = [
        "NEDTree-4R deterministic coding advice (opt-in).",
        DOMAIN_OBLIGATION_NOTE,
        f"Overall status: {overall_status}.",
    ]
    for item in items:
        source = item["source"]
        status = item["status"]
        lines.append(f"[{source}] status={status}; expression={item['expression']}")
        if status == "applied":
            lines.append(f"  Linearized Form: {item['linearized_form']}")
            definitions = "; ".join(item["new_definition_set"])
            lines.append(f"  New Definition Set: {definitions}")
            conditions = "; ".join(item["domain_conditions"]) or "none"
            lines.append(
                "  Domain conditions (obligations, not auto-enforced): " + conditions
            )
        else:
            lines.append(f"  Fallback: {item['fallback']}")
    return "\n".join(lines)


def build_nedtree_4r_advice(entry):
    """Return deterministic NEDTree-4R advice plus visible per-candidate fallbacks."""
    variables = _declared_names(entry.get("Variables_List", []), "symbol")
    parameters = _declared_names(entry.get("Parameters_List", []), "Name")

    candidates = []
    objective = _objective_candidate(entry.get("Objective", {}))
    if objective is not None:
        candidates.append(objective)
    candidates.extend(_constraint_candidates(entry.get("Constraint_Table", [])))

    items = [
        _serialise_candidate(candidate, variables, parameters, candidate_index)
        for candidate_index, candidate in enumerate(candidates)
    ]
    counts = {
        status: sum(item["status"] == status for item in items)
        for status in ("applied", "linear_skipped", "unsupported")
    }
    overall_status = "applied" if counts["applied"] else "fallback"
    overall_fallback = None
    if overall_status == "fallback":
        overall_fallback = _fallback("no supported nonlinear candidate was found")

    return {
        "status": overall_status,
        "variables": variables,
        "parameters": parameters,
        "summary": counts,
        "domain_conditions_note": DOMAIN_OBLIGATION_NOTE,
        "fallback": overall_fallback,
        "candidates": items,
        "coding_advice": _format_coding_advice(items, overall_status),
    }


def attach_nedtree_4r_advice(entry, math_model):
    """Copy the current model payload and append opt-in deterministic advice."""
    if isinstance(math_model, dict):
        combined = dict(math_model)
    else:
        combined = {"math_model": math_model}
    combined["nedtree_4r_advice"] = build_nedtree_4r_advice(entry)
    return combined
