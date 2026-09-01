"""Read-only NEXTOR/NORA validation with an optional audited diagnostic subset.

This script never rewrites the source dataset or historical NORA output.  When
requested, it writes a fresh subset that preserves canonical question, answer,
and index fields from NEXTOR while reusing the historical structured extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "data" / "20251021_origin_datasets" / "NExTNLP.json"
DEFAULT_PROCESSED = (
    ROOT
    / "NORA_process_data"
    / "NExTNLP_NORA_gpt-5.1.json"
)
REQUIRED_PROCESSED_FIELDS = {
    "index",
    "question",
    "answer",
    "Parameters_List",
    "Variables_List",
    "Constraint_Table",
    "Objective",
    "Problem_Type",
}
PARAMETER_FIELDS = {"Name", "Type", "Value"}
VARIABLE_FIELDS = {"symbol", "Meaning", "Type", "Range "}
OBJECTIVE_FIELDS = {"Objective_sentence", "Mathematical_expressions"}

# These inconsistencies are visible in the released artifact. They are kept out
# of new strict scalar diagnostics without changing the historical dataset.
KNOWN_EXCLUSIONS = {
    "0": "The question requests a location vector; the answer stores the objective.",
    "2": (
        "The question requests x_1, whereas the stored scalar answer is the "
        "objective-scale value; exclude from new targeted experiments."
    ),
    "4": "The question requests a production/price plan; the answer stores sales.",
    "7": "The question requests a warehouse location; the answer stores the objective.",
    "9": "The question requests a location; the answer stores weighted distance.",
    "11": "The unlabeled two-value answer does not match the requested three-variable design.",
    "17": "The question requests dispatch decisions; the answer stores total cost.",
    "18": "The question requests reduction decisions; the answer stores net cost.",
    "21": (
        "The question requests the pair (Q_1, Q_2), whereas the stored scalar "
        "answer is the minimum total cost; exclude from scalar-answer scoring."
    ),
    "22": "The question requests route-1 vehicle count; the answer stores travel time.",
    "27": "The question requests (P_1, P_2); the answer stores total cost.",
    "28": "The question requests f_A; the answer stores a non-optimal objective value.",
    "29": "The question requests three speeds and uses an unattained strict-bound optimum.",
    "30": "The question requests transport decisions and has unit/total cost ambiguity.",
    "31": "The stated trigger is unreachable under x_2 <= 90 and units conflict.",
    "33": "The question requests (T, C); the answer stores a weighted objective.",
    "35": "The question requests x_1; the answer stores a rounded objective.",
    "36": "The request includes a discharge decision; the answer stores only power.",
    "37": "The question requests flow A; the answer stores total cost.",
}

SCALAR_ONLY_PARTIAL = {
    case_id: "The stored scalar covers an objective-value subtask but not every requested decision."
    for case_id in ("1", "3", "5", "6", "8", "10", "12", "15", "19", "23", "25", "26", "34")
}

STRICT_SCALAR_SAFE = {
    "13": "The final request is the optimal portfolio variance.",
    "14": "The final request is the calculated maximum output.",
    "16": "The final request is the maximum profit.",
    "20": "The final request is the minimum transportation cost.",
    "24": "The final request is the minimum dosing objective.",
}

ROUTE_SPECIAL = {
    "32": (
        "The scalar maximum revenue is aligned, but the full model is MINLP; "
        "routing it as a single NLP is not a semantic repair."
    )
}


def _read_json(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by case id")
    return data


def validate_processed_entry(case_id: str, entry: Any) -> list[str]:
    """Validate the nested structure consumed by the modeling and coding agents."""

    errors: list[str] = []
    if not isinstance(entry, dict):
        return [f"case {case_id}: processed entry must be an object"]

    parameters = entry.get("Parameters_List")
    if not isinstance(parameters, list) or not parameters:
        errors.append(f"case {case_id}: Parameters_List must be a non-empty list")
    else:
        for position, parameter in enumerate(parameters):
            if not isinstance(parameter, dict) or set(parameter) != PARAMETER_FIELDS:
                errors.append(
                    f"case {case_id}: Parameters_List[{position}] must contain exactly "
                    f"{sorted(PARAMETER_FIELDS)}"
                )
                continue
            if not isinstance(parameter["Name"], str) or not parameter["Name"].strip():
                errors.append(f"case {case_id}: Parameters_List[{position}].Name must be text")
            if not isinstance(parameter["Type"], str) or not parameter["Type"].strip():
                errors.append(f"case {case_id}: Parameters_List[{position}].Type must be text")

    variables = entry.get("Variables_List")
    if not isinstance(variables, list) or not variables:
        errors.append(f"case {case_id}: Variables_List must be a non-empty list")
    else:
        for position, variable in enumerate(variables):
            if not isinstance(variable, dict) or set(variable) != VARIABLE_FIELDS:
                errors.append(
                    f"case {case_id}: Variables_List[{position}] must contain exactly "
                    f"{sorted(VARIABLE_FIELDS)}"
                )
                continue
            for field in VARIABLE_FIELDS:
                if not isinstance(variable[field], str) or not variable[field].strip():
                    errors.append(
                        f"case {case_id}: Variables_List[{position}].{field} must be text"
                    )

    constraints = entry.get("Constraint_Table")
    if not isinstance(constraints, list) or not constraints:
        errors.append(f"case {case_id}: Constraint_Table must be a non-empty list")
    else:
        for position, row in enumerate(constraints):
            if (
                not isinstance(row, list)
                or len(row) != 3
                or any(not isinstance(cell, str) or not cell.strip() for cell in row)
            ):
                errors.append(
                    f"case {case_id}: Constraint_Table[{position}] must be three non-empty text cells"
                )

    objective = entry.get("Objective")
    if not isinstance(objective, dict) or set(objective) != OBJECTIVE_FIELDS:
        errors.append(
            f"case {case_id}: Objective must contain exactly {sorted(OBJECTIVE_FIELDS)}"
        )
    else:
        for field in OBJECTIVE_FIELDS:
            if not isinstance(objective[field], str) or not objective[field].strip():
                errors.append(f"case {case_id}: Objective.{field} must be non-empty text")

    if not isinstance(entry.get("Problem_Type"), str) or not entry["Problem_Type"].strip():
        errors.append(f"case {case_id}: Problem_Type must be non-empty text")
    return errors


def validate(
    source_path: Path,
    processed_path: Path,
    subset_ids: list[str],
    allow_partial_scalar: bool = False,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source = _read_json(source_path)
    processed = _read_json(processed_path)
    errors: list[str] = []
    warnings: list[str] = []

    expected_ids = {str(i) for i in range(38)}
    if set(source) != expected_ids:
        errors.append("source ids are not exactly 0..37")
    if set(processed) != expected_ids:
        errors.append("processed ids are not exactly 0..37")

    for case_id in sorted(expected_ids, key=int):
        if case_id not in source or case_id not in processed:
            continue
        src = source[case_id]
        proc = processed[case_id]
        missing = sorted(REQUIRED_PROCESSED_FIELDS - set(proc))
        if missing:
            errors.append(f"case {case_id}: missing processed fields {missing}")
        errors.extend(validate_processed_entry(case_id, proc))
        if str(src.get("index")) != case_id:
            errors.append(f"case {case_id}: source index does not match key")
        for field in ("question", "answer", "index"):
            if src.get(field) != proc.get(field):
                errors.append(f"case {case_id}: processed {field} differs from source")
        for field in ("Parameters_List", "Variables_List", "Constraint_Table", "Objective"):
            if not proc.get(field):
                errors.append(f"case {case_id}: empty {field}")
        if proc.get("Problem_Type") != "NLP":
            warnings.append(
                f"case {case_id}: historical Problem_Type={proc.get('Problem_Type')!r}; "
                "requires an explicit routing audit and is not automatically repaired"
            )

    for case_id, reason in KNOWN_EXCLUSIONS.items():
        warnings.append(f"case {case_id}: {reason}")
    for case_id, reason in ROUTE_SPECIAL.items():
        warnings.append(f"case {case_id}: {reason}")

    selected: dict[str, dict[str, Any]] = {}
    for case_id in subset_ids:
        if case_id in KNOWN_EXCLUSIONS:
            errors.append(f"case {case_id}: known exclusion cannot enter diagnostic subset")
            continue
        if case_id in ROUTE_SPECIAL:
            errors.append(f"case {case_id}: route-special case cannot enter strict diagnostic subset")
            continue
        if case_id in SCALAR_ONLY_PARTIAL and not allow_partial_scalar:
            errors.append(
                f"case {case_id}: scalar answer covers only part of the request; "
                "use --allow-partial-scalar only for explicitly objective-value diagnostics"
            )
            continue
        if case_id not in source or case_id not in processed:
            errors.append(f"case {case_id}: requested subset id is unavailable")
            continue
        entry = dict(processed[case_id])
        entry["index"] = source[case_id]["index"]
        entry["question"] = source[case_id]["question"]
        entry["answer"] = source[case_id]["answer"]
        entry["Problem_Type"] = "NLP"
        selected[case_id] = entry

    manifest = {
        "status": "PASS" if not errors else "FAIL",
        "source": {
            "path": str(source_path.resolve()),
            "count": len(source),
        },
        "processed": {
            "path": str(processed_path.resolve()),
            "count": len(processed),
        },
        "diagnostic_agent_route": "NLP for selected strict/partial cases only",
        "selection_contract": (
            "objective_value_only" if allow_partial_scalar else "strict_scalar_request"
        ),
        "subset_ids": subset_ids,
        "subset_count": len(selected),
        "known_exclusions": KNOWN_EXCLUSIONS,
        "scalar_only_partial": SCALAR_ONLY_PARTIAL,
        "strict_scalar_safe": STRICT_SCALAR_SAFE,
        "route_special": ROUTE_SPECIAL,
        "warnings": warnings,
        "errors": errors,
    }
    return manifest, selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--processed", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument(
        "--subset-ids",
        default="13,14,16,20,24",
        help="comma-separated; defaults to the five audited strict-scalar cases",
    )
    parser.add_argument(
        "--allow-partial-scalar",
        action="store_true",
        help="allow cases whose scalar answer covers only the objective-value subtask",
    )
    parser.add_argument("--subset-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()

    subset_ids = [item.strip() for item in args.subset_ids.split(",") if item.strip()]
    manifest, selected = validate(
        args.source,
        args.processed,
        subset_ids,
        allow_partial_scalar=args.allow_partial_scalar,
    )

    if args.subset_output and manifest["status"] == "PASS":
        args.subset_output.parent.mkdir(parents=True, exist_ok=True)
        args.subset_output.write_text(
            json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest["subset_output"] = str(args.subset_output.resolve())

    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
