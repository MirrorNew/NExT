"""Validate and organize the decomposition returned by one Auxiliary Agent call.

This module deliberately does not invoke NEDTree-4R or decompose source formulas.
The Auxiliary Agent response is the sole source of candidate decompositions.
"""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any


AUXILIARY_SCHEMA_VERSION = "nextor-aux-tree-v1"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INDEXED_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\[[0-9]+\]$")
_SOURCE_SUBSCRIPT = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\[([^\[\]]*)\]")
_RELATION = re.compile(r"<=|>=|==|(?<![<>!=])=(?!=)|<|>")
_NONFINITE_NAMES = {"nan", "inf", "infinity"}
_ALLOWED_FUNCTIONS = {"exp", "log", "ln", "log2", "log10", "sin", "cos", "sqrt"}


def _normalise_expression(value: str) -> str:
    replacements = {
        "≤": "<=", "≥": ">=", "·": "*", "×": "*", "−": "-",
        "^": "**", "$": "",
    }
    result = value.strip()
    for old, new in replacements.items():
        result = result.replace(old, new)
    # Treat ln(...) as the common natural-log alias for log(...), while callers
    # retain the exact source text in the returned tree for provenance.
    result = re.sub(r"\bln\s*(?=\()", "log", result)
    return re.sub(r"\s+", "", result)


def _canonical_relation(value: str) -> str:
    return {"≤": "<=", "≥": ">=", "==": "="}.get(value, value)


def _source_expressions(math_input: dict[str, Any]) -> dict[str, str]:
    objective = math_input.get("Objective")
    if isinstance(objective, dict):
        expression = objective.get("Mathematical_expressions")
    else:
        expression = objective
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("math_input objective expression is missing")
    sources = {"objective": expression}
    constraints = math_input.get("Constraint_Table", [])
    if not isinstance(constraints, list):
        raise ValueError("Constraint_Table must be a list")
    for index, item in enumerate(constraints):
        if isinstance(item, dict):
            expression = item.get("Mathematical_expressions", item.get("expression"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            expression = item[1]
        else:
            expression = item
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError(f"Constraint_Table[{index}] expression is missing")
        sources[f"constraint:{index}"] = expression
    return sources


def _declared_symbols(math_input: dict[str, Any], sources: dict[str, str]) -> set[str]:
    symbols: set[str] = set()
    for item in math_input.get("Variables_List", []):
        value = item.get("symbol", item.get("Symbol")) if isinstance(item, dict) else item
        if isinstance(value, str):
            name = value.strip()
            if _IDENTIFIER.fullmatch(name) or _INDEXED_IDENTIFIER.fullmatch(name):
                symbols.add(name)

    source_indices: dict[str, set[int]] = {}
    dynamic_source_indices: set[str] = set()
    for expression in sources.values():
        for match in _SOURCE_SUBSCRIPT.finditer(expression):
            name, index = match.groups()
            if re.fullmatch(r"[0-9]+", index):
                source_indices.setdefault(name, set()).add(int(index))
            else:
                dynamic_source_indices.add(name)

    for item in math_input.get("Parameters_List", []):
        value = item.get("Name", item.get("name")) if isinstance(item, dict) else item
        if isinstance(value, str):
            name = value.strip()
            if _INDEXED_IDENTIFIER.fullmatch(name):
                symbols.add(name)
                continue
            if _IDENTIFIER.fullmatch(name):
                parameter_value = item.get("Value") if isinstance(item, dict) else None
                if not isinstance(parameter_value, list):
                    symbols.add(name)
                    continue
                if (name in dynamic_source_indices or not parameter_value
                        or not all(type(entry) is int
                                   or (type(entry) is float and math.isfinite(entry))
                                   for entry in parameter_value)):
                    continue
                indices = source_indices.get(name, set())
                size = len(parameter_value)
                zero_based = set(range(size))
                one_based = set(range(1, size + 1))
                if indices == zero_based or indices == one_based:
                    symbols.update(f"{name}[{index}]" for index in indices)
    return symbols


def _reference_is_declared(reference: str, available: set[str]) -> bool:
    return reference in available


def _names_from_ast(parsed: ast.AST, location: str) -> set[str]:
    indexed_bases: set[int] = set()
    references: set[str] = set()
    for node in ast.walk(parsed):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError(f"{location} contains a non-numeric constant")
            if isinstance(node.value, float) and not math.isfinite(node.value):
                raise ValueError(f"{location} contains a non-finite numeric constant")
        if isinstance(node, ast.Subscript):
            if (not isinstance(node.value, ast.Name)
                    or not isinstance(node.slice, ast.Constant)
                    or isinstance(node.slice.value, bool)
                    or not isinstance(node.slice.value, int)
                    or node.slice.value < 0):
                raise ValueError(f"{location} only supports nonnegative integer constant indices")
            indexed_bases.add(id(node.value))
            references.add(f"{node.value.id}[{node.slice.value}]")
    call_function_nodes = {
        id(node.func) for node in ast.walk(parsed)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    for node in ast.walk(parsed):
        if (isinstance(node, ast.Name) and id(node) not in call_function_nodes
                and id(node) not in indexed_bases):
            if node.id.casefold() in _NONFINITE_NAMES:
                raise ValueError(f"{location} contains a non-finite numeric name")
            references.add(node.id)
    return references


def _names_in_expression(expression: str, location: str) -> set[str]:
    """Parse a restricted arithmetic expression without evaluating it."""
    try:
        canonical = re.sub(r"\bln\s*(?=\()", "log", expression)
        parsed = ast.parse(canonical.replace("^", "**"), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"{location} is not a supported arithmetic expression") from exc
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult,
        ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Name, ast.Load, ast.Constant,
        ast.Call, ast.Attribute, ast.Subscript,
    )
    for node in ast.walk(parsed):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"{location} contains unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCTIONS:
                raise ValueError(f"{location} contains an unsupported function call")
            if node.keywords:
                raise ValueError(f"{location} function calls cannot use keyword arguments")
            if len(node.args) != 1:
                raise ValueError(f"{location} function calls must have exactly one argument")
        if isinstance(node, ast.Attribute):
            raise ValueError(f"{location} cannot use attribute access")
    return _names_from_ast(parsed, location)


def _validate_domains(values: list[Any], available: set[str], location: str) -> list[str]:
    if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values):
        raise ValueError(f"{location} must be a list of non-empty strings")
    allowed = (
        ast.Expression, ast.Compare, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
        ast.Mult, ast.Div, ast.Pow, ast.USub, ast.UAdd, ast.Name, ast.Load,
        ast.Constant, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
        ast.Subscript,
    )
    for index, condition in enumerate(values):
        try:
            parsed = ast.parse(condition.replace("^", "**"), mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"{location}[{index}] is not a supported domain relation") from exc
        if (not isinstance(parsed.body, ast.Compare)
                or len(parsed.body.ops) != 1
                or any(not isinstance(node, allowed) for node in ast.walk(parsed))):
            raise ValueError(f"{location}[{index}] must be a simple arithmetic relation")
        names = _names_from_ast(parsed, f"{location}[{index}]")
        unknown = {name for name in names if not _reference_is_declared(name, available)}
        if unknown:
            raise ValueError(f"{location}[{index}] uses undeclared symbols: {', '.join(sorted(unknown))}")
    return [item.strip() for item in values]


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Auxiliary Agent response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Auxiliary Agent response must be a JSON object")
    return payload


def parse_auxiliary_response(content: str, math_input: dict[str, Any]) -> dict[str, Any]:
    """Return validated, dependency-ordered decomposition records.

    Structural validation cannot prove algebraic equivalence. The returned
    ``equivalence_status`` therefore remains ``not_proven`` for every candidate.
    """
    payload = _parse_json(content)
    advice = payload.get("math_model_advice")
    if not isinstance(advice, list):
        raise ValueError("math_model_advice must be a list")
    for index, item in enumerate(advice):
        if (not isinstance(item, dict)
                or set(item) != {"variables", "constraints", "objective"}
                or not isinstance(item["variables"], list)
                or not all(isinstance(value, str) for value in item["variables"])
                or not isinstance(item["constraints"], list)
                or not all(isinstance(value, str) for value in item["constraints"])
                or not isinstance(item["objective"], str)):
            raise ValueError(f"math_model_advice[{index}] has an invalid schema")
    decomposition = payload.get("decomposition")
    if not isinstance(decomposition, dict):
        raise ValueError("decomposition object is required; refusing raw-expression redecomposition")
    if decomposition.get("schema_version") != AUXILIARY_SCHEMA_VERSION:
        raise ValueError(f"decomposition.schema_version must be {AUXILIARY_SCHEMA_VERSION}")
    candidates = decomposition.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("decomposition.candidates must be a list")

    sources = _source_expressions(math_input)
    declared = _declared_symbols(math_input, sources)
    seen_sources: set[str] = set()
    seen_auxiliary_symbols: set[str] = set()
    normalized_candidates = []
    for index, candidate in enumerate(candidates):
        location = f"decomposition.candidates[{index}]"
        if not isinstance(candidate, dict):
            raise ValueError(f"{location} must be an object")
        source_id = candidate.get("source_id")
        if source_id not in sources:
            raise ValueError(f"{location}.source_id does not identify a frozen source expression")
        if source_id in seen_sources:
            raise ValueError(f"duplicate decomposition source_id: {source_id}")
        seen_sources.add(source_id)
        source_expression = candidate.get("source_expression")
        if not isinstance(source_expression, str) or _normalise_expression(source_expression) != _normalise_expression(sources[source_id]):
            raise ValueError(f"{location}.source_expression differs from the frozen input")

        relation = candidate.get("relation")
        if not isinstance(relation, str):
            raise ValueError(f"{location}.relation must be text")
        relation = _canonical_relation(relation)
        normalized_source = _normalise_expression(sources[source_id])
        expected_relations = [
            _canonical_relation(match.group())
            for match in _RELATION.finditer(normalized_source)
        ]
        if source_id != "objective" and len(expected_relations) > 1:
            raise ValueError(f"{location} uses an unsupported chained relation")
        if source_id == "objective":
            if relation != "objective":
                raise ValueError(f"{location}.relation must be 'objective'")
        elif not expected_relations or relation not in expected_relations:
            raise ValueError(f"{location}.relation does not match the frozen constraint")

        linearized_form = candidate.get("linearized_form")
        if not isinstance(linearized_form, str) or not linearized_form.strip():
            raise ValueError(f"{location}.linearized_form must be non-empty text")
        definitions = candidate.get("definitions")
        if not isinstance(definitions, list):
            raise ValueError(f"{location}.definitions must be a list")
        available = set(declared)
        ordered_definitions = []
        for definition_index, definition in enumerate(definitions):
            def_location = f"{location}.definitions[{definition_index}]"
            if not isinstance(definition, dict):
                raise ValueError(f"{def_location} must be an object")
            symbol, rhs = definition.get("symbol"), definition.get("rhs")
            if not isinstance(symbol, str) or not _IDENTIFIER.fullmatch(symbol):
                raise ValueError(f"{def_location}.symbol must be a simple identifier")
            if symbol in available or symbol in declared:
                raise ValueError(f"{def_location}.symbol is duplicate or shadows a source symbol")
            if symbol in seen_auxiliary_symbols:
                raise ValueError(
                    f"{def_location}.symbol conflicts with an auxiliary definition in another candidate"
                )
            if not isinstance(rhs, str) or not rhs.strip():
                raise ValueError(f"{def_location}.rhs must be non-empty text")
            references = _names_in_expression(rhs, f"{def_location}.rhs")
            unknown = {reference for reference in references
                       if not _reference_is_declared(reference, available)}
            if unknown:
                raise ValueError(f"{def_location}.rhs has forward or undeclared dependencies: {', '.join(sorted(unknown))}")
            available.add(symbol)
            domains = _validate_domains(definition.get("domain_conditions", []), available,
                                        f"{def_location}.domain_conditions")
            seen_auxiliary_symbols.add(symbol)
            ordered_definitions.append({"symbol": symbol, "rhs": rhs.strip(), "dependencies": sorted(references),
                                        "domain_conditions": list(domains)})

        form_references = _names_in_expression(linearized_form, f"{location}.linearized_form")
        unknown_form_names = {reference for reference in form_references
                              if not _reference_is_declared(reference, available)}
        if unknown_form_names:
            raise ValueError(f"{location}.linearized_form has undeclared dependencies: {', '.join(sorted(unknown_form_names))}")
        defined_symbols = {definition["symbol"] for definition in ordered_definitions}
        used_auxiliary_symbols = defined_symbols.intersection(form_references)
        if defined_symbols and not used_auxiliary_symbols:
            raise ValueError(
                f"{location}.linearized_form does not use any declared auxiliary definition"
            )
        domains = _validate_domains(candidate.get("domain_conditions", []), available,
                                    f"{location}.domain_conditions")
        normalized_candidates.append({
            "source_id": source_id,
            "source_expression": sources[source_id],
            "relation": relation,
            "linearized_form": linearized_form.strip(),
            "definitions": ordered_definitions,
            "domain_conditions": domains,
            "equivalence_status": "not_proven",
        })

    return {
        "schema_version": AUXILIARY_SCHEMA_VERSION,
        "math_model_advice": advice,
        "candidates": normalized_candidates,
        "candidate_count": len(normalized_candidates),
        "status": "validated" if normalized_candidates else "no_decomposition",
        "equivalence_status": "not_proven",
    }
