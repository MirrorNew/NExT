"""NExTOR agent prompts routed through the audited LoggedModel interface."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any

from agents.extract_info_agents import (
    ExtractFactorsFromLongTextAgent,
    ExtractParameterFromLongTextAgent,
)
from agents.model_agents import AuxiliaryModelAgent, ModelingAgent
from agents.coding_agents import CodingAgent
from agents.repair_agents import RepairAgent

from .data import validate_math_input
from .auxiliary_tree import AUXILIARY_SCHEMA_VERSION, _source_expressions, parse_auxiliary_response


AUXILIARY_RESPONSE_FORMAT = {"type": "json_object"}


_DECOMPOSITION_PROMPT = f"""
Return only one JSON object containing math_model_advice and decomposition, with
the following structure. Do not call another model or request another decomposition:
{{
  "math_model_advice": [{{"variables": [], "constraints": [], "objective": "..."}}],
  "decomposition": {{
    "schema_version": "{AUXILIARY_SCHEMA_VERSION}",
    "candidates": [
      {{
        "source_id": "objective or constraint:<zero-based index>",
        "source_expression": "copy Context.source_expressions[source_id] verbatim",
        "relation": "exactly 'objective' for the objective; otherwise ONLY one operator token: '=', '==', '<=', '>=', '<', or '>'",
        "linearized_form": "objective expression, or for a constraint the lhs-rhs residual (without relation)",
        "definitions": [
          {{"symbol": "auxiliary_name", "rhs": "arithmetic expression", "domain_conditions": []}}
        ],
        "domain_conditions": []
      }}
    ]
  }}
}}
For each constraint keep its original relation; linearized_form must be lhs-rhs, so
the represented relation is linearized_form <relation> 0. For an objective, use its
objective expression without the Maximize/Minimize label or result-variable assignment.
When a candidate introduces auxiliary definitions to replace a nonlinear term, use
those defined symbols in linearized_form (for example, if y = ln(x), write y - ln(2),
not ln(x) - ln(2)). Do not include definitions that the candidate relation does not use.
The structured Context is authoritative for source expressions. The free-text Math model
is explanatory only and must not change or replace any expression in Context.
Copy source_expression from Context.source_expressions for the chosen source_id, including
any objective label, function name, assignment, and original punctuation. For example,
if that value is "Minimize f(x) = exp(x^2)", source_expression must be that whole string;
do not shorten it to "exp(x^2)". Only linearized_form omits the objective label/assignment.
The relation field must contain only the operator token (for example, ">=") and must
never contain either side of the constraint or the complete relation expression.
Use an empty candidates list if the existing advice proposes no auxiliary decomposition.
Each source_id must exactly match a Context.source_expressions key and occur at most once;
omit sources for which you propose no auxiliary decomposition. Use fresh simple auxiliary
identifiers, unique across all candidates and distinct from source variables/parameters.
List definitions in dependency order; each rhs may use only declared source symbols and
earlier definitions from the same candidate, never definitions from another candidate.
Preserve every required domain condition. Do not claim equivalence
has been machine-verified. No prose or markdown outside the JSON object.
""".strip()

def _json_object(text: str, label: str) -> dict[str, Any]:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\n?```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} response must be a JSON object")
    return parsed


def _persist(folder: str | Path, name: str, value: Any) -> None:
    path = Path(folder) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def _safe_error(exc: BaseException, model: Any) -> dict[str, str]:
    message = str(exc)
    api_key_name = getattr(getattr(model, "config", None), "api_key_env", None)
    secret = os.getenv(api_key_name) if isinstance(api_key_name, str) else None
    if secret:
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", message)
    message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", message)
    return {"error_type": type(exc).__name__, "message": message}


def _logged_agent(agent, model, *, stage: str, case_id: str, run_id: str,
                  call_ids: list[str], stage_costs: list[dict[str, Any]],
                  folder: str | Path,
                  stage_suffixes: tuple[str, ...] = ()) -> None:
    """Replace BaseAgent's direct client call with one logged completion per prompt."""
    count = 0

    async def query():
        nonlocal count
        current_stage = stage_suffixes[count] if count < len(stage_suffixes) else stage
        call_folder = Path(folder) / current_stage
        prompt = [dict(message) for message in agent.messages]
        _persist(call_folder, "prompt.json", prompt)
        started = time.perf_counter()
        try:
            response = await model.complete(
                prompt, stage=current_stage, case_id=case_id, run_id=run_id
            )
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            error = _safe_error(exc, model)
            _persist(call_folder, "error.json", {**error, "elapsed_seconds": elapsed})
            stage_costs.append({"stage": current_stage, "call_id": None, "usage": None,
                                "elapsed_seconds": elapsed, "status": "failed"})
            raise
        count += 1
        call_ids.append(response["call_id"])
        stage_costs.append({"stage": current_stage, "call_id": response["call_id"],
                            "usage": response.get("usage"),
                            "elapsed_seconds": response.get("elapsed_seconds", time.perf_counter() - started),
                            "status": "complete"})
        _persist(call_folder, "response.json", response)
        content = response["content"]
        agent.messages.append({"role": "assistant", "content": content})
        return content

    agent._query = query


async def prepare_ours_input(case_public: dict[str, Any], model, folder: str | Path,
                             run_id: str) -> dict[str, Any]:
    """Run NExTOR's parameter, factor, and ModelingAgent prompts once each stage.

    The actual extractors have an analysis prompt followed by a JSON-format prompt;
    both are preserved and routed through LoggedModel. No answer/reference fields
    are copied into the agent entry or any prompt.
    """
    question = case_public.get("question")
    case_id = str(case_public.get("case_id", "unknown"))
    if not isinstance(question, str) or not question.strip():
        raise ValueError("case_public.question must be non-empty")
    entry: dict[str, Any] = {"question": question, "index": case_id}
    call_ids: list[str] = []
    stage_costs: list[dict[str, Any]] = []

    parameter_agent = ExtractParameterFromLongTextAgent(None, model_name=model.config.model)
    _logged_agent(parameter_agent, model, stage="extract_parameters", case_id=case_id,
                  run_id=run_id, call_ids=call_ids, stage_costs=stage_costs,
                  folder=folder,
                  stage_suffixes=("extract_parameters_analysis", "extract_parameters_json"))
    parameter_text = await parameter_agent.generate(question)
    parameter_json_text = await parameter_agent.get_and_change_format_output(parameter_text)
    entry.update(_json_object(parameter_json_text, "parameter extraction"))

    factor_agent = ExtractFactorsFromLongTextAgent(None, model_name=model.config.model)
    _logged_agent(factor_agent, model, stage="extract_factors", case_id=case_id,
                  run_id=run_id, call_ids=call_ids, stage_costs=stage_costs,
                  folder=folder,
                  stage_suffixes=("extract_factors_analysis", "extract_factors_json"))
    factor_text = await factor_agent.generate(question)
    factor_json_text = await factor_agent.get_and_change_format_output(factor_text)
    entry.update(_json_object(factor_json_text, "factor extraction"))

    normalized = validate_math_input(entry)
    modeling_entry = {**normalized, "question": question}
    modeling_agent = ModelingAgent(None, model_name=model.config.model)
    _logged_agent(modeling_agent, model, stage="modeling", case_id=case_id,
                  run_id=run_id, call_ids=call_ids, stage_costs=stage_costs, folder=folder)
    math_model = await modeling_agent.generate(modeling_entry)

    result = {
        "math_input": normalized,
        "math_model": math_model,
        "call_ids": call_ids,
        "stage_costs": stage_costs,
        "status": "complete",
    }
    _persist(folder, "ours_upstream.json", result)
    return result


async def prepare_auxiliary(case_public: dict[str, Any], math_input: dict[str, Any],
                            model, folder: str | Path, run_id: str) -> dict[str, Any]:
    """Make exactly one existing AuxiliaryModelAgent completion and organize its output."""
    case_id = str(case_public.get("case_id", "unknown"))
    question = case_public.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("case_public.question must be non-empty")
    normalized = validate_math_input(math_input)
    # Preserve the agent's modeling instructions, but replace its old output-only
    # example so this single completion has exactly one JSON output contract.
    agent = AuxiliaryModelAgent(None, model_name=model.config.model)
    old_output_marker = "Output ONLY a JSON object with the following structure (no extra text):"
    instructions, marker, _ = agent.messages[0]["content"].partition(old_output_marker)
    if not marker:
        raise ValueError("AuxiliaryModelAgent output contract changed; review the adapter")
    agent.messages[0]["content"] = instructions.rstrip() + "\n\n" + _DECOMPOSITION_PROMPT
    context = json.dumps({
        "variables": normalized.get("Variables_List", []),
        "constraints": normalized.get("Constraint_Table", []),
        "objective": normalized.get("Objective", {}),
        "problem_type": normalized.get("Problem_Type", ""),
        "source_expressions": _source_expressions(normalized),
    }, ensure_ascii=False)
    math_model = case_public.get("math_model")
    if not isinstance(math_model, str) or not math_model.strip():
        math_model = json.dumps(normalized, ensure_ascii=False)
    agent.messages.append({"role": "user", "content": (
        f"Problem: {question}\nContext: {context}\nMath model:\n{math_model}\n"
        "Please review and suggest encoding improvements as specified above."
    )})
    prompt = [dict(message) for message in agent.messages]
    auxiliary_folder = Path(folder) / "auxiliary"
    _persist(auxiliary_folder, "prompt.json", prompt)
    started = time.perf_counter()
    try:
        response = await model.complete(
            prompt, stage="auxiliary", case_id=case_id, run_id=run_id,
            response_format=AUXILIARY_RESPONSE_FORMAT,
        )
    except BaseException as exc:
        elapsed = time.perf_counter() - started
        error = _safe_error(exc, model)
        _persist(auxiliary_folder, "error.json", {**error, "elapsed_seconds": elapsed})
        raise
    _persist(auxiliary_folder, "response.json", response)
    call_ids = [response["call_id"]]
    raw_response = response["content"]
    result: dict[str, Any] = {
        "tree": {"status": "fallback", "candidates": [],
                 "coding_advice": "Auxiliary decomposition unavailable; use the unchanged native coding path."},
        "math_model_advice": None,
        "text_advice": "Auxiliary decomposition unavailable; use the unchanged native coding path.",
        "call_ids": call_ids,
        "stage_costs": [{"stage": "auxiliary", "call_id": response["call_id"],
                         "usage": response.get("usage"),
                         "elapsed_seconds": response.get("elapsed_seconds", time.perf_counter() - started),
                         "status": "complete"}],
        "status": "invalid_decomposition",
        "raw_response": raw_response,
        "rawresponse": raw_response,
    }
    try:
        tree = parse_auxiliary_response(raw_response, normalized)
        text_advice = _format_tree_advice(tree)
        result.update(tree=tree, math_model_advice=tree["math_model_advice"],
                      text_advice=text_advice, status=tree["status"])
    except ValueError as exc:
        result["error"] = str(exc)
        result["tree"]["error"] = str(exc)
    _persist(folder, "auxiliary/result.json", result)
    return result


def _format_tree_advice(tree: dict[str, Any]) -> str:
    lines = ["Structured AuxiliaryModelAgent decomposition; algebraic equivalence is not proven."]
    if not tree["candidates"]:
        lines.append("No auxiliary definitions were returned; use the native coding path.")
    for candidate in tree["candidates"]:
        lines.append(f"[{candidate['source_id']}] relation={candidate['relation']}")
        lines.append(f"  Linearized Form: {candidate['linearized_form']}")
        for definition in candidate["definitions"]:
            lines.append(f"  Definition: {definition['symbol']} = {definition['rhs']}")
            lines.append(f"    Dependencies: {', '.join(definition['dependencies']) or 'none'}")
            lines.append(f"    Domain conditions: {', '.join(definition['domain_conditions']) or 'none'}")
        lines.append(f"  Domain conditions: {', '.join(candidate['domain_conditions']) or 'none'}")
    return "\n".join(lines)


async def generate_ours_code(
    case_public: dict[str, Any],
    math_input: dict[str, Any],
    math_model: str,
    advice: dict[str, Any] | str | None,
    model,
    folder: str | Path,
    run_id: str,
    feedback: dict[str, Any] | None = None,
    *,
    code_system: str | None = None,
) -> dict[str, Any]:
    """Run the repository CodingAgent or RepairAgent through LoggedModel.

    The agent-specific system prompt remains intact. ``code_system`` is appended
    as a shared experimental contract, and the answer specification is included
    explicitly in both initial and repair context. Native callers pass ``None``
    for advice; the NED caller passes the already prepared tree/advice.
    """
    case_id = str(case_public.get("case_id", "unknown"))
    question = case_public.get("question")
    if not isinstance(question, str):
        raise ValueError("case_public.question must be text")
    normalized = validate_math_input(math_input)
    answer_spec = case_public.get("answer_spec", {})
    if not isinstance(answer_spec, dict):
        raise ValueError("case_public.answer_spec must be an object")
    answer_spec_text = json.dumps(answer_spec, ensure_ascii=False)
    entry = {**normalized, "question": question}
    treatment = {"math_model": math_model}
    if advice is not None:
        treatment["nedtree_auxiliary_advice"] = advice
    shared_contract = ("\n\n" + code_system.strip()) if isinstance(code_system, str) and code_system.strip() else ""
    call_ids: list[str] = []
    stage_costs: list[dict[str, Any]] = []

    if feedback is None:
        # Keep CodingAgent's full NLP/MILP system prompt; append the controlled
        # experiment contract instead of replacing the project's prompt.
        agent = CodingAgent(None, model_name=model.config.model,
                            problem_type=normalized.get("Problem_Type", "MILP"))
        messages = [dict(message) for message in agent.messages]
        messages[0]["content"] += shared_contract
        context = json.dumps({
            "variables": normalized["Variables_List"],
            "constraints": normalized["Constraint_Table"],
            "objective": normalized["Objective"],
            "problem_type": normalized["Problem_Type"],
        }, ensure_ascii=False)
        messages.append({"role": "user", "content": (
            f"Problem: {question}\nContext: {context}\n"
            f"Answer specification (format only): {answer_spec_text}\n"
            "Authority: the frozen Context/math_input is the authoritative mathematical model. "
            "The free-text math_model below is explanatory only and must not override it. "
            "NED auxiliary advice, when supplied, is an encoding aid only and must not change the model.\n"
            f"Math model and coding advice:\n{json.dumps(treatment, ensure_ascii=False)}\n"
            f"Parameters List: {normalized['Parameters_List']}\n"
            "Output the requested answer according to answer specification."
        )})
        coding_folder = Path(folder) / "coding"
        _persist(coding_folder, "prompt.json", messages)
        started = time.perf_counter()
        try:
            response = await model.complete(messages, stage="code", case_id=case_id, run_id=run_id)
        except BaseException as exc:
            elapsed = time.perf_counter() - started
            _persist(coding_folder, "error.json", {**_safe_error(exc, model), "elapsed_seconds": elapsed})
            stage_costs.append({"stage": "code", "call_id": None, "usage": None,
                                "elapsed_seconds": elapsed, "status": "failed"})
            raise
        call_ids.append(response["call_id"])
        stage_costs.append({"stage": "code", "call_id": response["call_id"],
                            "usage": response.get("usage"),
                            "elapsed_seconds": response.get("elapsed_seconds", time.perf_counter() - started),
                            "status": "complete"})
        _persist(coding_folder, "response.json", response)
    else:
        code = feedback.get("code")
        error = feedback.get("error")
        if not isinstance(code, str) or not isinstance(error, str):
            raise ValueError("feedback requires string fields 'code' and 'error'")
        repair = RepairAgent(None, model_name=model.config.model)
        # RepairAgent.forward retains its actual code-first analysis and update
        # prompts. The answer spec is supplied in its system and math-model context
        # so both internal calls see the same output contract.
        repair.system_msg += shared_contract + "\n\nAnswer specification (format only): " + answer_spec_text
        repair.system_msg += (
            "\n\nAuthority: the frozen math_input is the authoritative mathematical model. "
            "The free-text math_model is explanatory only; auxiliary advice is an encoding aid and "
            "must not override or change the frozen model."
        )
        repair.messages = [{"role": "system", "content": repair.system_msg}]
        _logged_agent(
            repair, model, stage="repair", case_id=case_id, run_id=run_id,
            call_ids=call_ids, stage_costs=stage_costs, folder=Path(folder) / "repair",
            stage_suffixes=("repair_analysis", "repair_code"),
        )
        analysis, updated_code = await repair.forward(
            entry, error, code,
            json.dumps(treatment, ensure_ascii=False)
            + "\nAuthority: frozen math_input is authoritative; free-text math_model and auxiliary advice "
            "are explanatory/encoding aids and must not override it."
            + "\nAnswer specification (format only): " + answer_spec_text,
        )
        response = {"content": updated_code, "analysis": analysis}

    _persist(folder, "coding/latest_response.json", response)
    return {"response": response, "call_ids": call_ids, "stage_costs": stage_costs,
            "status": "complete"}
