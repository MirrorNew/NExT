import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.coding_agents import CodingAgent
from nedtree_4r_adapter import (
    DOMAIN_OBLIGATION_NOTE,
    attach_nedtree_4r_advice,
    build_nedtree_4r_advice,
)


def sample_entry():
    return {
        "index": 0,
        "question": "Maximize a nonlinear return subject to a budget.",
        "Problem_Type": "NLP",
        "Variables_List": [
            {"symbol": "x_1"},
            {"symbol": "x_2"},
        ],
        "Parameters_List": [{"Name": "alpha", "Value": 2.0}],
        "Objective": {
            "Mathematical_expressions": "maximize Z = log(x_1) + x_2^-1"
        },
        "Constraint_Table": [
            ["budget", "x_1 + x_2 <= 10", "sentence numbers:1"],
            ["risk", "x_1^2 + x_2^2 <= 9", "sentence numbers:1"],
            ["unsupported", "Abs(x_1) <= 4", "sentence numbers:2"],
            ["tuple_shape", "(x_1, x_2) <= 4", "sentence numbers:3"],
        ],
    }


def test_candidate_statuses_fallbacks_and_domain_obligations_are_visible():
    advice = build_nedtree_4r_advice(sample_entry())
    by_source = {item["source"]: item for item in advice["candidates"]}

    assert advice["status"] == "applied"
    assert by_source["objective"]["status"] == "applied"
    assert by_source["objective"]["fallback"] is None
    assert set(by_source["objective"]["domain_conditions"]) == {
        "x_1 > 0",
        "x_2 != 0",
    }
    assert by_source["objective"]["domain_conditions_note"] == DOMAIN_OBLIGATION_NOTE

    assert by_source["constraint:budget"]["status"] == "linear_skipped"
    assert "Fallback:" in advice["coding_advice"]
    assert "already linear" in by_source["constraint:budget"]["fallback"]

    assert by_source["constraint:risk"]["status"] == "applied"
    objective_names = {
        definition.split(" = ", 1)[0]
        for definition in by_source["objective"]["new_definition_set"]
    }
    risk_names = {
        definition.split(" = ", 1)[0]
        for definition in by_source["constraint:risk"]["new_definition_set"]
    }
    assert objective_names.isdisjoint(risk_names)
    assert by_source["objective"]["auxiliary_namespace"] != by_source[
        "constraint:risk"
    ]["auxiliary_namespace"]

    assert by_source["constraint:unsupported"]["status"] == "unsupported"
    assert by_source["constraint:unsupported"]["error"]
    assert "no NEDTree-4R decomposition was applied" in by_source[
        "constraint:unsupported"
    ]["fallback"]
    assert by_source["constraint:tuple_shape"]["status"] == "unsupported"
    assert (
        "UNSUPPORTED_OPERATOR: Tuple"
        in by_source["constraint:tuple_shape"]["error"]
    )
    assert by_source["constraint:tuple_shape"]["fallback"]
    assert "obligations, not auto-enforced" in advice["coding_advice"]


def test_overall_fallback_is_explicit_when_no_nonlinear_candidate_applies():
    entry = sample_entry()
    entry["Objective"] = {"Mathematical_expressions": "min Z = x_1 + x_2"}
    entry["Constraint_Table"] = [["budget", "x_1 + x_2 <= 10"]]

    advice = build_nedtree_4r_advice(entry)

    assert advice["status"] == "fallback"
    assert advice["fallback"]
    assert all(item["status"] == "linear_skipped" for item in advice["candidates"])
    assert all(item["fallback"] for item in advice["candidates"])


def test_adapter_splits_chained_bounds_and_accepts_plain_sqrt():
    entry = sample_entry()
    entry["Objective"] = {"Mathematical_expressions": "maximize sqrt(x_1)"}
    entry["Constraint_Table"] = [["bounds", "0 <= x_1 <= 10", "sentence 1"]]

    advice = build_nedtree_4r_advice(entry)
    by_source = {item["source"]: item for item in advice["candidates"]}

    assert by_source["objective"]["status"] == "applied"
    assert by_source["constraint:bounds:0"]["status"] == "linear_skipped"
    assert by_source["constraint:bounds:1"]["status"] == "linear_skipped"


def test_adapter_rejects_strict_relation_only_at_gurobi_boundary():
    entry = sample_entry()
    entry["Objective"] = {"Mathematical_expressions": "maximize x_1 + x_2"}
    entry["Constraint_Table"] = [
        ["strict", "log(x_1) + x_2^2 > alpha", "sentence 1"]
    ]

    advice = build_nedtree_4r_advice(entry)
    strict = next(
        item for item in advice["candidates"] if item["source"] == "constraint:strict"
    )

    assert strict["status"] == "unsupported"
    assert "UNSUPPORTED_STRICT_RELATION" in strict["error"]
    assert "Gurobi does not support > or <" in strict["error"]
    assert strict["fallback"]


class _FakeCompletions:
    def __init__(self):
        self.messages = None

    async def create(self, **kwargs):
        self.messages = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="```python\npass\n```"))],
            model=kwargs["model"],
            usage=SimpleNamespace(total_tokens=0),
        )


def test_adapter_output_enters_coding_agent_context_without_api():
    completions = _FakeCompletions()
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    coder = CodingAgent(client, model_name="gpt-5.1", problem_type="NLP")
    model_payload = attach_nedtree_4r_advice(sample_entry(), "base math model")

    asyncio.run(coder.generate(sample_entry(), model_payload))

    joined = "\n".join(message["content"] for message in completions.messages)
    assert "base math model" in joined
    assert "NEDTree-4R deterministic coding advice (opt-in)" in joined
    assert "Linearized Form" in joined
    assert "New Definition Set" in joined
    assert DOMAIN_OBLIGATION_NOTE in joined
    assert coder.model_name == "gpt-5.1"


def test_runner_defaults_preserve_gpt51_and_disable_adapter(monkeypatch):
    import RebuildNORA_easyread as runner

    monkeypatch.setattr(sys, "argv", ["RebuildNORA_easyread.py"])
    opts = runner.get_args()

    assert opts.model == "gpt-5.1"
    assert opts.nedtree_4r is False


def test_runner_opt_in_passes_adapter_payload_to_code_solver(monkeypatch, tmp_path):
    import RebuildNORA_easyread as runner

    captured = {}

    class _Agent:
        def __init__(self, client, model_name, **kwargs):
            self.model_name = model_name
            self.total_tokens = 0

    class _Modeler(_Agent):
        async def generate(self, entry):
            return "base math model"

    class _Auxiliary(_Agent):
        async def integrate_model(self, entry, math_model):
            return {"math_model": math_model, "math_model_advice": []}

    async def _solver(coder, repairer, entry, math_model, **kwargs):
        captured["math_model"] = math_model
        captured["model_name"] = coder.model_name
        return True, [1.0], ""

    monkeypatch.setattr(runner, "get_async_openai", lambda: object())
    monkeypatch.setattr(runner, "ModelingAgent", _Modeler)
    monkeypatch.setattr(runner, "AuxiliaryModelAgent", _Auxiliary)
    monkeypatch.setattr(runner, "CodingAgent", _Agent)
    monkeypatch.setattr(runner, "RepairAgent", _Agent)
    monkeypatch.setattr(runner, "async_code_solver", _solver)

    opts = SimpleNamespace(model="gpt-5.1", nedtree_4r=True, output_dir=str(tmp_path))
    success, result, _ = asyncio.run(runner.async_NExT_OR_Agent(sample_entry(), opts))

    assert success is True
    assert result == [1.0]
    assert captured["model_name"] == "gpt-5.1"
    assert captured["math_model"]["math_model"] == "base math model"
    advice = captured["math_model"]["nedtree_4r_advice"]
    assert advice["status"] == "applied"
    assert "Linearized Form" in advice["coding_advice"]
    trace_path = tmp_path / "case_0_nedtree_4r_trace.json"
    assert trace_path.exists()
    trace = __import__("json").loads(trace_path.read_text(encoding="utf-8"))
    assert trace["case_id"] == 0
    assert trace["model"] == "gpt-5.1"
    assert trace["advice"]["status"] == "applied"


def test_runner_without_opt_in_preserves_model_payload(monkeypatch):
    import RebuildNORA_easyread as runner

    captured = {}

    class _Agent:
        def __init__(self, client, model_name, **kwargs):
            self.total_tokens = 0

    class _Modeler(_Agent):
        async def generate(self, entry):
            return "base math model"

    class _Auxiliary(_Agent):
        async def integrate_model(self, entry, math_model):
            return {"math_model": math_model, "math_model_advice": []}

    async def _solver(coder, repairer, entry, math_model, **kwargs):
        captured["math_model"] = math_model
        return True, [1.0], ""

    monkeypatch.setattr(runner, "get_async_openai", lambda: object())
    monkeypatch.setattr(runner, "ModelingAgent", _Modeler)
    monkeypatch.setattr(runner, "AuxiliaryModelAgent", _Auxiliary)
    monkeypatch.setattr(runner, "CodingAgent", _Agent)
    monkeypatch.setattr(runner, "RepairAgent", _Agent)
    monkeypatch.setattr(runner, "async_code_solver", _solver)

    opts = SimpleNamespace(model="gpt-5.1", nedtree_4r=False, output_dir=".")
    asyncio.run(runner.async_NExT_OR_Agent(sample_entry(), opts))

    assert captured["math_model"] == {
        "math_model": "base math model",
        "math_model_advice": [],
    }
