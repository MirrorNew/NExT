"""Offline tests for baseline adapters; model and solver calls are faked."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rebuttal_runner import baselines
from rebuttal_runner.models import BudgetedModel


class FakeModel:
    def __init__(self, responses):
        self.config = SimpleNamespace(model="fake-model")
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, *, stage, case_id, run_id):
        self.calls.append({"messages": messages, "stage": stage,
                           "case_id": case_id, "run_id": run_id})
        response = self.responses.pop(0)
        return {"content": response, "call_id": f"call-{len(self.calls)}",
                "returned_model": "fake-model", "usage": {"total_tokens": 3},
                "elapsed_seconds": 0.001}


class BaselineAdapterTests(unittest.TestCase):
    def setUp(self):
        self.source = Path(r"D:\LLMProject\or_llm_agent-main")
        if not self.source.is_dir():
            self.skipTest("local OR-LLM-Agent source snapshot is unavailable")
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "or_llm_agent-main"
        self.root.mkdir()
        shutil.copyfile(self.source / "or_llm_eval_async.py", self.root / "or_llm_eval_async.py")
        shutil.copyfile(self.source / "utils.py", self.root / "utils.py")
        self.output = Path(self.temp.name) / "out"

    def tearDown(self):
        self.temp.cleanup()

    def test_preflight_hashes_sources_without_importing_clients(self):
        report = baselines.baseline_availability("or_llm_agent", self.root)
        self.assertTrue(report["available"])
        self.assertEqual(report["source"]["git_commit"], None)
        self.assertEqual(report["source"]["python_files"], 2)

    def test_original_async_agent_flow_uses_logged_model_and_shared_executor(self):
        model = FakeModel([
            "maximize x subject to x <= 5",
            "```python\nimport gurobipy as gp\nm=gp.Model()\nx=m.addVar()\nm.setObjective(x, gp.GRB.MAXIMIZE)\nm.addConstr(x <= 5)\nm.optimize()\nprint('FinalAnswer=【5】')\n```",
        ])
        case = {"case_id": "case-1", "question": "Maximize x subject to x <= 5.",
                "math_input": None,
                "reference": 987654321.0,
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        calls = []

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            calls.append(code)
            return {"status": "executed", "returncode": 0,
                    "stdout": "Best objective 5\nFinalAnswer=【5】", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "or_llm_agent", case, model, self.output, "run-1", level="end_to_end",
                source_root=self.root, python_executable="python", process_timeout=10,
                solver_time_limit=5, solver_threads=1, max_calls=5, max_repairs=1))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["prediction"], 5)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["call_ids"], ["call-1", "call-2"])
        call_log = self.output / "model_calls" / "call_0002"
        self.assertTrue((call_log / "prompt.json").is_file())
        self.assertTrue((call_log / "response.json").is_file())
        self.assertEqual(json.loads((call_log / "call.json").read_text(encoding="utf-8"))["status"], "completed")
        serialized_prompts = repr([c["messages"] for c in model.calls])
        self.assertIn(case["question"], serialized_prompts)
        self.assertNotIn("987654321", serialized_prompts)

    def test_missing_math_input_stops_conditioned_run_before_calls(self):
        model = FakeModel([])
        case = {"case_id": "case-2", "question": "question",
                "math_input": None, "answer_spec": {"kind": "scalar"}}
        result = asyncio.run(baselines.run_baseline(
            "or_llm_agent", case, model, self.output, "run-2", level="coding",
            source_root=self.root, python_executable="python", process_timeout=10,
            solver_time_limit=5, solver_threads=1, max_calls=5, max_repairs=1))
        self.assertEqual(result["status"], "missing_math_input")
        self.assertEqual(model.calls, [])

    def test_model_call_budget_stops_original_agent_without_fallback_call(self):
        model = FakeModel(["maximize x subject to x <= 5"])
        case = {"case_id": "case-3", "question": "Maximize x subject to x <= 5.",
                "math_input": None,
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        result = asyncio.run(baselines.run_baseline(
            "or_llm_agent", case, model, self.output, "run-3", level="end_to_end",
            source_root=self.root, python_executable="python", process_timeout=10,
            solver_time_limit=5, solver_threads=1, max_calls=1, max_repairs=1))
        self.assertEqual(result["status"], "call_budget_exhausted")
        self.assertEqual(len(model.calls), 1)

    def test_outer_budgeted_model_exhaustion_is_reported_as_budget_status(self):
        underlying = FakeModel(["maximize x subject to x <= 5"])
        underlying.ledger_path = self.output / "calls.jsonl"
        model = BudgetedModel(underlying, 1)
        case = {"case_id": "case-upstream-budget", "question": "Maximize x subject to x <= 5.",
                "math_input": None,
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        result = asyncio.run(baselines.run_baseline(
            "or_llm_agent", case, model, self.output, "run-upstream-budget",
            level="end_to_end", source_root=self.root, python_executable="python",
            process_timeout=10, solver_time_limit=5, solver_threads=1,
            max_calls=5, max_repairs=1))
        self.assertEqual(result["status"], "call_budget_exhausted")
        self.assertEqual(len(underlying.calls), 1)
        self.assertEqual(result["call_ids"], ["call-1"])

    def test_or_parse_failure_is_repaired_and_only_verified_answer_is_returned(self):
        model = FakeModel([
            "maximize x subject to x <= 5",
            "```python\nprint('not an answer')\n```",
            "```python\nprint('FinalAnswer=【5】')\n```",
        ])
        case = {"case_id": "case-parse-repair", "question": "Maximize x subject to x <= 5.",
                "math_input": None,
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        executed = []

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            executed.append(code)
            return {"status": "executed", "returncode": 0,
                    "stdout": "FinalAnswer=【5】" if len(executed) == 2 else "unparseable",
                    "stderr": "", "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "or_llm_agent", case, model, self.output, "run-parse-repair",
                level="end_to_end", source_root=self.root, python_executable="python",
                process_timeout=10, solver_time_limit=5, solver_threads=1,
                max_calls=5, max_repairs=1))

        self.assertEqual(result["status"], "completed", result.get("baseline_trace"))
        self.assertEqual(result["prediction"], 5)
        self.assertEqual(len(executed), 2)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIsNone(result["attempts"][0]["prediction"])
        self.assertEqual(result["attempts"][1]["prediction"], 5)

    def test_or_requires_single_code_block_and_preserves_invalid_attempt(self):
        model = FakeModel([
            "maximize x subject to x <= 5",
            "```python\nprint('first')\n```\n```python\nprint('second')\n```",
            "```python\nprint('FinalAnswer=【5】')\n```",
        ])
        case = {"case_id": "case-multicode", "question": "Maximize x subject to x <= 5.",
                "math_input": None,
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        executed = []

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            executed.append(code)
            return {"status": "executed", "returncode": 0,
                    "stdout": "FinalAnswer=【5】", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "or_llm_agent", case, model, self.output, "run-multicode",
                level="end_to_end", source_root=self.root, python_executable="python",
                process_timeout=10, solver_time_limit=5, solver_threads=1,
                max_calls=5, max_repairs=1))

        self.assertEqual(result["status"], "completed", result.get("baseline_trace"))
        self.assertEqual(result["prediction"], 5)
        self.assertEqual(len(executed), 1)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(result["attempts"][0]["execution"]["status"], "invalid_output")

    def test_real_coe_chain_runs_offline_through_logged_chat_proxy(self):
        source = Path(r"D:\LLMProject\Chain-of-Experts-main")
        available = baselines.baseline_availability("coe", source)
        if not available["available"]:
            self.skipTest(f"CoE snapshot/dependencies unavailable: {available}")
        model = FakeModel([
            "Parameter Extractor", "Extract parameters.",
            "Modeling Expert", "Model formulation.",
            "Programming Expert", "```python\nimport gurobipy as gp\nm=gp.Model()\nx=m.addVar()\nm.setObjective(x, gp.GRB.MAXIMIZE)\nm.optimize()\nprint('FinalAnswer=【5】')\n```",
            "```python\nprint('FinalAnswer=【5】')\n```",
        ])
        case = {"case_id": "coe-1", "question": "Maximize x subject to x <= 5.",
                "math_input": None,
                "answer_spec": {"kind": "scalar", "fields": ["answer"]}}

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            return {"status": "executed", "returncode": 0,
                    "stdout": "FinalAnswer=【5】", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "coe", case, model, self.output, "coe-run", level="end_to_end",
                source_root=source, python_executable="python", process_timeout=10,
                solver_time_limit=5, solver_threads=1, max_calls=10, max_repairs=0))

        self.assertEqual(result["status"], "completed", result.get("baseline_trace"))
        self.assertEqual(result["prediction"], 5)
        self.assertEqual(len(model.calls), 7)
        self.assertEqual(result["baseline_trace"]["algorithm"], "chain_of_experts")

    def test_real_optimus_stages_run_offline_through_logged_model(self):
        source = Path(r"D:\LLMProject\OptiMUS-main")
        available = baselines.baseline_availability("optimus", source)
        if not available["available"]:
            self.skipTest(f"OptiMUS snapshot/dependencies unavailable: {available}")
        model = FakeModel([
            "{}",  # parameter extraction
            "=====\nOBJECTIVE: Maximize profit\n=====",  # objective extraction
            "[]",  # constraint extraction
            "[]",  # redundancy check
            "===== $profit$ =====",  # objective formulation
            "=====\nmodel.setObjective(0, GRB.MAXIMIZE)\n=====",  # code generation
        ])
        case = {"case_id": "optimus-1", "question": "Maximize profit.",
                "math_input": None,
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        calls = []

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            calls.append(code)
            return {"status": "executed", "returncode": 0,
                    "stdout": "Optimal Objective Value: 7.0", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "optimus", case, model, self.output, "optimus-run", level="end_to_end",
                source_root=source, python_executable="python", process_timeout=10,
                solver_time_limit=5, solver_threads=1, max_calls=20, max_repairs=0))

        self.assertEqual(result["status"], "completed", result.get("baseline_trace"))
        self.assertEqual(result["prediction"], 7.0)
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["baseline_trace"]["algorithm"], "OptiMUS_staged_pipeline")

    def test_optimus_low_confidence_constraint_feedback_prompt_is_loaded(self):
        source = Path(r"D:\LLMProject\OptiMUS-main")
        available = baselines.baseline_availability("optimus", source)
        if not available["available"]:
            self.skipTest(f"OptiMUS snapshot/dependencies unavailable: {available}")

        feedback_prompts = []

        def fake_response(prompt, model=None):
            feedback_prompts.append(prompt)
            return '{"action": "yes", "updatedConstraint": null}'

        stages = baselines._load_optimus(source, fake_response)
        constraints = ["x >= 0"]
        valid, result = stages["constraint"]["extract_score_constraint"](
            "Maximize x.", "1 OUT OF 5", {}, {}, constraints, "x >= 0", None
        )

        self.assertTrue(valid)
        self.assertIs(result, constraints)
        self.assertEqual(len(feedback_prompts), 1)
        self.assertIn("Do you want to keep this constraint?", feedback_prompts[0])

    def test_optimus_constraint_model_prompts_are_loaded(self):
        source = Path(r"D:\LLMProject\OptiMUS-main")
        available = baselines.baseline_availability("optimus", source)
        if not available["available"]:
            self.skipTest(f"OptiMUS snapshot/dependencies unavailable: {available}")

        prompts = []

        def fake_response(prompt, model=None):
            prompts.append(prompt)
            return "not valid formulation output"

        stages = baselines._load_optimus(source, fake_response)
        constraints = [{"description": "x >= 2", "formulation": None, "code": None}]
        with patch("builtins.print"):
            result, variables = stages["constraint_model"]["get_constraint_formulations"](
                "Minimize x.", {}, constraints, "fake-model", True
            )

        self.assertEqual(len(prompts), 3)
        self.assertIn("model the following constraint mathematically", prompts[0])
        self.assertIn("Does this constraint logically make sense?", prompts[2])
        self.assertEqual(len(result), 1)
        self.assertEqual(variables, {})

    def test_optimus_repair_delimiters_are_unwrapped_and_validated(self):
        source = Path(r"D:\LLMProject\OptiMUS-main")
        available = baselines.baseline_availability("optimus", source)
        if not available["available"]:
            self.skipTest(f"OptiMUS snapshot/dependencies unavailable: {available}")
        stages = baselines._load_optimus(source, lambda prompt, model=None: "")
        original_extract_code = stages["execute_code"]["extract_code"]

        response = "=====\nprint('FinalAnswer=【2.0】')\n====="
        self.assertEqual(
            baselines._extract_optimus_repair_code(response, original_extract_code),
            "print('FinalAnswer=【2.0】')",
        )
        with self.assertRaisesRegex(ValueError, "parseable Python code"):
            baselines._extract_optimus_repair_code(
                "=====\nif True print('bad')\n=====", original_extract_code
            )
        with self.assertRaisesRegex(ValueError, "parseable Python code"):
            baselines._extract_optimus_repair_code(
                "=====\nprint('unclosed')", original_extract_code
            )

    def test_conditioned_coding_uses_frozen_input_and_supplied_advice(self):
        model = FakeModel([
            "```python\nimport gurobipy as gp\nm=gp.Model()\nx=m.addVar()\nm.setObjective(x, gp.GRB.MAXIMIZE)\nm.addConstr(x <= 5)\nm.optimize()\nprint('FinalAnswer=【5】')\n```",
        ])
        case = {"case_id": "case-4", "question": "Maximize x subject to x <= 5.",
                "math_input": {"Variables_List": [{"symbol": "x"}]},
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            return {"status": "executed", "returncode": 0,
                    "stdout": "Best objective 5\nFinalAnswer=【5】", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "or_llm_agent", case, model, self.output, "run-4", level="coding",
                ned_advice={"resultant": "r = log(x)"}, source_root=self.root,
                python_executable="python", process_timeout=10,
                solver_time_limit=5, solver_threads=1, max_calls=5, max_repairs=1))
        self.assertEqual(result["status"], "completed")
        text = repr(result["baseline_trace"]["model_calls"])
        self.assertIn("authoritative_math_input", text)
        self.assertIn("supplied_ned_advice", text)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["baseline_trace"]["level_label"], "baseline_conditioned_coding")

    def test_coe_coding_runs_only_programming_expert_with_frozen_math(self):
        source = Path(r"D:\LLMProject\Chain-of-Experts-main")
        if not baselines.baseline_availability("coe", source)["available"]:
            self.skipTest("CoE source/dependencies unavailable")
        model = FakeModel([
            "```python\nimport gurobipy as gp\nm=gp.Model()\nx=m.addVar()\nm.setObjective(x, gp.GRB.MAXIMIZE)\nm.addConstr(x <= 5)\nm.optimize()\nprint('FinalAnswer=【5】')\n```",
        ])
        case = {"case_id": "coe-coding", "question": "Maximize x subject to x <= 5.",
                "math_input": {"Objective": {"Mathematical_expressions": "maximize x"}},
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            return {"status": "executed", "returncode": 0,
                    "stdout": "FinalAnswer=【5】", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "coe", case, model, self.output, "coe-coding-run", level="coding",
                ned_advice={"resultant": "r = log(x)"}, source_root=source,
                python_executable="python", process_timeout=10,
                solver_time_limit=5, solver_threads=1, max_calls=5, max_repairs=0))
        self.assertEqual(result["status"], "completed", result.get("baseline_trace"))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["baseline_trace"]["stages"], ["programming_expert"])
        self.assertIn("authoritative frozen mathematical model", repr(model.calls).casefold())

    def test_optimus_coding_uses_frozen_model_only_in_code_generation(self):
        source = Path(r"D:\LLMProject\OptiMUS-main")
        available = baselines.baseline_availability("optimus", source)
        if not available["available"]:
            self.skipTest(f"OptiMUS snapshot/dependencies unavailable: {available}")
        model = FakeModel([
            "=====\nmodel.addConstr(x <= a)\n=====",
            "=====\nmodel.setObjective(x, GRB.MAXIMIZE)\n=====",
        ])
        case = {"case_id": "optimus-coding", "question": "Maximize x under its frozen model.",
                "math_input": {
                    "Parameters_List": [{"Name": "a", "Type": "float", "Value": 5}],
                    "Variables_List": [{"symbol": "x", "Meaning": "decision variable",
                                        "Type": "continuous", "Range": "x >= 0"}],
                    "Constraint_Table": [{"name": "limit", "Mathematical_expressions": "x <= a"}],
                    "Objective": {"Objective_sentence": "Maximize x",
                                  "Mathematical_expressions": "maximize x"},
                    "Problem_Type": "LP"},
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        executed = []

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            executed.append(code)
            return {"status": "executed", "returncode": 0,
                    "stdout": "Optimal Objective Value: 5.0", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "optimus", case, model, self.output, "optimus-coding-run", level="coding",
                source_root=source, python_executable="python", process_timeout=10,
                solver_time_limit=5, solver_threads=1, max_calls=10, max_repairs=0))
        self.assertEqual(result["status"], "completed", result.get("baseline_trace"))
        self.assertEqual(result["prediction"], 5.0)
        self.assertEqual(result["baseline_trace"]["stages"], ["code_generation"])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(executed), 1)

    def test_optimus_coding_preserves_explicit_scalar_bounds_in_original_codegen(self):
        source = Path(r"D:\LLMProject\OptiMUS-main")
        if not baselines.baseline_availability("optimus", source)["available"]:
            self.skipTest("OptiMUS source/dependencies unavailable")
        model = FakeModel([
            "=====\nmodel.addConstr(x <= a)\n=====",
            "=====\nmodel.setObjective(x + n + z, GRB.MAXIMIZE)\n=====",
        ])
        math_input = {
            "Parameters_List": [{"Name": "a", "Type": "float", "Value": 5}],
            "Variables_List": [
                {"symbol": "x", "Type": "continuous type", "Range ": "[-10, 200]"},
                {"symbol": "n", "Type": "integer type", "Range": "0 <= n <= 12"},
                {"symbol": "z", "Type": "continuous type", "Range": "unbounded real"},
            ],
            "Constraint_Table": [{"name": "limit", "Mathematical_expressions": "x <= a"}],
            "Objective": {"Mathematical_expressions": "maximize x + n + z"},
            "Problem_Type": "MILP",
        }
        case = {"case_id": "optimus-bounds", "question": "Frozen bound preservation test.",
                "math_input": math_input,
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        self.assertEqual(baselines.baseline_coding_support("optimus", math_input),
                         {"supported": True})
        executed = []

        async def fake_execute(code, directory, python_executable, timeout_seconds,
                               solver_time_limit, solver_threads=1):
            executed.append(code)
            return {"status": "executed", "returncode": 0,
                    "stdout": "Optimal Objective Value: 7.0", "stderr": "",
                    "solver_status": {"status": 2, "sol_count": 1},
                    "solver_limits_applied": True}

        with patch.object(baselines, "execute_code", fake_execute):
            result = asyncio.run(baselines.run_baseline(
                "optimus", case, model, self.output, "optimus-bounds-run", level="coding",
                source_root=source, python_executable="python", process_timeout=10,
                solver_time_limit=5, solver_threads=1, max_calls=10, max_repairs=0))
        self.assertEqual(result["status"], "completed", result.get("baseline_trace"))
        code = executed[0]
        self.assertIn('x = model.addVar(lb=-10.0, ub=200.0, vtype=GRB.CONTINUOUS, name="x")', code)
        self.assertIn('n = model.addVar(lb=0.0, ub=12.0, vtype=GRB.INTEGER, name="n")', code)
        self.assertIn('z = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="z")', code)
        self.assertEqual(baselines._parse_scalar_variable_bounds("x", "CONTINUOUS", "[0, +∞)"),
                         (0.0, float("inf")))
        self.assertEqual(baselines._parse_scalar_variable_bounds("x", "CONTINUOUS", "(-∞,+∞)"),
                         (float("-inf"), float("inf")))
        with self.assertRaises(baselines.BaselineUnavailable):
            baselines._parse_scalar_variable_bounds("X", "CONTINUOUS", "x >= 0")
        with self.assertRaises(baselines.BaselineUnavailable):
            baselines._parse_scalar_variable_bounds("x", "CONTINUOUS", "0 < x <= 10")
        with self.assertRaises(baselines.BaselineUnavailable):
            baselines._parse_scalar_variable_bounds("x", "CONTINUOUS", "x <= 1e999")

    def test_optimus_unmappable_domain_fails_before_model_calls(self):
        source = Path(r"D:\LLMProject\OptiMUS-main")
        if not baselines.baseline_availability("optimus", source)["available"]:
            self.skipTest("OptiMUS source/dependencies unavailable")
        model = FakeModel([])
        case = {"case_id": "optimus-unsupported", "question": "Maximize x.",
                "math_input": {
                    "Parameters_List": [],
                    "Variables_List": [{"symbol": "x", "Type": "continuous",
                                    "Range": "-5 < x < 5"}],
                    "Constraint_Table": [],
                    "Objective": {"Mathematical_expressions": "maximize x"},
                    "Problem_Type": "LP"},
                "answer_spec": {"kind": "scalar", "fields": ["objective_value"]}}
        result = asyncio.run(baselines.run_baseline(
            "optimus", case, model, self.output, "optimus-unsupported-run", level="coding",
            source_root=source, python_executable="python", process_timeout=10,
            solver_time_limit=5, solver_threads=1, max_calls=10, max_repairs=0))
        self.assertEqual(result["status"], "unsupported_coding_input")
        self.assertEqual(model.calls, [])

    def test_offline_coding_support_preflight_matches_optimus_mapper(self):
        supported = {
            "Parameters_List": [],
            "Variables_List": [{"symbol": "k", "Type": "integer", "Range": "k >= 0"}],
            "Constraint_Table": [],
            "Objective": {"Mathematical_expressions": "maximize k"},
            "Problem_Type": "IP",
        }
        unsupported = {
            **supported,
            "Variables_List": [{"symbol": "k", "Type": "integer", "Range": "-2 < k <= 8"}],
        }
        self.assertEqual(baselines.baseline_coding_support("OptiMUS", supported),
                         {"supported": True})
        result = baselines.baseline_coding_support("optimus", unsupported)
        self.assertFalse(result["supported"])
        self.assertFalse(baselines.baseline_coding_support("optimus", unsupported)["supported"])
        self.assertTrue(result["reason"])
        self.assertEqual(baselines.baseline_coding_support("coe", supported),
                         {"supported": True})
        self.assertFalse(baselines.baseline_coding_support("or_llm_agent", None)["supported"])


if __name__ == "__main__":
    unittest.main()
