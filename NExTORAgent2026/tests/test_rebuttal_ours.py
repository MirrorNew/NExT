import json
from pathlib import Path
import tempfile
import unittest

from rebuttal_runner.ours import generate_ours_code, prepare_auxiliary, prepare_ours_input
from rebuttal_runner.models import ModelConfig
from rebuttal_runner.auxiliary_tree import AUXILIARY_SCHEMA_VERSION


class FakeModel:
    def __init__(self, replies):
        self.config = ModelConfig(model="test-model", base_url="https://unit.invalid/v1")
        self.replies = list(replies)
        self.calls = []

    async def complete(self, messages, **identity):
        self.calls.append({"messages": [dict(message) for message in messages], **identity})
        content = self.replies.pop(0)
        return {"content": content, "call_id": f"aux-call-{len(self.calls)}",
                "returned_model": "test-model", "usage": {"total_tokens": 1}}


class OursIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_coder_keeps_project_prompt_and_receives_answer_spec(self):
        model = FakeModel(["```python\nprint('FinalAnswer=【2】')\n```"])
        case_public = {
            "case_id": "case-1", "question": "maximize x",
            "answer_spec": {"kind": "scalar", "fields": ["objective_value"]},
        }
        math_input = {
            "Parameters_List": [], "Variables_List": [{"symbol": "x"}],
            "Constraint_Table": [], "Objective": {"Mathematical_expressions": "Maximize x"},
            "Problem_Type": "NLP",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = await generate_ours_code(
                case_public, math_input, "maximize x", None, model, directory, "run-1",
                code_system="shared code system",
            )
            self.assertTrue((Path(directory) / "coding" / "prompt.json").is_file())
            self.assertTrue((Path(directory) / "coding" / "response.json").is_file())
        self.assertEqual(result["call_ids"], ["aux-call-1"])
        messages = model.calls[0]["messages"]
        self.assertIn("You are a Python program expert", messages[0]["content"])
        self.assertIn("shared code system", messages[0]["content"])
        self.assertIn("frozen Context/math_input is the authoritative mathematical model", messages[1]["content"])
        self.assertIn('"objective_value"', messages[1]["content"])
        self.assertGreaterEqual(result["stage_costs"][0]["elapsed_seconds"], 0)

    async def test_repair_uses_project_two_step_prompt_and_answer_spec(self):
        model = FakeModel(["{\"findings\": []}", "print('FinalAnswer=【2】')"])
        case_public = {
            "case_id": "case-1", "question": "maximize x",
            "answer_spec": {"kind": "scalar", "fields": ["objective_value"]},
        }
        math_input = {
            "Parameters_List": [], "Variables_List": [{"symbol": "x"}],
            "Constraint_Table": [], "Objective": {"Mathematical_expressions": "Maximize x"},
            "Problem_Type": "NLP",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = await generate_ours_code(
                case_public, math_input, "maximize x", {"tree": {}}, model, directory, "run-1",
                feedback={"code": "bad code", "error": "syntax error"},
                code_system="shared code system",
            )
            for stage in ("repair_analysis", "repair_code"):
                self.assertTrue((Path(directory) / "repair" / stage / "prompt.json").is_file())
                self.assertTrue((Path(directory) / "repair" / stage / "response.json").is_file())
        self.assertEqual(result["call_ids"], ["aux-call-1", "aux-call-2"])
        self.assertEqual(len(model.calls), 2)
        for call in model.calls:
            serialized = json.dumps(call["messages"], ensure_ascii=False)
            self.assertIn("objective_value", serialized)
            self.assertIn("shared code system", serialized)
            self.assertIn("frozen math_input is the authoritative mathematical model", serialized)
        self.assertIn("code-first", json.dumps(model.calls[0]["messages"], ensure_ascii=False))

    async def test_auxiliary_call_is_once_and_bad_schema_does_not_trigger_redecomposition(self):
        model = FakeModel([json.dumps({"math_model_advice": []})])
        math_input = {
            "Parameters_List": [], "Variables_List": [{"symbol": "x"}],
            "Constraint_Table": [], "Objective": {"Mathematical_expressions": "Maximize x"},
            "Problem_Type": "NLP",
        }
        public = {"case_id": "case-1", "question": "maximize x", "reference": "GOLD_SECRET_741852"}
        with tempfile.TemporaryDirectory() as directory:
            result = await prepare_auxiliary(public, math_input, model, directory, "run-1")
            self.assertTrue((Path(directory) / "auxiliary" / "prompt.json").is_file())
            self.assertTrue((Path(directory) / "auxiliary" / "response.json").is_file())
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["call_ids"], ["aux-call-1"])
        self.assertEqual(result["status"], "invalid_decomposition")
        self.assertIn("refusing raw-expression redecomposition", result["error"])
        self.assertNotIn("GOLD_SECRET_741852", json.dumps(model.calls, ensure_ascii=False))
        self.assertEqual(model.replies, [])
        self.assertIn("structured Context is authoritative", model.calls[0]["messages"][0]["content"])
        self.assertGreaterEqual(result["stage_costs"][0]["elapsed_seconds"], 0)

    async def test_auxiliary_tree_is_built_from_same_response_without_reference_leak(self):
        reply = {
            "math_model_advice": [{"variables": ["t=x**2"], "constraints": [], "objective": "t"}],
            "decomposition": {
                "schema_version": AUXILIARY_SCHEMA_VERSION,
                "candidates": [{
                    "source_id": "objective", "source_expression": "Maximize x**2",
                    "relation": "objective", "linearized_form": "t",
                    "definitions": [{"symbol": "t", "rhs": "x**2", "domain_conditions": []}],
                    "domain_conditions": [],
                }],
            },
        }
        model = FakeModel([json.dumps(reply)])
        math_input = {
            "Parameters_List": [], "Variables_List": [{"symbol": "x"}],
            "Constraint_Table": [], "Objective": {"Mathematical_expressions": "Maximize x**2"},
            "Problem_Type": "NLP",
        }
        with tempfile.TemporaryDirectory() as directory:
            result = await prepare_auxiliary(
                {"case_id": "case-1", "question": "maximize x squared", "reference": "GOLD_SECRET_741852"},
                math_input, model, directory, "run-1",
            )
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result["status"], "validated")
        self.assertEqual(result["tree"]["candidates"][0]["definitions"][0]["symbol"], "t")
        self.assertNotIn("GOLD_SECRET_741852", json.dumps(model.calls, ensure_ascii=False))

    async def test_end_to_end_upstream_uses_logged_project_prompts_and_excludes_reference(self):
        replies = [
            "parameter analysis", '{"Parameters_List": []}',
            "factor analysis", json.dumps({
                "Variables_List": [{"symbol": "x", "Meaning": "decision", "Type": "continuous", "Range ": "x >= 0"}],
                "Constraint_Table": [],
                "Objective": {"Objective_sentence": "maximize x", "Mathematical_expressions": "Maximize x"},
                "Problem_Type": "NLP",
            }),
            "model draft",
        ]
        model = FakeModel(replies)
        with tempfile.TemporaryDirectory() as directory:
            result = await prepare_ours_input(
                {"case_id": "case-1", "question": "maximize x", "reference": 123456},
                model, directory, "run-1",
            )
            self.assertEqual(
                json.loads((Path(directory) / "modeling" / "response.json").read_text(encoding="utf-8"))["content"],
                "model draft",
            )
            self.assertTrue((Path(directory) / "modeling" / "prompt.json").is_file())
        self.assertEqual(len(model.calls), 5)
        self.assertEqual(len(result["call_ids"]), 5)
        self.assertEqual(result["math_input"]["Problem_Type"], "NLP")
        self.assertEqual(result["math_model"], "model draft")
        self.assertNotIn("123456", json.dumps(model.calls, ensure_ascii=False))
        self.assertEqual([call["stage"] for call in model.calls], [
            "extract_parameters_analysis", "extract_parameters_json",
            "extract_factors_analysis", "extract_factors_json", "modeling",
        ])
        self.assertTrue(all(cost["elapsed_seconds"] >= 0 for cost in result["stage_costs"]))

    async def test_failed_logged_call_saves_prompt_and_error_without_response(self):
        class FailingModel(FakeModel):
            async def complete(self, messages, **identity):
                self.calls.append({"messages": [dict(message) for message in messages], **identity})
                raise RuntimeError("request failed with api_key=sk-abcdefgh12345678")

        model = FailingModel([])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "request failed"):
                await prepare_ours_input(
                    {"case_id": "case-fail", "question": "maximize x", "reference": 5},
                    model, directory, "run-fail",
                )
            stage = Path(directory) / "extract_parameters_analysis"
            self.assertTrue((stage / "prompt.json").is_file())
            self.assertTrue((stage / "error.json").is_file())
            self.assertFalse((stage / "response.json").exists())
            error = json.loads((stage / "error.json").read_text(encoding="utf-8"))
            self.assertNotIn("sk-abcdefgh12345678", json.dumps(error))
            self.assertEqual(len(model.calls), 1)

    async def test_failed_auxiliary_call_saves_prompt_and_error_once(self):
        class FailingModel(FakeModel):
            async def complete(self, messages, **identity):
                self.calls.append({"messages": [dict(message) for message in messages], **identity})
                raise RuntimeError("aux request failed")

        model = FailingModel([])
        math_input = {
            "Parameters_List": [], "Variables_List": [{"symbol": "x"}],
            "Constraint_Table": [], "Objective": {"Mathematical_expressions": "Maximize x"},
            "Problem_Type": "NLP",
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "aux request failed"):
                await prepare_auxiliary(
                    {"case_id": "case-fail", "question": "maximize x"},
                    math_input, model, directory, "run-fail",
                )
            stage = Path(directory) / "auxiliary"
            self.assertTrue((stage / "prompt.json").is_file())
            self.assertTrue((stage / "error.json").is_file())
            self.assertFalse((stage / "response.json").exists())
            self.assertEqual(len(model.calls), 1)


if __name__ == "__main__":
    unittest.main()
