import asyncio
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from rebuttal_runner.data import load_manifest
from rebuttal_runner.models import ModelConfig
from rebuttal_runner.runner import run_paired


EXAMPLE = Path(__file__).resolve().parents[1] / "rebuttal_runner/examples/math_smoke.json"


class FakeModel:
    def __init__(self, directory):
        self.config = ModelConfig(model="test-model", base_url="https://unit.invalid/v1")
        self.ledger_path = directory / "calls.jsonl"
        self.calls = []

    async def complete(self, messages, **identity):
        self.calls.append({"messages": copy.deepcopy(messages), **identity})
        content = '```python\nprint("FinalAnswer=【2】")\n```'
        if identity["stage"] == "auxiliary":
            content = json.dumps({"math_model_advice": [], "decomposition": {
                "schema_version": "nextor-aux-tree-v1", "candidates": []}})
        return {"content": content,
                "call_id": f"call-{len(self.calls)}", "usage": {"total_tokens": 10},
                "returned_model": "test-model", "elapsed_seconds": 0.01}


def executed(value=2):
    return {"status": "executed", "returncode": 0, "stdout": f"FinalAnswer=【{value}】\n",
            "stderr": "", "elapsed_seconds": 0.01, "solver_limits_applied": True,
            "solver_status": {"status": 2, "sol_count": 1}}


class PairedRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_is_not_resumed_as_a_finished_unit(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model = FakeModel(directory)
            model.complete = AsyncMock(side_effect=asyncio.CancelledError)
            manifest = load_manifest(EXAMPLE)
            with self.assertRaises(asyncio.CancelledError):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable)
            with self.assertRaisesRegex(RuntimeError, "Interrupted unit"):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable,
                                 resume=True)
            self.assertEqual(model.complete.await_count, 1)

    async def test_duplicate_cases_are_rejected_before_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model = FakeModel(directory)
            manifest = load_manifest(EXAMPLE)
            manifest["cases"].append(copy.deepcopy(manifest["cases"][0]))
            with self.assertRaisesRegex(ValueError, "Duplicate case_id"):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable)
            self.assertEqual(model.calls, [])

    async def test_answer_without_recorded_solver_solution_cannot_pass(self):
        for solver_count, applied in [(0, True), (1, False)]:
            with self.subTest(solver_count=solver_count, applied=applied), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                result = executed()
                result["solver_limits_applied"] = applied
                result["solver_status"]["sol_count"] = solver_count
                model = FakeModel(directory)
                with patch("rebuttal_runner.runner.execute_code", new=AsyncMock(return_value=result)):
                    records = await run_paired(load_manifest(EXAMPLE), model, directory / "run",
                                               python_executable=sys.executable)
                self.assertTrue(all(not record["score"]["correct"] for record in records))
                self.assertTrue(all(record["status"] == "invalid_output" for record in records))

    async def test_same_input_no_reference_leak_no_oracle_repair(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = load_manifest(EXAMPLE)
            manifest["cases"][0]["reference"] = 987654321
            original = copy.deepcopy(manifest)
            model = FakeModel(directory)
            with patch("rebuttal_runner.runner.execute_code", new=AsyncMock(return_value=executed())):
                records = await run_paired(manifest, model, directory / "run", max_repairs=3,
                                           python_executable=sys.executable)
            self.assertEqual(manifest, original)
            self.assertEqual(len(model.calls), 3)
            self.assertTrue(all(not r["score"]["correct"] for r in records))
            self.assertEqual(records[0]["input_hash"], records[1]["input_hash"])
            self.assertNotIn("987654321", json.dumps(model.calls))
            native = json.dumps(model.calls[0]["messages"])
            ned = json.dumps(model.calls[2]["messages"])
            self.assertNotIn("nedtree_auxiliary_advice", native)
            self.assertIn("nedtree_auxiliary_advice", ned)
            self.assertEqual(sum(c["stage"] == "auxiliary" for c in model.calls), 1)
            self.assertEqual(records[0]["representation_call_ids"], [])
            self.assertEqual(records[1]["representation_call_ids"], ["call-2"])

    async def test_zero_repairs_stops_after_one_execution_per_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            model = FakeModel(directory)
            failed = {"status": "executed", "returncode": 1, "stdout": "", "stderr": "error"}
            with patch("rebuttal_runner.runner.execute_code", new=AsyncMock(return_value=failed)):
                records = await run_paired(load_manifest(EXAMPLE), model, directory / "run",
                                           max_repairs=0, python_executable=sys.executable)
            self.assertEqual(len(model.calls), 3)
            self.assertTrue(all(len(r["attempts"]) == 1 and r["repair_count"] == 0 for r in records))

    async def test_missing_input_counts_both_arms_without_paid_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = load_manifest(EXAMPLE)
            manifest["cases"][0]["math_input"] = None
            manifest["cases"][0]["upstream"] = {"status": "failed", "call_ids": ["upstream-call"]}
            model = FakeModel(directory)
            records = await run_paired(manifest, model, directory / "run", python_executable=sys.executable)
            self.assertEqual(len(model.calls), 0)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(r["status"] == "upstream_failed" and not r["score"]["correct"] for r in records))

    async def test_resume_skips_failed_or_successful_finished_units(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = load_manifest(EXAMPLE)
            model = FakeModel(directory)
            with patch("rebuttal_runner.runner.execute_code", new=AsyncMock(return_value=executed(3))):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable)
                records = await run_paired(manifest, model, directory / "run", python_executable=sys.executable,
                                           resume=True)
            self.assertEqual(len(model.calls), 3)
            self.assertEqual(len(records), 2)
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable,
                                 max_repairs=1, resume=True)

    async def test_unreviewed_and_possible_leak_fail_before_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            manifest = load_manifest(EXAMPLE)
            manifest["cases"][0]["review_status"] = "needs_review"
            model = FakeModel(directory)
            with self.assertRaisesRegex(ValueError, "Unreviewed"):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable)
            manifest["cases"][0]["review_reasons"] = ["question_may_include_answer_material"]
            with self.assertRaisesRegex(ValueError, "answer material"):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable,
                                 allow_unreviewed=True)
            self.assertEqual(model.calls, [])


if __name__ == "__main__":
    unittest.main()
