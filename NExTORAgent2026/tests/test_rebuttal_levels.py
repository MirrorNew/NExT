import copy
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from rebuttal_runner.data import load_manifest
from rebuttal_runner.cli import main, read_arm_records, summarize_directory
from rebuttal_runner.models import ModelConfig
from rebuttal_runner.planning import attach_cached_inputs, plan_run
from rebuttal_runner.reporting import summarize_records
from rebuttal_runner.runner import digest, extract_inputs, run_paired
from test_rebuttal_runner import EXAMPLE, FakeModel, executed


class LevelTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_baseline_input_does_not_spend_auxiliary_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = FakeModel(directory)
            with patch("rebuttal_runner.baselines.baseline_availability", return_value={"available": True}), patch(
                    "rebuttal_runner.baselines.source_fingerprint", return_value={"sha256": "fixture"}), patch(
                    "rebuttal_runner.baselines.baseline_coding_support", return_value={"supported": False, "reason": "unsupported domain"}), patch(
                    "rebuttal_runner.ours.prepare_auxiliary", new=AsyncMock()) as auxiliary:
                records = await run_paired(load_manifest(EXAMPLE), model, directory / "run",
                    framework="optimus", level="coding", source_root="unused", python_executable=sys.executable)
            auxiliary.assert_not_awaited()
            self.assertEqual(model.calls, [])
            self.assertEqual(len(records), 2)
            self.assertTrue(all(row["status"] == "unsupported_coding_input" for row in records))

    async def test_parallel_extraction_cannot_charge_for_same_case_twice(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = load_manifest(EXAMPLE)
            entered, release = asyncio.Event(), asyncio.Event()
            async def upstream(*args):
                entered.set()
                await release.wait()
                return {"math_input": manifest["cases"][0]["math_input"], "math_model": "frozen"}
            with patch("rebuttal_runner.ours.prepare_ours_input", side_effect=upstream) as prepare:
                first = asyncio.create_task(extract_inputs(manifest, FakeModel(directory), directory / "frozen.json"))
                await entered.wait()
                try:
                    with self.assertRaisesRegex(RuntimeError, "Concurrent"):
                        await extract_inputs(manifest, FakeModel(directory), directory / "frozen.json")
                    self.assertEqual(prepare.await_count, 1)
                finally:
                    release.set()
                    await first

    async def test_same_directory_cannot_have_two_paid_runners(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            entered, release = asyncio.Event(), asyncio.Event()
            async def execute(*args):
                entered.set()
                await release.wait()
                return executed()
            with patch("rebuttal_runner.runner.execute_code", side_effect=execute):
                first = asyncio.create_task(run_paired(load_manifest(EXAMPLE), FakeModel(directory),
                    directory / "run", arms=("native",), python_executable=sys.executable))
                await entered.wait()
                second = FakeModel(directory)
                try:
                    with self.assertRaisesRegex(RuntimeError, "locked"):
                        await run_paired(load_manifest(EXAMPLE), second, directory / "run",
                            arms=("native",), resume=True, python_executable=sys.executable)
                    self.assertEqual(second.calls, [])
                finally:
                    release.set()
                    await first
            self.assertFalse((directory / "run/.run.lock").exists())

    async def test_extraction_saves_paid_upstream_before_auxiliary_interruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = load_manifest(EXAMPLE)
            prepared = {"math_input": manifest["cases"][0]["math_input"], "math_model": "frozen"}
            with patch("rebuttal_runner.ours.prepare_ours_input", new=AsyncMock(return_value=prepared)), patch(
                    "rebuttal_runner.ours.prepare_auxiliary", new=AsyncMock(side_effect=asyncio.CancelledError())):
                with self.assertRaises(asyncio.CancelledError):
                    await extract_inputs(manifest, FakeModel(directory), directory / "frozen.json")
            partial = load_manifest(directory / "frozen.partial.json")
            self.assertEqual(partial["cases"][0]["math_model"], "frozen")
            self.assertEqual(partial["cases"][0]["upstream"]["status"], "complete")
            self.assertFalse((directory / "frozen.json").exists())

    async def test_auxiliary_logs_are_not_scored_or_rescored_as_trials(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "run"
            with patch("rebuttal_runner.runner.execute_code", new=AsyncMock(return_value=executed())):
                await run_paired(load_manifest(EXAMPLE), FakeModel(directory), output,
                                 python_executable=sys.executable)
            self.assertEqual(len(list(output.glob("cases/*/auxiliary/result.json"))), 1)
            self.assertEqual(len(read_arm_records(output)), 2)
            summary = summarize_directory(output)
            self.assertEqual(sum(g["run_count"] for g in summary["groups"]), 2)
            rescored = directory / "rescored.jsonl"
            with patch("builtins.print"):
                main(["rescore", "--run-dir", str(output), "--output", str(rescored)])
            rows = [json.loads(line) for line in rescored.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["arm"] for row in rows}, {"native", "ned"})

    async def test_end_to_end_repeats_regenerate_shared_upstream_and_one_auxiliary(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = load_manifest(EXAMPLE)
            manifest["cases"][0]["reference"] = 987654321
            model = FakeModel(directory)
            count = 0

            async def upstream(public, scoped, folder, run_id):
                nonlocal count
                count += 1
                self.assertIsNone(public["math_input"])
                self.assertNotIn("reference", public)
                await scoped.complete([{"role": "user", "content": public["question"]}],
                                      stage="modeling", case_id=public["case_id"], run_id=run_id)
                return {"math_input": manifest["cases"][0]["math_input"],
                        "math_model": f"frozen modeled text for repeat {count}", "status": "complete"}

            with patch("rebuttal_runner.ours.prepare_ours_input", side_effect=upstream), patch(
                    "rebuttal_runner.runner.execute_code", new=AsyncMock(return_value=executed())):
                records = await run_paired(manifest, model, directory / "run", level="end_to_end",
                                           repeats=2, python_executable=sys.executable)
            self.assertEqual(count, 2)
            self.assertEqual(len(records), 4)
            self.assertEqual(sum(c["stage"] == "auxiliary" for c in model.calls), 2)
            self.assertEqual(records[0]["input_hash"], records[1]["input_hash"])
            self.assertNotEqual(records[0]["input_hash"], records[2]["input_hash"])
            self.assertEqual(records[0]["upstream_call_ids"], records[1]["upstream_call_ids"])
            self.assertNotIn("987654321", json.dumps(model.calls))

    async def test_external_end_to_end_ignores_ours_cache_and_excludes_only_explicit_data_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = load_manifest(EXAMPLE)
            excluded = copy.deepcopy(manifest["cases"][0])
            excluded.update(case_id="OptMATH:137", source_id="137", reference=None,
                            review_status="excluded_missing_critical_definition")
            manifest["cases"].append(excluded)
            baseline = AsyncMock(return_value={"status": "completed", "prediction": 2,
                                              "attempts": [], "execution": executed(), "call_ids": []})
            with patch("rebuttal_runner.baselines.baseline_availability", return_value={"available": True}), patch(
                    "rebuttal_runner.baselines.source_fingerprint", return_value={"sha256": "fixture"}), patch(
                    "rebuttal_runner.baselines.run_baseline", baseline), patch(
                    "rebuttal_runner.ours.prepare_ours_input", new=AsyncMock()) as upstream:
                records = await run_paired(manifest, FakeModel(directory), directory / "run",
                                           framework="coe", level="end_to_end", arms=("native",),
                                           source_root="unused", python_executable=sys.executable)
            self.assertEqual(len(records), 1)
            self.assertIsNone(baseline.call_args.args[1]["math_input"])
            self.assertNotIn("reference", baseline.call_args.args[1])
            upstream.assert_not_awaited()
            protocol = json.loads((directory / "run/protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["excluded_cases"][0]["case_id"], "OptMATH:137")

    async def test_stale_auxiliary_cache_fails_before_any_paid_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = load_manifest(EXAMPLE)
            case = manifest["cases"][0]
            case["auxiliary"] = {"input_hash": "wrong", "question_hash": digest(case["question"])}
            model = FakeModel(directory)
            with self.assertRaisesRegex(ValueError, "Cached Auxiliary input mismatch"):
                await run_paired(manifest, model, directory / "run", python_executable=sys.executable)
            self.assertEqual(model.calls, [])

    async def test_invalid_auxiliary_text_stays_out_of_coding_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = FakeModel(directory)
            original = model.complete

            async def complete(messages, **identity):
                response = await original(messages, **identity)
                if identity["stage"] == "auxiliary":
                    response["content"] = "INVALID_AUXILIARY_RAW_DO_NOT_FEED_TO_CODER"
                return response

            model.complete = complete
            with patch("rebuttal_runner.runner.execute_code", new=AsyncMock(return_value=executed())):
                records = await run_paired(load_manifest(EXAMPLE), model, directory / "run",
                                           python_executable=sys.executable)
            coding = [c for c in model.calls if c["stage"] == "code"]
            self.assertNotIn("INVALID_AUXILIARY_RAW_DO_NOT_FEED_TO_CODER", json.dumps(coding))
            self.assertEqual(records[1]["ned_trace"]["status"], "fallback")


class OfflinePlanningTests(unittest.TestCase):
    def test_plan_is_explicit_about_missing_inputs_and_does_not_construct_client(self):
        manifest = load_manifest(EXAMPLE)
        manifest["cases"][0]["math_input"] = None
        config = ModelConfig(model="test", base_url="https://unit.invalid/v1")
        with patch("rebuttal_runner.models.AsyncOpenAI", side_effect=AssertionError("online client")):
            plan = plan_run(manifest, config, framework="standard", level="coding")
        self.assertFalse(plan["offline_configuration_ready"])
        self.assertEqual(len(plan["missing_math_input"]), 1)
        self.assertFalse(plan["provider_checked_online"])

    def test_cache_alignment_preserves_reference_and_refuses_changed_questions(self):
        manifest = load_manifest(EXAMPLE)
        cached = copy.deepcopy(manifest)
        cached["cases"][0]["reference"] = 987654321
        manifest["cases"][0]["math_input"] = None
        attached = attach_cached_inputs(manifest, cached, cache_path="fixture.json")
        self.assertIsNotNone(attached["cases"][0]["math_input"])
        self.assertEqual(attached["cases"][0]["reference"], 2)
        manifest["cases"][0]["math_model"] = "old model"
        manifest["cases"][0]["auxiliary"] = {"stale": True}
        attached = attach_cached_inputs(manifest, cached, cache_path="fixture.json")
        self.assertNotIn("math_model", attached["cases"][0])
        self.assertNotIn("auxiliary", attached["cases"][0])
        cached["cases"][0]["question"] += " changed"
        rejected = attach_cached_inputs(manifest, cached, cache_path="fixture.json")
        self.assertIsNone(rejected["cases"][0]["math_input"])
        self.assertEqual(rejected["cases"][0]["cached_input_alignment"]["status"], "question_version_mismatch")

    def test_different_levels_never_form_a_pair_and_all_attempt_times_are_counted(self):
        base = {"dataset": "fixture", "dataset_version": "v1", "model": "test", "framework": "standard",
                "case_id": "c", "source_id": "c", "repeat": 0, "input_hash": "same", "status": "complete",
                "prediction": 2, "score": {"correct": True}, "call_ids": [],
                "attempts": [{"execution": {"elapsed_seconds": 2}}, {"execution": {"elapsed_seconds": 3}}]}
        summary = summarize_records([{**base, "arm": "native", "level": "coding"},
                                     {**base, "arm": "ned", "level": "end_to_end"}], [])
        self.assertTrue(all(c["valid_pair_run_count"] == 0 for c in summary["comparisons"]))
        self.assertEqual({g["execution"]["known_total_seconds"] for g in summary["groups"]}, {5})


if __name__ == "__main__":
    unittest.main()
