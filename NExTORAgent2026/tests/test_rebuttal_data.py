import copy
import json
import tempfile
import unittest
from pathlib import Path

from rebuttal_runner.data import (
    load_manifest,
    prepare_manifest,
    public_case,
    validate_math_input,
)


def nora_case(index=0):
    return {
        "index": index,
        "question": "Minimize x subject to x >= 1.",
        "answer": 1,
        "Parameters_List": [],
        "Variables_List": [{"Symbol": "X", "Type": "continuous", "Meaning": "decision"}],
        "Constraint_Table": [["lower", "x >= 1", "source note"]],
        "Objective": {"Mathematical_expressions": "min x", "answer": 999},
        "Problem_Type": "LP",
        "results": {"leak": 456},
        "gold": 789,
        "model_name": "model-example",
        "unit": "USD",
    }


REAL_OPTIBENCH_PATH = Path(r"D:\LLMProject\NExTORAgent2025\data\origin_data\OptiBench.json")


class RebuttalDataTests(unittest.TestCase):
    def test_manifest_rejects_duplicate_case_ids(self):
        manifest = {
            "schema_version": 1,
            "dataset": "sample",
            "dataset_version": "v1",
            "cases": [
                {
                    "case_id": "same",
                    "source_id": "0",
                    "question": "q",
                    "math_input": None,
                    "reference": 1,
                    "answer_spec": {"kind": "scalar"},
                    "scoring": {},
                    "review_status": "needs_review",
                },
                {
                    "case_id": "same",
                    "source_id": "1",
                    "question": "q2",
                    "math_input": None,
                    "reference": 2,
                    "answer_spec": {"kind": "scalar"},
                    "scoring": {},
                    "review_status": "needs_review",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                load_manifest(path)

    def test_source_and_public_case_are_not_mutated(self):
        source = {"0": nora_case()}
        before = copy.deepcopy(source)
        manifest = prepare_manifest(source, "NEXTOR", source_format="nora")
        self.assertEqual(source, before)
        case = manifest["cases"][0]
        self.assertEqual(case["question"], before["0"]["question"])
        public = public_case(case)
        public["math_input"]["Variables_List"][0]["symbol"] = "changed"
        self.assertEqual(case["math_input"]["Variables_List"][0]["symbol"], "X")
        self.assertEqual(source, before)

    def test_public_case_does_not_leak_reference_or_results(self):
        case = prepare_manifest({"0": nora_case()}, "NEXTOR", source_format="nora")["cases"][0]
        public = public_case(case)
        serialized = json.dumps(public).lower()
        self.assertNotIn('"reference"', serialized)
        self.assertNotIn('"results"', serialized)
        self.assertNotIn('"gold"', serialized)
        self.assertNotIn("999", serialized)
        self.assertNotIn("456", serialized)
        self.assertNotIn("789", serialized)
        self.assertEqual(set(public), {"question", "math_input", "answer_spec"})

    def test_nonlinear_filter_keeps_expected_133_of_133_plus_50_records(self):
        records = {}
        for index in range(183):
            record = {
                "index": index,
                "question": f"q{index}",
                "results": {"profit": str(index)},
                "type": "NLP" if index < 133 else "LP",
                "ori": f"source-{index}",
            }
            records[str(index)] = record

        all_cases = prepare_manifest(records, "OptiBench", "optibench")
        nonlinear = prepare_manifest(
            records, "OptiBench", "optibench", nonlinear_only=True
        )
        self.assertEqual(len(all_cases["cases"]), 183)
        self.assertEqual(len(nonlinear["cases"]), 133)
        self.assertTrue(all(case["source"]["label"].startswith("source-") for case in nonlinear["cases"]))
        self.assertTrue(all(case["reference_status"] == "candidate" for case in nonlinear["cases"]))
        self.assertTrue(all(isinstance(case["reference"], dict) for case in nonlinear["cases"]))
        self.assertEqual(nonlinear["cases"][0]["reference"], {"profit": 0})
        self.assertEqual(nonlinear["cases"][0]["reference_source"], {"profit": "0"})

    def test_nora_keeps_problem_types_without_prediction_filtering(self):
        source = {
            "0": {"index": 0, "question": "linear", "answer": 0, "Problem_Type": "LP"},
            "1": {"index": 1, "question": "nonlinear", "answer": 1, "Problem_Type": "NLP"},
        }
        manifest = prepare_manifest(source, "OptMATH", "nora")
        self.assertEqual(len(manifest["cases"]), 2)
        self.assertEqual(
            [case["math_input"] for case in manifest["cases"]],
            [None, None],
        )
        with self.assertRaisesRegex(ValueError, "only for optibench"):
            prepare_manifest(source, "OptiMATH", "nora", nonlinear_only=True)

    def test_nora_bad_math_is_retained_for_review(self):
        record = nora_case()
        record.pop("Objective")
        manifest = prepare_manifest({"0": record}, "NEXTOR", "nora")
        self.assertEqual(len(manifest["cases"]), 1)
        case = manifest["cases"][0]
        self.assertIsNone(case["math_input"])
        self.assertEqual(case["review_status"], "needs_review")
        self.assertTrue(any("math_input_unavailable" in reason for reason in case["review_reasons"]))

    def test_unsupported_reference_is_preserved_and_flagged(self):
        record = {"index": "unsupported", "question": "q", "answer": [1, "unknown"]}
        manifest = prepare_manifest({"one": record}, "OptiBench", "optibench")
        case = manifest["cases"][0]
        self.assertEqual(case["reference"], [1, "unknown"])
        self.assertEqual(case["reference_kind"], "unsupported_needs_review")
        self.assertEqual(case["answer_spec"]["kind"], "needs_review")
        self.assertEqual(len(manifest["cases"]), 1)

    def test_ready_manifest_accepts_finite_numeric_list_reference(self):
        manifest = {
            "schema_version": 1,
            "dataset": "sample",
            "dataset_version": "v1",
            "cases": [
                {
                    "case_id": "sample:1",
                    "source_id": "1",
                    "question": "q",
                    "math_input": None,
                    "reference": [1, 2.5],
                    "answer_spec": {"kind": "numeric_list", "fields": ["answer"], "order": ["answer"]},
                    "scoring": {},
                    "review_status": "ready",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(load_manifest(path)["cases"][0]["reference"], [1, 2.5])

    def test_math_input_uses_adapter_schema_and_excludes_answer_fields(self):
        normalized = validate_math_input(nora_case())
        self.assertEqual(
            set(normalized),
            {"Parameters_List", "Variables_List", "Constraint_Table", "Objective", "Problem_Type"},
        )
        self.assertEqual(normalized["Variables_List"][0]["symbol"], "X")
        self.assertEqual(normalized["Objective"], {"Mathematical_expressions": "min x"})
        self.assertEqual(normalized["Constraint_Table"][0]["Mathematical_expressions"], "x >= 1")
        serialized = json.dumps(normalized).lower()
        self.assertNotIn("answer", serialized)
        self.assertNotIn("results", serialized)
        self.assertNotIn("gold", serialized)

    def test_math_normalization_preserves_parameter_and_objective_semantics(self):
        entry = nora_case()
        entry["Parameters_List"] = [{"Name": "Q", "Value": 10, "Meaning": "demand"}]
        entry["Objective"].update({"Objective_sentence": "Minimize cost", "Sense": "min"})
        normalized = validate_math_input(entry)
        self.assertEqual(normalized["Parameters_List"][0]["Meaning"], "demand")
        self.assertEqual(normalized["Objective"]["Objective_sentence"], "Minimize cost")
        self.assertEqual(normalized["Objective"]["Sense"], "min")

    def test_scoring_protocol_is_explicit_and_validated(self):
        source = {"0": {"index": 0, "question": "q", "answer": 1}}
        prepared = prepare_manifest(source, "NEXTOR", "nora", scoring_name="nextor_relative")
        self.assertEqual(prepared["cases"][0]["scoring"], {"name": "nextor_relative"})
        self.assertNotIn("scoring_protocol_missing", prepared["cases"][0]["review_reasons"])
        missing = prepare_manifest(source, "NEXTOR", "nora")
        self.assertIn("scoring_protocol_missing", missing["cases"][0]["review_reasons"])
        with self.assertRaisesRegex(ValueError, "unsupported scoring_name"):
            prepare_manifest(source, "NEXTOR", "nora", scoring_name="guess")

    @unittest.skipUnless(REAL_OPTIBENCH_PATH.exists(), "historical source is not available")
    def test_real_optibench_nonlinear_subset_and_named_results(self):
        manifest = prepare_manifest(
            REAL_OPTIBENCH_PATH,
            "OptiBench",
            "optibench",
            nonlinear_only=True,
            scoring_name="optibench_absolute",
        )
        self.assertEqual(len(manifest["cases"]), 183)
        multi_output = next(case for case in manifest["cases"] if len(case["reference"]) > 1)
        self.assertEqual(len(multi_output["reference"]), len(multi_output["reference_source"]))
        self.assertTrue(all(isinstance(value, (int, float)) for value in multi_output["reference"].values()))
        self.assertEqual(multi_output["reference_status"], "candidate")
        self.assertEqual(multi_output["scoring"], {"name": "optibench_absolute"})

    def test_public_case_sanitizes_math_and_preserves_question_verbatim(self):
        case = {
            "question": "Compute x.",
            "math_input": {
                **nora_case(),
                "reference": 314159,
                "gold": 271828,
            },
            "answer_spec": {"kind": "scalar", "fields": ["answer"], "order": ["answer"]},
            "reference": 314159,
        }
        public = public_case(case)
        serialized = json.dumps(public).lower()
        self.assertNotIn("314159", serialized)
        self.assertNotIn("271828", serialized)
        self.assertNotIn('"reference"', serialized)
        self.assertNotIn('"gold"', serialized)
        self.assertEqual("Compute x.", public["question"])

    def test_question_with_answer_material_is_kept_and_flagged_for_review(self):
        record = {"index": 0, "question": "Find x.\nAnswer: x=4", "answer": 4}
        case = prepare_manifest({"0": record}, "NEXTOR", "nora")["cases"][0]
        self.assertEqual(case["question"], record["question"])
        self.assertEqual(case["review_status"], "needs_review")
        self.assertIn("question_may_include_answer_material", case["review_reasons"])


if __name__ == "__main__":
    unittest.main()
