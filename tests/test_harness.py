import pathlib
import os
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RebuildNORA_utils import (
    async_extract_and_execute_python_code,
    build_solver_subprocess_env,
    eval_model_result,
    extract_best_objective,
    extract_final_answer,
    extract_solver_objectives,
)
from validate_nextor_harness import validate_processed_entry


class HarnessScoringTests(unittest.TestCase):
    def test_explicit_final_answer_precedes_solver_log(self):
        output = "Optimal objective 999\nBest objective 999\nFinalAnswer=【12.5】"
        self.assertEqual(extract_best_objective(output), ["12.5"])

    def test_no_final_answer_marker_never_falls_back(self):
        output = "Optimal objective 12.5\nBest objective 12.5"
        self.assertEqual(extract_final_answer(output), ["None"])
        self.assertEqual(extract_best_objective(output), ["None"])
        self.assertEqual(extract_solver_objectives(output), ["12.5"])

    def test_nonnumeric_final_answer_does_not_fall_back_to_objective(self):
        output = "Optimal objective 486.49\nFinalAnswer=【Q1=200, Q2=150】"
        self.assertEqual(extract_best_objective(output), ["None"])

    def test_multiple_distinct_final_answers_are_ambiguous(self):
        output = "FinalAnswer=【10】\nFinalAnswer=【20】"
        self.assertEqual(extract_best_objective(output), ["None"])

    def test_one_invalid_marker_rejects_other_valid_marker(self):
        output = "FinalAnswer=【10】\nFinalAnswer=【Q1=10】"
        self.assertEqual(extract_final_answer(output), ["None"])

    def test_malformed_marker_is_rejected(self):
        self.assertEqual(extract_final_answer("FinalAnswer = 10"), ["None"])

    def test_final_answer_substring_is_not_a_marker(self):
        self.assertEqual(extract_final_answer("NotFinalAnswer=【10】"), ["None"])

    def test_numerically_close_but_distinct_answers_are_ambiguous(self):
        output = "FinalAnswer=【1】\nFinalAnswer=【1.0000000001】"
        self.assertEqual(extract_final_answer(output), ["None"])

    def test_repeated_identical_final_answer_is_allowed(self):
        output = "FinalAnswer=【10】\nFinalAnswer=【10.0】"
        self.assertEqual(extract_best_objective(output), ["10.0"])

    def test_magnitude_mismatch_is_not_correct(self):
        self.assertEqual(eval_model_result(True, [1.23456], 123456000), (True, False))

    def test_same_scale_rounding_is_allowed(self):
        self.assertEqual(eval_model_result(True, [123.4561], 123.456), (True, True))

    def test_tolerance_does_not_scale_with_large_objective(self):
        self.assertEqual(
            eval_model_result(True, [241137.150597], 241138.90237),
            (True, False),
        )

    def test_truncated_ground_truth_uses_reported_last_digit(self):
        self.assertEqual(
            eval_model_result(True, [198.418027716], 198.418027),
            (True, True),
        )

    def test_failed_execution_never_scores_correct(self):
        self.assertEqual(eval_model_result(False, [10], 10), (False, False))

    def test_scalar_cannot_match_one_element_of_vector_ground_truth(self):
        self.assertEqual(eval_model_result(True, [9.0], [1.96564, 9.0]), (True, False))

    def test_vector_ground_truth_requires_positional_match(self):
        self.assertEqual(
            eval_model_result(True, [1.96564, 9.0], [1.96564, 9.0]),
            (True, True),
        )
        self.assertEqual(
            eval_model_result(True, [9.0, 1.96564], [1.96564, 9.0]),
            (True, False),
        )

    def test_solver_subprocess_environment_is_allowlisted(self):
        source = {
            "Path": "solver-path",
            "SystemRoot": "C:\\Windows",
            "TEMP": "C:\\Temp",
            "GRB_LICENSE_FILE": "C:\\gurobi.lic",
            "OPENAI_API_KEY": "must-not-propagate",
            "HTTPS_PROXY": "must-not-propagate",
            "AWS_SECRET_ACCESS_KEY": "must-not-propagate",
            "AZURE_CLIENT_SECRET": "must-not-propagate",
            "GOOGLE_APPLICATION_CREDENTIALS": "must-not-propagate",
        }
        child = build_solver_subprocess_env(source)
        self.assertEqual(child["PATH"], "solver-path")
        self.assertEqual(child["GRB_LICENSE_FILE"], "C:\\gurobi.lic")
        self.assertEqual(child["PYTHONUTF8"], "1")
        for forbidden in (
            "OPENAI_API_KEY",
            "HTTPS_PROXY",
            "AWS_SECRET_ACCESS_KEY",
            "AZURE_CLIENT_SECRET",
            "GOOGLE_APPLICATION_CREDENTIALS",
        ):
            self.assertNotIn(forbidden, child)


class HarnessExecutionIsolationTests(unittest.IsolatedAsyncioTestCase):
    async def test_generated_code_cannot_read_parent_api_key(self):
        code = """```python
import os
if os.getenv("OPENAI_API_KEY"):
    print("FinalAnswer=【999】")
else:
    print("FinalAnswer=【1】")
```"""
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "parent-only"}):
            success, result = await async_extract_and_execute_python_code(
                code,
                {"index": "isolation"},
                timeout=20,
            )
        self.assertTrue(success)
        self.assertEqual(result, ["1.0"])


class DatasetSchemaTests(unittest.TestCase):
    @staticmethod
    def valid_entry():
        return {
            "Parameters_List": [{"Name": "a", "Type": "float", "Value": 1.0}],
            "Variables_List": [
                {
                    "symbol": "x",
                    "Meaning": "decision",
                    "Type": "continuous",
                    "Range ": "[0, 1]",
                }
            ],
            "Constraint_Table": [["capacity", "x <= 1", "sentence 1"]],
            "Objective": {
                "Objective_sentence": "minimize cost",
                "Mathematical_expressions": "min x",
            },
            "Problem_Type": "NLP",
        }

    def test_agent_consumed_nested_schema_is_accepted(self):
        self.assertEqual(validate_processed_entry("0", self.valid_entry()), [])

    def test_malformed_constraint_row_is_rejected(self):
        entry = self.valid_entry()
        entry["Constraint_Table"] = [["missing provenance", "x <= 1"]]
        self.assertTrue(validate_processed_entry("0", entry))


if __name__ == "__main__":
    unittest.main()
