import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebuttal_runner.execution import execute_code, extract_code, parse_solver_answer


PYTHON = sys.executable


class ExtractCodeTests(unittest.TestCase):
    def test_extracts_one_python_fenced_block(self):
        text = "Answer follows:\n```python\nx = 2\n```\nDone."
        self.assertEqual(extract_code(text), "x = 2")

    def test_accepts_parseable_bare_python(self):
        code = "x = 2\nprint(x)\n"
        self.assertEqual(extract_code(code), code)

    def test_rejects_multiple_python_blocks(self):
        text = "```python\nx = 1\n```\n```python\nx = 2\n```"
        with self.assertRaisesRegex(ValueError, "multiple python code blocks"):
            extract_code(text)

    def test_rejects_non_code_response(self):
        with self.assertRaisesRegex(ValueError, "parseable Python code"):
            extract_code("Here is the code you requested.")


class SolverAnswerContractTests(unittest.TestCase):
    def test_distinguishes_unverified_solver_no_solution_and_missing_marker(self):
        base = {"status": "executed", "returncode": 0, "stdout": "",
                "solver_limits_applied": True,
                "solver_status": {"status": 2, "sol_count": 1}}
        self.assertEqual(parse_solver_answer({**base, "solver_limits_applied": False})[1]["code"],
                         "solver_not_verified")
        self.assertEqual(parse_solver_answer({**base, "solver_status": {"status": 3, "sol_count": 0}})[1]["code"],
                         "no_solver_solution")
        self.assertEqual(parse_solver_answer(base)[1]["code"], "missing_final_answer")

    def test_accepts_only_finalanswer_after_verified_solution(self):
        execution = {"status": "executed", "returncode": 0,
                     "stdout": "FinalAnswer=【5】\n", "solver_limits_applied": True,
                     "solver_status": {"status": 2, "sol_count": 1}}
        self.assertEqual(parse_solver_answer(execution), (5, None))
        legacy = {**execution, "stdout": "Optimal Objective Value: 5.0\n"}
        self.assertEqual(parse_solver_answer(legacy)[1]["code"], "missing_final_answer")


class ExecuteCodeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tempdir.name)

    def tearDown(self):
        self.tempdir.cleanup()

    def case_dir(self, name):
        path = self.root / name
        path.mkdir(parents=True)
        return path

    async def test_syntax_error_is_saved_and_not_executed(self):
        folder = self.case_dir("syntax")
        result = await execute_code(
            "if True print('bad')",
            folder,
            PYTHON,
            timeout_seconds=2,
            solver_time_limit=1,
        )

        self.assertEqual(result["status"], "syntax_error")
        self.assertIsNone(result["returncode"])
        self.assertFalse(result["solver_limits_applied"])
        self.assertEqual((folder / "generated.py").read_text(encoding="utf-8"), "if True print('bad')")
        self.assertTrue((folder / "executed.py").exists())
        self.assertTrue((folder / "stdout.txt").exists())
        self.assertTrue((folder / "stderr.txt").exists())

    async def test_fake_model_call_is_rewritten_and_limits_are_injected(self):
        folder = self.case_dir("rewrite")
        source = '''from __future__ import annotations

class Params:
    pass

class FakeModel:
    def __init__(self):
        self.Params = Params()
        self.Status = 2
        self.SolCount = 1
        self.Runtime = 0.125
    def optimize(self):
        assert self.Params.TimeLimit == 1.25
        assert self.Params.Threads == 3

model = FakeModel()
model.optimize()
'''
        result = await execute_code(
            source,
            folder,
            PYTHON,
            timeout_seconds=3,
            solver_time_limit=1.25,
            solver_threads=3,
        )

        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["solver_status"]["status"], 2)
        self.assertEqual(result["solver_status"]["sol_count"], 1)
        self.assertEqual(result["solver_status"]["optimize_call_count"], 1)
        self.assertEqual(result["solver_status"]["optimize_calls"][0]["runtime_seconds"], 0.125)
        self.assertEqual(result["solver_status"]["solver_runtime_seconds"], 0.125)
        self.assertTrue(result["solver_limits_applied"])
        self.assertEqual((folder / "generated.py").read_text(encoding="utf-8"), source)
        executed = (folder / "executed.py").read_text(encoding="utf-8")
        self.assertIn("model.Params.TimeLimit = 1.25", executed)
        self.assertIn("model.Params.Threads = 3", executed)
        self.assertIn("__rebuttal_optimize__(model)", executed)
        self.assertEqual(result["stdout"].count("SolverStatus=JSON"), 1)

    async def test_multiple_optimize_calls_keep_each_runtime_and_total(self):
        folder = self.case_dir("multiple_optimize")
        source = '''class Params:
    pass
class FakeModel:
    def __init__(self):
        self.Params = Params()
        self.Status = 2
        self.SolCount = 1
        self.Runtime = 0.0
        self.calls = 0
    def optimize(self):
        self.calls += 1
        self.Runtime = self.calls / 10
model = FakeModel()
model.optimize()
model.optimize()
'''
        result = await execute_code(source, folder, PYTHON, 3, 1)
        solver = result["solver_status"]
        self.assertEqual(result["status"], "executed")
        self.assertEqual(solver["optimize_call_count"], 2)
        self.assertEqual([call["runtime_seconds"] for call in solver["optimize_calls"]], [0.1, 0.2])
        self.assertAlmostEqual(solver["solver_runtime_seconds"], 0.3)

    async def test_missing_runtime_is_null_and_does_not_become_zero(self):
        folder = self.case_dir("missing_runtime")
        source = '''class Params:
    pass
class FakeModel:
    def __init__(self):
        self.Params = Params()
        self.Status = 2
        self.SolCount = 1
    def optimize(self):
        pass
model = FakeModel()
model.optimize()
'''
        result = await execute_code(source, folder, PYTHON, 3, 1)
        solver = result["solver_status"]
        self.assertEqual(result["status"], "executed")
        self.assertIsNone(solver["optimize_calls"][0]["runtime_seconds"])
        self.assertEqual(solver["known_solver_runtime_seconds"], 0)
        self.assertIsNone(solver["solver_runtime_seconds"])
        self.assertEqual(solver["missing_runtime_count"], 1)

    async def test_optimize_exception_keeps_failure_and_records_runtime(self):
        folder = self.case_dir("optimize_failure")
        source = '''class Params:
    pass
class FakeModel:
    def __init__(self):
        self.Params = Params()
        self.Status = 3
        self.SolCount = 0
        self.Runtime = 0.25
    def optimize(self):
        raise RuntimeError("solver failure")
model = FakeModel()
model.optimize()
'''
        result = await execute_code(source, folder, PYTHON, 3, 1)
        self.assertEqual(result["status"], "invalid_program")
        self.assertEqual(result["solver_status"]["optimize_call_count"], 1)
        self.assertIsNone(result["solver_status"]["optimize_calls"][0]["runtime_seconds"])
        self.assertEqual(result["solver_status"]["missing_runtime_count"], 1)
        self.assertEqual(result["solver_status"]["sol_count"], 0)

    async def test_helper_name_conflict_is_rejected(self):
        folder = self.case_dir("conflict")
        source = '''def __rebuttal_optimize__(model):
    return model

model.optimize()
'''
        result = await execute_code(
            source,
            folder,
            PYTHON,
            timeout_seconds=2,
            solver_time_limit=1,
        )

        self.assertEqual(result["status"], "invalid_program")
        self.assertFalse(result["solver_limits_applied"])
        self.assertIn("conflicts with injected helper", result["stderr"])

    async def test_injected_math_alias_conflict_is_rejected(self):
        folder = self.case_dir("math_alias_conflict")
        result = await execute_code(
            "__rebuttal_math = object()\nmodel.optimize()\n",
            folder,
            PYTHON,
            timeout_seconds=2,
            solver_time_limit=1,
        )
        self.assertEqual(result["status"], "invalid_program")
        self.assertFalse(result["solver_limits_applied"])
        self.assertIn("__rebuttal_math", result["stderr"])

    async def test_unbound_optimize_reference_is_rejected(self):
        folder = self.case_dir("alias")
        result = await execute_code(
            "optimize = model.optimize\noptimize()\n",
            folder,
            PYTHON,
            timeout_seconds=2,
            solver_time_limit=1,
        )
        self.assertEqual(result["status"], "invalid_program")
        self.assertFalse(result["solver_limits_applied"])
        self.assertIn("cannot be reliably limited", result["stderr"])

    async def test_code_without_solver_call_is_allowed_and_marked_unlimited(self):
        folder = self.case_dir("no_solver")
        result = await execute_code(
            "print('plain python')\n",
            folder,
            PYTHON,
            timeout_seconds=2,
            solver_time_limit=1,
        )
        self.assertEqual(result["status"], "executed")
        self.assertFalse(result["solver_limits_applied"])
        self.assertIsNone(result["solver_status"])
        self.assertEqual(result["stdout"].splitlines(), ["plain python"])
        self.assertEqual(
            (folder / "executed.py").read_text(encoding="utf-8"),
            "print('plain python')\n",
        )

    async def test_timeout_kills_subprocess_and_saves_output_files(self):
        folder = self.case_dir("timeout")
        result = await execute_code(
            "import time\ntime.sleep(5)\n",
            folder,
            PYTHON,
            timeout_seconds=0.1,
            solver_time_limit=1,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertIsNotNone(result["returncode"])
        self.assertTrue((folder / "stdout.txt").exists())
        self.assertTrue((folder / "stderr.txt").exists())

    async def test_child_process_does_not_receive_secret_environment(self):
        folder = self.case_dir("environment")
        source = '''import os
names = ["OPENAI_API_KEY", "NED_REBUTTAL_API_KEY", "SERVICE_API_TOKEN", "AWS_SECRET_ACCESS_KEY"]
print([bool(os.getenv(name)) for name in names])
print(bool(os.getenv("PATH")))
'''
        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "mock-openai-key",
                "NED_REBUTTAL_API_KEY": "mock-rebuttal-key",
                "SERVICE_API_TOKEN": "mock-api-token",
                "AWS_SECRET_ACCESS_KEY": "mock-secret",
            },
        ):
            result = await execute_code(
                source,
                folder,
                PYTHON,
                timeout_seconds=2,
                solver_time_limit=1,
            )

        self.assertEqual(result["status"], "executed")
        self.assertIn("[False, False, False, False]", result["stdout"])
        self.assertIn("True", result["stdout"])
        for secret in ("mock-openai-key", "mock-rebuttal-key", "mock-api-token", "mock-secret"):
            self.assertNotIn(secret, result["stdout"] + result["stderr"])

    async def test_child_stdout_preserves_utf8_chinese_answer(self):
        folder = self.case_dir("utf8")
        result = await execute_code(
            'print("FinalAnswer=【42】")\nprint("中文输出")\n',
            folder,
            PYTHON,
            timeout_seconds=2,
            solver_time_limit=1,
        )

        self.assertEqual(result["status"], "executed")
        self.assertEqual(result["stdout"].splitlines(), ["FinalAnswer=【42】", "中文输出"])
        self.assertNotIn("�", result["stdout"])


if __name__ == "__main__":
    unittest.main()
