import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from rebuttal_runner.models import BudgetedModel, CallBudgetExceeded, ModelCallError, ModelConfig


class BudgetTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_request_consumes_budget_and_next_call_never_reaches_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            model = AsyncMock()
            model.config = ModelConfig(model="test", base_url="https://unit.invalid/v1")
            model.ledger_path = Path(directory) / "calls.jsonl"
            model.complete.side_effect = ModelCallError("provider unavailable", {"call_id": "failed-1"})
            bounded = BudgetedModel(model, 1)
            with self.assertRaises(ModelCallError):
                await bounded.complete([], stage="code", case_id="c", run_id="r")
            with self.assertRaises(CallBudgetExceeded):
                await bounded.complete([], stage="repair", case_id="c", run_id="r")
            self.assertEqual(model.complete.await_count, 1)
            self.assertEqual(bounded.call_ids, ["failed-1"])

    async def test_arm_budgets_are_independent_and_preserve_provider_configuration(self):
        model = AsyncMock()
        model.config = ModelConfig(model="test", base_url="https://unit.invalid/v1")
        model.ledger_path = Path("unused.jsonl")
        model.complete.side_effect = [{"call_id": "native-1"}, {"call_id": "ned-1"}]
        native, ned = BudgetedModel(model, 1), BudgetedModel(model, 1)
        await native.complete([], stage="code", case_id="c", run_id="r:native")
        await ned.complete([], stage="code", case_id="c", run_id="r:ned")
        self.assertIs(native.config, ned.config)
        self.assertEqual(native.call_ids, ["native-1"])
        self.assertEqual(ned.call_ids, ["ned-1"])


if __name__ == "__main__":
    unittest.main()
