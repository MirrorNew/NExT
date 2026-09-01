import pathlib
from types import SimpleNamespace
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import RebuildNORA_easyread as runner
from agents.base_agents import BaseAgent


class RunnerConfigurationTests(unittest.TestCase):
    def test_agent_default_input_uses_requested_model(self):
        opts = SimpleNamespace(
            nora_file_input_path=None,
            agent=True,
            dataset_name="NExTLP",
            model="gpt-5.1",
        )
        self.assertEqual(
            runner.resolve_runner_input_path(opts, "unused"),
            str(pathlib.Path("NORA_process_data") / "NExTLP_NORA_gpt-5.1.json"),
        )

    def test_explicit_input_path_is_preserved(self):
        opts = SimpleNamespace(
            nora_file_input_path="fixed/input.json",
            agent=True,
            dataset_name="NExTLP",
            model="gpt-5.1",
        )
        self.assertEqual(
            runner.resolve_runner_input_path(opts, "unused"),
            "fixed/input.json",
        )


class RunnerRoundsTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_mode_initializes_origin_path_without_agent_flag(self):
        opts = SimpleNamespace(
            nora_file_input_path=None,
            agent=False,
            dataset_name="NExTLP",
            model="gpt-5.1",
            output_dir=None,
        )
        fake_open = mock.mock_open(read_data="{}")
        with mock.patch("builtins.open", fake_open):
            await runner.main_process_data(opts)
        self.assertEqual(
            opts.data_path,
            str(pathlib.Path("data/20251021_origin_datasets") / "NExTLP.json"),
        )

    async def test_rounds_accept_six_item_case_result(self):
        opts = SimpleNamespace(
            nora_file_input_path="fixed/input.json",
            agent=True,
            dataset_name="NExTLP",
            model="gpt-5.1",
            output_dir=None,
        )
        fake_open = mock.mock_open(read_data='{"0": {"answer": 1}}')
        fake_case = mock.AsyncMock(
            return_value=(True, True, "0", 0.1, 10, {1: 0, 2: 0, 3: 0})
        )
        with (
            mock.patch("builtins.open", fake_open),
            mock.patch.object(runner.os, "makedirs"),
            mock.patch.object(runner, "process_single_case", fake_case),
        ):
            await runner.main_round_10(opts=opts, num_rounds=1)
        fake_case.assert_awaited_once()


class AgentModelIdentityTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def fake_client(returned_model):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            model=returned_model,
            usage=SimpleNamespace(total_tokens=1),
        )
        return SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=mock.AsyncMock(return_value=response),
                )
            )
        )

    async def test_every_agent_call_accepts_exact_requested_model(self):
        agent = BaseAgent(self.fake_client("gpt-5.1"), model_name="gpt-5.1")
        self.assertEqual(await agent._query(), "ok")
        self.assertEqual(agent.returned_models, ["gpt-5.1"])

    async def test_every_agent_call_rejects_model_alias_or_upgrade(self):
        agent = BaseAgent(self.fake_client("newer-model"), model_name="gpt-5.1")
        with self.assertRaisesRegex(RuntimeError, "MODEL_ID_MISMATCH"):
            await agent._query()
        self.assertEqual(agent.returned_models, ["newer-model"])


if __name__ == "__main__":
    unittest.main()
