import json
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rebuttal_runner.models import LoggedModel, ModelCallError, ModelConfig, ModelConfigError


def fake_response(*, model="test-model", content="ok", usage=None, finish_reason="stop"):
    return SimpleNamespace(
        id="provider-call-id",
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
    )


def fake_client(create):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create),
        ),
    )


class ModelConfigTests(unittest.TestCase):
    def test_model_is_required_and_never_inferred(self):
        with self.assertRaisesRegex(ModelConfigError, "model is required"):
            ModelConfig.from_dict({})

    def test_deepseek_requires_explicit_or_environment_base_url(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ModelConfigError, "base_url or OPENAI_API_BASE"):
                ModelConfig.from_dict({"model": "deepseek-chat"})

    def test_every_model_requires_an_explicit_endpoint(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ModelConfigError, "model endpoint is required"):
                ModelConfig.from_dict({"model": "gpt-5.1"})

    def test_streaming_is_rejected_for_non_streaming_complete(self):
        with self.assertRaisesRegex(ModelConfigError, "stream=True is unsupported"):
            ModelConfig.from_dict(
                {
                    "model": "test-model",
                    "base_url": "https://example.test/v1",
                    "request_parameters": {"stream": True},
                }
            )

    def test_base_url_can_come_from_openai_api_base(self):
        with mock.patch.dict("os.environ", {"OPENAI_API_BASE": "https://example.test/v1"}):
            config = ModelConfig.from_dict({"model": "deepseek-chat"})
        self.assertEqual(config.base_url, "https://example.test/v1")
        self.assertEqual(config.allowed_returned_models, ("deepseek-chat",))


class LoggedModelTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.ledger = pathlib.Path(self.tempdir.name) / "calls.jsonl"
        self.config = ModelConfig.from_dict(
            {
                "model": "test-model",
                "base_url": "https://example.test/v1",
                "api_key_env": "TEST_MODEL_API_KEY",
                "request_parameters": {"max_completion_tokens": 256, "temperature": 0},
            }
        )

    def tearDown(self):
        self.tempdir.cleanup()

    def read_records(self):
        return [json.loads(line) for line in self.ledger.read_text(encoding="utf-8").splitlines()]

    async def test_request_parameters_and_provider_usage_are_preserved(self):
        usage = {
            "prompt_tokens": 7,
            "completion_tokens": 4,
            "total_tokens": 11,
            "completion_tokens_details": {"reasoning_tokens": 3},
        }
        create = mock.AsyncMock(return_value=fake_response(usage=usage))
        client = fake_client(create)
        wrapper = LoggedModel(self.config, self.ledger, client=client)

        with mock.patch.dict("os.environ", {"TEST_MODEL_API_KEY": "secret-value"}):
            result = await wrapper.complete(
                [{"role": "user", "content": "question"}],
                stage="coding",
                case_id="case-1",
                run_id="run-1",
            )

        create.assert_awaited_once_with(
            model="test-model",
            messages=[{"role": "user", "content": "question"}],
            max_completion_tokens=256,
            temperature=0,
        )
        self.assertEqual(result["content"], "ok")
        self.assertEqual(result["usage"], usage)
        record = self.read_records()[0]
        self.assertEqual(record["usage"], usage)
        self.assertEqual(record["request"]["parameters"]["max_completion_tokens"], 256)
        self.assertEqual(record["finish_reason"], "stop")
        self.assertEqual(record["call_id"], result["call_id"])
        self.assertNotIn("content", record["request"])
        self.assertNotIn("secret-value", self.ledger.read_text(encoding="utf-8"))

    async def test_model_mismatch_is_logged_with_usage_before_error(self):
        usage = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        create = mock.AsyncMock(return_value=fake_response(model="provider-alias", usage=usage))
        wrapper = LoggedModel(self.config, self.ledger, client=fake_client(create))

        with mock.patch.dict("os.environ", {"TEST_MODEL_API_KEY": "secret-value"}):
            with self.assertRaises(ModelCallError) as caught:
                await wrapper.complete([], stage="modeling", case_id="c2", run_id="r2")

        record = self.read_records()[0]
        self.assertEqual(caught.exception.record, record)
        self.assertEqual(record["returned_model"], "provider-alias")
        self.assertEqual(record["usage"], usage)
        self.assertIn("MODEL_ID_MISMATCH", record["error"])

    async def test_provider_error_is_logged_and_secret_is_redacted(self):
        secret = "private-api-secret"
        create = mock.AsyncMock(side_effect=RuntimeError(f"request failed with key {secret}"))
        config = ModelConfig.from_dict(
            {
                "model": "test-model",
                "base_url": "https://example.test/v1",
                "api_key_env": "TEST_MODEL_API_KEY",
                "request_parameters": {"api_key": secret, "temperature": 0},
            }
        )
        wrapper = LoggedModel(config, self.ledger, client=fake_client(create))

        with mock.patch.dict("os.environ", {"TEST_MODEL_API_KEY": secret}):
            with self.assertRaises(ModelCallError) as caught:
                await wrapper.complete([], stage="repair", case_id="c3", run_id="r3")

        contents = self.ledger.read_text(encoding="utf-8")
        self.assertNotIn(secret, contents)
        self.assertNotIn(secret, str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        record = self.read_records()[0]
        self.assertEqual(record["request"]["parameters"]["api_key"], "[REDACTED]")
        self.assertIn("[REDACTED]", record["error"])
        self.assertIsNone(record["usage"])

    async def test_missing_key_is_logged_without_calling_provider(self):
        create = mock.AsyncMock()
        wrapper = LoggedModel(self.config, self.ledger, client=fake_client(create))

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ModelCallError, "Missing required environment variable"):
                await wrapper.complete([], stage="extract", case_id="c4", run_id="r4")

        create.assert_not_awaited()
        record = self.read_records()[0]
        self.assertIn("Missing required environment variable TEST_MODEL_API_KEY", record["error"])
        self.assertIsNone(record["usage"])

    async def test_empty_response_is_logged_before_error(self):
        create = mock.AsyncMock(return_value=fake_response(content="  "))
        wrapper = LoggedModel(self.config, self.ledger, client=fake_client(create))

        with mock.patch.dict("os.environ", {"TEST_MODEL_API_KEY": "secret-value"}):
            with self.assertRaisesRegex(ModelCallError, "EMPTY_MODEL_RESPONSE"):
                await wrapper.complete([], stage="coding", case_id="c5", run_id="r5")

        self.assertIn("EMPTY_MODEL_RESPONSE", self.read_records()[0]["error"])

    async def test_truncation_is_explicit_and_preserves_provider_usage(self):
        usage = {"completion_tokens": 4096, "total_tokens": 4600,
                 "completion_tokens_details": {"reasoning_tokens": 4096}}
        for content in ("", "partially returned code"):
            with self.subTest(content=content):
                create = mock.AsyncMock(return_value=fake_response(
                    content=content, usage=usage, finish_reason="length"))
                wrapper = LoggedModel(self.config, self.ledger, client=fake_client(create))
                with mock.patch.dict("os.environ", {"TEST_MODEL_API_KEY": "secret-value"}):
                    with self.assertRaisesRegex(ModelCallError, "MODEL_OUTPUT_TRUNCATED"):
                        await wrapper.complete([], stage="coding", case_id="truncated", run_id="r")
                record = self.read_records()[-1]
                self.assertEqual(record["finish_reason"], "length")
                self.assertEqual(record["usage"], usage)
                create.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
