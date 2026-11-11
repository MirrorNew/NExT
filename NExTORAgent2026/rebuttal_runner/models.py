"""Small, auditable async model client for rebuttal experiments."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI


class ModelConfigError(ValueError):
    """Raised when a model configuration is incomplete or inconsistent."""


class ModelCallError(RuntimeError):
    """A failed model call with its already-persisted metadata record."""

    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


class CallBudgetExceeded(RuntimeError):
    """The declared call budget was exhausted before another provider request."""


class BudgetedModel:
    """Per-run view of the same provider; failures consume one attempted call."""

    def __init__(self, model, max_calls: int):
        if isinstance(max_calls, bool) or not isinstance(max_calls, int) or max_calls < 1:
            raise ValueError("max_calls must be a positive integer")
        self.model = model
        self.config = model.config
        self.ledger_path = model.ledger_path
        self.max_calls = max_calls
        self.attempted_calls = 0
        self.call_ids = []

    async def complete(self, messages, **identity):
        if self.attempted_calls >= self.max_calls:
            raise CallBudgetExceeded(f"Call budget {self.max_calls} exhausted")
        self.attempted_calls += 1
        try:
            response = await self.model.complete(messages, **identity)
        except ModelCallError as exc:
            self.call_ids.append(exc.record["call_id"])
            raise
        self.call_ids.append(response["call_id"])
        return response


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    request_parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 120
    allowed_returned_models: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        model = self.model.strip() if isinstance(self.model, str) else ""
        if not model:
            raise ModelConfigError("model must be a non-empty string")
        object.__setattr__(self, "model", model)

        if self.base_url is not None and not isinstance(self.base_url, str):
            raise ModelConfigError("base_url must be a string or None")
        base_url = (self.base_url or "").strip() or os.getenv("OPENAI_API_BASE")
        if base_url is not None:
            if not isinstance(base_url, str):
                raise ModelConfigError("OPENAI_API_BASE must be a string")
            base_url = base_url.strip() or None
        object.__setattr__(self, "base_url", base_url)
        if not base_url:
            raise ModelConfigError(
                "model endpoint is required; set base_url or OPENAI_API_BASE"
            )
        if not isinstance(self.api_key_env, str) or not self.api_key_env.strip():
            raise ModelConfigError("api_key_env must name an environment variable")
        object.__setattr__(self, "api_key_env", self.api_key_env.strip())
        if not isinstance(self.request_parameters, dict):
            raise ModelConfigError("request_parameters must be a dictionary")
        reserved = {"model", "messages"}.intersection(self.request_parameters)
        if reserved:
            raise ModelConfigError(
                "request_parameters cannot override " + ", ".join(sorted(reserved))
            )
        if self.request_parameters.get("stream") is True:
            raise ModelConfigError("stream=True is unsupported; complete() is non-streaming")
        if self.timeout_seconds <= 0:
            raise ModelConfigError("timeout_seconds must be greater than zero")

        allowed = self.allowed_returned_models
        if allowed is None:
            allowed = (model,)
        elif isinstance(allowed, str):
            allowed = (allowed,)
        else:
            allowed = tuple(allowed)
        if not allowed or any(not isinstance(item, str) or not item for item in allowed):
            raise ModelConfigError("allowed_returned_models must contain model ids")
        object.__setattr__(self, "allowed_returned_models", allowed)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ModelConfig":
        """Build a config from a mapping without inventing provider/model defaults."""
        if not isinstance(values, Mapping):
            raise ModelConfigError("model configuration must be a mapping")
        known = {
            "model",
            "base_url",
            "api_key_env",
            "request_parameters",
            "timeout_seconds",
            "allowed_returned_models",
        }
        unknown = set(values) - known
        if unknown:
            raise ModelConfigError("unknown model config fields: " + ", ".join(sorted(unknown)))
        if "model" not in values:
            raise ModelConfigError("model is required; no model will be selected implicitly")
        return cls(**dict(values))


_LEDGER_LOCK = asyncio.Lock()
_CAUSE_CHAIN_LIMIT = 8
_CAUSE_MESSAGE_LIMIT = 512
_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{8,}\b"),
)


def _redact(value: Any, secrets: Sequence[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: (match.group(1) if match.lastindex else "") + "[REDACTED]", text)
    return text


def _exception_chain(exc: BaseException) -> tuple[list[BaseException], bool]:
    """Return a bounded cause/context chain, stopping safely on cycles."""
    chain: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and len(chain) < _CAUSE_CHAIN_LIMIT:
        marker = id(current)
        if marker in seen:
            break
        seen.add(marker)
        chain.append(current)
        cause = current.__cause__
        if cause is None and not current.__suppress_context__:
            cause = current.__context__
        if cause is not None and id(cause) in seen:
            break
        current = cause
    truncated = (current is not None and len(chain) == _CAUSE_CHAIN_LIMIT
                 and id(current) not in seen)
    return chain, truncated


def _error_code(record: Mapping[str, Any], chain: Sequence[BaseException]) -> str:
    message = str(record.get("error") or "")
    normalized_message = message.casefold()
    if record.get("finish_reason") == "length" or "MODEL_OUTPUT_TRUNCATED" in message:
        return "truncated"
    if "MODEL_ID_MISMATCH" in message:
        return "model_id"
    if "EMPTY_MODEL_RESPONSE" in message:
        return "empty"
    if "Missing required environment variable" in message:
        return "configuration"

    names = [type(item).__name__.casefold() for item in chain]
    if (any(isinstance(item, (TimeoutError, asyncio.TimeoutError)) or
            "timeout" in name or "timedout" in name for item, name in zip(chain, names)) or
            "timeout" in normalized_message or "timed out" in normalized_message):
        return "timeout"
    if any(isinstance(getattr(item, "status_code", None), int) or
           ("http" in name and "error" in name) or name == "apistatuserror"
           for item, name in zip(chain, names)) or "http error" in normalized_message:
        return "http"
    if any(isinstance(item, OSError) or
           name in {"apiconnectionerror", "connectionerror", "connecterror", "networkerror"}
           or "connectionerror" in name for item, name in zip(chain, names)) or any(
               phrase in normalized_message for phrase in
               ("connection error", "connection failed", "connection refused", "failed to connect")):
        return "connection"
    return "unknown"


def _cause_chain_record(chain: Sequence[BaseException], secrets: Sequence[str]) -> list[dict[str, str]]:
    result = []
    for item in chain:
        try:
            message = str(item)
        except Exception:
            message = "<exception message unavailable>"
        result.append({"type": type(item).__name__,
                       "message": _redact(message, secrets)[:_CAUSE_MESSAGE_LIMIT]})
    return result


def _safe_metadata(value: Any, secrets: Sequence[str] = ()) -> Any:
    if isinstance(value, Mapping):
        safe = {}
        for key, item in value.items():
            name = str(key)
            if re.search(
                r"(?i)(api[_-]?key|authorization|secret|access[_-]?token|refresh[_-]?token|^token$)",
                name,
            ):
                safe[name] = "[REDACTED]"
            else:
                safe[name] = _safe_metadata(item, secrets)
        return safe
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact(value, secrets)
    return _json_safe(value)


def _redact_values(value: Any, secrets: Sequence[str] = ()) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _redact_values(item, secrets) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_values(item, secrets) for item in value]
    if isinstance(value, str):
        return _redact(value, secrets)
    return _json_safe(value)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _json_safe(model_dump(mode="json", exclude_none=True))
        except TypeError:
            return _json_safe(model_dump())
    return str(value)


def _usage_record(usage: Any) -> Any:
    """Preserve provider-reported fields; do not derive or sum token categories."""
    if usage is None:
        return None
    dumped = _json_safe(usage)
    if isinstance(dumped, dict):
        return dumped
    names = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "prompt_tokens",
        "completion_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
    )
    values = {name: getattr(usage, name) for name in names if hasattr(usage, name)}
    return _json_safe(values) if values else dumped


class LoggedModel:
    """AsyncOpenAI chat-completions wrapper with one JSONL record per attempt."""

    def __init__(
        self,
        config: ModelConfig | Mapping[str, Any],
        ledger_path: str | os.PathLike[str],
        *,
        client: Any | None = None,
    ) -> None:
        self.config = config if isinstance(config, ModelConfig) else ModelConfig.from_dict(config)
        self.ledger_path = Path(ledger_path)
        self.client = client
        self._owns_client = client is None

    def _get_client(self, api_key: str) -> Any:
        if self.client is None:
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=self.config.base_url,
                max_retries=0,
                timeout=self.config.timeout_seconds,
            )
        return self.client

    async def _append_record(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        async with _LEDGER_LOCK:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)

    async def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        stage: str,
        case_id: str,
        run_id: str,
        response_format: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        call_id = str(uuid.uuid4())
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        message_list = [dict(message) for message in messages]
        parameters = dict(self.config.request_parameters)
        if response_format is not None:
            parameters["response_format"] = dict(response_format)
        request = {
            "model": self.config.model,
            "messages": message_list,
            **parameters,
        }
        api_key = os.getenv(self.config.api_key_env)
        secrets = (api_key,) if api_key else ()
        record: dict[str, Any] = {
            "call_id": call_id,
            "stage": stage,
            "case_id": case_id,
            "run_id": run_id,
            "started_at": started_at,
            "request": {
                "model": self.config.model,
                "parameters": _safe_metadata(parameters, secrets),
                "message_count": len(message_list),
                "message_roles": [message.get("role") for message in message_list],
            },
            "returned_model": None,
            "usage": None,
            "finish_reason": None,
            "elapsed_seconds": None,
            "error": None,
            "error_type": None,
            "error_code": None,
            "cause_chain": [],
            "cause_chain_truncated": False,
        }

        try:
            if not api_key:
                raise RuntimeError(
                    f"Missing required environment variable {self.config.api_key_env}"
                )
            client = self._get_client(api_key)
            response = await asyncio.wait_for(
                client.chat.completions.create(**request),
                timeout=self.config.timeout_seconds,
            )
            returned_model = getattr(response, "model", None)
            record["returned_model"] = returned_model
            record["usage"] = _usage_record(getattr(response, "usage", None))
            choices = getattr(response, "choices", None) or []
            choice = choices[0] if choices else None
            message = getattr(choice, "message", None) if choice is not None else None
            content = getattr(message, "content", None) if message is not None else None
            record["finish_reason"] = getattr(choice, "finish_reason", None) if choice else None

            if returned_model not in self.config.allowed_returned_models:
                raise RuntimeError(
                    f"MODEL_ID_MISMATCH: requested {self.config.model!r}, "
                    f"returned {returned_model!r}"
                )
            if record["finish_reason"] == "length":
                raise RuntimeError("MODEL_OUTPUT_TRUNCATED: completion token limit reached; inspect usage before changing the budget")
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("EMPTY_MODEL_RESPONSE")

            record["elapsed_seconds"] = time.monotonic() - started
            await self._append_record(_redact_values(record, secrets))
            return {
                "content": _redact(content, secrets),
                "returned_model": _redact(returned_model, secrets),
                "usage": _redact_values(record["usage"], secrets),
                "elapsed_seconds": record["elapsed_seconds"],
                "call_id": call_id,
            }
        except Exception as exc:
            record["elapsed_seconds"] = time.monotonic() - started
            chain, chain_truncated = _exception_chain(exc)
            try:
                error_message = str(exc)
            except Exception:
                error_message = "<exception message unavailable>"
            record["error"] = _redact(error_message, secrets) or type(exc).__name__
            record["error_type"] = type(exc).__name__
            record["error_code"] = _error_code(record, chain)
            record["cause_chain"] = _cause_chain_record(chain, secrets)
            record["cause_chain_truncated"] = chain_truncated
            safe_record = _redact_values(record, secrets)
            await self._append_record(safe_record)
            raise ModelCallError(safe_record["error"], safe_record) from None

    async def close(self) -> None:
        if self._owns_client and self.client is not None:
            await self.client.close()
            self.client = None
