from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
import sys
import time
from typing import Any, Type
import asyncio

import httpx
import litellm
import numpy as np
import truststore
from json_repair import repair_json

from fast_graphrag._llm._base import BaseEmbeddingService, BaseLLMService, T_model
from fast_graphrag._models import BaseModelAlias


JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

if sys.platform == "darwin":
    # Standalone Python distributions on macOS do not always read Keychain CAs.
    # Use the OS trust store instead of disabling TLS verification.
    truststore.inject_into_ssl()


def _value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class UsageStats:
    calls: int = 0
    cache_hits: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    elapsed_seconds: float = 0.0
    failures: int = 0


@dataclass
class LLMUsageLedger:
    by_stage: dict[str, UsageStats] = field(default_factory=dict)

    def record(
        self,
        stage: str,
        *,
        response: Any = None,
        elapsed_seconds: float = 0.0,
        failed: bool = False,
    ) -> None:
        stats = self.by_stage.setdefault(stage, UsageStats())
        stats.calls += 1
        stats.elapsed_seconds += max(float(elapsed_seconds), 0.0)
        if failed:
            stats.failures += 1

        usage = _value(response, "usage")
        stats.input_tokens += int(_value(usage, "prompt_tokens", 0) or 0)
        stats.output_tokens += int(_value(usage, "completion_tokens", 0) or 0)
        prompt_details = _value(usage, "prompt_tokens_details")
        stats.cached_input_tokens += int(
            _value(prompt_details, "cached_tokens", 0) or 0
        )
        completion_details = _value(usage, "completion_tokens_details")
        stats.reasoning_tokens += int(
            _value(completion_details, "reasoning_tokens", 0) or 0
        )

        hidden = _value(response, "_hidden_params", {}) or {}
        headers = _value(hidden, "additional_headers", {}) or {}
        cache_status = str(
            _value(headers, "x-openrouter-cache", "")
            or _value(headers, "X-OpenRouter-Cache", "")
        ).lower()
        if cache_status in {"hit", "true", "1"}:
            stats.cache_hits += 1
        cost = _value(hidden, "response_cost")
        if not isinstance(cost, (int, float)):
            cost = _value(usage, "cost", 0.0)
        if isinstance(cost, (int, float)):
            stats.cost_usd += float(cost)

    def to_dict(self) -> dict[str, dict[str, int | float]]:
        return {
            stage: {
                "calls": stats.calls,
                "cache_hits": stats.cache_hits,
                "input_tokens": stats.input_tokens,
                "cached_input_tokens": stats.cached_input_tokens,
                "output_tokens": stats.output_tokens,
                "reasoning_tokens": stats.reasoning_tokens,
                "cost_usd": stats.cost_usd,
                "elapsed_seconds": stats.elapsed_seconds,
                "failures": stats.failures,
            }
            for stage, stats in sorted(self.by_stage.items())
        }


def extract_json_text(content: str) -> str:
    fenced = JSON_BLOCK_PATTERN.search(content)
    if fenced:
        return fenced.group(1).strip()
    return content.strip()


def validate_response_model(response_model: Type[T_model], content: str) -> T_model:
    cleaned = extract_json_text(content)
    repaired = repair_json(cleaned)
    payload = json.loads(repaired)

    # MiniMax can occasionally serialize mixed content blocks as
    # ["", {structured payload}].  Preserve strict model validation while
    # unwrapping the only object payload instead of discarding the focus result.
    if isinstance(payload, list):
        object_payloads = [item for item in payload if isinstance(item, dict)]
        if len(object_payloads) == 1 and all(
            isinstance(item, dict) or not str(item or "").strip()
            for item in payload
        ):
            payload = object_payloads[0]

    if issubclass(response_model, BaseModelAlias):
        parsed = response_model.Model.model_validate(payload)
        return parsed.to_dataclass(parsed)

    return response_model.model_validate(payload)


@dataclass
class LiteLLMService(BaseLLMService):
    temperature: float = field(default=0.0)
    response_cache: bool = field(default=True)
    request_timeout_seconds: float = field(default=60.0)
    max_retries: int = field(default=1)
    usage: LLMUsageLedger = field(default_factory=LLMUsageLedger, init=False)

    def _uses_openrouter(self) -> bool:
        return (
            self.model.startswith("openrouter/")
            or "openrouter.ai" in str(self.base_url or "").lower()
        )

    async def send_message(
        self,
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, str]] | None = None,
        response_model: Type[T_model] | None = None,
        **kwargs: Any,
    ) -> tuple[T_model, list[dict[str, str]]]:
        stage = str(kwargs.pop("gos_stage", "unspecified"))
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        if self.response_cache and self._uses_openrouter():
            extra_headers = dict(kwargs.pop("extra_headers", {}) or {})
            extra_headers.setdefault("X-OpenRouter-Cache", "true")
            kwargs["extra_headers"] = extra_headers

        started = time.perf_counter()
        response: Any = None
        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                api_key=self.api_key,
                base_url=self.base_url,
                temperature=kwargs.pop("temperature", self.temperature),
                timeout=kwargs.pop("timeout", self.request_timeout_seconds),
                num_retries=kwargs.pop("num_retries", self.max_retries),
                **kwargs,
            )
            content = response.choices[0].message.content or ""
            if response_model is None:
                parsed_response = content
            else:
                parsed_response = validate_response_model(response_model, content)
        except Exception:
            self.usage.record(
                stage,
                response=response,
                elapsed_seconds=time.perf_counter() - started,
                failed=True,
            )
            raise
        self.usage.record(
            stage,
            response=response,
            elapsed_seconds=time.perf_counter() - started,
        )

        updated_history = messages + [{"role": "assistant", "content": content}]
        return parsed_response, updated_history


@dataclass
class LiteLLMEmbeddingService(BaseEmbeddingService):
    # Gemini BatchEmbedContentsRequest allows at most 100 items per call.
    embedding_batch_size: int = field(default=100)
    embedding_concurrency: int = field(default=4)
    response_cache: bool = field(default=True)
    http_client: Any = field(default=None, repr=False)
    openrouter_max_retries: int = field(default=2)
    usage: LLMUsageLedger = field(default_factory=LLMUsageLedger, init=False)

    def _uses_openrouter(self) -> bool:
        return "openrouter.ai" in str(self.base_url or "").lower()

    async def _encode_openrouter_batch(
        self,
        batch: list[str],
        model: str,
    ) -> list[list[float]]:
        if not self.api_key:
            raise RuntimeError("OpenRouter embeddings require an API key.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.response_cache:
            headers["X-OpenRouter-Cache"] = "true"

        client = self.http_client or httpx.AsyncClient()
        owns_client = self.http_client is None
        endpoint = f"{str(self.base_url).rstrip('/')}/embeddings"
        try:
            for attempt in range(max(self.openrouter_max_retries, 0) + 1):
                started = time.perf_counter()
                ledger_response: dict[str, Any] | None = None
                response: httpx.Response | None = None
                try:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json={"model": model, "input": batch},
                        timeout=60.0,
                    )
                    try:
                        payload = response.json()
                    except ValueError:
                        payload = {}
                    if not isinstance(payload, dict):
                        payload = {}
                    ledger_response = {
                        "usage": payload.get("usage", {}),
                        "_hidden_params": {
                            "additional_headers": dict(response.headers),
                        },
                    }
                    response.raise_for_status()

                    rows = payload.get("data", [])
                    if not isinstance(rows, list) or len(rows) != len(batch):
                        raise ValueError(
                            "OpenRouter embedding response row count does not match input."
                        )
                    ordered_rows = sorted(
                        rows,
                        key=lambda item: int(item.get("index", 0)),
                    )
                    vectors = [item.get("embedding", []) for item in ordered_rows]
                    if any(
                        not isinstance(vector, list) or not vector for vector in vectors
                    ):
                        raise ValueError(
                            "OpenRouter embedding response contains an empty vector."
                        )
                except Exception:
                    self.usage.record(
                        "embedding",
                        response=ledger_response,
                        elapsed_seconds=time.perf_counter() - started,
                        failed=True,
                    )
                    retryable = response is not None and response.status_code in {
                        429,
                        500,
                        502,
                        503,
                        529,
                    }
                    if retryable and attempt < max(self.openrouter_max_retries, 0):
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    raise

                self.usage.record(
                    "embedding",
                    response=ledger_response,
                    elapsed_seconds=time.perf_counter() - started,
                )
                return vectors
        finally:
            if owns_client:
                await client.aclose()

        raise RuntimeError("OpenRouter embedding request exhausted retries.")

    async def _encode_batch(self, batch: list[str], model: str) -> list[list[float]]:
        if self._uses_openrouter():
            return await self._encode_openrouter_batch(batch, model)

        kwargs: dict[str, Any] = {}

        started = time.perf_counter()
        try:
            response = await litellm.aembedding(
                model=model,
                input=batch,
                api_key=self.api_key,
                api_base=self.base_url or None,
                **kwargs,
            )
        except Exception:
            self.usage.record(
                "embedding",
                elapsed_seconds=time.perf_counter() - started,
                failed=True,
            )
            raise
        self.usage.record(
            "embedding",
            response=response,
            elapsed_seconds=time.perf_counter() - started,
        )
        vectors = []
        for item in response.data:
            if isinstance(item, dict):
                vectors.append(item["embedding"])
            else:
                vectors.append(item.embedding)
        return vectors

    async def encode(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> np.ndarray[Any, np.dtype[np.float32]]:
        resolved_model = model or self.model
        batches = [
            texts[i : i + self.embedding_batch_size]
            for i in range(0, len(texts), self.embedding_batch_size)
        ]
        concurrency = max(int(self.embedding_concurrency), 1)
        semaphore = asyncio.Semaphore(concurrency)

        async def encode_bounded(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._encode_batch(batch, resolved_model)

        results = await asyncio.gather(
            *[encode_bounded(batch) for batch in batches]
        )
        all_vectors = [v for batch_vectors in results for v in batch_vectors]
        return np.array(all_vectors, dtype=np.float32)
