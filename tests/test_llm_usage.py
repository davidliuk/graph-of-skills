import asyncio
from types import SimpleNamespace
from types import MethodType
from unittest.mock import AsyncMock

import httpx
import litellm
import pytest

from gos.core.litellm_services import LiteLLMEmbeddingService, LiteLLMService
from gos.core.litellm_services import LLMUsageLedger
from gos.core.schema import GOSRelationList


def test_openrouter_call_enables_response_cache_and_records_usage(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=80),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
        _hidden_params={
            "additional_headers": {"x-openrouter-cache": "HIT"},
            "response_cost": 0.0012,
        },
    )
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(litellm, "acompletion", completion)

    async def scenario():
        service = LiteLLMService(
            model="openrouter/minimax/minimax-m2.7",
            api_key="test-secret",
            base_url="https://openrouter.ai/api/v1",
            response_cache=True,
        )
        parsed, _ = await service.send_message(
            "hello",
            response_model=None,
            gos_stage="relation_validation",
        )

        assert parsed == "ok"
        kwargs = completion.await_args.kwargs
        assert kwargs["extra_headers"]["X-OpenRouter-Cache"] == "true"
        assert kwargs["timeout"] == 60.0
        assert kwargs["num_retries"] == 1

        stats = service.usage.by_stage["relation_validation"]
        assert stats.calls == 1
        assert stats.cache_hits == 1
        assert stats.input_tokens == 100
        assert stats.cached_input_tokens == 80
        assert stats.output_tokens == 20
        assert stats.reasoning_tokens == 5
        assert stats.cost_usd == 0.0012

    asyncio.run(scenario())


def test_non_openrouter_call_does_not_add_openrouter_cache_header(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage=None,
        _hidden_params={},
    )
    completion = AsyncMock(return_value=response)
    monkeypatch.setattr(litellm, "acompletion", completion)

    async def scenario():
        service = LiteLLMService(
            model="openai/gpt-4o-mini",
            api_key="test-secret",
            response_cache=True,
        )
        await service.send_message("hello", response_model=None)
        assert "extra_headers" not in completion.await_args.kwargs

    asyncio.run(scenario())


def test_openrouter_embedding_uses_native_endpoint_cache_and_usage():
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(
            200,
            request=request,
            headers={"x-openrouter-cache": "HIT"},
            json={
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {
                    "prompt_tokens": 12,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 12},
                    "cost": 0.0001,
                },
            },
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            service = LiteLLMEmbeddingService(
                model="qwen/qwen3-embedding-8b",
                embedding_dim=3,
                api_key="test-secret",
                base_url="https://openrouter.ai/api/v1",
                response_cache=True,
                http_client=client,
            )
            vectors = await service.encode(["catalog"])

        assert vectors.shape == (1, 3)
        request = captured["request"]
        assert str(request.url) == "https://openrouter.ai/api/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert request.headers["X-OpenRouter-Cache"] == "true"
        stats = service.usage.by_stage["embedding"]
        assert stats.calls == 1
        assert stats.cache_hits == 1
        assert stats.input_tokens == 12
        assert stats.cached_input_tokens == 12
        assert stats.cost_usd == 0.0001

    asyncio.run(scenario())


def test_usage_ledger_reads_provider_cost_when_hidden_cost_is_absent():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=2,
            cost=0.0042,
        ),
        _hidden_params={},
    )
    ledger = LLMUsageLedger()

    ledger.record("relation_validation", response=response)

    assert ledger.by_stage["relation_validation"].cost_usd == 0.0042


def test_embedding_batches_use_bounded_concurrency_and_preserve_order():
    async def scenario():
        service = LiteLLMEmbeddingService(
            model="test-embedding",
            embedding_dim=1,
            embedding_batch_size=1,
            embedding_concurrency=2,
        )
        active = 0
        maximum_active = 0

        async def fake_batch(self, batch, model):
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return [[float(batch[0])]]

        service._encode_batch = MethodType(fake_batch, service)
        vectors = await service.encode(["0", "1", "2", "3", "4"])

        assert maximum_active == 2
        assert vectors[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    asyncio.run(scenario())


def test_invalid_structured_response_counts_as_failure_without_losing_usage(
    monkeypatch,
):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="not valid json"))],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=3),
        _hidden_params={"response_cost": 0.002},
    )
    monkeypatch.setattr(litellm, "acompletion", AsyncMock(return_value=response))

    async def scenario():
        service = LiteLLMService(
            model="openrouter/minimax/minimax-m2.7",
            api_key="test-secret",
            base_url="https://openrouter.ai/api/v1",
        )
        with pytest.raises(Exception):
            await service.send_message(
                "return relations",
                response_model=GOSRelationList,
                gos_stage="relation_validation",
            )

        stats = service.usage.by_stage["relation_validation"]
        assert stats.calls == 1
        assert stats.failures == 1
        assert stats.input_tokens == 9
        assert stats.output_tokens == 3
        assert stats.cost_usd == 0.002

    asyncio.run(scenario())
