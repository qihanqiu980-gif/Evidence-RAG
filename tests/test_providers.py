import json
from dataclasses import asdict
from pathlib import Path

import httpx
import pytest

from rag_app.errors import ProviderResponseError, ProviderUnavailableError
from rag_app.evaluation import estimate_provider_cost
from rag_app.providers.base import (
    ModelProvider,
    ProviderMetricsTracker,
    ProviderTokenUsage,
    RerankItem,
)
from rag_app.providers.online import OnlineModelProvider
from tests.conftest import make_settings
from tests.fakes.provider import FakeProvider


def test_fake_provider_is_deterministic_and_matches_protocol():
    provider = FakeProvider(dimension=8)
    assert isinstance(provider, ModelProvider)
    first = provider.embed_query("WiFi 6 规格")
    second = provider.embed_query("WiFi 6 规格")
    assert first == second
    assert len(first) == 8
    assert provider.rerank("wifi", ["wifi router", "power supply"], 1) == [
        RerankItem(index=0, score=1.0)
    ]


def test_online_provider_rejects_wrong_embedding_dimension(tmp_path: Path):
    settings = make_settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"embedding": [1.0, 0.0]}]})

    provider = OnlineModelProvider(
        settings,
        client=httpx.Client(
            base_url="https://provider.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        with pytest.raises(ProviderResponseError) as error:
            provider.embed_query("hello")
    finally:
        provider.close()
    assert settings.api_key not in str(error.value)


def test_online_provider_records_retry_usage_and_failure_metrics(tmp_path: Path):
    settings = make_settings(
        tmp_path,
        RAG_APP_MAX_RETRIES="2",
        RAG_APP_EMBEDDING_DIMENSION="2",
    )
    calls = 0
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return httpx.Response(500, json={"error": "unavailable"})
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            },
        )

    provider = OnlineModelProvider(
        settings,
        client=httpx.Client(
            base_url="https://provider.test",
            transport=httpx.MockTransport(handler),
        ),
        sleep_function=sleeps.append,
    )
    try:
        assert provider.chat_json([{"role": "user", "content": "test"}], task="judge")
        metrics = provider.metrics_tracker.snapshot().operation("chat")
    finally:
        provider.close()

    assert calls == 3
    assert metrics.request_count == 1
    assert metrics.attempt_count == 3
    assert metrics.retry_count == 2
    assert metrics.http_5xx_retry_count == 2
    assert metrics.transport_retry_count == 0
    assert metrics.failed_request_count == 0
    assert sleeps == [0.2, 0.4]
    assert metrics.input_tokens == 11
    assert metrics.output_tokens == 7
    assert metrics.total_tokens == 18


def test_online_provider_records_final_transport_failure(tmp_path: Path):
    settings = make_settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed")
    sleeps: list[float] = []

    provider = OnlineModelProvider(
        settings,
        client=httpx.Client(
            base_url="https://provider.test",
            transport=httpx.MockTransport(handler),
        ),
        sleep_function=sleeps.append,
    )
    try:
        with pytest.raises(ProviderUnavailableError):
            provider.chat_json([{"role": "user", "content": "test"}], task="judge")
        metrics = provider.metrics_tracker.snapshot().operation("chat")
    finally:
        provider.close()

    assert metrics.request_count == 1
    assert metrics.attempt_count == 2
    assert metrics.retry_count == 1
    assert metrics.transport_retry_count == 1
    assert metrics.failed_request_count == 1
    assert metrics.provider_unavailable_failure_count == 1
    assert sleeps == [0.2]


def test_online_provider_estimates_cost_from_usage_metrics(tmp_path: Path):
    settings = make_settings(
        tmp_path,
        RAG_APP_CHAT_INPUT_PRICE_PER_1K="0.01",
        RAG_APP_CHAT_OUTPUT_PRICE_PER_1K="0.03",
    )
    usage = ProviderTokenUsage(input_tokens=1500, output_tokens=500, total_tokens=2000)
    metrics = ProviderMetricsTracker()
    metrics.record_request("chat")
    metrics.record_usage("chat", usage)

    assert estimate_provider_cost(metrics.snapshot(), settings) == pytest.approx(0.03)

    total_only = ProviderMetricsTracker()
    total_only.record_request("rerank")
    total_only.record_usage(
        "rerank", ProviderTokenUsage(total_tokens=20)
    )
    assert estimate_provider_cost(total_only.snapshot(), settings) is None
    assert settings.api_key not in json.dumps(asdict(metrics.snapshot()))


@pytest.mark.parametrize(
    "content",
    [
        '{"ok":true}',
        '```json\n{"ok":true}\n```',
        '"{\\"ok\\":true}"',
        'prefix {"ok":true} suffix',
    ],
)
def test_online_provider_parses_compatible_json_variants(
    tmp_path: Path,
    content: str,
):
    settings = make_settings(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    provider = OnlineModelProvider(
        settings,
        client=httpx.Client(
            base_url="https://provider.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        assert provider.chat_json(
            [{"role": "user", "content": "test"}],
            task="judge",
        ) == {"ok": True}
    finally:
        provider.close()


def test_online_chat_json_supports_task_model_and_json_response_format(tmp_path: Path):
    settings = make_settings(tmp_path)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    provider = OnlineModelProvider(
        settings,
        client=httpx.Client(
            base_url="https://provider.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        provider.chat_json(
            [{"role": "user", "content": "test"}],
            task="discovery_topics",
            model="qwen-flash",
            response_format={"type": "json_object"},
        )
    finally:
        provider.close()

    assert requests[0]["model"] == "qwen-flash"
    assert requests[0]["response_format"] == {"type": "json_object"}


def test_online_chat_json_uses_prompt_contract_without_response_format(tmp_path: Path):
    settings = make_settings(tmp_path)
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    provider = OnlineModelProvider(
        settings,
        client=httpx.Client(
            base_url="https://provider.test",
            transport=httpx.MockTransport(handler),
        ),
    )
    try:
        provider.chat_json([{"role": "user", "content": "test"}], task="decompose")
    finally:
        provider.close()

    assert requests[0]["model"] == settings.chat_model
    assert "response_format" not in requests[0]
