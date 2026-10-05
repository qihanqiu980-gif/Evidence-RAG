import pytest

from rag_app.config import Settings
from rag_app.errors import ConfigurationError


def test_settings_accept_explicit_environment_without_dotenv(tmp_path):
    settings = Settings.from_env(
        tmp_path,
        {
            "RAG_APP_API_KEY": "key",
            "RAG_APP_CHUNK_SIZE": "20",
            "RAG_APP_CHUNK_OVERLAP": "5",
        },
    )
    assert settings.configured
    assert settings.chunk_size == 20
    assert settings.sqlite_path == tmp_path / "data" / "app.db"


def test_missing_api_key_is_not_configured_but_is_valid(tmp_path):
    settings = Settings.from_env(tmp_path, {"RAG_APP_DATA_DIR": "data"})
    assert not settings.configured
    assert settings.api_key == ""


def test_blank_rerank_configuration_is_not_ready(tmp_path):
    settings = Settings.from_env(
        tmp_path,
        {"RAG_APP_API_KEY": "key", "RAG_APP_RERANK_MODEL": ""},
    )
    assert not settings.configured


def test_chunk_overlap_must_be_less_than_size(tmp_path):
    with pytest.raises(ConfigurationError):
        Settings.from_env(
            tmp_path,
            {"RAG_APP_CHUNK_SIZE": "10", "RAG_APP_CHUNK_OVERLAP": "10"},
        )


def test_optional_cost_prices_and_currency_are_validated(tmp_path):
    settings = Settings.from_env(
        tmp_path,
        {
            "RAG_APP_COST_CURRENCY": "usd",
            "RAG_APP_CHAT_INPUT_PRICE_PER_1K": "0.01",
            "RAG_APP_CHAT_OUTPUT_PRICE_PER_1K": "0.03",
        },
    )
    assert settings.cost_currency == "USD"
    assert settings.chat_input_price_per_1k == 0.01
    assert settings.chat_output_price_per_1k == 0.03
    assert settings.embedding_input_price_per_1k is None

    with pytest.raises(ConfigurationError):
        Settings.from_env(
            tmp_path,
            {"RAG_APP_RERANK_INPUT_PRICE_PER_1K": "-1"},
        )

    with pytest.raises(ConfigurationError):
        Settings.from_env(tmp_path, {"RAG_APP_COST_CURRENCY": "dollars"})


def test_retry_stability_settings_are_validated(tmp_path):
    settings = Settings.from_env(
        tmp_path,
        {
            "RAG_APP_MAX_RETRIES": "3",
            "RAG_APP_RETRY_BACKOFF_SECONDS": "0.05",
        },
    )
    assert settings.max_retries == 3
    assert settings.retry_backoff_seconds == 0.05

    with pytest.raises(ConfigurationError):
        Settings.from_env(tmp_path, {"RAG_APP_MAX_RETRIES": "6"})

    with pytest.raises(ConfigurationError):
        Settings.from_env(tmp_path, {"RAG_APP_RETRY_BACKOFF_SECONDS": "10.1"})
