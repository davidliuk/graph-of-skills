from gos.core.engine import (
    DEFAULT_OPENROUTER_API_BASE,
    _resolve_openrouter_api_key,
    _resolve_openrouter_base_url,
    build_default_embedding_service,
    build_default_llm_service,
)


def test_openrouter_embedding_service_uses_openai_compat_env(monkeypatch):
    monkeypatch.setattr(
        "gos.core.engine.settings.EMBEDDING_MODEL",
        "openai/text-embedding-3-large",
    )
    monkeypatch.setattr("gos.core.engine.settings.EMBEDDING_DIM", 3072)
    monkeypatch.setattr("gos.core.engine.settings.OPENROUTER_API_KEY", None)
    monkeypatch.setattr(
        "gos.core.engine.settings.OPENAI_API_KEY",
        type("Secret", (), {"get_secret_value": lambda self: "openai-key"})(),
    )
    monkeypatch.setattr(
        "gos.core.engine.settings.OPENAI_BASE_URL",
        "https://openrouter.ai/api",
    )

    service = build_default_embedding_service()

    assert service.model == "openai/text-embedding-3-large"
    assert service.api_key == "openai-key"
    assert service.base_url == DEFAULT_OPENROUTER_API_BASE
    assert service.embedding_dim == 3072


def test_openrouter_llm_service_prefers_openrouter_api_key(monkeypatch):
    monkeypatch.setattr(
        "gos.core.engine.settings.LLM_MODEL",
        "openrouter/openai/gpt-4o-mini",
    )
    monkeypatch.setattr(
        "gos.core.engine.settings.OPENROUTER_API_KEY",
        type("Secret", (), {"get_secret_value": lambda self: "or-key"})(),
    )
    monkeypatch.setattr(
        "gos.core.engine.settings.OPENAI_API_KEY",
        type("Secret", (), {"get_secret_value": lambda self: "openai-key"})(),
    )
    monkeypatch.setattr(
        "gos.core.engine.settings.OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
    )

    service = build_default_llm_service()

    assert service.model == "openrouter/openai/gpt-4o-mini"
    assert service.api_key == "or-key"
    assert service.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_helpers_fallback_to_defaults(monkeypatch):
    monkeypatch.setattr("gos.core.engine.settings.OPENROUTER_API_KEY", None)
    monkeypatch.setattr("gos.core.engine.settings.OPENAI_API_KEY", None)
    monkeypatch.setattr("gos.core.engine.settings.OPENAI_BASE_URL", None)

    assert _resolve_openrouter_api_key() is None
    assert _resolve_openrouter_base_url() == DEFAULT_OPENROUTER_API_BASE
