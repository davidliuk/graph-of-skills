from gos.utils.config import Settings


def test_settings_reads_gos_prefixed_env_vars(monkeypatch):
    monkeypatch.setenv("GOS_WORKING_DIR", "./tmp_workspace")
    monkeypatch.setenv("GOS_PREBUILT_WORKING_DIR", "./prebuilt_workspace")
    monkeypatch.setenv("GOS_LLM_MODEL", "gemini/gemini-3.1-flash-lite-preview")
    monkeypatch.setenv("GOS_EMBEDDING_MODEL", "gemini/gemini-embedding-001")
    monkeypatch.setenv("GOS_SEED_TOP_K", "7")
    monkeypatch.setenv("GOS_ENABLE_SEMANTIC_LINKING", "false")

    settings = Settings(_env_file=None)

    assert settings.WORKING_DIR == "./tmp_workspace"
    assert settings.PREBUILT_WORKING_DIR == "./prebuilt_workspace"
    assert settings.LLM_MODEL == "gemini/gemini-3.1-flash-lite-preview"
    assert settings.EMBEDDING_MODEL == "gemini/gemini-embedding-001"
    assert settings.SEED_TOP_K == 7
    assert settings.ENABLE_SEMANTIC_LINKING is False


def test_settings_reads_openrouter_embedding_env_vars(monkeypatch):
    monkeypatch.setenv("GOS_EMBEDDING_MODEL", "openrouter/openai/text-embedding-3-large")
    monkeypatch.setenv("GOS_EMBEDDING_DIM", "3072")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    settings = Settings(_env_file=None)

    assert settings.EMBEDDING_MODEL == "openrouter/openai/text-embedding-3-large"
    assert settings.EMBEDDING_DIM == 3072
    assert settings.OPENAI_API_KEY.get_secret_value() == "openai-key"
    assert settings.OPENAI_BASE_URL == "https://openrouter.ai/api/v1"
