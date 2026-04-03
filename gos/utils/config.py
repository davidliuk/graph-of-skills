from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="GOS_",
        extra="ignore",
    )

    WORKING_DIR: str = "./gos_workspace"
    PREBUILT_WORKING_DIR: str | None = None
    DOMAIN: str = "Agent Skills and Tool Dependencies"

    LLM_MODEL: str = "openrouter/google/gemini-2.5-flash"
    EMBEDDING_MODEL: str = "openrouter/openai/text-embedding-3-large"
    EMBEDDING_DIM: int = 3072
    GEMINI_API_KEY: SecretStr | None = Field(default=None, alias="GEMINI_API_KEY")
    OPENAI_API_KEY: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    OPENROUTER_API_KEY: SecretStr | None = Field(default=None, alias="OPENROUTER_API_KEY")
    OPENAI_BASE_URL: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    LINK_TOP_K: int = 8
    SEED_TOP_K: int = 5
    RETRIEVAL_TOP_N: int = 8
    USE_FULL_MARKDOWN: bool = True
    ENABLE_SEMANTIC_LINKING: bool = True
    DEPENDENCY_MATCH_THRESHOLD: float = 0.6

    PPR_DAMPING: float = 0.2
    PPR_MAX_ITER: int = 50
    PPR_TOLERANCE: float = 1e-6

    SNIPPET_CHARS: int = 900
    MAX_SKILL_CHARS: int = 2400
    MAX_CONTEXT_CHARS: int = 12000
    RERANK_CANDIDATE_MULTIPLIER: int = 4
    SEED_CANDIDATE_TOP_K_SEMANTIC: int = 20
    SEED_CANDIDATE_TOP_K_LEXICAL: int = 20
    ENABLE_QUERY_REWRITE: bool = False

    SKILL_FILENAME: str = "SKILL.md"
    ALLOW_FRONTMATTER_DOCS: bool = True

    # When set, graphskills-query rewrites Source: paths to {SKILLS_DIR}/{skill_name}/SKILL.md
    # so agents in containerised environments can find skill scripts at the mounted path.
    SKILLS_DIR: str = ""


settings = Settings()
