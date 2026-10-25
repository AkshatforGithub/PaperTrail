from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Database
    postgres_user: str = "papertrail"
    postgres_password: str = "papertrail"
    postgres_db: str = "papertrail"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 5

    # LLM (Phase 7)
    groq_api_key: str = ""
    llm_model: str = ""
    judge_model: str = ""  # faithfulness judge; empty = same as llm_model

    # Computed database URL (filled in at initialization)
    database_url: str = ""

    # Guardrails
    max_question_length: int = 500
    daily_llm_call_cap: int = 100
    spend_limit_usd: float = 10.0
    cost_per_llm_call_usd: float = 0.001

    def _compute_database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.database_url = s._compute_database_url()
    return s