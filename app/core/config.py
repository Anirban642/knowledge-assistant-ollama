"""Application configuration loaded from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.security import MIN_API_KEY_LENGTH

# Project root folder (knowledge-assistant/), so paths work no matter where uvicorn is started.
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central place for every configurable value in the application."""

    # Application
    app_name: str = "Company Knowledge Assistant"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"
    cors_allowed_origins: list[str] = []  # browser origins allowed to call the API, e.g. a React app

    # Documents and chunking
    documents_dir: Path = BASE_DIR / "data" / "documents"
    max_upload_size_mb: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 150

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096
    llm_num_gpu: int | None = None  # 0 = force CPU, unset = let Ollama decide
    llm_timeout_seconds: float = 180.0

    # Vector database
    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma"
    chroma_collection_name: str = "company_knowledge"

    # Retrieval
    retrieval_top_k: int = 4
    retrieval_max_distance: float = 0.45
    source_distance_margin: float = 0.08  # only show sources this close to the best match
    max_sources: int = 3

    # Conversations
    conversation_history_turns: int = 3  # recent question/answer pairs remembered per session
    conversation_max_sessions: int = 500  # oldest idle sessions are dropped beyond this
    
    # Authentication: JSON lists of keys. Admin keys can manage documents, user keys can only ask questions.
    admin_api_keys: list[str] = []
    user_api_keys: list[str] = []

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("documents_dir", "chroma_persist_dir")
    @classmethod
    def _make_path_absolute(cls, value: Path) -> Path:
        """Treat relative paths as relative to the project root."""
        return value if value.is_absolute() else BASE_DIR / value
    
    @field_validator("admin_api_keys", "user_api_keys")
    @classmethod
    def _reject_weak_api_keys(cls, keys: list[str]) -> list[str]:
        """Refuse to start with guessable keys."""
        if any(len(key) < MIN_API_KEY_LENGTH for key in keys):
            raise ValueError(f"every API key must be at least {MIN_API_KEY_LENGTH} characters long")
        return keys

    @property
    def max_upload_bytes(self) -> int:
        """Upload size limit expressed in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so the .env file is read only once."""
    return Settings()