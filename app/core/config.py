"""Application configuration loaded from environment variables and the .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root folder (knowledge-assistant/), so paths work no matter where uvicorn is started.
BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Central place for every configurable value in the application."""

    app_name: str = "Company Knowledge Assistant"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    api_v1_prefix: str = "/api/v1"

    documents_dir: Path = BASE_DIR / "data" / "documents"
    max_upload_size_mb: int = 10
    chunk_size: int = 800
    chunk_overlap: int = 150

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"

    chroma_persist_dir: Path = BASE_DIR / "data" / "chroma"
    chroma_collection_name: str = "company_knowledge"
    retrieval_top_k: int = 4
    retrieval_max_distance: float = 0.45

    llm_model: str = "llama3"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 4096
    llm_num_gpu: int | None = None  # 0 = force CPU, unset = let Ollama decide
    llm_timeout_seconds: float = 180.0
    
    conversation_history_turns: int = 3  # how many recent question/answer pairs are remembered per session
    conversation_max_sessions: int = 500  # oldest idle sessions are dropped beyond this

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def max_upload_bytes(self) -> int:
        """Upload size limit expressed in bytes."""
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so the .env file is read only once."""
    return Settings()