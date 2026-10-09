"""Shared pytest fixtures. API tests replace every external dependency with a fake."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_chat_model,
    get_chunk_repository,
    get_conversation_repository,
    get_ollama_status_checker,
)
from app.core.config import Settings, get_settings
from app.main import app
from app.repositories.conversation_repository import InMemoryConversationRepository
from tests.fakes import FakeChunkRepository, FakeOllamaStatusChecker, ScriptedChatModel


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings that point at a throwaway folder and ignore the developer's real .env file."""
    return Settings(
        _env_file=None,
        environment="test",
        documents_dir=tmp_path / "documents",
        chroma_persist_dir=tmp_path / "chroma",
        max_upload_size_mb=1,
    )


@pytest.fixture
def chunk_repository() -> FakeChunkRepository:
    return FakeChunkRepository()


@pytest.fixture
def chat_model() -> ScriptedChatModel:
    return ScriptedChatModel("Default fake answer.")


@pytest.fixture
def ollama_checker() -> FakeOllamaStatusChecker:
    return FakeOllamaStatusChecker()


@pytest.fixture
def client(
    settings: Settings,
    chunk_repository: FakeChunkRepository,
    chat_model: ScriptedChatModel,
    ollama_checker: FakeOllamaStatusChecker,
) -> Iterator[TestClient]:
    """A test client whose external dependencies (Ollama, ChromaDB) are all fakes."""
    conversation_repository = InMemoryConversationRepository(max_messages_per_session=6, max_sessions=10)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_chunk_repository] = lambda: chunk_repository
    app.dependency_overrides[get_chat_model] = lambda: chat_model
    app.dependency_overrides[get_conversation_repository] = lambda: conversation_repository
    app.dependency_overrides[get_ollama_status_checker] = lambda: ollama_checker

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()