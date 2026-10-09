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
from tests.constants import ADMIN_API_KEY, USER_API_KEY
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
        admin_api_keys=[ADMIN_API_KEY],
        user_api_keys=[USER_API_KEY],
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
    """A test client that sends the ADMIN key. Ollama and ChromaDB are replaced by fakes."""
    conversation_repository = InMemoryConversationRepository(max_messages_per_session=6, max_sessions=10)

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_chunk_repository] = lambda: chunk_repository
    app.dependency_overrides[get_chat_model] = lambda: chat_model
    app.dependency_overrides[get_conversation_repository] = lambda: conversation_repository
    app.dependency_overrides[get_ollama_status_checker] = lambda: ollama_checker

    with TestClient(app, headers={"X-API-Key": ADMIN_API_KEY}) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def user_client(client: TestClient) -> TestClient:
    """A client that sends the USER key. It shares the fakes set up by the `client` fixture."""
    return TestClient(app, headers={"X-API-Key": USER_API_KEY})


@pytest.fixture
def anonymous_client(client: TestClient) -> TestClient:
    """A client that sends no API key at all."""
    return TestClient(app)