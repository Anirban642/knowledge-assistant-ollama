"""Lightweight stand-ins for external systems (vector database, Ollama) so tests run fast and offline."""

from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig

from app.core.exceptions import ExternalServiceError
from app.repositories.chunk_repository import RetrievedChunk


def make_chunk(
    text: str = "Employees get 12 days of paid sick leave per year.",
    source: str = "handbook.txt",
    chunk_index: int = 0,
    distance: float = 0.2,
    page: int | None = None,
) -> RetrievedChunk:
    """Build a retrieved chunk with sensible defaults."""
    return RetrievedChunk(text=text, source=source, page=page, chunk_index=chunk_index, distance=distance)


class ScriptedChatModel(Runnable[Any, AIMessage]):
    """Fake LLM: returns the scripted responses in order (looping) and counts how often it was called."""

    def __init__(self, *responses: str) -> None:
        self.script(*responses)

    def script(self, *responses: str) -> None:
        """Replace the scripted responses and reset the call counter."""
        self.responses = list(responses)
        self.calls = 0

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> AIMessage:
        response = self.responses[self.calls % len(self.responses)]
        self.calls += 1
        return AIMessage(content=response)


class FailingChatModel(Runnable[Any, AIMessage]):
    """Fake LLM that always fails, like Ollama being down."""

    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> AIMessage:
        raise ConnectionError("Ollama is down")


class StubRetrievalService:
    """Returns fixed chunks and records the queries it received."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks


class FakeChunkRepository:
    """In-memory replacement for ChunkRepository (the vector database)."""

    def __init__(self, search_results: list[RetrievedChunk] | None = None) -> None:
        self.search_results = search_results or []
        self.queries: list[str] = []
        self.fail_on_write = False
        self.is_down = False
        self._chunks_by_source: dict[str, list[Document]] = {}

    def replace_source_chunks(self, source: str, chunks: list[Document]) -> int:
        if self.fail_on_write:
            raise ExternalServiceError("Simulated vector store failure.")
        previous_count = len(self._chunks_by_source.get(source, []))
        self._chunks_by_source[source] = list(chunks)
        return previous_count

    def delete_by_source(self, source: str) -> int:
        return len(self._chunks_by_source.pop(source, []))

    def list_source_counts(self) -> dict[str, int]:
        return {source: len(chunks) for source, chunks in sorted(self._chunks_by_source.items())}

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.search_results[:top_k]

    def ping(self) -> None:
        if self.is_down:
            raise ExternalServiceError("Simulated vector store outage.")


class FakeOllamaStatusChecker:
    """Replacement for OllamaStatusChecker."""

    def __init__(self, models: set[str] | None = None, is_down: bool = False) -> None:
        self.models = models if models is not None else {"llama3:latest", "nomic-embed-text:latest"}
        self.is_down = is_down

    def list_installed_models(self) -> set[str]:
        if self.is_down:
            raise ConnectionError("Ollama is down")
        return self.models