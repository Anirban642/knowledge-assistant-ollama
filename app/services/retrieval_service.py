"""Service layer: finds the chunks most relevant to a question."""

from app.core.exceptions import BadRequestError
from app.repositories.chunk_repository import ChunkRepository, RetrievedChunk


class RetrievalService:
    """Validates the question and searches the knowledge base."""

    def __init__(self, chunk_repository: ChunkRepository, default_top_k: int) -> None:
        self._chunk_repository = chunk_repository
        self._default_top_k = default_top_k

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return the best matching chunks for the query."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise BadRequestError("The search query must not be empty.")
        return self._chunk_repository.similarity_search(cleaned_query, top_k or self._default_top_k)