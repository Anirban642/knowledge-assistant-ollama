"""Repository layer: the only place in the app that talks to the vector database."""

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.exceptions import ExternalServiceError
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned from a similarity search. Lower distance means a better match."""

    text: str
    source: str
    page: int | None
    chunk_index: int
    distance: float


@contextmanager
def _translate_errors(action: str) -> Iterator[None]:
    """Convert any low-level vector store / embedding failure into one clear app error."""
    try:
        yield
    except Exception as exc:
        logger.exception("Vector store failure while %s", action)
        raise ExternalServiceError(
            f"Could not complete the vector store operation ({action}). Check that Ollama is running."
        ) from exc


class ChunkRepository:
    """Stores and searches document chunks in the vector database."""

    def __init__(self, vector_store: Chroma) -> None:
        self._store = vector_store

    def replace_source_chunks(self, source: str, chunks: list[Document]) -> int:
        """Store the new chunks first, then remove leftovers from the previous version.

        Chunk IDs are deterministic ('file.txt::3'), so storing overwrites the matching old
        chunks. If embedding fails halfway, the previous version stays mostly intact instead
        of being deleted up front. Returns how many chunks the file had before.
        """
        new_ids = [self._build_chunk_id(chunk) for chunk in chunks]
        new_id_set = set(new_ids)

        with _translate_errors("replacing document chunks"):
            previous_ids = self._store.get(where={"source": source})["ids"]
            self._store.add_documents(documents=chunks, ids=new_ids)
            stale_ids = [chunk_id for chunk_id in previous_ids if chunk_id not in new_id_set]
            if stale_ids:
                self._store.delete(ids=stale_ids)

        return len(previous_ids)

    def delete_by_source(self, source: str) -> int:
        """Remove every chunk that came from the given file. Returns how many were removed."""
        with _translate_errors("deleting chunks"):
            existing_ids = self._store.get(where={"source": source})["ids"]
            if existing_ids:
                self._store.delete(ids=existing_ids)
        return len(existing_ids)

    def list_source_counts(self) -> dict[str, int]:
        """Return {file name: number of chunks} for every indexed document, sorted by name."""
        with _translate_errors("listing documents"):
            metadatas = self._store.get(include=["metadatas"])["metadatas"] or []
        counts = Counter(metadata.get("source", "unknown") for metadata in metadatas)
        return dict(sorted(counts.items()))

    def similarity_search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Return the top_k chunks whose meaning is closest to the query."""
        with _translate_errors("searching"):
            results = self._store.similarity_search_with_score(query, k=top_k)

        return [
            RetrievedChunk(
                text=document.page_content,
                source=document.metadata.get("source", "unknown"),
                page=document.metadata.get("page"),
                chunk_index=document.metadata.get("chunk_index", -1),
                distance=float(distance),
            )
            for document, distance in results
        ]

    @staticmethod
    def _build_chunk_id(chunk: Document) -> str:
        """Example: 'company_handbook.txt::3' (file name + chunk number)."""
        return f"{chunk.metadata['source']}::{chunk.metadata['chunk_index']}"