"""Creates the ChromaDB vector store (local, persistent, free)."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


def create_vector_store(
    collection_name: str,
    persist_directory: Path,
    embeddings: Embeddings,
) -> Chroma:
    """Open (or create) a persistent Chroma collection that uses cosine distance."""
    persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},  # distance: 0 = identical, 2 = opposite
    )