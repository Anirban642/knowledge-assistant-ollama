"""Embedding model client: turns text into vectors using Ollama's nomic-embed-text."""

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

# nomic-embed-text was trained with these prefixes. Without them, retrieval quality drops.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class NomicOllamaEmbeddings(Embeddings):
    """LangChain-compatible embeddings that add the prefixes nomic-embed-text expects."""

    def __init__(self, model: str, base_url: str) -> None:
        self._client = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed chunks that will be stored in the vector database."""
        return self._client.embed_documents([f"{DOCUMENT_PREFIX}{text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question used to search the vector database."""
        return self._client.embed_query(f"{QUERY_PREFIX}{text}")