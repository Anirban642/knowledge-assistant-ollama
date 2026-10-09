"""Splits loaded documents into smaller, overlapping chunks ready for embedding."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingService:
    """Wraps LangChain's recursive splitter with our chunking rules."""

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],  # paragraph > line > sentence > word
            add_start_index=True,  # records where each chunk starts in the original text
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks and number them in order (chunk_index)."""
        chunks = self._splitter.split_documents(documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index
        return chunks