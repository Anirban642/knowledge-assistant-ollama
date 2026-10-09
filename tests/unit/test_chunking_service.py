import pytest
from langchain_core.documents import Document

from app.services.chunking_service import ChunkingService


def test_chunks_respect_size_limit_and_get_sequential_indexes():
    service = ChunkingService(chunk_size=100, chunk_overlap=20)
    text = " ".join(f"Sentence number {i} is here." for i in range(40))

    chunks = service.split([Document(page_content=text, metadata={"source": "doc.txt"})])

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.metadata["source"] == "doc.txt" for chunk in chunks)


def test_short_text_stays_in_a_single_chunk():
    service = ChunkingService(chunk_size=800, chunk_overlap=150)

    chunks = service.split([Document(page_content="A short policy.", metadata={"source": "doc.txt"})])

    assert len(chunks) == 1
    assert chunks[0].metadata["chunk_index"] == 0


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        ChunkingService(chunk_size=100, chunk_overlap=100)