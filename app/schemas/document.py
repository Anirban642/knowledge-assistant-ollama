"""Response models for document-related endpoints."""

from pydantic import BaseModel, ConfigDict


class ChunkPreview(BaseModel):
    """A single chunk as shown to API clients."""

    chunk_index: int
    source: str
    page: int | None = None
    characters: int
    text: str


class ChunkPreviewResponse(BaseModel):
    """Summary of how a document was chunked, plus the first few chunks."""

    filename: str
    total_chunks: int
    chunks: list[ChunkPreview]


class IngestionResponse(BaseModel):
    """Outcome of indexing a document into the knowledge base."""

    model_config = ConfigDict(from_attributes=True)

    source: str
    chunks_stored: int
    replaced_chunks: int
    
    
class DocumentInfo(BaseModel):
    """One indexed document."""

    model_config = ConfigDict(from_attributes=True)

    source: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    """All documents currently in the knowledge base."""

    total_documents: int
    documents: list[DocumentInfo]


class DeleteDocumentResponse(BaseModel):
    """What was removed when a document was deleted."""

    model_config = ConfigDict(from_attributes=True)

    source: str
    chunks_deleted: int
    file_deleted: bool    