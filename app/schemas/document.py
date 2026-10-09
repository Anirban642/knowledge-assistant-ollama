"""Response models for document-related endpoints."""

from pydantic import BaseModel, ConfigDict


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