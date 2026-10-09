"""TEMPORARY routes for testing. Delete this file in a later step."""

from typing import NoReturn

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import (
    get_document_processing_service,
    get_ingestion_service,
    get_retrieval_service,
)
from app.core.exceptions import NotFoundError
from app.schemas.document import ChunkPreview, ChunkPreviewResponse, IngestionResponse
from app.schemas.search import SearchHit, SearchResponse
from app.services.document_processing_service import DocumentProcessingService
from app.services.ingestion_service import IngestionService
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/debug", tags=["Debug (temporary)"])


@router.get("/not-found", response_model=None, summary="Raise a NotFoundError")
def trigger_not_found() -> NoReturn:
    raise NotFoundError("Sample document was not found.")


@router.get("/crash", response_model=None, summary="Raise an unexpected error")
def trigger_crash() -> NoReturn:
    raise RuntimeError("Simulated crash for testing.")


@router.get("/validate", summary="Trigger a validation error with ?number=abc")
def trigger_validation(number: int) -> dict[str, int]:
    return {"number": number}


@router.get("/chunks", response_model=ChunkPreviewResponse, summary="Preview how a document is chunked")
def preview_chunks(
    filename: str,
    limit: int = Query(default=3, ge=1, le=50, description="How many chunks to return"),
    service: DocumentProcessingService = Depends(get_document_processing_service),
) -> ChunkPreviewResponse:
    """Load a file from data/documents and show how it was split into chunks."""
    chunks = service.prepare_chunks(filename)
    previews = [
        ChunkPreview(
            chunk_index=chunk.metadata["chunk_index"],
            source=chunk.metadata["source"],
            page=chunk.metadata.get("page"),
            characters=len(chunk.page_content),
            text=chunk.page_content,
        )
        for chunk in chunks[:limit]
    ]
    return ChunkPreviewResponse(filename=filename, total_chunks=len(chunks), chunks=previews)


@router.post("/index", response_model=IngestionResponse, summary="Index a file into the vector database")
def index_document(
    filename: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResponse:
    """Embed a file from data/documents and store its chunks in ChromaDB."""
    result = service.ingest_file(filename)
    return IngestionResponse.model_validate(result)


@router.get("/search", response_model=SearchResponse, summary="Search the vector database")
def search_chunks(
    query: str,
    top_k: int | None = Query(default=None, ge=1, le=20, description="How many chunks to return"),
    service: RetrievalService = Depends(get_retrieval_service),
) -> SearchResponse:
    """Find the chunks closest in meaning to the query."""
    hits = service.search(query, top_k)
    return SearchResponse(
        query=query,
        total_hits=len(hits),
        hits=[SearchHit.model_validate(hit) for hit in hits],
    )