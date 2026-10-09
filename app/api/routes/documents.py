"""Routes for uploading, listing and deleting knowledge-base documents."""

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.auth import authenticate, require_admin
from app.api.dependencies import get_document_service
from app.core.config import Settings, get_settings
from app.schemas.document import (
    DeleteDocumentResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestionResponse,
)
from app.services.document_service import DocumentService

# Any valid key may read. Writing needs an admin key (added per route below).
router = APIRouter(prefix="/documents", tags=["Documents"], dependencies=[Depends(authenticate)])


@router.post(
    "",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document (admin)",
    dependencies=[Depends(require_admin)],
)
def upload_document(
    file: UploadFile = File(..., description="A .pdf, .txt or .md file"),
    settings: Settings = Depends(get_settings),
    service: DocumentService = Depends(get_document_service),
) -> IngestionResponse:
    """Save the file and add it to the knowledge base. Uploading the same file name updates it."""
    # Read one byte past the limit so the service can detect oversized files without loading them fully.
    content = file.file.read(settings.max_upload_bytes + 1)
    result = service.upload(file.filename, content)
    return IngestionResponse.model_validate(result)


@router.get("", response_model=DocumentListResponse, summary="List indexed documents")
def list_documents(service: DocumentService = Depends(get_document_service)) -> DocumentListResponse:
    """Show every document currently in the knowledge base."""
    documents = service.list_documents()
    return DocumentListResponse(
        total_documents=len(documents),
        documents=[DocumentInfo.model_validate(document) for document in documents],
    )


@router.delete(
    "/{filename}",
    response_model=DeleteDocumentResponse,
    summary="Delete a document (admin)",
    dependencies=[Depends(require_admin)],
)
def delete_document(
    filename: str,
    service: DocumentService = Depends(get_document_service),
) -> DeleteDocumentResponse:
    """Remove a document from the knowledge base and from disk."""
    result = service.delete_document(filename)
    return DeleteDocumentResponse.model_validate(result)