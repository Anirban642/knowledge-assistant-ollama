"""Service layer: manages the lifecycle of documents (upload, list, delete)."""

import re
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import BadRequestError, NotFoundError, PayloadTooLargeError
from app.core.logging import get_logger
from app.infrastructure.file_storage import LocalDocumentStorage
from app.repositories.chunk_repository import ChunkRepository
from app.services.ingestion_service import IngestionResult, IngestionService

logger = get_logger(__name__)

_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]")


@dataclass(frozen=True)
class DocumentSummary:
    """One indexed document and how many chunks it was split into."""

    source: str
    chunk_count: int


@dataclass(frozen=True)
class DeletionResult:
    """What was removed when a document was deleted."""

    source: str
    chunks_deleted: int
    file_deleted: bool


class DocumentService:
    """Validates uploads, stores files, and keeps disk and vector database in sync."""

    def __init__(
        self,
        storage: LocalDocumentStorage,
        ingestion_service: IngestionService,
        chunk_repository: ChunkRepository,
        supported_extensions: list[str],
        max_upload_bytes: int,
    ) -> None:
        self._storage = storage
        self._ingestion_service = ingestion_service
        self._chunk_repository = chunk_repository
        self._supported_extensions = supported_extensions
        self._max_upload_bytes = max_upload_bytes

    def upload(self, filename: str | None, content: bytes) -> IngestionResult:
        """Validate, save and index an uploaded file. Uploading the same name again updates it."""
        safe_name = self._sanitize_filename(filename)
        self._validate_upload(safe_name, content)

        already_existed = self._storage.exists(safe_name)
        self._storage.save(safe_name, content)

        try:
            return self._ingestion_service.ingest_file(safe_name)
        except Exception:
            # Do not leave an un-indexed file behind when a brand-new upload fails.
            if not already_existed:
                self._storage.delete(safe_name)
            raise

    def list_documents(self) -> list[DocumentSummary]:
        """Return every indexed document with its chunk count."""
        counts = self._chunk_repository.list_source_counts()
        return [DocumentSummary(source=source, chunk_count=count) for source, count in counts.items()]

    def delete_document(self, filename: str) -> DeletionResult:
        """Remove a document's chunks from the vector database and its file from disk."""
        source = Path(filename).name
        chunks_deleted = self._chunk_repository.delete_by_source(source)
        file_deleted = self._storage.delete(source)

        if chunks_deleted == 0 and not file_deleted:
            raise NotFoundError(f"Document '{source}' was not found.")

        logger.info("Deleted '%s' (%d chunk(s), file removed: %s)", source, chunks_deleted, file_deleted)
        return DeletionResult(source=source, chunks_deleted=chunks_deleted, file_deleted=file_deleted)

    @staticmethod
    def _sanitize_filename(filename: str | None) -> str:
        """Strip folder parts and replace unsafe characters, e.g. 'My Policy (1).pdf' -> 'My_Policy__1_.pdf'."""
        name = _UNSAFE_FILENAME_CHARS.sub("_", Path(filename or "").name.strip())
        if not name or name.startswith("."):
            raise BadRequestError("A valid file name is required.")
        return name

    def _validate_upload(self, safe_name: str, content: bytes) -> None:
        """Reject unsupported types, empty files and files over the size limit."""
        extension = Path(safe_name).suffix.lower()
        if extension not in self._supported_extensions:
            supported = ", ".join(self._supported_extensions)
            raise BadRequestError(f"Unsupported file type '{extension}'. Supported types: {supported}.")
        if not content:
            raise BadRequestError("The uploaded file is empty.")
        if len(content) > self._max_upload_bytes:
            limit_mb = self._max_upload_bytes // (1024 * 1024)
            raise PayloadTooLargeError(f"The file is larger than the {limit_mb} MB upload limit.")