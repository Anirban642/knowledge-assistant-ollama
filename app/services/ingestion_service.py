"""Service layer: puts a document into the knowledge base."""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.repositories.chunk_repository import ChunkRepository
from app.services.document_processing_service import DocumentProcessingService

logger = get_logger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Summary of one ingestion run."""

    source: str
    chunks_stored: int
    replaced_chunks: int


class IngestionService:
    """Loads, chunks, embeds and stores a document. Re-ingesting a file updates its chunks."""

    def __init__(
        self,
        processing_service: DocumentProcessingService,
        chunk_repository: ChunkRepository,
    ) -> None:
        self._processing_service = processing_service
        self._chunk_repository = chunk_repository

    def ingest_file(self, filename: str) -> IngestionResult:
        """Index a file from the documents folder."""
        chunks = self._processing_service.prepare_chunks(filename)
        source = chunks[0].metadata["source"]

        previous_count = self._chunk_repository.replace_source_chunks(source, chunks)

        logger.info("Ingested '%s': %d chunk(s) stored (had %d before)", source, len(chunks), previous_count)
        return IngestionResult(source=source, chunks_stored=len(chunks), replaced_chunks=previous_count)