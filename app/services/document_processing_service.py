"""Turns a file in the documents folder into a list of ready-to-embed chunks."""

from pathlib import Path

from langchain_core.documents import Document

from app.core.logging import get_logger
from app.infrastructure.file_document_loader import FileDocumentLoader
from app.services.chunking_service import ChunkingService

logger = get_logger(__name__)


class DocumentProcessingService:
    """Coordinates loading and chunking for a single document."""

    def __init__(
        self,
        loader: FileDocumentLoader,
        chunking_service: ChunkingService,
        documents_dir: Path,
    ) -> None:
        self._loader = loader
        self._chunking_service = chunking_service
        self._documents_dir = documents_dir

    def prepare_chunks(self, filename: str) -> list[Document]:
        """Load the named file from the documents folder and split it into chunks."""
        file_path = self._resolve_path(filename)
        sections = self._loader.load(file_path)
        chunks = self._chunking_service.split(sections)
        logger.info("Prepared %d chunk(s) from '%s'", len(chunks), file_path.name)
        return chunks

    def _resolve_path(self, filename: str) -> Path:
        """Build a safe path inside the documents folder.

        Path(filename).name drops any folder parts, so input like '../../.env'
        can never escape the documents directory.
        """
        return self._documents_dir / Path(filename).name