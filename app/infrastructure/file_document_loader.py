"""Reads files from disk and converts them into LangChain Document objects."""

from collections.abc import Callable
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

from app.core.exceptions import BadRequestError, NotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Maps a file extension to a function that builds the matching LangChain loader.
_LOADER_FACTORIES: dict[str, Callable[[Path], BaseLoader]] = {
    ".pdf": lambda path: PyPDFLoader(str(path)),
    ".txt": lambda path: TextLoader(str(path), encoding="utf-8"),
    ".md": lambda path: TextLoader(str(path), encoding="utf-8"),
}


class FileDocumentLoader:
    """Loads a single file from disk into a list of LangChain Documents."""

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this loader can read."""
        return sorted(_LOADER_FACTORIES)

    def load(self, file_path: Path) -> list[Document]:
        """Read the file and return one Document per PDF page (or one for a text file)."""
        if not file_path.is_file():
            raise NotFoundError(f"Document '{file_path.name}' was not found.")

        extension = file_path.suffix.lower()
        loader_factory = _LOADER_FACTORIES.get(extension)
        if loader_factory is None:
            supported = ", ".join(self.supported_extensions)
            raise BadRequestError(f"Unsupported file type '{extension}'. Supported types: {supported}.")

        try:
            documents = loader_factory(file_path).load()
        except Exception as exc:
            logger.exception("Failed to read '%s'", file_path.name)
            raise BadRequestError(f"Could not read '{file_path.name}'. The file may be corrupted.") from exc

        documents = [document for document in documents if document.page_content.strip()]
        if not documents:
            raise BadRequestError(
                f"No readable text found in '{file_path.name}'. Scanned PDFs are not supported."
            )

        for document in documents:
            self._normalise_metadata(document, file_path)

        logger.info("Loaded %d section(s) from '%s'", len(documents), file_path.name)
        return documents

    @staticmethod
    def _normalise_metadata(document: Document, file_path: Path) -> None:
        """Keep only clean, simple metadata: the file name and (for PDFs) a 1-based page number."""
        metadata: dict[str, str | int] = {"source": file_path.name}
        if "page" in document.metadata:
            metadata["page"] = int(document.metadata["page"]) + 1
        document.metadata = metadata