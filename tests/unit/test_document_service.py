import pytest

from app.core.exceptions import BadRequestError, ExternalServiceError, NotFoundError, PayloadTooLargeError
from app.infrastructure.file_document_loader import FileDocumentLoader
from app.infrastructure.file_storage import LocalDocumentStorage
from app.services.chunking_service import ChunkingService
from app.services.document_processing_service import DocumentProcessingService
from app.services.document_service import DocumentService, DocumentSummary
from app.services.ingestion_service import IngestionService
from tests.fakes import FakeChunkRepository

MAX_UPLOAD_BYTES = 1000


@pytest.fixture
def documents_dir(tmp_path):
    return tmp_path / "documents"


@pytest.fixture
def repository():
    return FakeChunkRepository()


@pytest.fixture
def service(documents_dir, repository):
    processing_service = DocumentProcessingService(
        loader=FileDocumentLoader(),
        chunking_service=ChunkingService(chunk_size=200, chunk_overlap=20),
        documents_dir=documents_dir,
    )
    return DocumentService(
        storage=LocalDocumentStorage(documents_dir),
        ingestion_service=IngestionService(processing_service, repository),
        chunk_repository=repository,
        supported_extensions=[".md", ".pdf", ".txt"],
        max_upload_bytes=MAX_UPLOAD_BYTES,
    )


def test_upload_saves_the_file_and_indexes_its_chunks(service, documents_dir, repository):
    result = service.upload("policy.txt", b"Leave policy. " * 30)

    assert result.source == "policy.txt"
    assert result.chunks_stored >= 1
    assert (documents_dir / "policy.txt").is_file()
    assert repository.list_source_counts() == {"policy.txt": result.chunks_stored}


def test_upload_sanitizes_unsafe_characters_in_file_names(service):
    result = service.upload("My Policy (1).txt", b"hello world")

    assert result.source == "My_Policy__1_.txt"


def test_upload_cannot_escape_the_documents_folder(service, documents_dir):
    result = service.upload("../../evil.txt", b"hello world")

    assert result.source == "evil.txt"
    assert (documents_dir / "evil.txt").is_file()


def test_upload_rejects_hidden_file_names(service):
    with pytest.raises(BadRequestError, match="valid file name"):
        service.upload(".env", b"SECRET=1")


def test_upload_rejects_unsupported_types(service):
    with pytest.raises(BadRequestError, match="Unsupported file type"):
        service.upload("macro.exe", b"data")


def test_upload_rejects_empty_files(service):
    with pytest.raises(BadRequestError, match="empty"):
        service.upload("policy.txt", b"")


def test_upload_rejects_files_over_the_size_limit(service):
    with pytest.raises(PayloadTooLargeError):
        service.upload("policy.txt", b"a" * (MAX_UPLOAD_BYTES + 1))


def test_uploading_the_same_name_replaces_the_old_chunks(service):
    first = service.upload("policy.txt", b"Version one of the policy.")
    second = service.upload("policy.txt", b"Version two of the policy.")

    assert second.replaced_chunks == first.chunks_stored


def test_failed_new_upload_leaves_no_file_behind(service, repository, documents_dir):
    repository.fail_on_write = True

    with pytest.raises(ExternalServiceError):
        service.upload("policy.txt", b"Some policy text.")

    assert not (documents_dir / "policy.txt").exists()


def test_failed_reupload_restores_the_previous_file(service, repository, documents_dir):
    service.upload("policy.txt", b"Original text.")
    repository.fail_on_write = True

    with pytest.raises(ExternalServiceError):
        service.upload("policy.txt", b"New text.")

    assert (documents_dir / "policy.txt").read_bytes() == b"Original text."


def test_list_and_delete_document(service, documents_dir):
    result = service.upload("policy.txt", b"Leave policy text.")

    assert service.list_documents() == [DocumentSummary(source="policy.txt", chunk_count=result.chunks_stored)]

    deletion = service.delete_document("policy.txt")

    assert deletion.chunks_deleted == result.chunks_stored
    assert deletion.file_deleted is True
    assert not (documents_dir / "policy.txt").exists()
    assert service.list_documents() == []

    with pytest.raises(NotFoundError):
        service.delete_document("policy.txt")