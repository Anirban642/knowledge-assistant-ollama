import pytest

from app.core.exceptions import BadRequestError, NotFoundError
from app.infrastructure.file_document_loader import FileDocumentLoader


def test_loads_a_text_file_with_clean_metadata(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Hello policy world", encoding="utf-8")

    documents = FileDocumentLoader().load(path)

    assert len(documents) == 1
    assert documents[0].page_content == "Hello policy world"
    assert documents[0].metadata == {"source": "notes.txt"}


def test_missing_file_raises_not_found(tmp_path):
    with pytest.raises(NotFoundError):
        FileDocumentLoader().load(tmp_path / "missing.txt")


def test_unsupported_extension_raises_bad_request(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(BadRequestError, match="Unsupported file type"):
        FileDocumentLoader().load(path)


def test_file_without_readable_text_raises_bad_request(tmp_path):
    path = tmp_path / "blank.txt"
    path.write_text("   \n  ", encoding="utf-8")

    with pytest.raises(BadRequestError, match="No readable text"):
        FileDocumentLoader().load(path)