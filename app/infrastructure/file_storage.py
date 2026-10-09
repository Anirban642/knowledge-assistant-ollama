"""Local disk storage for uploaded documents."""

from pathlib import Path


class LocalDocumentStorage:
    """Saves, checks and deletes document files inside a single folder."""

    def __init__(self, documents_dir: Path) -> None:
        self._documents_dir = documents_dir
        self._documents_dir.mkdir(parents=True, exist_ok=True)

    def exists(self, filename: str) -> bool:
        """Return True if the file is already stored."""
        return self._path_for(filename).is_file()
    
    def read(self, filename: str) -> bytes | None:
        """Return the stored file's bytes, or None if it does not exist."""
        path = self._path_for(filename)
        return path.read_bytes() if path.is_file() else None

    def save(self, filename: str, content: bytes) -> Path:
        """Write the file to disk, replacing any existing file with the same name."""
        path = self._path_for(filename)
        path.write_bytes(content)
        return path

    def delete(self, filename: str) -> bool:
        """Delete the file. Returns False if it did not exist."""
        path = self._path_for(filename)
        if not path.is_file():
            return False
        path.unlink()
        return True

    def _path_for(self, filename: str) -> Path:
        """Build a path that is guaranteed to stay inside the documents folder."""
        return self._documents_dir / Path(filename).name