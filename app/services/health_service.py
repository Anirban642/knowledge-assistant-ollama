"""Service layer: checks whether the application's dependencies are ready to serve requests."""

from dataclasses import dataclass

from app.core.logging import get_logger
from app.infrastructure.ollama_status import OllamaStatusChecker
from app.repositories.chunk_repository import ChunkRepository

logger = get_logger(__name__)


@dataclass(frozen=True)
class ComponentCheck:
    """The result of checking one dependency."""

    name: str
    healthy: bool
    detail: str


class HealthService:
    """Reports whether Ollama, the required models and the vector database are usable."""

    def __init__(
        self,
        ollama_status_checker: OllamaStatusChecker,
        chunk_repository: ChunkRepository,
        required_models: list[str],
    ) -> None:
        self._ollama_status_checker = ollama_status_checker
        self._chunk_repository = chunk_repository
        self._required_models = required_models

    def check_readiness(self) -> list[ComponentCheck]:
        """Run every check and return one result per dependency."""
        checks: list[ComponentCheck] = []

        try:
            installed_models = self._ollama_status_checker.list_installed_models()
        except Exception:
            logger.warning("Ollama readiness check failed", exc_info=True)
            checks.append(ComponentCheck("ollama", False, "Cannot reach the Ollama server. Is it running?"))
        else:
            checks.append(ComponentCheck("ollama", True, "Ollama server is reachable."))
            checks.extend(self._check_model(model, installed_models) for model in self._required_models)

        checks.append(self._check_vector_store())
        return checks

    @staticmethod
    def _check_model(model: str, installed_models: set[str]) -> ComponentCheck:
        """A model counts as installed if its name matches exactly or with a tag, e.g. 'llama3:latest'."""
        is_installed = any(name == model or name.startswith(f"{model}:") for name in installed_models)
        if is_installed:
            return ComponentCheck(f"model:{model}", True, "Model is installed.")
        return ComponentCheck(f"model:{model}", False, f"Model is not installed. Run: ollama pull {model}")

    def _check_vector_store(self) -> ComponentCheck:
        try:
            self._chunk_repository.ping()
        except Exception:
            return ComponentCheck("vector_store", False, "ChromaDB is not accessible.")
        return ComponentCheck("vector_store", True, "ChromaDB is accessible.")