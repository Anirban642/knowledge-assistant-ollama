"""Checks whether the Ollama server is reachable and which models it has installed."""

import httpx


class OllamaStatusChecker:
    """Asks the Ollama server for its list of installed models."""

    def __init__(self, base_url: str, timeout_seconds: float = 3.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    def list_installed_models(self) -> set[str]:
        """Return installed model names such as 'llama3:latest'. Raises if Ollama cannot be reached."""
        response = httpx.get(f"{self._base_url}/api/tags", timeout=self._timeout_seconds)
        response.raise_for_status()
        return {model["name"] for model in response.json().get("models", [])}