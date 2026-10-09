"""Response models for the health-check endpoints."""

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    """Basic information that proves the service is up and running."""

    status: str
    app_name: str
    version: str
    environment: str


class ComponentCheckResponse(BaseModel):
    """The state of one dependency (Ollama, a model, the vector database)."""

    model_config = ConfigDict(from_attributes=True)

    name: str
    healthy: bool
    detail: str


class ReadinessResponse(BaseModel):
    """Overall readiness plus the result of every individual check."""

    status: str
    checks: list[ComponentCheckResponse]