"""Response models for the health-check endpoint."""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Basic information that proves the service is up and running."""

    status: str
    app_name: str
    version: str
    environment: str