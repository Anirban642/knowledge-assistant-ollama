"""Health-check route used by developers and monitoring tools."""

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.schemas.health import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Check if the service is running")
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return the service status along with basic application details."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )