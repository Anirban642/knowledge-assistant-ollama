"""Health routes used by developers, load balancers and container platforms."""

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import get_health_service
from app.core.config import Settings, get_settings
from app.schemas.health import ComponentCheckResponse, HealthResponse, ReadinessResponse
from app.services.health_service import HealthService

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse, summary="Liveness: is the service running?")
def get_health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Return the service status along with basic application details."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness: are Ollama, the models and the vector database usable?",
)
def get_readiness(
    response: Response,
    service: HealthService = Depends(get_health_service),
) -> ReadinessResponse:
    """Return 200 when everything is usable, or 503 with details about what is broken."""
    checks = service.check_readiness()
    is_ready = all(check.healthy for check in checks)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        checks=[ComponentCheckResponse.model_validate(check) for check in checks],
    )