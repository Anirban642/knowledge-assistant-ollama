"""Application entry point. Run with: uvicorn app.main:app --reload"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.error_handlers import register_exception_handlers
from app.api.middleware import request_logging_middleware
from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _check_security_settings(settings: Settings) -> None:
    """Fail fast in production if auth is not configured, and warn loudly in other environments."""
    if not settings.admin_api_keys and settings.environment == "production":
        raise RuntimeError("ADMIN_API_KEYS must contain at least one key when ENVIRONMENT=production.")
    if not settings.admin_api_keys and not settings.user_api_keys:
        logger.warning(
            "No API keys are configured, so every protected endpoint will return 401. "
            "Set ADMIN_API_KEYS and USER_API_KEYS in .env."
        )


def create_app() -> FastAPI:
    """Build and configure the FastAPI application (application factory pattern)."""
    settings = get_settings()
    configure_logging(settings.log_level)
    _check_security_settings(settings)

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Ask questions about your company documents. Answers come only from the uploaded files. "
            "Click **Authorize** and enter an API key to call the protected endpoints."
        ),
    )

    application.middleware("http")(request_logging_middleware)
    if settings.cors_allowed_origins:
        # Added last, so it is the outermost layer and can answer browser preflight requests first.
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID"],
        )
    register_exception_handlers(application)
    application.include_router(api_router, prefix=settings.api_v1_prefix)

    logger.info("%s v%s started in '%s' mode", settings.app_name, settings.app_version, settings.environment)
    return application


app = create_app()