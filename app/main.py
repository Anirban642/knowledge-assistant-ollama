"""Application entry point. Run with: uvicorn app.main:app --reload"""

from fastapi import FastAPI

from app.api.error_handlers import register_exception_handlers
from app.api.middleware import request_logging_middleware
from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Build and configure the FastAPI application (application factory pattern)."""
    settings = get_settings()
    configure_logging(settings.log_level)

    # Note: we deliberately do not pass debug=True to FastAPI. In debug mode Starlette
    # shows its own HTML traceback page and bypasses our JSON 500 handler.
    application = FastAPI(title=settings.app_name, version=settings.app_version)

    application.middleware("http")(request_logging_middleware)
    register_exception_handlers(application)
    application.include_router(api_router, prefix=settings.api_v1_prefix)

    logger.info("%s v%s started in '%s' mode", settings.app_name, settings.app_version, settings.environment)
    return application


app = create_app()