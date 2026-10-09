"""Application entry point. Run with: uvicorn app.main:app --reload"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Ask questions about your company documents. Answers come only from the uploaded files.",
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