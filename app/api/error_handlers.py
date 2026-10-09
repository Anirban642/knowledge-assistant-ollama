"""Global exception handlers that turn every error into a consistent JSON response."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.exceptions import AppError
from app.core.logging import get_logger
from app.schemas.error import ErrorBody, ErrorResponse, FieldError

logger = get_logger(__name__)


def _build_error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    fields: list[FieldError] | None = None,
) -> JSONResponse:
    """Create the standard error JSON, including the request ID for easy debugging."""
    body = ErrorResponse(
        error=ErrorBody(
            code=code,
            message=message,
            request_id=getattr(request.state, "request_id", None),
            fields=fields,
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump(exclude_none=True))


async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Handle expected errors raised by our own code."""
    logger.warning("Application error [%s]: %s", exc.code, exc.message)
    return _build_error_response(request, exc.status_code, exc.code, exc.message)


async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle framework errors such as 404 for an unknown URL."""
    return _build_error_response(request, exc.status_code, "http_error", str(exc.detail))


async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle invalid request data (wrong types, missing fields, etc.)."""
    fields = [
        FieldError(field=".".join(str(part) for part in error["loc"]), message=error["msg"])
        for error in exc.errors()
    ]
    return _build_error_response(request, 422, "validation_error", "Request validation failed.", fields)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Handle any bug we did not anticipate. Log everything, reveal nothing to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _build_error_response(request, 500, "internal_error", "An unexpected error occurred.")


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all handlers to the application."""
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(StarletteHTTPException, handle_http_exception)
    app.add_exception_handler(RequestValidationError, handle_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)