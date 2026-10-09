"""HTTP middleware: assigns a request ID and logs every request with its duration."""

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from app.core.logging import get_logger

logger = get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


async def request_logging_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Log method, path, status code, duration and the calling client for every request."""
    request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started_at = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.error("[%s] %s %s -> 500 (%.1f ms)", request_id, request.method, request.url.path, duration_ms)
        raise

    duration_ms = (time.perf_counter() - started_at) * 1000
    response.headers[REQUEST_ID_HEADER] = request_id
    logger.info(
        "[%s] %s %s -> %d (%.1f ms) client=%s",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
        getattr(request.state, "client_id", "-"),
    )
    return response