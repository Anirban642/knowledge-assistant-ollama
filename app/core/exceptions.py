"""Custom application exceptions.

Services and repositories raise these. The API layer converts them into HTTP
responses (see app/api/error_handlers.py), so business code never needs to
know anything about HTTP.
"""


class AppError(Exception):
    """Base class for all expected, application-level errors."""

    status_code: int = 500
    code: str = "internal_error"
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class BadRequestError(AppError):
    """The client sent something we cannot process."""

    status_code = 400
    code = "bad_request"
    default_message = "The request could not be processed."


class NotFoundError(AppError):
    """A requested resource does not exist."""

    status_code = 404
    code = "not_found"
    default_message = "The requested resource was not found."


class ExternalServiceError(AppError):
    """A dependency such as Ollama or ChromaDB is unavailable or failed."""

    status_code = 503
    code = "external_service_unavailable"
    default_message = "A required external service is currently unavailable."