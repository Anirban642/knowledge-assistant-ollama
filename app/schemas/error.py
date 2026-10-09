"""Response models describing the shape of every error returned by the API."""

from pydantic import BaseModel


class FieldError(BaseModel):
    """A single invalid field in a request."""

    field: str
    message: str


class ErrorBody(BaseModel):
    """Details about what went wrong."""

    code: str
    message: str
    request_id: str | None = None
    fields: list[FieldError] | None = None


class ErrorResponse(BaseModel):
    """Top-level error envelope: {"error": {...}}."""

    error: ErrorBody