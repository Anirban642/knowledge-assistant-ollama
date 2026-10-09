"""Authentication and authorization dependencies for the API layer."""

from fastapi import Depends, Request, Security
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.core.exceptions import ForbiddenError, UnauthorizedError
from app.core.security import AuthenticatedClient, Role, build_client_id, matches_any_key

# auto_error=False lets us return our own consistent JSON error instead of the framework's default.
_api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="Your API key. Admin keys can manage documents; user keys can ask questions.",
)


def authenticate(
    request: Request,
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedClient:
    """Require a valid API key and return who is calling."""
    if not api_key:
        raise UnauthorizedError("Missing API key. Send it in the X-API-Key header.")

    if matches_any_key(api_key, settings.admin_api_keys):
        role = Role.ADMIN
    elif matches_any_key(api_key, settings.user_api_keys):
        role = Role.USER
    else:
        raise UnauthorizedError("Invalid API key.")

    client = AuthenticatedClient(client_id=build_client_id(api_key), role=role)
    request.state.client_id = client.client_id  # picked up by the request logger
    return client


def require_admin(client: AuthenticatedClient = Depends(authenticate)) -> AuthenticatedClient:
    """Allow only admin keys."""
    if client.role is not Role.ADMIN:
        raise ForbiddenError("This action requires an admin API key.")
    return client