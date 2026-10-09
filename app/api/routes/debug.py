"""TEMPORARY routes for testing error handling. Delete this file in a later step."""

from typing import NoReturn

from fastapi import APIRouter

from app.core.exceptions import NotFoundError

router = APIRouter(prefix="/debug", tags=["Debug (temporary)"])


@router.get("/not-found", response_model=None, summary="Raise a NotFoundError")
def trigger_not_found() -> NoReturn:
    raise NotFoundError("Sample document was not found.")


@router.get("/crash", response_model=None, summary="Raise an unexpected error")
def trigger_crash() -> NoReturn:
    raise RuntimeError("Simulated crash for testing.")


@router.get("/validate", summary="Trigger a validation error with ?number=abc")
def trigger_validation(number: int) -> dict[str, int]:
    return {"number": number}