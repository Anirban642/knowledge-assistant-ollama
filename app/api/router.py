"""Aggregates every route module into a single API router."""

from fastapi import APIRouter

from app.api.routes import chat, debug, documents, health
from app.core.config import get_settings

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)

if get_settings().environment == "development":
    api_router.include_router(debug.router)