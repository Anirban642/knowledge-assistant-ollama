"""Response models for search endpoints."""

from pydantic import BaseModel, ConfigDict


class SearchHit(BaseModel):
    """One matching chunk. Lower distance means a closer match."""

    model_config = ConfigDict(from_attributes=True)

    text: str
    source: str
    page: int | None = None
    chunk_index: int
    distance: float


class SearchResponse(BaseModel):
    """Results of a similarity search."""

    query: str
    total_hits: int
    hits: list[SearchHit]