"""Request and response models for the chat endpoints."""

from pydantic import BaseModel, ConfigDict, Field

SESSION_ID_PATTERN = r"^[A-Za-z0-9_-]+$"


class ChatRequest(BaseModel):
    """A question from the user, optionally part of an ongoing conversation."""

    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000, examples=["How many sick leave days do I get?"])
    session_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=SESSION_ID_PATTERN,
        description="Reuse the session_id from an earlier response to ask a follow-up. Leave empty to start fresh.",
        examples=["demo-1"],
    )


class SourceReference(BaseModel):
    """A document passage that supports the answer."""

    source: str
    page: int | None = None
    chunk_index: int
    distance: float
    excerpt: str


class ChatResponse(BaseModel):
    """The assistant's answer plus where it came from."""

    session_id: str
    question: str
    standalone_question: str
    answer: str
    answered_from_documents: bool
    sources: list[SourceReference]


class ClearSessionResponse(BaseModel):
    """Confirmation that a conversation was forgotten."""

    session_id: str
    cleared: bool