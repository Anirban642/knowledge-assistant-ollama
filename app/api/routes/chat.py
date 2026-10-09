"""Chat routes: ask the assistant questions about the company documents."""

from fastapi import APIRouter, Depends, Path

from app.api.auth import authenticate
from app.api.dependencies import get_chat_service
from app.core.security import AuthenticatedClient
from app.repositories.chunk_repository import RetrievedChunk
from app.schemas.chat import (
    SESSION_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    ClearSessionResponse,
    SourceReference,
)
from app.services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])

_EXCERPT_LENGTH = 200


def _to_source_reference(chunk: RetrievedChunk) -> SourceReference:
    """Convert a retrieved chunk into the public source format."""
    return SourceReference(
        source=chunk.source,
        page=chunk.page,
        chunk_index=chunk.chunk_index,
        distance=round(chunk.distance, 3),
        excerpt=chunk.text[:_EXCERPT_LENGTH],
    )


@router.post("", response_model=ChatResponse, summary="Ask a question about the company documents")
def ask_question(
    request: ChatRequest,
    client: AuthenticatedClient = Depends(authenticate),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Answer using only the indexed documents. Send the same session_id to ask follow-up questions."""
    result = service.ask(request.question, request.session_id, client.client_id)
    answer = result.rag_answer
    return ChatResponse(
        session_id=result.session_id,
        question=request.question,
        standalone_question=result.standalone_question,
        answer=answer.answer,
        answered_from_documents=answer.answered_from_documents,
        sources=[_to_source_reference(chunk) for chunk in answer.sources],
    )


@router.delete("/sessions/{session_id}", response_model=ClearSessionResponse, summary="Forget a conversation")
def clear_session(
    session_id: str = Path(min_length=1, max_length=64, pattern=SESSION_ID_PATTERN),
    client: AuthenticatedClient = Depends(authenticate),
    service: ChatService = Depends(get_chat_service),
) -> ClearSessionResponse:
    """Delete the stored history of one of your conversation sessions."""
    service.clear_session(session_id, client.client_id)
    return ClearSessionResponse(session_id=session_id, cleared=True)