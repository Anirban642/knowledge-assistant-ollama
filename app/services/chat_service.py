"""Service layer: runs one chat turn by combining conversation history with the RAG pipeline."""

import uuid
from dataclasses import dataclass

from app.core.exceptions import NotFoundError
from app.core.logging import get_logger
from app.repositories.conversation_repository import ConversationRepository
from app.services.question_rewriter import QuestionRewriter
from app.services.rag_service import RagAnswer, RagService

logger = get_logger(__name__)


@dataclass(frozen=True)
class ChatResult:
    """Everything the API needs to build a chat response."""

    session_id: str
    standalone_question: str
    rag_answer: RagAnswer


class ChatService:
    """Handles a user's question inside a conversation session."""

    def __init__(
        self,
        rag_service: RagService,
        question_rewriter: QuestionRewriter,
        conversation_repository: ConversationRepository,
    ) -> None:
        self._rag_service = rag_service
        self._question_rewriter = question_rewriter
        self._conversations = conversation_repository

    def ask(self, question: str, session_id: str | None) -> ChatResult:
        """Answer the question, remembering it for follow-ups. A new session is created when none is given."""
        active_session_id = session_id or uuid.uuid4().hex

        history = self._conversations.get_history(active_session_id)
        standalone_question = self._question_rewriter.rewrite(question, history)
        rag_answer = self._rag_service.answer(standalone_question)

        # Saved only after a successful answer, so a failed turn never pollutes the history.
        self._conversations.append_exchange(active_session_id, question, rag_answer.answer)

        return ChatResult(
            session_id=active_session_id,
            standalone_question=standalone_question,
            rag_answer=rag_answer,
        )

    def clear_session(self, session_id: str) -> None:
        """Forget a conversation. Raises NotFoundError if the session does not exist."""
        if not self._conversations.clear(session_id):
            raise NotFoundError(f"Chat session '{session_id}' was not found.")
        logger.info("Cleared chat session '%s'", session_id)