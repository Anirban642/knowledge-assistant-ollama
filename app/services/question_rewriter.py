"""Service layer: turns a follow-up question into a standalone question using the chat history."""

import re

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from app.core.logging import get_logger
from app.repositories.conversation_repository import ConversationMessage
from app.services.prompts import REWRITE_PROMPT

logger = get_logger(__name__)

_MAX_CHARS_PER_HISTORY_MESSAGE = 300  # keeps the prompt small so CPU inference stays fast

# Words that usually mean "this question points back at something said earlier".
_REFERENCE_WORDS = frozenset(
    {"it", "its", "that", "this", "these", "those", "they", "them", "their", "same", "above", "previous"}
)
# Openers typical of fragments such as "And for managers?" or "What about hotels?".
_FOLLOW_UP_OPENERS = ("and ", "or ", "but ", "also ", "what about", "how about")
_MAX_FRAGMENT_WORDS = 2  # e.g. "Hotels?" or "For managers?"


class QuestionRewriter:
    """Makes follow-up questions understandable on their own, so document search works."""

    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chain = REWRITE_PROMPT | chat_model | StrOutputParser()

    def rewrite(self, question: str, history: list[ConversationMessage]) -> str:
        """Return a standalone version of the question. Falls back to the original if anything looks off."""
        if not history:
            return question  # first question of a conversation: nothing to resolve

        if not self._looks_like_follow_up(question):
            logger.info("Question looks standalone; skipping rewrite")
            return question

        try:
            raw_output = self._chain.invoke({"history": self._format_history(history), "question": question})
        except Exception:
            logger.warning("Question rewriting failed; using the original question", exc_info=True)
            return question

        rewritten = self._clean(raw_output)
        if not rewritten or len(rewritten) > 2 * len(question) + 200:
            logger.warning("Discarded an unusable rewrite: %r", raw_output)
            return question

        logger.info("Rewrote question: %r -> %r", question, rewritten)
        return rewritten

    @staticmethod
    def _looks_like_follow_up(question: str) -> bool:
        """Cheap rule-based check: does this question seem to depend on earlier messages?"""
        lowered = question.lower().strip()
        words = re.findall(r"[a-z']+", lowered)
        return (
            lowered.startswith(_FOLLOW_UP_OPENERS)
            or len(words) <= _MAX_FRAGMENT_WORDS
            or any(word in _REFERENCE_WORDS for word in words)
        )

    @staticmethod
    def _format_history(history: list[ConversationMessage]) -> str:
        """Render history as 'User: ...' / 'Assistant: ...' lines, trimming long messages."""
        lines = []
        for message in history:
            speaker = "User" if message.role == "user" else "Assistant"
            content = message.content
            if len(content) > _MAX_CHARS_PER_HISTORY_MESSAGE:
                content = content[:_MAX_CHARS_PER_HISTORY_MESSAGE] + "..."
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _clean(raw_output: str) -> str:
        """Keep only the question itself: drop intro lines like 'Here is the question:' and quote marks."""
        lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
        candidates = [line for line in lines if not line.endswith(":")]
        if not candidates:
            return ""
        return candidates[0].strip("\"'` ")