"""Repository layer: stores chat history per session."""

from collections import OrderedDict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol


@dataclass(frozen=True)
class ConversationMessage:
    """One message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


class ConversationRepository(Protocol):
    """What the services need from a history store. Swap the implementation without touching them."""

    def get_history(self, session_id: str) -> list[ConversationMessage]:
        """Return the stored messages for a session, oldest first (empty if unknown)."""
        ...

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        """Record one question and its answer."""
        ...

    def clear(self, session_id: str) -> bool:
        """Delete a session's history. Returns False if the session did not exist."""
        ...


class InMemoryConversationRepository:
    """Keeps history in process memory. Fast and simple, but lost on restart and not shared across workers."""

    def __init__(self, max_messages_per_session: int, max_sessions: int) -> None:
        self._max_messages_per_session = max_messages_per_session
        self._max_sessions = max_sessions
        self._sessions: OrderedDict[str, deque[ConversationMessage]] = OrderedDict()
        self._lock = Lock()  # FastAPI runs sync routes in a thread pool, so access must be locked

    def get_history(self, session_id: str) -> list[ConversationMessage]:
        with self._lock:
            messages = self._sessions.get(session_id)
            return list(messages) if messages else []

    def append_exchange(self, session_id: str, question: str, answer: str) -> None:
        with self._lock:
            messages = self._sessions.get(session_id)
            if messages is None:
                # maxlen drops the oldest messages automatically once the limit is reached
                messages = deque(maxlen=self._max_messages_per_session)
                self._sessions[session_id] = messages
            messages.append(ConversationMessage(role="user", content=question))
            messages.append(ConversationMessage(role="assistant", content=answer))

            self._sessions.move_to_end(session_id)
            while len(self._sessions) > self._max_sessions:
                self._sessions.popitem(last=False)  # evict the least recently used session

    def clear(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None