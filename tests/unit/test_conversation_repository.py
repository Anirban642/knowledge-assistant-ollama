from app.repositories.conversation_repository import InMemoryConversationRepository


def make_repository(max_messages: int = 4, max_sessions: int = 2) -> InMemoryConversationRepository:
    return InMemoryConversationRepository(max_messages_per_session=max_messages, max_sessions=max_sessions)


def test_unknown_session_has_empty_history():
    assert make_repository().get_history("nope") == []


def test_history_keeps_order_and_roles():
    repository = make_repository()
    repository.append_exchange("s1", "question one", "answer one")

    history = repository.get_history("s1")

    assert [(message.role, message.content) for message in history] == [
        ("user", "question one"),
        ("assistant", "answer one"),
    ]


def test_history_keeps_only_the_most_recent_messages():
    repository = make_repository(max_messages=4)
    for number in range(1, 4):
        repository.append_exchange("s1", f"question {number}", f"answer {number}")

    contents = [message.content for message in repository.get_history("s1")]

    assert contents == ["question 2", "answer 2", "question 3", "answer 3"]


def test_least_recently_used_session_is_evicted():
    repository = make_repository(max_sessions=2)
    repository.append_exchange("a", "q", "a")
    repository.append_exchange("b", "q", "a")
    repository.append_exchange("a", "q2", "a2")  # touching "a" makes "b" the oldest
    repository.append_exchange("c", "q", "a")

    assert repository.get_history("b") == []
    assert repository.get_history("a") != []
    assert repository.get_history("c") != []


def test_clear_reports_whether_the_session_existed():
    repository = make_repository()
    repository.append_exchange("s1", "q", "a")

    assert repository.clear("s1") is True
    assert repository.clear("s1") is False
    assert repository.get_history("s1") == []