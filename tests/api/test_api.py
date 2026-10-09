"""End-to-end API tests. Ollama and ChromaDB are replaced by fakes (see tests/conftest.py)."""

import re

from app.api.dependencies import get_chat_model
from app.main import app
from tests.fakes import FailingChatModel, make_chunk

API = "/api/v1"
POLICY_FILE = ("policy.txt", b"Employees get 12 days of paid sick leave per year.", "text/plain")


def upload(client, file=POLICY_FILE):
    return client.post(f"{API}/documents", files={"file": file})


def ask(client, question, session_id=None):
    payload = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    return client.post(f"{API}/chat", json=payload)


# ---------- health ----------

def test_liveness(client):
    response = client.get(f"{API}/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_when_everything_is_up(client):
    response = client.get(f"{API}/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert all(check["healthy"] for check in response.json()["checks"])


def test_readiness_returns_503_when_ollama_is_down(client, ollama_checker):
    ollama_checker.is_down = True

    response = client.get(f"{API}/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


# ---------- documents ----------

def test_upload_list_and_delete_a_document(client, settings):
    response = upload(client)
    assert response.status_code == 201
    body = response.json()
    assert body["source"] == "policy.txt"
    assert body["replaced_chunks"] == 0
    assert (settings.documents_dir / "policy.txt").is_file()

    listing = client.get(f"{API}/documents").json()
    assert listing == {
        "total_documents": 1,
        "documents": [{"source": "policy.txt", "chunk_count": body["chunks_stored"]}],
    }

    deleted = client.delete(f"{API}/documents/policy.txt")
    assert deleted.status_code == 200
    assert deleted.json()["file_deleted"] is True
    assert not (settings.documents_dir / "policy.txt").exists()

    again = client.delete(f"{API}/documents/policy.txt")
    assert again.status_code == 404
    assert again.json()["error"]["code"] == "not_found"


def test_uploading_the_same_file_again_replaces_its_chunks(client):
    first = upload(client).json()
    second = upload(client).json()

    assert second["replaced_chunks"] == first["chunks_stored"]


def test_upload_sanitizes_the_file_name(client):
    response = upload(client, ("My Policy (1).txt", b"hello world", "text/plain"))

    assert response.json()["source"] == "My_Policy__1_.txt"


def test_upload_rejects_unsupported_types(client):
    response = upload(client, ("virus.exe", b"data", "application/octet-stream"))

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "bad_request"


def test_upload_rejects_empty_files(client):
    response = upload(client, ("empty.txt", b"", "text/plain"))

    assert response.status_code == 400


def test_upload_rejects_files_over_the_limit(client):
    response = upload(client, ("big.txt", b"a" * (1024 * 1024 + 1), "text/plain"))

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "payload_too_large"


# ---------- chat ----------

def test_question_is_answered_with_sources(client, chunk_repository, chat_model):
    chunk_repository.search_results = [make_chunk(distance=0.2)]
    chat_model.script("You get 12 days of paid sick leave per year.")

    response = ask(client, "How many sick leave days do I get?")

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "You get 12 days of paid sick leave per year."
    assert body["answered_from_documents"] is True
    assert body["standalone_question"] == "How many sick leave days do I get?"
    assert re.fullmatch(r"[0-9a-f]{32}", body["session_id"])
    assert body["sources"][0]["source"] == "handbook.txt"
    assert body["sources"][0]["distance"] == 0.2


def test_out_of_scope_question_is_refused_without_calling_the_llm(client, chunk_repository, chat_model):
    chunk_repository.search_results = [make_chunk(distance=0.6)]

    response = ask(client, "What is the capital of France?")

    body = response.json()
    assert body["answered_from_documents"] is False
    assert body["sources"] == []
    assert chat_model.calls == 0


def test_follow_up_question_uses_the_session_history(client, chunk_repository, chat_model):
    chunk_repository.search_results = [make_chunk(distance=0.2)]
    rewritten = "What is the notice period for confirmed employees?"
    # LLM calls in order: answer 1, rewrite of the follow-up, answer 2
    chat_model.script("15 days.", rewritten, "60 days.")

    first = ask(client, "What is the notice period if I am on probation?", "test-1").json()
    second = ask(client, "And for confirmed employees?", "test-1").json()

    assert first["answer"] == "15 days."
    assert second["session_id"] == "test-1"
    assert second["standalone_question"] == rewritten
    assert second["answer"] == "60 days."
    assert chunk_repository.queries[-1] == rewritten


def test_clearing_a_session(client, chunk_repository):
    chunk_repository.search_results = [make_chunk(distance=0.2)]
    ask(client, "How many sick leave days do I get?", "test-1")

    cleared = client.delete(f"{API}/chat/sessions/test-1")
    assert cleared.status_code == 200
    assert cleared.json() == {"session_id": "test-1", "cleared": True}

    again = client.delete(f"{API}/chat/sessions/test-1")
    assert again.status_code == 404


def test_llm_outage_returns_a_clean_503(client, chunk_repository):
    chunk_repository.search_results = [make_chunk(distance=0.2)]
    app.dependency_overrides[get_chat_model] = lambda: FailingChatModel()

    response = ask(client, "How many sick leave days do I get?")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "external_service_unavailable"


# ---------- validation and error format ----------

def test_blank_question_is_a_validation_error(client):
    response = ask(client, "   ")

    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["fields"][0]["field"] == "body.question"


def test_invalid_session_id_is_a_validation_error(client):
    response = ask(client, "Hi", "bad id!")

    assert response.status_code == 422
    assert response.json()["error"]["fields"][0]["field"] == "body.session_id"


def test_errors_carry_a_request_id_matching_the_response_header(client):
    response = client.get(f"{API}/does-not-exist")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_error"
    assert error["request_id"] == response.headers["x-request-id"]