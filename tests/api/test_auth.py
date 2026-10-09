"""Authentication and authorization behaviour of the API."""

import pytest

from tests.fakes import make_chunk

API = "/api/v1"
TXT_FILE = {"file": ("policy.txt", b"Employees get 12 days of paid sick leave per year.", "text/plain")}

PROTECTED_ENDPOINTS = [
    ("GET", f"{API}/documents", {}),
    ("POST", f"{API}/documents", {"files": TXT_FILE}),
    ("DELETE", f"{API}/documents/policy.txt", {}),
    ("POST", f"{API}/chat", {"json": {"question": "How many sick leave days do I get?"}}),
    ("DELETE", f"{API}/chat/sessions/demo-1", {}),
]


@pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED_ENDPOINTS)
def test_requests_without_a_key_are_rejected(anonymous_client, method, path, kwargs):
    response = anonymous_client.request(method, path, **kwargs)

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
    assert "Missing API key" in response.json()["error"]["message"]


@pytest.mark.parametrize(("method", "path", "kwargs"), PROTECTED_ENDPOINTS)
def test_requests_with_a_wrong_key_are_rejected(anonymous_client, method, path, kwargs):
    response = anonymous_client.request(method, path, headers={"X-API-Key": "wrong-key-0123456789"}, **kwargs)

    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid API key."


def test_health_endpoints_stay_public(anonymous_client):
    assert anonymous_client.get(f"{API}/health").status_code == 200
    assert anonymous_client.get(f"{API}/health/ready").status_code == 200


def test_user_key_can_chat_and_list_documents(user_client, chunk_repository):
    chunk_repository.search_results = [make_chunk(distance=0.2)]

    assert user_client.get(f"{API}/documents").status_code == 200
    chat = user_client.post(f"{API}/chat", json={"question": "How many sick leave days do I get?"})
    assert chat.status_code == 200


def test_user_key_cannot_upload_or_delete_documents(user_client):
    upload = user_client.post(f"{API}/documents", files=TXT_FILE)
    delete = user_client.delete(f"{API}/documents/policy.txt")

    for response in (upload, delete):
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"


def test_admin_key_can_manage_documents(client):
    upload = client.post(f"{API}/documents", files=TXT_FILE)
    assert upload.status_code == 201

    delete = client.delete(f"{API}/documents/policy.txt")
    assert delete.status_code == 200


def test_sessions_are_private_to_each_api_key(client, user_client, chunk_repository):
    chunk_repository.search_results = [make_chunk(distance=0.2)]
    client.post(f"{API}/chat", json={"question": "How many sick leave days do I get?", "session_id": "shared-1"})

    # Same session name, different key: the user cannot see or delete the admin's session.
    assert user_client.delete(f"{API}/chat/sessions/shared-1").status_code == 404
    assert client.delete(f"{API}/chat/sessions/shared-1").status_code == 200