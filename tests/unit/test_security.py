import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.core.security import build_client_id, hash_api_key, matches_any_key

KEY = "a-sufficiently-long-secret-key"


def test_matching_is_exact():
    assert matches_any_key(KEY, [KEY]) is True
    assert matches_any_key(KEY, ["other-key-aaaaaaaaaaaaaaaa", KEY]) is True
    assert matches_any_key(KEY + "x", [KEY]) is False
    assert matches_any_key("", [KEY]) is False


def test_no_configured_keys_means_nothing_matches():
    assert matches_any_key(KEY, []) is False


def test_client_id_is_stable_short_and_does_not_reveal_the_key():
    client_id = build_client_id(KEY)

    assert client_id == build_client_id(KEY)
    assert len(client_id) == 12
    assert client_id != build_client_id(KEY + "x")
    assert KEY not in client_id
    assert hash_api_key(KEY).startswith(client_id)


def test_weak_api_keys_are_rejected_at_startup():
    with pytest.raises(ValidationError, match="at least 16 characters"):
        Settings(_env_file=None, admin_api_keys=["short"])