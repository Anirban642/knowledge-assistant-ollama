"""Security helpers: roles, safe API key comparison and a log-friendly client identifier."""

import hashlib
import hmac
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

MIN_API_KEY_LENGTH = 16


class Role(str, Enum):
    """What a client is allowed to do."""

    USER = "user"    # ask questions, list documents
    ADMIN = "admin"  # everything a user can do, plus upload and delete documents


@dataclass(frozen=True)
class AuthenticatedClient:
    """The identity behind a valid API key. The key itself is never stored here."""

    client_id: str
    role: Role


def hash_api_key(api_key: str) -> str:
    """One-way SHA-256 hash of a key."""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def build_client_id(api_key: str) -> str:
    """A short, stable ID that is safe to log and to use as a session namespace."""
    return hash_api_key(api_key)[:12]


def matches_any_key(candidate: str, valid_keys: Iterable[str]) -> bool:
    """Check a key against every valid key in constant time (no early exit, no timing leaks)."""
    candidate_hash = hash_api_key(candidate)
    matched = False
    for valid_key in valid_keys:
        if hmac.compare_digest(candidate_hash, hash_api_key(valid_key)):
            matched = True
    return matched