"""Opaque development bearer tokens map to server-configured identities only."""

import hmac
from dataclasses import dataclass
from uuid import UUID

from fastapi import HTTPException


@dataclass(frozen=True)
class Principal:
    user_id: UUID


def authenticate(authorization: str | None, tokens: dict[str, UUID]) -> Principal:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            401, "Authentication required", headers={"WWW-Authenticate": "Bearer"}
        )
    supplied = authorization[7:]
    if len(supplied) > 256:
        raise HTTPException(401, "Invalid development token")
    for token, user_id in tokens.items():
        if hmac.compare_digest(supplied.encode(), token.encode()):
            return Principal(user_id)
    raise HTTPException(
        401, "Invalid development token", headers={"WWW-Authenticate": "Bearer"}
    )
