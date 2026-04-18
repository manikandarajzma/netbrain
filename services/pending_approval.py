"""Owned pending approval state for network-ops write actions."""
from __future__ import annotations

import json
import os
from typing import Any


class PendingApprovalStore:
    """Stores one pending approval payload per session with Redis fallback."""

    def __init__(self, *, ttl_seconds: int = 1800) -> None:
        self._mem: dict[str, str] = {}
        self._ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"atlas:pending_approval:{session_id}"

    def _redis_client(self):
        try:
            import redis as _redis

            return _redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )
        except Exception:
            return None

    def set(self, session_id: str, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload)
        client = self._redis_client()
        if client is not None:
            try:
                client.setex(self._key(session_id), self._ttl_seconds, raw)
                return
            except Exception:
                pass
        self._mem[session_id] = raw

    def get(self, session_id: str) -> dict[str, Any] | None:
        client = self._redis_client()
        if client is not None:
            try:
                raw = client.get(self._key(session_id))
            except Exception:
                raw = self._mem.get(session_id)
        else:
            raw = self._mem.get(session_id)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def has(self, session_id: str) -> bool:
        client = self._redis_client()
        if client is not None:
            try:
                return client.exists(self._key(session_id)) > 0
            except Exception:
                pass
        return session_id in self._mem

    def clear(self, session_id: str) -> None:
        client = self._redis_client()
        if client is not None:
            try:
                client.delete(self._key(session_id))
                return
            except Exception:
                pass
        self._mem.pop(session_id, None)


pending_approval_store = PendingApprovalStore()
