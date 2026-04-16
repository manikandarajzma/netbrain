"""Owned pending clarification state backed by Redis with in-memory fallback."""
from __future__ import annotations

import json
import os


class PendingContextStore:
    """Owns pending clarification storage for multi-turn follow-up handling."""

    def __init__(self, *, ttl_seconds: int = 600) -> None:
        self._mem: dict[str, str] = {}
        self._ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"atlas:pending_ts:{session_id}"

    def _redis_client(self):
        try:
            import redis as _redis

            return _redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            )
        except Exception:
            return None

    def set(self, session_id: str, prompt: str, issue_type: str = "general") -> None:
        payload = json.dumps({"prompt": prompt, "issue_type": issue_type})
        client = self._redis_client()
        if client is not None:
            try:
                client.setex(self._key(session_id), self._ttl_seconds, payload)
                return
            except Exception:
                pass
        self._mem[session_id] = payload

    def get(self, session_id: str) -> tuple[str | None, str | None]:
        client = self._redis_client()
        if client is not None:
            try:
                raw = client.get(self._key(session_id))
            except Exception:
                raw = self._mem.get(session_id)
        else:
            raw = self._mem.get(session_id)
        if not raw:
            return None, None
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data.get("prompt", ""), data.get("issue_type", "general")
            return str(data), "general"
        except Exception:
            return str(raw), "general"

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


pending_context_store = PendingContextStore()
