"""Pending clarification state backed by Redis with in-memory fallback."""
from __future__ import annotations

import json

_PENDING_TS_MEM: dict[str, str] = {}
_PENDING_TS_TTL = 600


def set_pending_context(session_id: str, prompt: str, issue_type: str = "general") -> None:
    payload = json.dumps({"prompt": prompt, "issue_type": issue_type})
    try:
        import os
        import redis as _redis

        _redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        ).setex(f"atlas:pending_ts:{session_id}", _PENDING_TS_TTL, payload)
        return
    except Exception:
        pass
    _PENDING_TS_MEM[session_id] = payload


def get_pending_context(session_id: str) -> tuple[str | None, str | None]:
    try:
        import os
        import redis as _redis

        raw = _redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        ).get(f"atlas:pending_ts:{session_id}")
    except Exception:
        raw = _PENDING_TS_MEM.get(session_id)
    if not raw:
        return None, None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("prompt", ""), data.get("issue_type", "general")
        return str(data), "general"
    except Exception:
        return str(raw), "general"


def has_pending_context(session_id: str) -> bool:
    try:
        import os
        import redis as _redis

        return (
            _redis.from_url(
                os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                decode_responses=True,
            ).exists(f"atlas:pending_ts:{session_id}")
            > 0
        )
    except Exception:
        pass
    return session_id in _PENDING_TS_MEM


def clear_pending_context(session_id: str) -> None:
    try:
        import os
        import redis as _redis

        _redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        ).delete(f"atlas:pending_ts:{session_id}")
        return
    except Exception:
        pass
    _PENDING_TS_MEM.pop(session_id, None)

