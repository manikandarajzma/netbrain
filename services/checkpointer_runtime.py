"""Checkpointer lifecycle helpers for Atlas graph execution."""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("atlas.checkpointer_runtime")

_checkpointer_lock = asyncio.Lock()
_checkpointer_ready = False
_checkpointer_state = "pending"
_checkpointer_error: str | None = None


async def ensure_checkpointer() -> None:
    """
    Build and wire the AsyncRedisSaver into atlas_graph on first call.

    If Redis is unavailable the graph keeps running without persistence.
    """
    global _checkpointer_ready, _checkpointer_state, _checkpointer_error
    if _checkpointer_ready:
        return
    async with _checkpointer_lock:
        if _checkpointer_ready:
            return
        try:
            import atlas.graph_builder as _gb
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver

            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            cp = AsyncRedisSaver(
                redis_url=redis_url,
                ttl={"default_collection_ttl": 86400},
            )
            await cp.asetup()
            _gb.atlas_graph = _gb.build_graph(cp)
            _checkpointer_ready = True
            _checkpointer_state = "enabled"
            _checkpointer_error = None
            logger.info(
                "Atlas graph re-compiled with AsyncRedisSaver (thread_id=%s, Redis=%s)",
                "session_id",
                redis_url,
            )
        except Exception as exc:
            logger.warning(
                "Could not initialise Redis checkpointer (%s) — "
                "running without conversation state persistence",
                exc,
            )
            _checkpointer_ready = True
            _checkpointer_state = "disabled"
            _checkpointer_error = str(exc)


def get_checkpointer_status() -> dict[str, bool | str | None]:
    """Return checkpointer lifecycle state for diagnostics."""
    labels = {
        "pending": "Pending first graph run",
        "enabled": "Enabled",
        "disabled": "Disabled (running without Redis persistence)",
    }
    return {
        "ready": _checkpointer_state == "enabled",
        "state": _checkpointer_state,
        "label": labels.get(_checkpointer_state, _checkpointer_state),
        "error": _checkpointer_error,
    }
