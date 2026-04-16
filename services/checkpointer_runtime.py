"""Owned checkpointer lifecycle for Atlas graph execution."""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger("atlas.checkpointer_runtime")


class CheckpointerRuntime:
    """Owns lazy graph recompilation with an optional Redis-backed checkpointer."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._ready = False
        self._state = "pending"
        self._error: str | None = None

    async def ensure_ready(self) -> None:
        """
        Build and wire the AsyncRedisSaver into the graph on first call.

        If Redis is unavailable the graph keeps running without persistence.
        """
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            try:
                try:
                    import atlas.graph.graph_builder as _gb
                except ImportError:
                    import graph.graph_builder as _gb  # type: ignore
                from langgraph.checkpoint.redis.aio import AsyncRedisSaver

                redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
                cp = AsyncRedisSaver(
                    redis_url=redis_url,
                    ttl={"default_collection_ttl": 86400},
                )
                await cp.asetup()
                _gb.graph_builder.set_graph(_gb.graph_builder.build(cp))
                self._ready = True
                self._state = "enabled"
                self._error = None
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
                self._ready = True
                self._state = "disabled"
                self._error = str(exc)

    def get_status(self) -> dict[str, bool | str | None]:
        """Return checkpointer lifecycle state for diagnostics."""
        labels = {
            "pending": "Pending first graph run",
            "enabled": "Enabled",
            "disabled": "Disabled (running without Redis persistence)",
        }
        return {
            "ready": self._state == "enabled",
            "state": self._state,
            "label": labels.get(self._state, self._state),
            "error": self._error,
        }


checkpointer_runtime = CheckpointerRuntime()
