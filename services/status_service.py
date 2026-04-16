"""Owned status-bus service for UI progress updates."""
from __future__ import annotations


class StatusService:
    """Owns best-effort status delivery to the UI."""

    async def push(self, session_id: str, message: str) -> None:
        try:
            try:
                import atlas.status_bus as status_bus
            except ImportError:
                import status_bus  # type: ignore
            await status_bus.push(session_id, message)
        except Exception:
            pass


status_service = StatusService()
