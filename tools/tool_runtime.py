"""Shared runtime helpers for agent-facing tool modules."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

try:
    from atlas.services.status_service import status_service
except ImportError:
    from services.status_service import status_service  # type: ignore


def sid_from_config(config: RunnableConfig) -> str:
    return (config or {}).get("configurable", {}).get("session_id", "default")


async def push_status(session_id: str, message: str) -> None:
    await status_service.push(session_id, message)
