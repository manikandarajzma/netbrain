"""LangGraph invocation helpers for Atlas chat entrypoints."""
from __future__ import annotations

from typing import Any

try:
    from atlas.services.checkpointer_runtime import ensure_checkpointer
    from atlas.services.graph_payloads import build_graph_config, build_initial_state
except ImportError:
    from services.checkpointer_runtime import ensure_checkpointer  # type: ignore
    from services.graph_payloads import build_graph_config, build_initial_state  # type: ignore


async def invoke_atlas_graph(
    prompt: str,
    conversation_history: list[dict[str, str]],
    *,
    username: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    await ensure_checkpointer()
    from atlas.graph_builder import atlas_graph

    initial_state = build_initial_state(prompt, conversation_history, username, session_id)
    config = build_graph_config(session_id)
    return await atlas_graph.ainvoke(initial_state, config=config)
