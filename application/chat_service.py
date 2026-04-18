"""
Atlas chat service.

Thin entry point that wires a user HTTP request into the LangGraph pipeline:

    process_message()
        └─► atlas_application.process_query()

The graph itself lives in ``graph/graph_builder.py``. All reasoning happens
inside the specialized LangGraph ReAct agents.

``_IP_OR_CIDR_RE`` is exported for reuse in ``graph_nodes.py``.
"""
import re
from typing import Any

# ---------------------------------------------------------------------------
# Shared regex — imported by graph_nodes.py to detect IP/CIDR in prompts
# ---------------------------------------------------------------------------

_IP_OR_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")

try:
    from atlas.application.atlas_application import atlas_application
except ImportError:
    from application.atlas_application import atlas_application  # type: ignore

async def process_message(
    prompt: str,
    conversation_history: list[dict[str, str]],
    *,
    username: str | None = None,
    session_id: str | None = None,
    ui_action: dict[str, Any] | None = None,
    # Legacy kwargs accepted but ignored so callers don't need immediate updates
    **_ignored: Any,
) -> dict[str, Any]:
    """
    Process one user message through the Atlas LangGraph pipeline.

    Parameters
    ----------
    prompt:
        The user's raw text input for this turn.
    conversation_history:
        List of prior ``{"role": "user"|"assistant", "content": ...}`` dicts.
        Up to the last 10 are forwarded to the LLM for multi-turn context.
    username:
        Authenticated username from the session cookie.  Passed to the
        orchestrator for ServiceNow ticket attribution.
    session_id:
        Opaque browser-session ID.  Used as the LangGraph ``thread_id`` so
        the Redis checkpointer stores per-session conversation state.

    Returns
    -------
    dict
        Always a ``{"role": "assistant", "content": ...}`` dict, optionally
        with a ``path_hops`` key for the PathVisualization frontend component.
    """
    kwargs: dict[str, Any] = {
        "username": username,
        "session_id": session_id,
    }
    if ui_action is not None:
        kwargs["ui_action"] = ui_action
    return await atlas_application.process_query(
        prompt,
        conversation_history,
        **kwargs,
    )
