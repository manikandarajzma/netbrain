"""
Atlas chat service.

Thin entry point that wires a user HTTP request into the LangGraph pipeline:

    process_message()
        └─► services.graph_runtime.invoke_atlas_graph()
        └─► services.graph_runtime.extract_final_response()

The graph itself lives in ``graph_builder.py``.  All reasoning happens inside
``agents/orchestrator.py`` via LangGraph's ``create_react_agent``.

``_IP_OR_CIDR_RE`` is exported for reuse in ``graph_nodes.py``.
"""
import re
from typing import Any

# ---------------------------------------------------------------------------
# Shared regex — imported by graph_nodes.py to detect IP/CIDR in prompts
# ---------------------------------------------------------------------------

_IP_OR_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")

try:
    from atlas.services.graph_invoker import invoke_atlas_graph
    from atlas.services.graph_payloads import extract_final_response
except ImportError:
    from services.graph_invoker import invoke_atlas_graph  # type: ignore
    from services.graph_payloads import extract_final_response  # type: ignore

async def process_message(
    prompt: str,
    conversation_history: list[dict[str, str]],
    *,
    username: str | None = None,
    session_id: str | None = None,
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
    result_state = await invoke_atlas_graph(
        prompt,
        conversation_history,
        username=username,
        session_id=session_id,
    )
    return extract_final_response(result_state)
