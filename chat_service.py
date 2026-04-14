"""
Atlas chat service.

Thin entry point that wires a user HTTP request into the LangGraph pipeline:

    process_message()
        └─► _ensure_checkpointer()   (Redis-backed conversation persistence)
        └─► atlas_graph.ainvoke()    (classify → troubleshoot → respond)
        └─► returns final_response dict

The graph itself lives in ``graph_builder.py``.  All reasoning happens inside
``agents/orchestrator.py`` via LangGraph's ``create_react_agent``.

``_IP_OR_CIDR_RE`` is exported for reuse in ``graph_nodes.py``.
"""
import asyncio
import logging
import os
import re
from typing import Any

logger = logging.getLogger("atlas.chat_service")

# ---------------------------------------------------------------------------
# Shared regex — imported by graph_nodes.py to detect IP/CIDR in prompts
# ---------------------------------------------------------------------------

_IP_OR_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")


# ---------------------------------------------------------------------------
# Redis checkpointer — lazily initialised once the event loop is running.
#
# LangGraph's AsyncRedisSaver uses get_running_loop() in __init__, so it must
# be constructed *after* the asyncio event loop starts (i.e. inside an async
# context).  We defer initialisation to the first call to process_message()
# and guard it with a double-checked lock so it only runs once.
# ---------------------------------------------------------------------------

_checkpointer_lock = asyncio.Lock()
_checkpointer_ready = False


async def _ensure_checkpointer() -> None:
    """
    Build and wire the AsyncRedisSaver into atlas_graph on first call.

    Replaces the module-level ``atlas_graph`` (compiled without a checkpointer)
    with one that persists conversation state to Redis keyed by ``session_id``
    (= LangGraph ``thread_id``).  This gives multi-turn memory: each follow-up
    question within the same browser session can see prior graph state.

    If Redis is unavailable the graph keeps running without persistence — the
    warning is logged once and the flag is set so we do not retry on every
    request.
    """
    global _checkpointer_ready
    if _checkpointer_ready:
        return
    async with _checkpointer_lock:
        if _checkpointer_ready:
            return  # double-checked locking — another coroutine beat us here
        try:
            import atlas.graph_builder as _gb
            from langgraph.checkpoint.redis.aio import AsyncRedisSaver
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            cp = AsyncRedisSaver(
                redis_url=redis_url,
                ttl={"default_collection_ttl": 86400},  # 24-hour TTL
            )
            await cp.asetup()
            _gb.atlas_graph = _gb.build_graph(cp)
            _checkpointer_ready = True
            logger.info(
                "Atlas graph re-compiled with AsyncRedisSaver "
                "(thread_id=%s, Redis=%s)", "session_id", redis_url,
            )
        except Exception as exc:
            logger.warning(
                "Could not initialise Redis checkpointer (%s) — "
                "running without conversation state persistence", exc,
            )
            _checkpointer_ready = True  # don't retry; accept stateless mode


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _build_initial_state(
    prompt: str,
    conversation_history: list[dict[str, str]],
    username: str | None,
    session_id: str | None,
) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "conversation_history": conversation_history or [],
        "username": username,
        "session_id": session_id,
        "intent": None,
        "rbac_error": None,
        "final_response": None,
    }


def _build_graph_config(session_id: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {"recursion_limit": 50}
    if session_id:
        # thread_id ties this invocation to the session's checkpoint slot in Redis
        config["configurable"] = {"thread_id": session_id}
    return config


async def _invoke_atlas_graph(
    prompt: str,
    conversation_history: list[dict[str, str]],
    *,
    username: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    await _ensure_checkpointer()
    from atlas.graph_builder import atlas_graph

    initial_state = _build_initial_state(prompt, conversation_history, username, session_id)
    config = _build_graph_config(session_id)
    return await atlas_graph.ainvoke(initial_state, config=config)


def _extract_final_response(result_state: dict[str, Any]) -> dict[str, Any]:
    return result_state.get("final_response") or {
        "role": "assistant",
        "content": "Something went wrong — please try again.",
    }

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
    result_state = await _invoke_atlas_graph(
        prompt,
        conversation_history,
        username=username,
        session_id=session_id,
    )
    return _extract_final_response(result_state)
