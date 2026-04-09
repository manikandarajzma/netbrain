"""
Atlas LangGraph graph construction.

The graph has exactly three nodes in a linear path:

    classify_intent
        │
        ├─► troubleshoot_orchestrator  (all real queries)
        │
        └─► build_final_response  (dismiss / early-exit)
                │
               END

``classify_intent`` inspects the incoming prompt and either:
  - routes to ``troubleshoot_orchestrator`` for any genuine troubleshooting
    query (including pending-clarification replies), or
  - short-circuits to ``build_final_response`` for bare acknowledgements with
    nothing pending (e.g. "ok", "thanks") so the LLM never runs unnecessarily.

``troubleshoot_orchestrator`` is the heavy node.  It delegates to
``agents.orchestrator.orchestrate_troubleshoot``, which runs a full
``create_react_agent`` loop — the LLM picks tools, sees results, and reasons
until it can write a final answer.  This node may ask the user for
clarification instead (returning a ``final_response`` directly) when the
initial prompt is too vague.

``build_final_response`` is the terminal pass-through node.  By the time the
graph reaches it, ``state["final_response"]`` is always populated; the node
only handles the edge-case where an RBAC error was stored on the state.
"""

from langgraph.graph import StateGraph, END

try:
    from atlas.graph_state import AtlasState
    from atlas.graph_nodes import (
        classify_intent,
        troubleshoot_orchestrator,
        build_final_response,
    )
except ImportError:
    from graph_state import AtlasState  # type: ignore[assignment]
    from graph_nodes import (  # type: ignore[assignment]
        classify_intent,
        troubleshoot_orchestrator,
        build_final_response,
    )


# ---------------------------------------------------------------------------
# Conditional edge: where does classify_intent send us?
# ---------------------------------------------------------------------------

def _route_intent(state: AtlasState) -> str:
    """
    Read the ``intent`` key set by ``classify_intent`` and return the name of
    the next node to execute.

    Possible values
    ---------------
    "troubleshoot"
        The prompt looks like a genuine network problem — hand it to the
        orchestrator.
    "dismiss"
        Bare acknowledgement with no pending context — skip the LLM and go
        straight to the terminal node (which will return a canned reply).
    """
    return state.get("intent", "troubleshoot")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """
    Compile and return the Atlas LangGraph.

    Parameters
    ----------
    checkpointer:
        Optional LangGraph checkpointer (e.g. Redis-backed) for conversation
        persistence.  When ``None`` the graph runs without persistence — fine
        for development; the web server replaces this on first request via
        ``chat_service._ensure_checkpointer()``.

    Returns
    -------
    CompiledGraph
        The compiled LangGraph ready to call with ``ainvoke()``.
    """
    g = StateGraph(AtlasState)

    # Register the three nodes
    g.add_node("classify_intent", classify_intent)
    g.add_node("troubleshoot_orchestrator", troubleshoot_orchestrator)
    g.add_node("build_final_response", build_final_response)

    # Entry point
    g.set_entry_point("classify_intent")

    # classify_intent → either troubleshoot or immediate dismiss
    g.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "troubleshoot": "troubleshoot_orchestrator",
            "dismiss": "build_final_response",
        },
    )

    # Both paths terminate at build_final_response → END
    g.add_edge("troubleshoot_orchestrator", "build_final_response")
    g.add_edge("build_final_response", END)

    return g.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Compiled without a checkpointer initially.
# ``chat_service._ensure_checkpointer()`` hot-swaps this with a Redis-backed
# instance on first use so that conversation history survives across requests.
atlas_graph = build_graph()
