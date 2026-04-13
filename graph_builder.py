"""
Atlas LangGraph graph construction.

    classify_intent
        ├─► call_troubleshoot_agent   (connectivity / device-health investigation)
        ├─► call_network_ops_agent    (firewall requests, policy review, spreadsheets)
        └─► build_final_response      (dismiss / early-exit)
                │
               END

classify_intent classifies the prompt into one of three intents:
  "troubleshoot" — layered diagnostic investigation
  "network_ops"  — structured change / document workflow
  "dismiss"      — bare acknowledgement, skip LLM

Both agent nodes share tools/all_tools.py as the single tool registry.
Each gets a different system prompt scoped to its workflow.
"""

from langgraph.graph import StateGraph, END

try:
    from atlas.graph_state import AtlasState
    from atlas.graph_nodes import (
        classify_intent,
        call_troubleshoot_agent,
        call_network_ops_agent,
        build_final_response,
    )
except ImportError:
    from graph_state import AtlasState  # type: ignore[assignment]
    from graph_nodes import (           # type: ignore[assignment]
        classify_intent,
        call_troubleshoot_agent,
        call_network_ops_agent,
        build_final_response,
    )


def _route_intent(state: AtlasState) -> str:
    return state.get("intent") or "dismiss"


def build_graph(checkpointer=None):
    """
    Compile and return the Atlas LangGraph.

    Pass a LangGraph checkpointer (MemorySaver, RedisSaver, etc.) for
    conversation persistence across requests.
    """
    g = StateGraph(AtlasState)

    g.add_node("classify_intent",          classify_intent)
    g.add_node("call_troubleshoot_agent",  call_troubleshoot_agent)
    g.add_node("call_network_ops_agent",   call_network_ops_agent)
    g.add_node("build_final_response",     build_final_response)

    g.set_entry_point("classify_intent")

    g.add_conditional_edges(
        "classify_intent",
        _route_intent,
        {
            "troubleshoot": "call_troubleshoot_agent",
            "network_ops":  "call_network_ops_agent",
            "dismiss":      "build_final_response",
        },
    )

    g.add_edge("call_troubleshoot_agent", "build_final_response")
    g.add_edge("call_network_ops_agent",  "build_final_response")
    g.add_edge("build_final_response",    END)

    return g.compile(checkpointer=checkpointer)


atlas_graph = build_graph()
