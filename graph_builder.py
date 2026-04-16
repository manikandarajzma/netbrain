"""
Atlas LangGraph graph construction.

    classify_intent
        ├─► call_troubleshoot_agent   (connectivity / device-health investigation)
        ├─► call_network_ops_agent    (incident, change, and operational workflows)
        └─► build_final_response      (dismiss / early-exit)
                │
               END
"""

from langgraph.graph import StateGraph, END

try:
    from atlas.graph_state import AtlasState
    from atlas.graph_nodes import (
        build_final_response,
        call_network_ops_agent,
        call_troubleshoot_agent,
        classify_intent,
    )
except ImportError:
    from graph_state import AtlasState  # type: ignore[assignment]
    from graph_nodes import (  # type: ignore[assignment]
        build_final_response,
        call_network_ops_agent,
        call_troubleshoot_agent,
        classify_intent,
    )


class GraphBuilder:
    """Owns Atlas graph compilation and the current compiled graph instance."""

    def __init__(self) -> None:
        self._graph = self.build()

    def _route_intent(self, state: AtlasState) -> str:
        return state.get("intent") or "dismiss"

    def build(self, checkpointer=None):
        """
        Compile and return the Atlas LangGraph.

        Pass a LangGraph checkpointer (MemorySaver, RedisSaver, etc.) for
        conversation persistence across requests.
        """
        g = StateGraph(AtlasState)

        g.add_node("classify_intent", classify_intent)
        g.add_node("call_troubleshoot_agent", call_troubleshoot_agent)
        g.add_node("call_network_ops_agent", call_network_ops_agent)
        g.add_node("build_final_response", build_final_response)

        g.set_entry_point("classify_intent")

        g.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "troubleshoot": "call_troubleshoot_agent",
                "network_ops": "call_network_ops_agent",
                "dismiss": "build_final_response",
            },
        )

        g.add_edge("call_troubleshoot_agent", "build_final_response")
        g.add_edge("call_network_ops_agent", "build_final_response")
        g.add_edge("build_final_response", END)

        return g.compile(checkpointer=checkpointer)

    def set_graph(self, graph) -> None:
        self._graph = graph
        global atlas_graph
        atlas_graph = graph

    def get_graph(self):
        return self._graph


graph_builder = GraphBuilder()
atlas_graph = graph_builder.get_graph()
