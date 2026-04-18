"""
Atlas LangGraph graph construction.

    classify_intent
        ├─► dispatch_agent          (generic workflow dispatch for routed agents)
        └─► build_final_response      (dismiss / early-exit)
                │
               END
"""

from langgraph.graph import StateGraph, END

try:
    from atlas.graph.graph_state import AtlasState
    from atlas.graph.graph_nodes import (
        build_final_response,
        classify_intent,
        dispatch_agent,
    )
except ImportError:
    from graph.graph_state import AtlasState  # type: ignore[assignment]
    from graph.graph_nodes import (  # type: ignore[assignment]
        build_final_response,
        classify_intent,
        dispatch_agent,
    )


class GraphBuilder:
    """Owns Atlas graph compilation and the current compiled graph instance."""

    def __init__(self) -> None:
        self._graph = self.build()

    def _route_intent(self, state: AtlasState) -> str:
        intent = str(state.get("intent") or "").strip()
        return "build_final_response" if intent == "dismiss" else "dispatch_agent"

    def build(self, checkpointer=None):
        """
        Compile and return the Atlas LangGraph.

        Pass a LangGraph checkpointer (MemorySaver, RedisSaver, etc.) for
        conversation persistence across requests.
        """
        g = StateGraph(AtlasState)

        g.add_node("classify_intent", classify_intent)
        g.add_node("dispatch_agent", dispatch_agent)
        g.add_node("build_final_response", build_final_response)

        g.set_entry_point("classify_intent")

        g.add_conditional_edges(
            "classify_intent",
            self._route_intent,
            {
                "dispatch_agent": "dispatch_agent",
                "build_final_response": "build_final_response",
            },
        )

        g.add_edge("dispatch_agent", "build_final_response")
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
