"""
Atlas LangGraph graph construction.
"""
from typing import Any
from langgraph.graph import StateGraph, END

from atlas.graph_state import AtlasState
from atlas.graph_nodes import (
    classify_intent,
    check_rbac,
    fetch_mcp_tools,
    tool_selector,
    doc_tool_caller,
    prefilled_tool_executor,
    tool_executor,
    normalize_result,
    synthesize_error,
    enrich_with_insights,
)
from atlas.chat_service import MAX_AGENT_ITERATIONS


# ---------------------------------------------------------------------------
# Conditional edge functions
# ---------------------------------------------------------------------------

def route_intent(state: AtlasState) -> str:
    return state.get("intent", "network")


def route_rbac(state: AtlasState) -> str:
    return "blocked" if state.get("rbac_error") else "allowed"


def route_after_doc(state: AtlasState) -> str:
    if state.get("final_response"):
        return "done"
    return "network"  # fall through to LLM


def route_after_tool_selector(state: AtlasState) -> str:
    if state.get("final_response"):
        return "done"
    return "check_rbac"


def route_after_tool_executor(state: AtlasState) -> str:
    if state.get("final_response"):
        return "done"
    if state.get("tool_error") is None:
        return "success"
    iteration = state.get("iteration", 0)
    if iteration < MAX_AGENT_ITERATIONS:
        return "retry"
    return "error"


def route_after_rbac_prefilled(state: AtlasState) -> str:
    return "blocked" if state.get("rbac_error") else "execute"


# ---------------------------------------------------------------------------
# Terminal node: packages final_response as passthrough
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    """Terminal node — final_response is already set, just pass through."""
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> Any:
    g = StateGraph(AtlasState)

    # Add nodes
    g.add_node("classify_intent", classify_intent)
    g.add_node("check_rbac_prefilled", check_rbac)
    g.add_node("prefilled_tool_executor", prefilled_tool_executor)
    g.add_node("doc_tool_caller", doc_tool_caller)
    g.add_node("fetch_mcp_tools", fetch_mcp_tools)
    g.add_node("tool_selector", tool_selector)
    g.add_node("check_rbac", check_rbac)
    g.add_node("tool_executor", tool_executor)
    g.add_node("normalize_result", normalize_result)
    g.add_node("synthesize_error", synthesize_error)
    g.add_node("enrich_with_insights", enrich_with_insights)
    g.add_node("build_final_response", build_final_response)

    # Entry point
    g.set_entry_point("classify_intent")

    # Route by intent
    g.add_conditional_edges("classify_intent", route_intent, {
        "prefilled": "check_rbac_prefilled",
        "doc": "doc_tool_caller",
        "network": "fetch_mcp_tools",
        "dismiss": "build_final_response",
    })

    # Prefilled path
    g.add_conditional_edges("check_rbac_prefilled", route_after_rbac_prefilled, {
        "blocked": "build_final_response",
        "execute": "prefilled_tool_executor",
    })
    g.add_edge("prefilled_tool_executor", "build_final_response")

    # Doc path
    g.add_conditional_edges("doc_tool_caller", route_after_doc, {
        "done": "build_final_response",
        "network": "fetch_mcp_tools",
    })

    # Network path
    g.add_edge("fetch_mcp_tools", "tool_selector")
    g.add_conditional_edges("tool_selector", route_after_tool_selector, {
        "done": "enrich_with_insights",
        "check_rbac": "check_rbac",
    })
    g.add_conditional_edges("check_rbac", route_rbac, {
        "blocked": "build_final_response",
        "allowed": "tool_executor",
    })
    g.add_conditional_edges("tool_executor", route_after_tool_executor, {
        "done": "enrich_with_insights",
        "success": "tool_selector",    # feed result back to LLM — it may chain another tool or stop
        "retry": "tool_selector",      # back-edge: retry with error context
        "error": "synthesize_error",
    })
    g.add_edge("normalize_result", "build_final_response")
    g.add_edge("synthesize_error", "build_final_response")
    g.add_edge("enrich_with_insights", "build_final_response")
    g.add_edge("build_final_response", END)

    return g.compile()


# Singleton — compiled once on import
atlas_graph = build_graph()
