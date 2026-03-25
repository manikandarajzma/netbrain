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
    planner_node,
    tool_selector,
    doc_tool_caller,
    prefilled_tool_executor,
    tool_executor,
    normalize_result,
    synthesize_error,
    enrich_with_insights,
    risk_orchestrator,
    netbrain_agent,
    troubleshoot_orchestrator,
    servicenow_agent,
)
from atlas.chat_service import MAX_AGENT_ITERATIONS

_MAX_SUCCESS_ITERATIONS = 5


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
    iteration = state.get("iteration", 0)
    if state.get("tool_error") is None:
        if iteration >= _MAX_SUCCESS_ITERATIONS:
            return "done"
        return "success"
    if iteration < MAX_AGENT_ITERATIONS:
        return "retry"
    return "error"


def route_after_rbac_prefilled(state: AtlasState) -> str:
    return "blocked" if state.get("rbac_error") else "execute"


# ---------------------------------------------------------------------------
# Terminal node: packages final_response as passthrough
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    """Terminal node — final_response is already set, or synthesise from accumulated results."""
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    if not state.get("final_response"):
        accumulated = state.get("accumulated_results") or []
        if len(accumulated) > 1:
            return {"final_response": {"role": "assistant", "content": {"multi_results": accumulated}}}
        if len(accumulated) == 1:
            return {"final_response": {"role": "assistant", "content": accumulated[0]}}
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None) -> Any:
    g = StateGraph(AtlasState)

    # Add nodes
    g.add_node("classify_intent", classify_intent)
    g.add_node("check_rbac_prefilled", check_rbac)
    g.add_node("prefilled_tool_executor", prefilled_tool_executor)
    g.add_node("doc_tool_caller", doc_tool_caller)
    g.add_node("fetch_mcp_tools", fetch_mcp_tools)
    g.add_node("planner_node", planner_node)
    g.add_node("tool_selector", tool_selector)
    g.add_node("check_rbac", check_rbac)
    g.add_node("tool_executor", tool_executor)
    g.add_node("normalize_result", normalize_result)
    g.add_node("synthesize_error", synthesize_error)
    g.add_node("enrich_with_insights", enrich_with_insights)
    g.add_node("risk_orchestrator", risk_orchestrator)
    g.add_node("netbrain_agent", netbrain_agent)
    g.add_node("troubleshoot_orchestrator", troubleshoot_orchestrator)
    g.add_node("servicenow_agent", servicenow_agent)
    g.add_node("build_final_response", build_final_response)

    # Entry point
    g.set_entry_point("classify_intent")

    # Route by intent
    g.add_conditional_edges("classify_intent", route_intent, {
        "prefilled": "check_rbac_prefilled",
        "doc": "doc_tool_caller",
        "network": "fetch_mcp_tools",
        "dismiss": "build_final_response",
        "risk": "risk_orchestrator",
        "netbrain": "netbrain_agent",
        "troubleshoot": "troubleshoot_orchestrator",
        "servicenow": "servicenow_agent",
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

    # Network path: fetch_mcp_tools → planner_node → tool_selector → ...
    g.add_edge("fetch_mcp_tools", "planner_node")
    g.add_edge("planner_node", "tool_selector")
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
        "success": "tool_selector",    # feed result back to LLM — follows its plan
        "retry": "tool_selector",      # back-edge: retry with error context
        "error": "synthesize_error",
    })
    g.add_edge("risk_orchestrator", "build_final_response")
    g.add_edge("netbrain_agent", "build_final_response")
    g.add_edge("troubleshoot_orchestrator", "build_final_response")
    g.add_edge("servicenow_agent", "build_final_response")
    g.add_edge("normalize_result", "build_final_response")
    g.add_edge("synthesize_error", "build_final_response")
    g.add_edge("enrich_with_insights", "build_final_response")
    g.add_edge("build_final_response", END)

    return g.compile(checkpointer=checkpointer)


# Singleton — compiled without checkpointer initially.
# chat_service._ensure_checkpointer() replaces this with a Redis-backed instance on first use.
atlas_graph = build_graph()
