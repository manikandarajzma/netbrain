"""
Atlas LangGraph nodes.
All logic is extracted from chat_service.py — no new behaviour introduced.
"""
import asyncio
import json
import logging
from typing import Any

from atlas.graph_state import AtlasState
from atlas.chat_service import (
    _is_doc_query,
    _build_llm_messages,
    _to_openai_tool,
    _fetch_mcp_tools,
    _normalize_result,
    _check_tool_access,
    _TOOL_TIMEOUTS,
    TOOL_DISPLAY_NAMES,
    get_tool_display_name,
    MAX_AGENT_ITERATIONS,
)

logger = logging.getLogger("atlas.graph_nodes")


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

_CONFIRMATIONS = {"yes", "yeah", "sure", "yep", "ok", "okay", "please", "go ahead", "do it", "show me", "yep please"}


async def classify_intent(state: AtlasState) -> dict[str, Any]:
    prompt = state["prompt"]
    prefilled = state.get("prefilled_tool_name")

    if prefilled and state.get("prefilled_tool_params") is not None:
        messages = _build_llm_messages(prompt, state.get("conversation_history") or [])
        return {"intent": "prefilled", "messages": messages, "iteration": 0}

    # If the last assistant message offered a follow-up, classify user's reply
    conversation_history = state.get("conversation_history") or []
    follow_up_action = None
    follow_up_text = None
    for msg in reversed(conversation_history):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("{"):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(content, dict):
                follow_up_action = content.get("follow_up_action")
                follow_up_text = content.get("follow_up")
                if not follow_up_action and content.get("multi_results"):
                    for r in reversed(content["multi_results"]):
                        if isinstance(r, dict) and r.get("follow_up_action"):
                            follow_up_action = r["follow_up_action"]
                            follow_up_text = r.get("follow_up", "")
                            break
            break  # only check most recent assistant message

    if follow_up_action and follow_up_action.get("tool"):
        user_lower = prompt.lower().strip().rstrip("!.")
        # Pure confirmation → execute follow-up directly
        if user_lower in _CONFIRMATIONS:
            messages = _build_llm_messages(prompt, conversation_history)
            return {
                "intent": "prefilled",
                "prefilled_tool_name": follow_up_action["tool"],
                "prefilled_tool_params": follow_up_action.get("params", {}),
                "messages": messages,
                "iteration": 0,
            }
        # User asks for something adjacent to the offer, e.g. "no. find the policies though"
        # when the offer was "see members?" — or "give me the members" when offer was "see policies?"
        user_wants_policies = any(w in user_lower for w in ("policies", "policy"))
        user_wants_members = "members" in user_lower

        if user_wants_policies or user_wants_members:
            fa_tool = follow_up_action.get("tool", "")
            fa_params = dict(follow_up_action.get("params", {}))

            # If the follow_up_action targets query_panorama_address_group_members we can
            # inject _policies_only when the user is asking for policies instead of members.
            if fa_tool == "query_panorama_address_group_members":
                if user_wants_policies:
                    fa_params["_policies_only"] = True
                # user_wants_members: params already correct (no _policies_only)
                messages = _build_llm_messages(prompt, conversation_history)
                return {
                    "intent": "prefilled",
                    "prefilled_tool_name": fa_tool,
                    "prefilled_tool_params": fa_params,
                    "messages": messages,
                    "iteration": 0,
                }

            # Generic: follow_up offered something and user is referencing it by concept
            if follow_up_text:
                fu_lower = follow_up_text.lower()
                offered_policies = "policies" in fu_lower or "policy" in fu_lower
                offered_members = "members" in fu_lower
                if (user_wants_policies and offered_policies) or (user_wants_members and offered_members):
                    messages = _build_llm_messages(prompt, conversation_history)
                    return {
                        "intent": "prefilled",
                        "prefilled_tool_name": fa_tool,
                        "prefilled_tool_params": fa_params,
                        "messages": messages,
                        "iteration": 0,
                    }
        # Anything else → fall through to normal routing with full history

    if not state.get("discover_only") and _is_doc_query(prompt):
        intent = "doc"
    else:
        intent = "network"

    messages = _build_llm_messages(prompt, state.get("conversation_history") or [])
    return {"intent": intent, "messages": messages, "iteration": 0}


# ---------------------------------------------------------------------------
# Node 2: check_rbac
# ---------------------------------------------------------------------------

async def check_rbac(state: AtlasState) -> dict[str, Any]:
    tool_name = state.get("selected_tool_name") or state.get("prefilled_tool_name")
    username = state.get("username")
    session_id = state.get("session_id")
    err = _check_tool_access(username, tool_name, session_id) if tool_name else None
    return {"rbac_error": err}


# ---------------------------------------------------------------------------
# Node 3: fetch_mcp_tools
# ---------------------------------------------------------------------------

async def fetch_mcp_tools(state: AtlasState) -> dict[str, Any]:
    # Just a pass-through — tools are fetched inside tool_selector
    # This node exists as a named step for graph clarity
    return {}


# ---------------------------------------------------------------------------
# Node 4: tool_selector  (LLM picks a tool)
# ---------------------------------------------------------------------------

async def tool_selector(state: AtlasState) -> dict[str, Any]:
    from langchain_openai import ChatOpenAI

    mcp_tools = await _fetch_mcp_tools()
    if not mcp_tools:
        return {"final_response": {"role": "assistant", "content": "Could not connect to MCP server."}}

    openai_tools = [_to_openai_tool(t) for t in mcp_tools]

    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    llm = ChatOpenAI(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, api_key="docker")

    iteration = state.get("iteration", 0)
    tool_choice = "required" if iteration == 0 else "auto"
    llm_with_tools = llm.bind_tools(openai_tools, tool_choice=tool_choice)
    messages = state["messages"]

    try:
        ai_msg = await asyncio.wait_for(llm_with_tools.ainvoke(messages), timeout=90.0)
    except asyncio.TimeoutError:
        return {"final_response": {"role": "assistant", "content": "Tool selection timed out. Please try again."}}

    # LLM chose to stop — use its text as the final answer
    if not ai_msg.tool_calls:
        accumulated = state.get("accumulated_results") or []
        if len(accumulated) > 1:
            return {"final_response": {"role": "assistant", "content": {"multi_results": accumulated}}}
        if len(accumulated) == 1:
            return {"final_response": {"role": "assistant", "content": accumulated[0]}}
        return {"final_response": {"role": "assistant", "content": ai_msg.content or "I could not determine how to answer that."}}

    tool_call = ai_msg.tool_calls[0]
    sel_tool_name = tool_call["name"]
    tool_args = {k: v for k, v in dict(tool_call["args"]).items() if v not in ({}, "")}
    tool_call_id = tool_call.get("id") or f"call_{sel_tool_name}_{iteration}"

    # discover_only: return tool metadata without executing
    if state.get("discover_only"):
        return {
            "final_response": {
                "tool_name": sel_tool_name,
                "parameters": tool_args,
                "tool_display_name": get_tool_display_name(sel_tool_name),
                "intent": tool_args.get("intent"),
                "format": "table",
            }
        }

    updated_messages = list(messages) + [ai_msg]
    return {
        "selected_tool_name": sel_tool_name,
        "selected_tool_args": tool_args,
        "tool_call_id": tool_call_id,
        "messages": updated_messages,
    }


# ---------------------------------------------------------------------------
# Node 5: doc_tool_caller
# ---------------------------------------------------------------------------

async def doc_tool_caller(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client import call_mcp_tool
    prompt = state["prompt"]
    result = await call_mcp_tool("search_documentation", {"query": prompt}, timeout=30.0)
    if result and not (isinstance(result, dict) and "error" in result):
        return {"final_response": {"role": "assistant", "content": result}}
    # Fall through to network path by setting intent
    return {"intent": "network", "tool_error": "doc search returned nothing"}


# ---------------------------------------------------------------------------
# Node 6: prefilled_tool_executor
# ---------------------------------------------------------------------------

async def prefilled_tool_executor(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client import call_mcp_tool
    from atlas.mcp_client_tool_selection import synthesize_final_answer

    tool_name = state["prefilled_tool_name"]
    tool_params = dict(state["prefilled_tool_params"] or {})
    prompt = state["prompt"]

    # Internal flag: caller wants policies from this group, not members
    policies_only = tool_params.pop("_policies_only", False)

    result = await call_mcp_tool(tool_name, tool_params, timeout=_TOOL_TIMEOUTS.get(tool_name, 65.0))

    if isinstance(result, dict) and result.get("requires_site"):
        return {"final_response": {"role": "assistant", "content": result}}
    if result is None or (isinstance(result, dict) and not result):
        msg = await synthesize_final_answer(prompt, tool_name, "No result returned.")
        return {"final_response": {"role": "assistant", "content": msg}}
    if isinstance(result, dict) and "error" in result:
        msg = await synthesize_final_answer(prompt, tool_name, result)
        return {"final_response": {"role": "assistant", "content": msg}}

    # policies_only: show policies from the group result instead of members
    if policies_only and isinstance(result, dict):
        group_name = result.get("address_group_name", tool_params.get("address_group_name", "this group"))
        policies = result.get("policies", [])
        if policies:
            count = len(policies)
            return {"final_response": {"role": "assistant", "content": {
                "direct_answer": f"Found {count} polic{'ies' if count != 1 else 'y'} referencing '{group_name}'",
                "policies": policies,
            }}}
        return {"final_response": {"role": "assistant", "content": {
            "direct_answer": f"No policies found referencing '{group_name}'.",
        }}}

    return {"final_response": {"role": "assistant", "content": _normalize_result(tool_name, result, prompt)}}


# ---------------------------------------------------------------------------
# Node 7: tool_executor
# ---------------------------------------------------------------------------

async def tool_executor(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client import call_mcp_tool
    from langchain_core.messages import ToolMessage

    tool_name = state["selected_tool_name"]
    tool_args = state.get("selected_tool_args") or {}
    tool_call_id = state.get("tool_call_id") or f"call_{tool_name}"
    iteration = state.get("iteration", 0)
    messages = state["messages"]

    result = await call_mcp_tool(tool_name, tool_args, timeout=_TOOL_TIMEOUTS.get(tool_name, 65.0))

    if isinstance(result, dict) and result.get("requires_site"):
        return {"final_response": {"role": "assistant", "content": result}}

    if result is None or (isinstance(result, dict) and not result):
        error = "No result returned. Check that the MCP server is running."
        updated_messages = messages + [ToolMessage(content=error, tool_call_id=tool_call_id)]
        return {"tool_error": error, "messages": updated_messages, "iteration": iteration + 1}

    if isinstance(result, dict) and "error" in result:
        updated_messages = messages + [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)]
        return {"tool_error": result, "messages": updated_messages, "iteration": iteration + 1}

    # Success: normalize result and accumulate
    result_summary = json.dumps(result) if isinstance(result, dict) else str(result)
    updated_messages = messages + [ToolMessage(content=result_summary, tool_call_id=tool_call_id)]
    normalized = _normalize_result(tool_name, result, state.get("prompt", ""))
    accumulated = list(state.get("accumulated_results") or [])
    accumulated.append(normalized)

    # If this result has a follow_up_action, stop here — don't let the LLM auto-chain the next step
    if isinstance(normalized, dict) and normalized.get("follow_up_action"):
        if len(accumulated) > 1:
            final = {"multi_results": accumulated}
        else:
            final = normalized
        return {"tool_raw_result": result, "accumulated_results": accumulated, "tool_error": None,
                "final_response": {"role": "assistant", "content": final}, "messages": updated_messages, "iteration": iteration + 1}

    return {"tool_raw_result": result, "accumulated_results": accumulated, "tool_error": None, "messages": updated_messages, "iteration": iteration + 1}


# ---------------------------------------------------------------------------
# Node 8: normalize_result
# ---------------------------------------------------------------------------

async def normalize_result(state: AtlasState) -> dict[str, Any]:
    tool_name = state["selected_tool_name"]
    result = state["tool_raw_result"]
    prompt = state.get("prompt", "")
    normalized = _normalize_result(tool_name, result, prompt)
    return {"final_response": {"role": "assistant", "content": normalized}}


# ---------------------------------------------------------------------------
# Node 9: synthesize_error
# ---------------------------------------------------------------------------

async def synthesize_error(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client_tool_selection import synthesize_final_answer
    tool_name = state.get("selected_tool_name") or state.get("prefilled_tool_name") or "tool"
    error = state.get("tool_error") or "Unknown error"
    prompt = state.get("prompt", "")
    msg = await synthesize_final_answer(prompt, tool_name, error)
    return {"final_response": {"role": "assistant", "content": msg}}
