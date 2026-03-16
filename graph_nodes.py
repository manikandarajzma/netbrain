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
    _IP_OR_CIDR_RE,
)

logger = logging.getLogger("atlas.graph_nodes")


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

_CONFIRMATIONS = {"yes", "yeah", "sure", "yep", "ok", "okay", "please", "go ahead", "do it", "show me", "yep please"}
_DISMISSALS = {"no", "nope", "nah", "no thanks", "don't", "dont", "never mind", "nevermind", "skip", "not now", "no need", "i'm good", "im good", "all good", "that's fine", "thats fine"}


async def classify_intent(state: AtlasState) -> dict[str, Any]:
    prompt = state["prompt"]
    prefilled = state.get("prefilled_tool_name")

    if prefilled and state.get("prefilled_tool_params") is not None:
        messages = _build_llm_messages(prompt, state.get("conversation_history") or [])
        return {"intent": "prefilled", "messages": messages, "iteration": 0}

    conversation_history = state.get("conversation_history") or []
    prompt_lower_strip = prompt.lower().strip().rstrip("!.")

    # last_follow_up_action is extracted once from history before graph entry (chat_service.py)
    follow_up_action = state.get("last_follow_up_action")
    follow_up_text = None
    if follow_up_action:
        # Find the matching follow_up text for display context
        for msg in reversed(conversation_history):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("{"):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(content, dict):
                if content.get("follow_up_action") == follow_up_action:
                    follow_up_text = content.get("follow_up")
                    break
                for r in (content.get("multi_results") or []):
                    if isinstance(r, dict) and r.get("follow_up_action") == follow_up_action:
                        follow_up_text = r.get("follow_up")
                        break
            if follow_up_text is not None:
                break

    if follow_up_action and follow_up_action.get("tool"):
        user_lower = prompt.lower().strip().rstrip("!.")
        # Pure dismissal → acknowledge and stop
        if user_lower in _DISMISSALS:
            return {"intent": "dismiss", "final_response": {"role": "assistant", "content": "Sure, let me know if you need anything else."}, "messages": [], "iteration": 0}
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
        # Only applies to short follow-up replies — not new queries with IPs or many words.
        user_wants_policies = any(w in user_lower for w in ("policies", "policy", "rules", "security rules", "firewall rules"))
        user_wants_members = any(w in user_lower for w in ("members", "contents", "what's in", "whats in"))
        is_short_followup = len(user_lower.split()) <= 10 and not _IP_OR_CIDR_RE.search(user_lower)

        if is_short_followup and (user_wants_policies or user_wants_members):
            fa_tool = follow_up_action.get("tool", "")
            fa_params = dict(follow_up_action.get("params", {}))

            # If the follow_up_action targets query_panorama_address_group_members we can
            # inject _policies_only when the user is asking for policies instead of members.
            if fa_tool == "query_panorama_address_group_members":
                # Always set explicitly — never inherit stale _policies_only from previous context
                fa_params["_policies_only"] = user_wants_policies
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

    # Short acknowledgement with no follow_up pending → dismiss rather than re-run a tool
    _ACKNOWLEDGEMENTS = _CONFIRMATIONS | _DISMISSALS | {"ok", "okay", "great", "thanks", "thank you", "cool", "got it", "noted", "perfect", "sounds good"}
    if prompt.lower().strip().rstrip("!.") in _ACKNOWLEDGEMENTS and not state.get("discover_only"):
        return {"intent": "dismiss", "final_response": {"role": "assistant", "content": "Sure, let me know if you need anything else."}, "messages": [], "iteration": 0}

    # Risk assessment: requires an IP + risk-related keywords → fan out to A2A agents
    _RISK_KEYWORDS = ("suspicious", "risk", "threat", "malicious", "deny", "blocked", "attack", "dangerous", "compromised")
    prompt_lower = prompt.lower()
    if (
        not state.get("discover_only")
        and _IP_OR_CIDR_RE.search(prompt)
        and any(kw in prompt_lower for kw in _RISK_KEYWORDS)
    ):
        return {"intent": "risk", "messages": [], "iteration": 0}

    # Path queries: two IPs, or path keywords + at least one IP → NetBrain agent
    _PATH_KEYWORDS = ("path", "route", "trace", "hops", "reach")
    if not state.get("discover_only"):
        ip_matches = _IP_OR_CIDR_RE.findall(prompt)
        if len(ip_matches) >= 2 or (
            len(ip_matches) >= 1 and any(kw in prompt_lower for kw in _PATH_KEYWORDS)
        ):
            return {"intent": "netbrain", "messages": [], "iteration": 0}

    if not state.get("discover_only") and _is_doc_query(prompt):
        intent = "doc"
    else:
        intent = "network"

    messages = _build_llm_messages(prompt, state.get("conversation_history") or [])
    return {"intent": intent, "messages": messages, "iteration": 0}


# ---------------------------------------------------------------------------
# Node: risk_orchestrator — fans out to Panorama + Splunk A2A agents
# ---------------------------------------------------------------------------

async def risk_orchestrator(state: AtlasState) -> dict[str, Any]:
    """Fan out to Panorama and Splunk A2A agents, synthesize with Ollama."""
    from agents.orchestrator import orchestrate_ip_risk
    prompt = state["prompt"]
    username = state.get("username")
    session_id = state.get("session_id")
    result = await orchestrate_ip_risk(prompt, username=username, session_id=session_id)
    return {"final_response": result}


# ---------------------------------------------------------------------------
# Node: netbrain_agent — delegates path queries to the NetBrain A2A agent
# ---------------------------------------------------------------------------

async def netbrain_agent(state: AtlasState) -> dict[str, Any]:
    """Send the path query to the NetBrain A2A agent and return its response."""
    import uuid
    import httpx

    prompt = state["prompt"]
    netbrain_url = "http://localhost:8004"

    task = {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": prompt}],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(netbrain_url, json=task)
            response.raise_for_status()
            data = response.json()
            artifacts = data.get("artifacts", [])
            text = None
            if artifacts:
                text = next(
                    (p.get("text") for p in artifacts[0].get("parts", []) if p.get("type") == "text"),
                    None,
                )
            if text:
                return {"final_response": {"role": "assistant", "content": {"direct_answer": text}}}
            return {"final_response": {"role": "assistant", "content": "NetBrain agent returned no data."}}
    except Exception as e:
        logger.warning("NetBrain agent call failed: %s", e)
        return {"final_response": {"role": "assistant", "content": f"NetBrain agent unavailable: {e}"}}


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
    normalized = _normalize_result(tool_name, result, state.get("prompt", ""))
    accumulated = list(state.get("accumulated_results") or [])
    accumulated.append(normalized)

    result_summary = json.dumps(result) if isinstance(result, dict) else str(result)
    # Truncate to avoid exceeding the model's context window on large Panorama results
    if len(result_summary) > 6000:
        result_summary = result_summary[:6000] + "... [truncated for context length]"
    updated_messages = messages + [ToolMessage(content=result_summary, tool_call_id=tool_call_id)]

    # Deterministic chaining: if IP lookup completed and user asked for members/policies,
    # call query_panorama_address_group_members directly — do not rely on LLM to chain.
    if tool_name == "query_panorama_ip_object_group" and isinstance(result, dict):
        prompt_lower_chain = (state.get("prompt") or "").lower()
        wants_members = any(w in prompt_lower_chain for w in ("members", "contents", "what's in", "whats in"))
        wants_policies = any(w in prompt_lower_chain for w in ("policies", "policy", "rules", "security rules", "firewall rules"))
        if wants_members or wants_policies:
            address_groups = result.get("address_groups", [])
            if address_groups:
                first_group = address_groups[0]
                group_name = first_group.get("name")
                device_group = first_group.get("device_group")
                if group_name:
                    members_params = {"address_group_name": group_name}
                    if device_group:
                        members_params["device_group"] = device_group
                    if wants_policies and not wants_members:
                        members_params["_policies_only"] = True
                    members_result = await call_mcp_tool(
                        "query_panorama_address_group_members",
                        members_params,
                        timeout=_TOOL_TIMEOUTS.get("query_panorama_address_group_members", 65.0),
                    )
                    if members_result and isinstance(members_result, dict) and "error" not in members_result:
                        members_normalized = _normalize_result("query_panorama_address_group_members", members_result, state.get("prompt", ""))
                        accumulated.append(members_normalized)
                        final = {"multi_results": accumulated} if len(accumulated) > 1 else members_normalized
                        return {"tool_raw_result": members_result, "accumulated_results": accumulated, "tool_error": None,
                                "final_response": {"role": "assistant", "content": final}, "messages": updated_messages, "iteration": iteration + 1}

    # If user asked for members + policies and this result already has both inline
    # (no follow_up_action needed), finalize immediately — don't let the LLM loop.
    prompt_lower_te = (state.get("prompt") or "").lower()
    if (
        tool_name == "query_panorama_address_group_members"
        and any(w in prompt_lower_te for w in ("policies", "policy"))
        and isinstance(normalized, dict)
        and not normalized.get("follow_up_action")
        and "members" in normalized
    ):
        final = {"multi_results": accumulated} if len(accumulated) > 1 else normalized
        return {"tool_raw_result": result, "accumulated_results": accumulated, "tool_error": None,
                "final_response": {"role": "assistant", "content": final}, "messages": updated_messages, "iteration": iteration + 1}

    # Stop auto-chaining when the result has a follow_up_action, unless the original
    # prompt explicitly asked for what this follow_up is offering.
    # e.g. "find group AND show members" → allow chaining to members step
    # but stop at members→policies unless prompt also asked for "policies".
    if isinstance(normalized, dict) and normalized.get("follow_up_action"):
        prompt_lower = (state.get("prompt") or "").lower()
        fu_lower = (normalized.get("follow_up") or "").lower()
        offered_members = "members" in fu_lower
        offered_policies = "policies" in fu_lower or "policy" in fu_lower
        user_asked_for_offer = (
            (offered_members and "members" in prompt_lower) or
            (offered_policies and any(w in prompt_lower for w in ("policies", "policy")))
        )
        if not user_asked_for_offer:
            final = {"multi_results": accumulated} if len(accumulated) > 1 else normalized
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


# ---------------------------------------------------------------------------
# Node 10: enrich_with_insights
# ---------------------------------------------------------------------------

def _extract_member_ips(result: dict) -> list[str]:
    """Extract host IPs from address group member objects."""
    ips = []
    for m in result.get("members", []):
        if not isinstance(m, dict):
            continue
        if m.get("type") not in ("ip-netmask", "ip-range"):
            continue
        val = m.get("value", "")
        ip = val.split("/")[0] if "/" in val else val.split("-")[0]
        if ip and ip not in ips:
            ips.append(ip)
    return ips


async def enrich_with_insights(state: AtlasState) -> dict[str, Any]:
    """Add proactive insights to the final response before it is returned."""
    from atlas.mcp_client import call_mcp_tool

    final = state.get("final_response")
    if not final:
        return {}
    content = final.get("content") if isinstance(final, dict) else None
    if not content or isinstance(content, str):
        return {}

    # Collect all result dicts to analyse
    if isinstance(content, dict) and content.get("multi_results"):
        results = [r for r in content["multi_results"] if isinstance(r, dict)]
    elif isinstance(content, dict):
        results = [content]
    else:
        return {}

    insights: list[str] = []

    for result in results:
        # --- Hint-only: no extra tool calls ---
        # Address group with members but no policies → may be unused
        if "members" in result and "policies" in result and not result.get("policies"):
            group = result.get("address_group_name", "This group")
            insights.append(f"'{group}' is not referenced by any security policies — it may be unused.")

        # --- Inline Splunk check for member IPs ---
        ips = _extract_member_ips(result)
        if not ips:
            continue
        # RBAC: only check Splunk if the user has access
        rbac_err = _check_tool_access(state.get("username"), "get_splunk_recent_denies", state.get("session_id"))
        if rbac_err:
            continue
        tasks = [
            call_mcp_tool("get_splunk_recent_denies", {"ip_address": ip}, timeout=25.0)
            for ip in ips[:3]
        ]
        splunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        for ip, sr in zip(ips, splunk_results):
            if isinstance(sr, dict) and sr.get("count", 0) > 0:
                count = sr["count"]
                insights.append(f"{ip} has {count} recent deny event{'s' if count != 1 else ''} in Splunk.")

    if not insights:
        return {}

    # Attach insights to content
    if isinstance(content, dict) and content.get("multi_results"):
        updated = list(content["multi_results"])
        last = dict(updated[-1]) if isinstance(updated[-1], dict) else updated[-1]
        if isinstance(last, dict):
            last["insights"] = insights
            updated[-1] = last
        new_content = dict(content)
        new_content["multi_results"] = updated
    else:
        new_content = dict(content)
        new_content["insights"] = insights

    return {"final_response": {"role": "assistant", "content": new_content}}
