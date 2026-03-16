"""
Chat service: native LLM tool calling via LangChain bind_tools().
The LLM selects and calls tools directly from their MCP schemas — no custom routing logic.
"""
import asyncio
import json
import logging
import re
from typing import Any

logger = logging.getLogger("atlas.chat_service")


# ---------------------------------------------------------------------------
# Scope check
# ---------------------------------------------------------------------------

_IP_OR_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")


def _is_doc_query(prompt: str) -> bool:
    """Return True if the query is a documentation/how-it-works question.

    Small local LLMs often output tool calls as plain text instead of structured
    function calls when there are many tools. Documentation queries are detected
    here and routed directly to search_documentation, bypassing LLM tool selection.
    """
    if not prompt:
        return False
    # Queries with IPs are always network queries, never docs
    if _IP_OR_CIDR_RE.search(prompt):
        return False
    lower = prompt.lower()
    # Action keywords that indicate a live network query (not a doc question)
    # Product names alone (netbrain, panorama, splunk) are NOT excluded —
    # "end to end flow for netbrain" is a doc query even though it says "netbrain".
    network_action_kw = any(k in lower for k in (
        "firewall rule", "security rule", "address group",
        "path from", "path to", "can reach",
        "orphan", "unused object", "address object",
        "deny event", "recent deny",
    ))
    if network_action_kw:
        return False
    # Documentation question patterns
    doc_kw = any(k in lower for k in (
        "how does", "how do", "how is", "how are",
        "what is", "what are", "what does",
        "explain", "describe", "tell me about",
        "end to end", "end-to-end", "flow for", "flow of",
        "authentication", "session", "rbac", "tool access",
        "add a new tool", "add a tool", "add a new domain",
        "troubleshoot", "troubleshooting",
        "documentation", "how it works", "how atlas",
    ))
    return doc_kw


# ---------------------------------------------------------------------------
# Tool display names + timeouts
# ---------------------------------------------------------------------------

TOOL_DISPLAY_NAMES: dict[str, str] = {
    "check_path_allowed": "NetBrain",
    "query_network_path": "NetBrain",
    "query_panorama_ip_object_group": "Panorama",
    "query_panorama_address_group_members": "Panorama",
    "find_unused_panorama_objects": "Panorama",
    "get_splunk_recent_denies": "Splunk",
}

_TOOL_TIMEOUTS: dict[str, float] = {
    "query_network_path": 385.0,
    "check_path_allowed": 370.0,
    "get_splunk_recent_denies": 95.0,
    "query_panorama_ip_object_group": 65.0,
    "query_panorama_address_group_members": 65.0,
    "find_unused_panorama_objects": 180.0,
}

MAX_AGENT_ITERATIONS = 3


def get_tool_display_name(tool_name: str) -> str:
    return TOOL_DISPLAY_NAMES.get(tool_name, tool_name)


# ---------------------------------------------------------------------------
# MCP -> LangChain tool format conversion
# ---------------------------------------------------------------------------

def _to_openai_tool(t) -> dict:
    """Convert an MCP tool object to OpenAI function-calling format for LangChain bind_tools()."""
    name = t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
    desc = t.get("description") if isinstance(t, dict) else getattr(t, "description", None)
    schema = t.get("inputSchema") if isinstance(t, dict) else getattr(t, "inputSchema", None)
    # Include description up to Args: section (keeps Use for, Do NOT use for, Examples)
    if desc:
        args_idx = desc.find("\n    Args:")
        if args_idx > 0:
            desc = desc[:args_idx].strip()
        desc = desc[:600]
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc or "",
            "parameters": schema or {"type": "object", "properties": {}},
        },
    }


_mcp_tools_cache: list | None = None  # reset on server restart


async def _fetch_mcp_tools() -> list:
    """Fetch the tool list from the MCP server, cached for the process lifetime."""
    global _mcp_tools_cache
    if _mcp_tools_cache is not None:
        return _mcp_tools_cache
    from atlas.mcp_client import get_mcp_session
    async for client in get_mcp_session():
        tools_result = await client.list_tools()
        result = tools_result if isinstance(tools_result, list) else (getattr(tools_result, "tools", None) or [])
        if result:
            _mcp_tools_cache = result
        return result
    return []


def _load_skill(name: str) -> str:
    """Load a skill file from the skills/ directory relative to this file."""
    import pathlib
    skills_dir = pathlib.Path(__file__).parent / "skills"
    path = skills_dir / name
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def _build_llm_messages(prompt: str, conversation_history: list) -> list:
    """Convert prompt + conversation history to LangChain messages."""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    system_content = _load_skill("base.md")
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in ("group", "members", "policies", "policy", "panorama", "address", "orphan", "unused", "stale")):
        panorama_skill = _load_skill("panorama_agent.md")
        if panorama_skill:
            system_content += "\n\n" + panorama_skill
    logger.info("Loaded system prompt (%d chars): %s", len(system_content), system_content[:200])
    messages = [
        SystemMessage(content=system_content)
    ]
    for msg in (conversation_history or [])[-10:]:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # Normalise: parse JSON strings so the same logic applies whether the
        # frontend sent a dict or JSON.stringify()'d it.
        if isinstance(content, str) and content.startswith("{"):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                pass
        if isinstance(content, dict):
            if content.get("requires_site"):
                rack = content.get("rack", "rack")
                content = f"Asked user which site for rack '{rack}'."
            elif content.get("multi_results"):
                # Summarise each result in the chain so the LLM has context
                summaries = []
                for r in content["multi_results"]:
                    if isinstance(r, dict):
                        part = r.get("direct_answer") or r.get("message") or ""
                        fup = r.get("follow_up") or ""
                        summaries.append((part + " " + fup).strip())
                content = " | ".join(s for s in summaries if s) or "[Results shown above]"
            else:
                # Replace structured result with a short note so the model cannot
                # answer the next query from memory instead of calling a tool.
                # Include direct_answer and follow_up so the LLM has context for
                # short replies like "yes" that refer back to the previous question.
                parts = []
                if content.get("direct_answer"):
                    parts.append(content["direct_answer"])
                if content.get("follow_up"):
                    parts.append(content["follow_up"])
                content = " ".join(parts) if parts else (content.get("message") or content.get("reason") or "[Result shown above]")
        if role == "user":
            messages.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            messages.append(AIMessage(content=str(content)))
    messages.append(HumanMessage(content=prompt))
    return messages


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------

def _strip_l2_noise(result: dict[str, Any]) -> dict[str, Any]:
    """Remove noisy path status messages like 'L2 connections has not been discovered'."""
    noise = ["l2 connections has not been discovered", "l2 connection has not been discovered"]
    for key in ("path_status_description", "statusDescription"):
        val = result.get(key)
        if isinstance(val, str) and any(p in val.lower() for p in noise):
            result[key] = ""
    return result


def _normalize_result(
    tool_name: str,
    result: dict[str, Any] | str | None,
    prompt: str = "",
) -> dict[str, Any] | str | None:
    """Apply display normalizations to a successful tool result."""
    if result is None or (isinstance(result, dict) and len(result) == 0):
        return result
    if isinstance(result, dict) and result.get("path_hops"):
        result = dict(result)
        _strip_l2_noise(result)
    if tool_name == "get_splunk_recent_denies" and isinstance(result, dict):
        if result.get("count") == 0 and "error" not in result:
            ip = result.get("ip_address", "this IP")
            result = dict(result)
            result["message"] = (
                f"No deny events found for {ip} in the last 24 hours. "
                "Try a different IP or time range, or check that Splunk has Palo Alto logs for that period."
            )
    if tool_name == "query_panorama_address_group_members" and isinstance(result, dict) and "error" not in result:
        members = result.get("members", [])
        group_name = result.get("address_group_name", "this group")
        prompt_lower = prompt.lower()
        wants_policies = any(w in prompt_lower for w in ("policies", "policy"))
        if members:
            count = len(members)
            direct_answer = f"Address group '{group_name}' contains {count} member{'s' if count != 1 else ''}"
            result = dict(result)
            result.pop("direct_answer", None)
            if wants_policies:
                # User explicitly asked for policies — include them inline, no follow_up needed
                new_result = {"direct_answer": direct_answer}
                for k, v in result.items():
                    new_result[k] = v
                return new_result
            result.pop("policies", None)
            new_result = {"direct_answer": direct_answer}
            for k, v in result.items():
                new_result[k] = v
            new_result["follow_up"] = f"Would you like to see the policies that reference '{group_name}'?"
            fu_params: dict = {"address_group_name": group_name, "_policies_only": True}
            if result.get("device_group"):
                fu_params["device_group"] = result["device_group"]
            new_result["follow_up_action"] = {"tool": "query_panorama_address_group_members", "params": fu_params}
            return new_result

    if tool_name == "find_unused_panorama_objects" and isinstance(result, dict) and "error" not in result:
        orphaned = len(result.get("orphaned_address_objects", []))
        unused = len(result.get("unused_address_groups", []))
        result = dict(result)
        result["direct_answer"] = (
            f"Found {orphaned} orphaned address object{'s' if orphaned != 1 else ''} "
            f"and {unused} unused address group{'s' if unused != 1 else ''}"
        )
        return result

    if tool_name == "query_panorama_ip_object_group" and isinstance(result, dict) and prompt and "error" not in result:
        queried_ip = result.get("queried_ip") or result.get("ip_address")
        if not queried_ip:
            ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b", prompt)
            if ip_match:
                queried_ip = ip_match.group(0)
        if "message" in result and result.get("address_groups"):
            result = dict(result)
            result.pop("message", None)
        if queried_ip and result.get("address_groups"):
            address_groups = result.get("address_groups", [])
            group_names = [ag.get("name") for ag in address_groups if ag.get("name")]
            if group_names:
                if len(group_names) == 1:
                    direct_answer = f"{queried_ip} is part of address group '{group_names[0]}'"
                else:
                    groups_str = "', '".join(group_names)
                    direct_answer = f"{queried_ip} is part of address groups: '{groups_str}'"
                address_objects = result.get("address_objects", [])
                if address_objects:
                    network_names = [obj.get("name") for obj in address_objects if obj.get("name")]
                    if network_names:
                        direct_answer += f" (via {', '.join(network_names)})"
                # Return only group-level info — members/policies shown separately via query_panorama_address_group_members
                group_name_hint = group_names[0] if len(group_names) == 1 else group_names[0]
                # Pass device_group so query_panorama_address_group_members can find it
                first_group = next((ag for ag in address_groups if ag.get("name") == group_name_hint), address_groups[0])
                follow_up_params = {"address_group_name": group_name_hint}
                if first_group.get("device_group"):
                    follow_up_params["device_group"] = first_group["device_group"]
                prompt_lower = prompt.lower()
                already_asked_members = "members" in prompt_lower
                already_asked_policies = any(w in prompt_lower for w in ("policies", "policy"))
                result_out = {
                    "direct_answer": direct_answer,
                    "address_groups": address_groups,
                    "queried_ip": queried_ip,
                }
                if not already_asked_members and not already_asked_policies:
                    result_out["follow_up"] = f"Would you like to see the members of '{group_name_hint}'?"
                    result_out["follow_up_action"] = {"tool": "query_panorama_address_group_members", "params": follow_up_params}
                return result_out

    return result


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------

def _check_tool_access(username: str | None, tool_name: str, session_id: str | None = None) -> str | None:
    """Return an error message if the user's group forbids tool_name, else None.

    Group is resolved from the signed session cookie (session_id) when available,
    which is the normal path for browser requests. The username fallback is used
    when no session cookie is present and always resolves to 'guest' (no access),
    so it effectively blocks unauthenticated tool calls.

    This check runs on every tool call and cannot be bypassed by prompt injection
    because it reads the group from the server-signed cookie, not from user input.
    """
    if username is None:
        return None
    from atlas.auth import get_group_for_session, get_user_group, get_allowed_tools
    group = get_group_for_session(session_id) if session_id else get_user_group(username)
    allowed = get_allowed_tools(group)
    if allowed is not None and tool_name not in allowed:
        display = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        return f"Your group ({group}) does not have access to {display} queries."
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _extract_last_follow_up_action(conversation_history: list) -> dict | None:
    """Scan conversation history (newest first) and return the most recent follow_up_action."""
    for msg in reversed(conversation_history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.startswith("{"):
            try:
                content = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                continue
        if isinstance(content, dict):
            fa = content.get("follow_up_action")
            if fa:
                return fa
            if content.get("multi_results"):
                for r in reversed(content["multi_results"]):
                    if isinstance(r, dict) and r.get("follow_up_action"):
                        return r["follow_up_action"]
    return None


async def process_message(
    prompt: str,
    conversation_history: list[dict[str, str]],
    *,
    default_live: bool = True,
    discover_only: bool = False,
    tool_name: str | None = None,
    parameters: dict[str, Any] | None = None,
    max_iterations: int | None = None,
    username: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Process one user message via the Atlas LangGraph agent.
    """
    from atlas.graph_builder import atlas_graph

    last_follow_up_action = _extract_last_follow_up_action(conversation_history or [])

    initial_state = {
        "prompt": prompt,
        "conversation_history": conversation_history or [],
        "username": username,
        "session_id": session_id,
        "discover_only": discover_only,
        "prefilled_tool_name": tool_name,
        "prefilled_tool_params": parameters,
        "max_iterations": max_iterations or MAX_AGENT_ITERATIONS,
        "intent": None,
        "rbac_error": None,
        "messages": [],
        "selected_tool_name": None,
        "selected_tool_args": None,
        "tool_call_id": None,
        "iteration": 0,
        "tool_raw_result": None,
        "accumulated_results": [],
        "requires_site": False,
        "tool_error": None,
        "final_response": None,
        "last_follow_up_action": last_follow_up_action,
    }

    result_state = await atlas_graph.ainvoke(initial_state)
    return result_state.get("final_response") or {"role": "assistant", "content": "Something went wrong. Please try again."}
