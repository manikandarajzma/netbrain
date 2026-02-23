"""
Chat service: native LLM tool calling via LangChain bind_tools().
The LLM selects and calls tools directly from their MCP schemas — no custom routing logic.
"""
import asyncio
import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger("atlas.chat_service")


# ---------------------------------------------------------------------------
# Scope check
# ---------------------------------------------------------------------------

_IP_OR_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")


def _is_obviously_in_scope(prompt: str) -> bool:
    """Fast keyword check: return True if query clearly matches known tool patterns."""
    lower = (prompt or "").lower()
    has_ip = bool(_IP_OR_CIDR_RE.search(prompt or ""))
    panorama_kw = any(k in lower for k in (
        "object group", "address group", "panorama", "palo alto",
        "firewall rule", "security rule", "security policy",
        "device group", "address object", "ip group",
        "orphan", "unused", "not referenced", "cleanup panorama",
    ))
    path_kw = any(k in lower for k in (
        "network path", "path from", "path to", "traffic allowed",
        "path allowed", "can reach", "connectivity", "path",
    ))
    splunk_kw = any(k in lower for k in (
        "splunk", "deny", "denied", "denies", "firewall log",
        "recent deny", "deny event",
    ))
    if has_ip and (panorama_kw or path_kw or splunk_kw):
        return True
    if panorama_kw or path_kw or splunk_kw:
        return True
    if len(_IP_OR_CIDR_RE.findall(prompt or "")) >= 2:
        return True
    return False


def is_query_in_scope(prompt: str) -> Dict[str, Any]:
    """Keyword-only scope check — no LLM call."""
    if _is_obviously_in_scope(prompt):
        return {"in_scope": True, "reason": "Keyword match"}
    return {"in_scope": True, "reason": "Passed through to LLM"}


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


def _build_llm_messages(prompt: str, conversation_history: list) -> list:
    """Convert prompt + conversation history to LangChain messages."""
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    messages = [
        SystemMessage(content=(
            "You are a network infrastructure assistant. "
            "Always call a tool — never answer from memory or prior context. "
            "Tool selection rules: "
            "IP addresses → query_panorama_ip_object_group or get_splunk_recent_denies; "
            "address group names → query_panorama_address_group_members; "
            "orphaned/unused objects or address groups → find_unused_panorama_objects; "
            "path/connectivity queries → query_network_path or check_path_allowed. "
            "When the user's reply is short, check the conversation history "
            "to understand what they are clarifying and combine it with the original request."
        ))
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
            else:
                # Replace structured result with a short note so the model cannot
                # answer the next query from memory instead of calling a tool.
                content = content.get("message") or content.get("reason") or "[Result shown above]"
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
        if members:
            count = len(members)
            direct_answer = f"Address group '{group_name}' contains {count} member{'s' if count != 1 else ''}"
            result = dict(result)
            result.pop("direct_answer", None)
            new_result = {"direct_answer": direct_answer}
            for k, v in result.items():
                new_result[k] = v
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
                result = dict(result)
                result.pop("direct_answer", None)
                new_result = {"direct_answer": direct_answer}
                for k, v in result.items():
                    new_result[k] = v
                return new_result

    return result


# ---------------------------------------------------------------------------
# RBAC
# ---------------------------------------------------------------------------

def _check_tool_access(username: str | None, tool_name: str, session_id: str | None = None) -> str | None:
    """Return an error message if the user's role forbids tool_name, else None."""
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
    Process one user message using native LLM tool calling via bind_tools().
    The LLM receives MCP tool schemas directly and selects the appropriate tool.
    """
    from atlas.mcp_client_tool_selection import synthesize_final_answer
    from atlas.mcp_client import call_mcp_tool

    if max_iterations is None:
        max_iterations = MAX_AGENT_ITERATIONS

    # Pre-filled tool: skip discovery, execute directly
    if tool_name and parameters is not None:
        access_err = _check_tool_access(username, tool_name)
        if access_err:
            return {"role": "assistant", "content": access_err}
        result = await call_mcp_tool(tool_name, parameters, timeout=_TOOL_TIMEOUTS.get(tool_name, 65.0))
        if isinstance(result, dict) and result.get("requires_site"):
            return {"role": "assistant", "content": result}
        if result is None or (isinstance(result, dict) and not result):
            msg = await synthesize_final_answer(prompt, tool_name, "No result returned.")
            return {"role": "assistant", "content": msg}
        if isinstance(result, dict) and "error" in result:
            msg = await synthesize_final_answer(prompt, tool_name, result)
            return {"role": "assistant", "content": msg}
        return {"role": "assistant", "content": _normalize_result(tool_name, result, prompt)}

    # Fetch tools from MCP server and build LLM with bind_tools()
    mcp_tools = await _fetch_mcp_tools()
    if not mcp_tools:
        return {"role": "assistant", "content": "Could not connect to MCP server."}

    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    from langchain_ollama import ChatOllama
    openai_tools = [_to_openai_tool(t) for t in mcp_tools]
    llm = ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
    )
    llm_with_tools = llm.bind_tools(openai_tools, tool_choice="required")
    messages = _build_llm_messages(prompt, conversation_history)

    last_tool_name: str | None = None
    last_error: str | dict | None = None

    for iteration in range(max_iterations):
        try:
            ai_msg = await asyncio.wait_for(llm_with_tools.ainvoke(messages), timeout=90.0)
        except asyncio.TimeoutError:
            return {"role": "assistant", "content": "Tool selection timed out. Please try again."}

        if not ai_msg.tool_calls:
            return {"role": "assistant", "content": ai_msg.content or "I could not determine how to answer that."}

        tool_call = ai_msg.tool_calls[0]
        sel_tool_name = tool_call["name"]
        tool_args = dict(tool_call["args"])
        # Strip invalid optional args — llama3.1:8b sends {} or "" instead of omitting them
        tool_args = {k: v for k, v in tool_args.items() if v not in ({}, "")}
        tool_call_id = tool_call.get("id") or f"call_{sel_tool_name}_{iteration}"
        last_tool_name = sel_tool_name

        if discover_only:
            return {
                "tool_name": sel_tool_name,
                "parameters": tool_args,
                "tool_display_name": get_tool_display_name(sel_tool_name),
                "intent": tool_args.get("intent"),
                "format": "table",
            }

        access_err = _check_tool_access(username, sel_tool_name, session_id)
        if access_err:
            return {"role": "assistant", "content": access_err}

        result = await call_mcp_tool(sel_tool_name, tool_args, timeout=_TOOL_TIMEOUTS.get(sel_tool_name, 65.0))

        if isinstance(result, dict) and result.get("requires_site"):
            return {"role": "assistant", "content": result}

        if result is None or (isinstance(result, dict) and not result):
            last_error = "No result returned. Check that the MCP server is running."
            from langchain_core.messages import ToolMessage
            messages = messages + [ai_msg, ToolMessage(content=last_error, tool_call_id=tool_call_id)]
            continue

        if isinstance(result, dict) and "error" in result:
            last_error = result
            from langchain_core.messages import ToolMessage
            messages = messages + [ai_msg, ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)]
            continue

        return {"role": "assistant", "content": _normalize_result(sel_tool_name, result, prompt)}

    if last_error is not None:
        msg = await synthesize_final_answer(prompt, last_tool_name or "tool", last_error)
        return {"role": "assistant", "content": msg}
    return {"role": "assistant", "content": "Something went wrong. Please try again."}
