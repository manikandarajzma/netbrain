"""
Chat service: tool discovery + execution for FastAPI.
Uses MCP client and tool selection from mcp_client / mcp_client_tool_selection.
"""
import asyncio
import json
import logging
import re
from typing import Any, Dict

logger = logging.getLogger("netbrain.chat_service")

# Lazy imports to avoid circular deps and heavy Streamlit import at module load
def _get_mcp_client():
    from netbrain.mcp_client import (
        get_mcp_session,
        execute_network_query,
        execute_rack_details_query,
        execute_racks_list_query,
        execute_rack_location_query,
        execute_panorama_ip_object_group_query,
        execute_panorama_address_group_members_query,
        execute_splunk_recent_denies_query,
        execute_path_allowed_check,
    )
    from netbrain.mcp_client import FASTMCP_CLIENT_AVAILABLE, FastMCPClient
    return (
        get_mcp_session,
        execute_network_query,
        execute_rack_details_query,
        execute_racks_list_query,
        execute_rack_location_query,
        execute_panorama_ip_object_group_query,
        execute_panorama_address_group_members_query,
        execute_splunk_recent_denies_query,
        execute_path_allowed_check,
        FASTMCP_CLIENT_AVAILABLE,
        FastMCPClient,
    )


# Extract IP/CIDR from user message (restores parsing that existed in pre-Streamlit UI)
_IP_OR_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")

# Path override: "path allowed" / "traffic allowed" / "find path from X to Y" with two IPs → force check_path_allowed
_PATH_ALLOWED_RE = re.compile(
    r"\b(?:is\s+)?path\s+allowed\b|\b(?:is\s+)?traffic\s+allowed\b|"
    r"check\s+if\s+traffic\s+.*?\s+allowed|can\s+traffic\s+.*\s+reach|"
    r"\b(?:does\s+path\s+exist|is\s+connectivity\s+allowed|can\s+.+\s+reach\s+.+|"
    r"connectivity\s+from\s+.+\s+to\s+|traffic\s+from\s+.+\s+to\s+.+\s+allowed)",
    re.IGNORECASE,
)
_PATH_FROM_TO_RE = re.compile(
    r"\b(?:find|get|show|query|trace)?\s*(?:network\s+)?path\s+from\s+.+\s+to\s+",
    re.IGNORECASE,
)


def _parse_path_allowed_query(prompt: str) -> dict[str, Any] | None:
    """If prompt asks path/traffic allowed or path from A to B between two IPs, return params for check_path_allowed."""
    if not (prompt or "").strip():
        return None
    if not _PATH_ALLOWED_RE.search(prompt) and not _PATH_FROM_TO_RE.search(prompt):
        return None
    ips = _IP_OR_CIDR_RE.findall(prompt)
    if len(ips) < 2:
        return None
    protocol = "TCP"
    if re.search(r"\bUDP\b", prompt, re.I):
        protocol = "UDP"
    port = "443"
    port_m = re.search(r"\b(?:port\s+)?(\d{1,5})\b", prompt)
    if port_m:
        port = port_m.group(1)
    return {
        "source": ips[0],
        "destination": ips[1],
        "protocol": protocol,
        "port": port,
    }


# Device rack override: all variations for "where is device X" / "is X in rack Y" with X = device name (dashes) → get_device_rack_location
_DEVICE_RACK_RE = re.compile(
    r"\b(?:where\s+is|which\s+rack\s+is|what\s+rack\s+is)\s+(.+?)\s+racked\b|"
    r"\b(?:where\s+is|which\s+rack)\s+(.+?)\s+in\s+(?:the\s+)?rack\b|"
    r"\brack\s+location\s+(?:of|for)\s+(.+?)(?:\?|$|\s)|"
    r"\b(?:locate|find|look\s+up|show)\s+(?:device\s+)?([a-z0-9\-]+)(?:\s+in\s+netbox|\s+in\s+NetBox)?(?:\?|$|\s)|"
    r"\b(?:which\s+rack\s+has|what\s+rack\s+is|which\s+rack\s+is)\s+([a-z0-9\-]+)(?:\s+in)?(?:\?|$|\s)|"
    r"\bdevice\s+([a-z0-9\-]+)\s+(?:location|rack|position)(?:\?|$|\s)|"
    r"\b([a-z0-9\-]+)\s+rack\s+position(?:\?|$|\s)|"
    r"\bis\s+([a-z0-9\-]+)\s+in\s+[A-Za-z0-9]{1,15}(?:\?|$|\s)|"
    r"\bis\s+([a-z0-9\-]+)\s+racked\s+in\s+[A-Za-z0-9]{1,15}(?:\?|$|\s)|"
    r"\b(?:on|in)\s+netbox\s+where\s+is\s+([a-z0-9\-]+)(?:\?|$|\s)|"
    r"\bnetbox\s+(?:where\s+is|find\s+device)\s+([a-z0-9\-]+)(?:\?|$|\s)|"
    r"\bwhere\s+is\s+([a-z0-9\-]+)(?:\?|$|\s)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_device_rack_query(prompt: str) -> dict[str, Any] | None:
    """If prompt asks where a device is racked / on netbox where is X, return params for get_device_rack_location."""
    if not (prompt or "").strip():
        return None
    m = _DEVICE_RACK_RE.search(prompt)
    if not m:
        return None
    # First non-empty group is the device name
    device_name = next((g for g in m.groups() if g), "")
    device_name = device_name.strip().strip("?.,")
    if not device_name or len(device_name) > 120:
        return None
    # Device names have dashes; IPs have dots — require dash for "where is X" / "netbox where is X" (avoid IP)
    if "." in device_name or (not re.search(r"-", device_name) and len(device_name) <= 4):
        return None  # likely IP or short rack name (A4), not device
    return {"device_name": device_name}


# Rack details override: all variations for "rack A4" / "details for A1" / "A4 in Leander" → get_rack_details (NetBox), not Panorama
_RACK_NAME_RE = re.compile(
    r"^(?:rack\s+)?([A-Za-z0-9]{1,15})$|"
    r"\brack\s+([A-Za-z0-9]{1,15})(?:\s|$|\?)|"
    r"\b(?:show|display|get)\s+(?:me\s+)?(?:rack\s+)?([A-Za-z0-9]{1,15})(?:\s|$|\?)|"
    r"\b(?:details?|info)\s+(?:for\s+)?(?:rack\s+)?([A-Za-z0-9]{1,15})(?:\s|$|\?)|"
    r"\b(?:space\s+)?utilization\s+(?:of\s+)?([A-Za-z0-9]{1,15})(?:\s|$|\?)|"
    r"\b([A-Za-z0-9]{1,15})\s+(?:rack|utilization|space\s+usage)(?:\s|$|\?)|"
    r"\b(?:what'?s?|which\s+devices?)\s+in\s+rack\s+([A-Za-z0-9]{1,15})(?:\?|$|\s)|"
    r"\bdevices?\s+in\s+([A-Za-z0-9]{1,15})(?:\?|$|\s)",
    re.IGNORECASE,
)
# "rack A1 in Leander", "rack A4 at Round Rock DC", "A1 in Leander", "A4 at Leander DC"
_RACK_AND_SITE_RE = re.compile(
    r"\b(?:rack\s+)?([A-Za-z0-9]{1,15})\s+(?:in|at)\s+([A-Za-z0-9\s]{1,50}?)(?:\?|$|\s*$)",
    re.IGNORECASE,
)


def _parse_rack_details_query(prompt: str) -> dict[str, Any] | None:
    """If prompt is a short rack name or 'rack X' or 'rack X in/at <site>', return params for get_rack_details (avoids Panorama)."""
    raw = (prompt or "").strip()
    if not raw:
        return None
    # "rack A1 in Leander" / "rack A4 at Leander DC" → pass site so server can resolve "Leander" → "Leander DC"
    m_site = _RACK_AND_SITE_RE.search(raw)
    if m_site:
        rack_name = m_site.group(1).strip()
        site_name = m_site.group(2).strip()
        if rack_name and "." not in rack_name and "-" not in rack_name and site_name:
            # Check if multiple known site names are mentioned (e.g., "leander round rock")
            # Common site name patterns: "Leander", "Round Rock", "Austin", etc.
            site_lower = site_name.lower()
            known_sites = ["leander", "round rock", "roundrock", "austin", "dallas", "houston"]
            sites_found = [site for site in known_sites if site in site_lower]

            if len(sites_found) >= 2:
                # Multiple site names detected - return None to let LLM ask for clarification
                logger.debug(f"Multiple sites detected in '{site_name}': {sites_found} - deferring to LLM")
                return None

            # Also check word count as fallback (e.g., "leander round rock" = 3 words is unusual)
            site_words = site_name.split()
            if len(site_words) >= 3 and "dc" not in site_lower:
                # 3+ words without "DC" suggests multiple sites - let LLM handle it
                logger.debug(f"Ambiguous site name detected: '{site_name}' (3+ words) - deferring to LLM")
                return None

            return {"rack_name": rack_name, "site_name": site_name}
    # Whole query is a single word (rack name) — no dots, no dashes
    if re.match(r"^[A-Za-z0-9]{1,15}$", raw):
        return {"rack_name": raw}
    m = _RACK_NAME_RE.search(raw)
    if not m:
        return None
    rack_name = next((g for g in m.groups() if g), "").strip()
    if not rack_name or "." in rack_name or "-" in rack_name:
        return None
    return {"rack_name": rack_name}


# List racks: all variations for "list racks at X" / "racks in Leander" / "how many racks in X" → list_racks(site_name=...)
# Use plural "racks" so "rack A1 in Leander" stays get_rack_details, not list_racks
_LIST_RACKS_AT_SITE_RE = re.compile(
    r"\b(?:what|which|list|show|get|display|how\s+many|count\s+of|number\s+of)\s+(?:all\s+)?racks\s+(?:are\s+)?(?:at|in)\s+([A-Za-z0-9\s]{1,50}?)(?:\?|$|\s*$)|"
    r"\b(?:list|show|get|display)\s+(?:me\s+)?(?:all\s+)?racks\s+(?:at|in)\s+([A-Za-z0-9\s]{1,50}?)(?:\?|$|\s*$)|"
    r"\b(?:all\s+)?racks\s+(?:at|in)\s+([A-Za-z0-9\s]{1,50}?)(?:\?|$|\s*$)|"
    r"\bracks\s+at\s+site\s+([A-Za-z0-9\s]{1,50}?)(?:\?|$|\s*$)|"
    r"\b(?:give\s+me|show\s+me)\s+all\s+racks\s+(?:at|in)\s+([A-Za-z0-9\s]{1,50}?)(?:\?|$|\s*$)",
    re.IGNORECASE,
)


def _parse_list_racks_query(prompt: str) -> dict[str, Any] | None:
    """If prompt is 'list racks at X' / 'what racks are in X' / 'list all racks', return params for list_racks."""
    raw = (prompt or "").strip()
    if not raw:
        return None
    # "list all racks" / "show all racks" (no site) → list_racks with no filter
    if re.match(r"^(?:list|show|get|display)\s+(?:all\s+)?racks\s*$", raw, re.IGNORECASE):
        return {"site_name": None}
    m = _LIST_RACKS_AT_SITE_RE.search(raw)
    if not m:
        return None
    site_name = next((g for g in m.groups() if g), "").strip()
    if not site_name:
        return None
    return {"site_name": site_name}


def _parse_rack_follow_up_site(
    conversation_history: list[dict[str, Any]],
    prompt: str,
) -> dict[str, Any] | None:
    """If the last assistant message asked 'Which site?' for a rack and the user replied with a site name, return get_rack_details params."""
    if not conversation_history or not (prompt or "").strip():
        return None
    raw = prompt.strip()

    # CRITICAL: Reject if prompt is clearly a NEW query, not a follow-up answer to "Which site?"
    # These checks prevent misrouting new queries when conversation history contains old "Which site?" messages

    # Check for IP addresses (dots) - these are Panorama/Splunk/NetBrain queries, NOT rack follow-ups
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw):
        # Contains an IP address - this is a new query
        return None

    # Check for device names (dashes) - these are device queries, NOT rack follow-ups
    if re.search(r"[a-z]+-[a-z]+-[a-z]+", raw, re.IGNORECASE):
        # Contains dashes in a device name pattern - this is a new device query
        return None

    # Check for query keywords that indicate a NEW query (not a follow-up answer)
    query_keywords = r"\b(what|where|show|list|which|find|query|path|address group|panorama|splunk|deny|events|allowed|traffic)\b"
    if re.search(query_keywords, raw, re.IGNORECASE):
        # Contains query keywords - this is a new question, not a site name answer
        return None

    # User reply should look like a site name: short phrase, no "rack" keyword
    if re.search(r"\brack\b", raw, re.IGNORECASE) or len(raw) > 60:
        return None
    # Find last assistant message (content can be string or dict with error/requires_site/sites)
    last_assistant_content = None
    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]
        if (msg.get("role") or "").lower() == "assistant":
            c = msg.get("content")
            if isinstance(c, dict):
                last_assistant_content = str(c.get("error") or "") + " " + str(c.get("message", ""))
                if c.get("requires_site") and c.get("sites"):
                    last_assistant_content += " Which site? " + ", ".join(str(s) for s in c.get("sites", [])) + "."
            else:
                last_assistant_content = str(c or "")
            break
    if not last_assistant_content:
        return None

    # Try multiple patterns for rack clarification questions:
    # Pattern 1: "Multiple racks named 'A4' found at different sites"
    rack_match = re.search(r"Multiple racks named\s+'([^']+)'", last_assistant_content, re.IGNORECASE)

    # Pattern 2: "Which site are you referring to for rack A4?" (LLM-generated)
    if not rack_match:
        rack_match = re.search(r"(?:which site|site).*?rack\s+([A-Z0-9]{1,5})", last_assistant_content, re.IGNORECASE)

    # Pattern 3: Any message with "which site" or "Which site?" asking about site
    if not rack_match and re.search(r"which site", last_assistant_content, re.IGNORECASE):
        # Try to extract rack name from the prompt itself (might be in recent context)
        # Check if previous user message mentioned a rack
        for i in range(len(conversation_history) - 1, -1, -1):
            msg = conversation_history[i]
            if (msg.get("role") or "").lower() == "user":
                user_msg = str(msg.get("content") or "")
                rack_in_user_msg = re.search(r"\brack\s+([A-Z0-9]{1,5})\b", user_msg, re.IGNORECASE)
                if rack_in_user_msg:
                    rack_match = rack_in_user_msg
                    break

    if not rack_match:
        return None

    rack_name = rack_match.group(1).strip()
    if not rack_name:
        return None
    return {"rack_name": rack_name, "site_name": raw}


def _extract_ip_or_cidr_from_prompt(text: str) -> str | None:
    """Extract the first IPv4 or CIDR from text. Used when tool needs ip_address but LLM did not fill it."""
    if not (text or "").strip():
        return None
    m = _IP_OR_CIDR_RE.search(text)
    return m.group(0) if m else None


def _is_obviously_in_scope(prompt: str) -> bool:
    """Fast keyword check: return True if query clearly matches known tool patterns.

    This avoids sending obvious network-infra queries through the LLM scope check
    which can misclassify valid queries (e.g. "object group" vs "address group").
    """
    lower = (prompt or "").lower()
    # Has an IP address or CIDR?
    has_ip = bool(_IP_OR_CIDR_RE.search(prompt or ""))

    # Panorama / firewall keywords
    panorama_kw = any(k in lower for k in (
        "object group", "address group", "panorama", "palo alto",
        "firewall rule", "security rule", "security policy",
        "device group", "address object", "ip group",
    ))
    # NetBrain / path keywords
    netbrain_kw = any(k in lower for k in (
        "network path", "path from", "path to", "traffic allowed",
        "path allowed", "can reach", "connectivity", "netbrain",
    ))
    # NetBox / rack keywords
    netbox_kw = any(k in lower for k in (
        "rack", "netbox", "racked", "device location",
    ))
    # Splunk keywords
    splunk_kw = any(k in lower for k in (
        "splunk", "deny", "denied", "denies", "firewall log",
        "recent deny", "deny event",
    ))

    # IP + any domain keyword → clearly in scope
    if has_ip and (panorama_kw or netbrain_kw or splunk_kw):
        return True
    # Domain keyword alone (no IP needed for racks, path queries, etc.)
    if panorama_kw or netbrain_kw or netbox_kw or splunk_kw:
        return True
    # Two IPs likely means path/traffic query
    ips = _IP_OR_CIDR_RE.findall(prompt or "")
    if len(ips) >= 2:
        return True
    return False


async def is_query_in_scope(prompt: str) -> Dict[str, Any]:
    """
    Use LLM to quickly check if the query is related to network infrastructure tools.
    Returns: {"in_scope": True/False, "reason": str}
    """
    # Fast path: skip LLM if query obviously matches known tool patterns
    if _is_obviously_in_scope(prompt):
        logger.debug("Scope check: keyword match → IN_SCOPE (skipped LLM)")
        return {"in_scope": True, "reason": "Keyword match - clearly in scope"}

    try:
        from langchain_ollama import ChatOllama

        scope_check_prompt = f"""You are a scope classifier for a network infrastructure assistant.

The assistant can ONLY handle queries related to:
• Network device locations and rack details (NetBox) — e.g. "where is device X racked", "rack A4", "list racks"
• Network path queries and traffic allowed checks (NetBrain) — e.g. "path from A to B", "is traffic allowed"
• Panorama / Palo Alto firewall lookups — e.g. "what object group is IP in", "address group members", "IP object group", "which group contains IP"
• Splunk deny event searches (firewall logs) — e.g. "recent denies for IP", "firewall deny events"

Any query mentioning IP addresses, firewalls, network paths, racks, Panorama, Splunk, object groups, address groups, or network devices is IN SCOPE.

Determine if the following query is IN SCOPE or OUT OF SCOPE.

Query: "{prompt}"

RESPOND WITH ONLY "IN_SCOPE" OR "OUT_OF_SCOPE" (one word, no explanation).
"""

        import os
        llm = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:14b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            temperature=0.0,
        )

        response = await asyncio.wait_for(
            llm.ainvoke(scope_check_prompt),
            timeout=5.0
        )

        result_text = response.content.strip().upper() if hasattr(response, "content") else str(response).strip().upper()

        if "OUT_OF_SCOPE" in result_text or "OUT OF SCOPE" in result_text:
            return {"in_scope": False, "reason": "Query is not related to network infrastructure tools"}
        else:
            return {"in_scope": True, "reason": "Query is related to network infrastructure"}

    except asyncio.TimeoutError:
        logger.warning("Scope check timed out, assuming in-scope")
        return {"in_scope": True, "reason": "Timeout - assuming in scope"}
    except Exception as e:
        logger.warning(f"Scope check failed ({e}), assuming in-scope")
        return {"in_scope": True, "reason": f"Error during scope check: {e}"}


async def discover_tool(prompt: str, conversation_history: list[dict[str, Any]]) -> dict[str, Any]:
    """Run tool discovery: get MCP tools, call LLM to select tool, return selection or clarification/error."""
    from netbrain.mcp_client_tool_selection import select_tool_with_llm

    (
        get_mcp_session,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        FASTMCP_CLIENT_AVAILABLE,
        FastMCPClient,
    ) = _get_mcp_client()

    async for client_or_session in get_mcp_session():
        try:
            if FASTMCP_CLIENT_AVAILABLE and hasattr(client_or_session, "list_tools"):
                tools_result = await client_or_session.list_tools()
                tools = tools_result if isinstance(tools_result, list) else (getattr(tools_result, "tools", None) or [])
            else:
                tools_result = await client_or_session.list_tools()
                tools = tools_result if isinstance(tools_result, list) else (getattr(tools_result, "tools", None) or [])
            if not tools:
                return {"success": False, "error": "No tools available from MCP server."}

            # No term matching: use tool list order from server; selection is LLM + docstrings only
            def _tool_name(t):
                return t.get("name") if isinstance(t, dict) else getattr(t, "name", None)
            tool_names_list = [_tool_name(t) for t in tools]
            def _params(t):
                schema = getattr(t, "inputSchema", None)
                if isinstance(schema, dict):
                    return ", ".join((schema.get("properties") or {}).keys())
                return ""

            # Extract concise tool descriptions (first meaningful line only)
            def _tool_description_concise(t):
                """Extract just the essential purpose from the docstring, removing all the warnings and examples."""
                full_desc = (t.get("description") if isinstance(t, dict) else getattr(t, "description", None)) or "No description"

                # Take only the first line before any double newline or before any "**" section
                lines = full_desc.split('\n')
                essential = []
                for line in lines:
                    stripped = line.strip()
                    # Stop at section markers
                    if stripped.startswith('**') or stripped.startswith('CRITICAL') or stripped.startswith('Args:') or stripped.startswith('Returns:'):
                        break
                    # Skip empty lines at the start
                    if not stripped and not essential:
                        continue
                    # Add non-empty lines until we hit a section
                    if stripped:
                        essential.append(stripped)
                        # Stop after first sentence or two (max 2 lines)
                        if len(essential) >= 2:
                            break

                result = ' '.join(essential)
                # Truncate at first period if too long
                if len(result) > 250:
                    first_period = result.find('. ')
                    if first_period > 50:
                        result = result[:first_period + 1]

                return result or full_desc[:200]

            # Build concise description text per tool
            tool_descriptions_list = [
                f"{_tool_name(t)}: {_tool_description_concise(t)} | Params: {_params(t)}"
                for t in tools
            ]
            # Log concise descriptions for debugging
            for t in tools:
                name = _tool_name(t)
                if name in ("check_path_allowed", "query_panorama_ip_object_group", "get_device_rack_location"):
                    desc = _tool_description_concise(t)
                    logger.debug(f"Concise description for {name}: {desc}")

            # Format as a clean numbered list (easier for LLM to parse)
            tools_description = "\n".join([
                f"{i+1}. {desc}"
                for i, desc in enumerate(tool_descriptions_list)
            ])

            # Check for rack follow-up responses FIRST (before LLM)
            # If user is answering "which site?" for a rack query, handle it immediately
            rack_follow_up_params = _parse_rack_follow_up_site(conversation_history, prompt)
            if rack_follow_up_params and "get_rack_details" in tool_names_list:
                logger.debug(f"RACK FOLLOW-UP DETECTED: get_rack_details with {rack_follow_up_params}")
                return {
                    "success": True,
                    "tool_name": "get_rack_details",
                    "parameters": rack_follow_up_params,
                    "format": "table",
                    "intent": None,
                }

            # Check for out-of-scope queries EARLY (before LLM or tool execution)
            # Use LLM to classify scope - scalable approach that catches any out-of-scope query
            scope_result = await is_query_in_scope(prompt)
            if not scope_result.get("in_scope"):
                logger.debug(f"OUT-OF-SCOPE DETECTED: {scope_result.get('reason')}")
                return {
                    "success": False,
                    "needs_clarification": True,
                    "clarification_question": (
                        "I'm not equipped to answer that question. I can help with:\n"
                        "• Network device locations and rack details (NetBox)\n"
                        "• Network path queries and traffic allowed checks (NetBrain)\n"
                        "• Panorama address group lookups\n"
                        "• Splunk deny event searches\n\n"
                        "Please ask a question related to these areas."
                    )
                }

            # OPTIONAL: Regex-based fast-path (can be disabled by setting USE_REGEX_FALLBACK=False)
            # Use LLM as primary selector, regex as fallback only when LLM fails
            USE_REGEX_FALLBACK = False  # Set to True to enable regex fast-path

            tool_name_override = None
            tool_params_override = {}

            if USE_REGEX_FALLBACK:
                # Check for "is path allowed" / "traffic allowed" with two IPs
                path_allowed_params = _parse_path_allowed_query(prompt)
                if path_allowed_params and "check_path_allowed" in tool_names_list:
                    tool_name_override = "check_path_allowed"
                    tool_params_override = path_allowed_params

                # Check for "where is X racked?" with device name
                if not tool_name_override:
                    device_rack_params = _parse_device_rack_query(prompt)
                    if device_rack_params and "get_device_rack_location" in tool_names_list:
                        tool_name_override = "get_device_rack_location"
                        tool_params_override = device_rack_params

                # Check for "list racks at Leander"
                if not tool_name_override:
                    list_racks_params = _parse_list_racks_query(prompt)
                    if list_racks_params and "list_racks" in tool_names_list:
                        tool_name_override = "list_racks"
                        tool_params_override = list_racks_params

                # Check for "Rack A4" / rack details
                if not tool_name_override:
                    rack_details_params = _parse_rack_follow_up_site(conversation_history, prompt) or _parse_rack_details_query(prompt)
                    if rack_details_params and "get_rack_details" in tool_names_list:
                        tool_name_override = "get_rack_details"
                        tool_params_override = rack_details_params

                # If regex override matched, skip LLM and return immediately
                if tool_name_override:
                    logger.debug(f"REGEX FAST-PATH: {tool_name_override} with params {tool_params_override}")
                    return {
                        "success": True,
                        "tool_name": tool_name_override,
                        "parameters": tool_params_override,
                        "format": "table",
                        "intent": None,
                    }

            # Proceed with LLM selection (primary method)
            tool_selection_result = await select_tool_with_llm(
                prompt=prompt,
                tools_description=tools_description,
                conversation_history=conversation_history,
            )
            tool_name = tool_selection_result.get("tool_name")
            needs_clarification = tool_selection_result.get("needs_clarification", False)

            if not tool_selection_result.get("success"):
                return tool_selection_result

            if tool_name is None and not needs_clarification:
                # LLM failed to select a tool - try regex patterns as safety net
                logger.warning("LLM returned tool_name=None, trying regex safety net...")

                # Try all regex patterns as fallback
                device_rack_params = _parse_device_rack_query(prompt)
                if device_rack_params and "get_device_rack_location" in tool_names_list:
                    logger.debug("REGEX SAFETY NET: get_device_rack_location")
                    return {
                        "success": True,
                        "tool_name": "get_device_rack_location",
                        "parameters": device_rack_params,
                        "format": "table",
                        "intent": None,
                    }

                path_allowed_params = _parse_path_allowed_query(prompt)
                if path_allowed_params and "check_path_allowed" in tool_names_list:
                    logger.debug("REGEX SAFETY NET: check_path_allowed")
                    return {
                        "success": True,
                        "tool_name": "check_path_allowed",
                        "parameters": path_allowed_params,
                        "format": "table",
                        "intent": None,
                    }

                rack_details_params = _parse_rack_follow_up_site(conversation_history, prompt) or _parse_rack_details_query(prompt)
                if rack_details_params and "get_rack_details" in tool_names_list:
                    logger.debug("REGEX SAFETY NET: get_rack_details")
                    return {
                        "success": True,
                        "tool_name": "get_rack_details",
                        "parameters": rack_details_params,
                        "format": "table",
                        "intent": None,
                    }

                list_racks_params = _parse_list_racks_query(prompt)
                if list_racks_params and "list_racks" in tool_names_list:
                    logger.debug("REGEX SAFETY NET: list_racks")
                    return {
                        "success": True,
                        "tool_name": "list_racks",
                        "parameters": list_racks_params,
                        "format": "table",
                        "intent": None,
                    }

                # No regex match either
                logger.debug("REGEX SAFETY NET: No pattern matched")

                clarification_msg = tool_selection_result.get("clarification_question") or (
                    "I'm sorry, but this system is not equipped to process that type of query."
                )

                return {
                    "success": False,
                    "tool_name": None,
                    "needs_clarification": False,
                    "clarification_question": clarification_msg,
                    "error": "Query cannot be processed by any available tool",
                }
            if needs_clarification:
                # LLM wants clarification - but check regex fallback first in case LLM is wrong
                logger.debug("LLM asked for clarification, checking regex fallback...")

                # Try all regex patterns as fallback
                device_rack_params = _parse_device_rack_query(prompt)
                if device_rack_params and "get_device_rack_location" in tool_names_list:
                    logger.debug("REGEX FALLBACK: Overriding LLM -> get_device_rack_location")
                    return {
                        "success": True,
                        "tool_name": "get_device_rack_location",
                        "parameters": device_rack_params,
                        "format": "table",
                        "intent": None,
                    }

                path_allowed_params = _parse_path_allowed_query(prompt)
                if path_allowed_params and "check_path_allowed" in tool_names_list:
                    logger.debug("REGEX FALLBACK: Overriding LLM -> check_path_allowed")
                    return {
                        "success": True,
                        "tool_name": "check_path_allowed",
                        "parameters": path_allowed_params,
                        "format": "table",
                        "intent": None,
                    }

                list_racks_params = _parse_list_racks_query(prompt)
                if list_racks_params and "list_racks" in tool_names_list:
                    logger.debug("REGEX FALLBACK: Overriding LLM -> list_racks")
                    return {
                        "success": True,
                        "tool_name": "list_racks",
                        "parameters": list_racks_params,
                        "format": "table",
                        "intent": None,
                    }

                rack_details_params = _parse_rack_follow_up_site(conversation_history, prompt) or _parse_rack_details_query(prompt)
                if rack_details_params and "get_rack_details" in tool_names_list:
                    logger.debug("REGEX FALLBACK: Overriding LLM -> get_rack_details")
                    return {
                        "success": True,
                        "tool_name": "get_rack_details",
                        "parameters": rack_details_params,
                        "format": "table",
                        "intent": None,
                    }

                # No regex match - use LLM's clarification
                logger.debug("REGEX FALLBACK: No pattern matched, using LLM clarification")
                return {
                    "success": False,
                    "needs_clarification": True,
                    "clarification_question": tool_selection_result.get("clarification_question", "Could you please clarify?"),
                }
            if not tool_name:
                return {"success": False, "error": "LLM did not select a tool."}
            tool_params = dict(tool_selection_result.get("parameters") or {})
            if "intent" in tool_params:
                tool_params = {k: v for k, v in tool_params.items() if k != "intent"}

            # Log successful LLM selection
            logger.debug(f"LLM successfully selected tool: {tool_name} with params {tool_params}")
            if "expected_rack" in tool_params:
                logger.debug(f"expected_rack extracted: {repr(tool_params.get('expected_rack'))}")

            return {
                "success": True,
                "tool_name": tool_name,
                "parameters": tool_params,
                "format": tool_selection_result.get("format", "table"),
                "intent": tool_selection_result.get("intent"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
        break
    return {"success": False, "error": "Could not connect to MCP server."}


async def execute_tool(
    tool_name: str,
    tool_params: dict[str, Any],
    *,
    default_live: bool = True,
) -> dict[str, Any] | str:
    """Execute the selected MCP tool and return result dict or error string."""
    (
        _,
        execute_network_query,
        execute_rack_details_query,
        execute_racks_list_query,
        execute_rack_location_query,
        execute_panorama_ip_object_group_query,
        execute_panorama_address_group_members_query,
        execute_splunk_recent_denies_query,
        execute_path_allowed_check,
        _,
        _,
    ) = _get_mcp_client()

    def run(coro):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(lambda: asyncio.run(coro)).result(timeout=400)

    try:
        if tool_name == "get_device_rack_location":
            device_name = (tool_params.get("device_name") or "").strip()
            if not device_name:
                return {"error": "Device name not found in query."}
            result = await asyncio.wait_for(
                execute_rack_location_query(
                    device_name,
                    tool_params.get("format") or "table",
                    None,
                    tool_params.get("intent"),
                    tool_params.get("expected_rack"),
                ),
                timeout=65.0,
            )
            return result or {"error": "No result from rack location query."}

        if tool_name == "list_racks":
            result = await asyncio.wait_for(
                execute_racks_list_query(
                    tool_params.get("site_name"),
                    tool_params.get("format") or "table",
                    None,
                ),
                timeout=65.0,
            )
            return result or {"error": "No result."}

        if tool_name == "get_rack_details":
            rack_name = tool_params.get("rack_name")
            if not rack_name:
                return {"error": "Rack name not found in query."}
            result = await asyncio.wait_for(
                execute_rack_details_query(
                    rack_name,
                    tool_params.get("format") or "table",
                    None,
                    tool_params.get("site_name"),
                ),
                timeout=65.0,
            )
            return result or {"error": "No result."}

        if tool_name == "check_path_allowed":
            source = (tool_params.get("source") or "").strip()
            destination = (tool_params.get("destination") or "").strip()
            if not source or not destination:
                return {"error": "Source and destination IP addresses are required."}
            result = await asyncio.wait_for(
                execute_path_allowed_check(
                    source,
                    destination,
                    tool_params.get("protocol") or "TCP",
                    tool_params.get("port") or "0",
                    True,
                ),
                timeout=370.0,
            )
            return result or {"error": "No result."}

        if tool_name == "query_network_path":
            source = tool_params.get("source")
            destination = tool_params.get("destination")
            if not source or not destination:
                return {"error": "Source and destination are required."}
            is_live = 1 if default_live else 0
            result = await asyncio.wait_for(
                execute_network_query(
                    source,
                    destination,
                    tool_params.get("protocol") or "TCP",
                    tool_params.get("port") or "0",
                    is_live,
                ),
                timeout=385.0,
            )
            if isinstance(result, dict) and "result" in result and len(result) == 1:
                inner = result["result"]
                if isinstance(inner, str):
                    try:
                        import json
                        result = json.loads(inner)
                    except Exception:
                        pass
            return result or {"error": "No result."}

        if tool_name == "query_panorama_ip_object_group":
            ip_address = (tool_params.get("ip_address") or "").strip()
            if not ip_address:
                return {"error": "IP address not found in query."}
            result = await asyncio.wait_for(
                execute_panorama_ip_object_group_query(
                    ip_address,
                    tool_params.get("device_group"),
                    tool_params.get("vsys", "vsys1"),
                ),
                timeout=65.0,
            )
            return result or {"error": "No result."}

        if tool_name == "query_panorama_address_group_members":
            address_group_name = (tool_params.get("address_group_name") or "").strip()
            if not address_group_name:
                return {"error": "Address group name not found in query."}
            result = await asyncio.wait_for(
                execute_panorama_address_group_members_query(
                    address_group_name,
                    tool_params.get("device_group"),
                    tool_params.get("vsys", "vsys1"),
                ),
                timeout=65.0,
            )
            return result or {"error": "No result."}

        if tool_name == "get_splunk_recent_denies":
            ip_address = (tool_params.get("ip_address") or "").strip()
            if not ip_address:
                return {"error": "IP address not found in query."}
            # Handle limit: use 100 if None or not provided
            limit = tool_params.get("limit")
            if limit is None:
                limit = 100
            result = await asyncio.wait_for(
                execute_splunk_recent_denies_query(
                    ip_address,
                    limit,
                    tool_params.get("earliest_time") or "-24h",
                ),
                timeout=95.0,
            )
            return result or {"error": "No result."}

        return {"error": f"Unknown tool: {tool_name}"}
    except asyncio.TimeoutError:
        return {"error": "Request timed out. Please try again."}
    except Exception as e:
        return {"error": str(e)}


# Tool name → display label for status ("Querying NetBrain", etc.)
TOOL_DISPLAY_NAMES: dict[str, str] = {
    "check_path_allowed": "NetBrain",
    "query_network_path": "NetBrain",
    "query_panorama_ip_object_group": "Panorama",
    "query_panorama_address_group_members": "Panorama",
    "get_rack_details": "NetBox",
    "list_racks": "NetBox",
    "get_device_rack_location": "NetBox",
    "get_splunk_recent_denies": "Splunk",
}


def get_tool_display_name(tool_name: str) -> str:
    """Return human-readable label for status message, e.g. 'Querying NetBrain'."""
    return TOOL_DISPLAY_NAMES.get(tool_name, "backend")


# Agent loop: max tool discovery + execute attempts before forcing a final synthesized answer
MAX_AGENT_ITERATIONS = 3


def _apply_tool_param_fixes(
    tool_name: str,
    tool_params: dict[str, Any],
    selection: dict[str, Any],
    prompt: str,
) -> None:
    """Apply intent and parameter extraction fixes before execute_tool."""
    if tool_name == "get_device_rack_location":
        if selection.get("intent"):
            tool_params["intent"] = selection.get("intent")
        if tool_params.get("intent") == "rack_locatior":
            tool_params["intent"] = "rack_location_only"
    if tool_name in ("query_panorama_ip_object_group", "get_splunk_recent_denies"):
        if not (tool_params.get("ip_address") or "").strip():
            extracted = _extract_ip_or_cidr_from_prompt(prompt)
            if extracted:
                tool_params["ip_address"] = extracted

    # Extract limit from prompt if not provided (e.g., "latest 10", "recent 5 events")
    if tool_name == "get_splunk_recent_denies" and tool_params.get("limit") is None:
        limit_match = re.search(r"\b(?:latest|recent|last|top)\s+(\d+)\b", prompt, re.IGNORECASE)
        if limit_match:
            try:
                tool_params["limit"] = int(limit_match.group(1))
                logger.debug(f"limit extracted from prompt via regex: {tool_params['limit']}")
            except ValueError:
                pass


def _strip_l2_noise(result: dict[str, Any]) -> dict[str, Any]:
    """Remove noisy NetBrain status messages like 'L2 connections has not been discovered'."""
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
    """Apply Splunk/LLM normalizations to a successful result."""
    if result is None or (isinstance(result, dict) and len(result) == 0):
        return result
    # Strip noisy L2 messages from path results
    if isinstance(result, dict) and result.get("path_hops"):
        result = dict(result)
        _strip_l2_noise(result)
    if tool_name == "get_splunk_recent_denies" and isinstance(result, dict):
        if result.get("count") == 0 and "error" not in result:
            ip = result.get("ip_address", "this IP")
            result = dict(result)
            result["message"] = f"No deny events found for {ip} in the last 24 hours. Try a different IP or time range, or check that Splunk has Palo Alto logs for that period."
    if isinstance(result, dict) and result.get("intent") == "rack_locatior":
        result = dict(result)
        result["intent"] = "rack_location_only"

    # Handle yes/no answers - prepend them to the result display
    if isinstance(result, dict) and "yes_no_answer" in result:
        # Move yes_no_answer to the front by creating a new dict
        yes_no_msg = result.pop("yes_no_answer")
        # Create a new result dict with yes_no_answer first
        new_result = {"yes_no_answer": yes_no_msg}
        new_result.update(result)
        return new_result

    # Handle specific metric queries for get_rack_details - provide direct answers
    if tool_name == "get_rack_details" and isinstance(result, dict) and prompt and "error" not in result:
        prompt_lower = prompt.lower()
        rack_name = result.get("rack_name", "rack")
        site_name = result.get("site", "")
        location_str = f"{rack_name} at {site_name}" if site_name else rack_name

        # Detect specific metric queries and extract the answer
        metric_answer = None

        if "space utilization" in prompt_lower or "utilization" in prompt_lower:
            if "space_utilization" in result:
                util_value = result.get("space_utilization")
                if util_value is not None:
                    metric_answer = f"Space utilization of {location_str} is {util_value}%"

        elif "occupied" in prompt_lower and "unit" in prompt_lower:
            if "occupied_units" in result:
                occupied = result.get("occupied_units")
                if occupied is not None:
                    metric_answer = f"{location_str} has {occupied} occupied units"

        elif "available" in prompt_lower and "unit" in prompt_lower:
            height = result.get("height")
            occupied = result.get("occupied_units")
            if height is not None and occupied is not None:
                available = height - occupied
                metric_answer = f"{location_str} has {available} available units (out of {height} total)"

        elif "status" in prompt_lower:
            if "status" in result:
                status = result.get("status")
                if status:
                    metric_answer = f"Status of {location_str} is: {status}"

        elif "height" in prompt_lower:
            if "height" in result:
                height = result.get("height")
                if height is not None:
                    metric_answer = f"{location_str} has a height of {height} units"

        # If we detected a specific metric query, prepend the direct answer
        if metric_answer:
            result = dict(result)
            result["metric_answer"] = metric_answer
            # Move metric_answer to the front
            new_result = {"metric_answer": metric_answer}
            for k, v in result.items():
                if k != "metric_answer":
                    new_result[k] = v
            return new_result

    # Handle Panorama IP lookup queries - provide direct answers
    if tool_name == "query_panorama_ip_object_group" and isinstance(result, dict) and prompt and "error" not in result:
        # Extract the queried IP from the result or prompt
        queried_ip = result.get("queried_ip") or result.get("ip_address")
        if not queried_ip:
            # Try to extract from prompt
            ip_match = re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b", prompt)
            if ip_match:
                queried_ip = ip_match.group(0)

        # Remove duplicate message if it exists (fix duplicate issue)
        if "message" in result and result.get("address_groups"):
            # The message is already in the result, we'll create a better direct_answer instead
            result = dict(result)
            result.pop("message", None)  # Remove the generic message

        # Create a direct answer
        if queried_ip and result.get("address_groups"):
            address_groups = result.get("address_groups", [])
            group_names = [ag.get("name") for ag in address_groups if ag.get("name")]

            if group_names:
                # Create direct answer
                if len(group_names) == 1:
                    direct_answer = f"{queried_ip} is part of address group '{group_names[0]}'"
                else:
                    groups_str = "', '".join(group_names)
                    direct_answer = f"{queried_ip} is part of address groups: '{groups_str}'"

                # Add network info if available
                address_objects = result.get("address_objects", [])
                if address_objects:
                    network_names = [obj.get("name") for obj in address_objects if obj.get("name")]
                    if network_names:
                        direct_answer += f" (via {', '.join(network_names)})"

                result["direct_answer"] = direct_answer
                # Move direct_answer to the front
                new_result = {"direct_answer": direct_answer}
                for k, v in result.items():
                    if k != "direct_answer":
                        new_result[k] = v
                return new_result

    return result


async def process_message(
    prompt: str,
    conversation_history: list[dict[str, str]],
    *,
    default_live: bool = True,
    discover_only: bool = False,
    tool_name: str | None = None,
    parameters: dict[str, Any] | None = None,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """
    Process one user message: discover tool, optionally execute, return assistant response.
    If discover_only=True, returns { tool_name, parameters, tool_display_name } (no execution).
    If tool_name and parameters are provided, skips discovery and executes that tool.
    Otherwise runs an agent loop (up to max_iterations, default MAX_AGENT_ITERATIONS): discover → execute;
    on tool failure, re-discovers with error in context; after max attempts or on any failure, forces a
    synthesized final answer via LLM instead of returning only the raw error.
    """
    if max_iterations is None:
        max_iterations = MAX_AGENT_ITERATIONS

    # Pre-filled tool: single execution, then synthesize on error
    if tool_name and parameters is not None:
        selection = {"success": True, "tool_name": tool_name, "parameters": parameters}
        tool_params = dict(parameters)
        _apply_tool_param_fixes(tool_name, tool_params, selection, prompt)
        result = await execute_tool(tool_name, tool_params, default_live=default_live)
        if result is None or (isinstance(result, dict) and len(result) == 0):
            from netbrain.mcp_client_tool_selection import synthesize_final_answer
            msg = await synthesize_final_answer(
                prompt, tool_name or "tool",
                "No result returned. Check that the MCP server is running and NetBox is reachable (e.g. NETBOX_URL).",
            )
            return {"role": "assistant", "content": msg}
        if isinstance(result, str):
            return {"role": "assistant", "content": result}
        if isinstance(result, dict) and "error" in result:
            if result.get("requires_site") and result.get("sites"):
                return {"role": "assistant", "content": result}
            from netbrain.mcp_client_tool_selection import synthesize_final_answer
            msg = await synthesize_final_answer(prompt, tool_name or "tool", result)
            return {"role": "assistant", "content": msg}
        return {"role": "assistant", "content": _normalize_result(tool_name, result, prompt)}

    # Discover-only: no loop, no execution
    if discover_only:
        selection = await discover_tool(prompt, conversation_history)
        if not selection.get("success"):
            if selection.get("needs_clarification"):
                return {"role": "assistant", "content": selection.get("clarification_question", "Could you please clarify?")}
            return {"role": "assistant", "content": selection.get("clarification_question") or selection.get("error", "Something went wrong.")}
        return {
            "tool_name": selection.get("tool_name"),
            "parameters": dict(selection.get("parameters") or {}),
            "tool_display_name": get_tool_display_name(selection.get("tool_name") or ""),
            "intent": selection.get("intent"),
            "format": selection.get("format"),
        }

    # Agent loop: up to max_iterations discover → execute; on error, add to history and retry; then force final answer
    from netbrain.mcp_client_tool_selection import synthesize_final_answer
    history_so_far = list(conversation_history)
    last_tool_name: str | None = None
    last_error: str | dict[str, Any] | None = None

    for iteration in range(max_iterations):
        selection = await discover_tool(prompt, history_so_far)
        if not selection.get("success"):
            if selection.get("needs_clarification"):
                return {"role": "assistant", "content": selection.get("clarification_question", "Could you please clarify?")}
            if selection.get("clarification_question"):
                return {"role": "assistant", "content": selection["clarification_question"]}
            last_error = selection.get("error", "Something went wrong.")
            if iteration == max_iterations - 1:
                break
            history_so_far = history_so_far + [{"role": "assistant", "content": last_error}]
            continue

        tool_name = selection.get("tool_name")
        tool_params = dict(selection.get("parameters") or {})
        _apply_tool_param_fixes(tool_name, tool_params, selection, prompt)
        result = await execute_tool(tool_name, tool_params, default_live=default_live)
        last_tool_name = tool_name

        if result is None or (isinstance(result, dict) and len(result) == 0):
            last_error = "No result returned. Check that the MCP server is running and NetBox is reachable (e.g. NETBOX_URL)."
            history_so_far = history_so_far + [{"role": "assistant", "content": last_error}]
            continue
        if isinstance(result, str):
            return {"role": "assistant", "content": result}
        if isinstance(result, dict) and "error" in result:
            if result.get("requires_site") and result.get("sites"):
                return {"role": "assistant", "content": result}
            last_error = result
            history_so_far = history_so_far + [{"role": "assistant", "content": result.get("error", "An error occurred.")}]
            continue

        # Success
        return {"role": "assistant", "content": _normalize_result(tool_name, result, prompt)}

    # All iterations exhausted or last step was discovery failure: force synthesized final answer
    if last_error is not None:
        err_str = last_error if isinstance(last_error, str) else (last_error.get("error") or str(last_error))
        msg = await synthesize_final_answer(prompt, last_tool_name or "tool", err_str)
        return {"role": "assistant", "content": msg}
    return {"role": "assistant", "content": "Something went wrong. Please try rephrasing or try again later."}
