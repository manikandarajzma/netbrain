# LLM Role and Resilience

This document covers what the LLM (Ollama / `qwen2.5:14b`) does in the system, and exactly what happens at each stage when it is unavailable.

---

## 1. What the LLM Does

The LLM has two distinct jobs, in two distinct parts of the stack.

### 1.1 Tool Selection (client side — `mcp_client_tool_selection.py`)

This is the primary job. When a user sends a message, the LLM reads the query and decides:

- **Which MCP tool to call** (e.g. `query_network_path` vs `get_rack_details`)
- **What parameters to extract** from natural language (e.g. pull `"10.0.0.1"` into `source`, `"leander-dc-leaf1"` into `device_name`)
- **Whether to ask for clarification** if the query is genuinely ambiguous

It does this through a structured prompt that lists all available tools and their descriptions, then uses Pydantic structured outputs to guarantee a typed, parseable response:

```python
# mcp_client_tool_selection.py (lines 197-294)

async def select_tool_with_llm(prompt, tools_description, conversation_history):

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    # Primary path: Pydantic structured output (typed, no JSON parsing required)
    structured_llm = llm.with_structured_output(ToolSelection)
    prompt_text = build_tool_selection_prompt(prompt, tools_description, conversation_history)
    response: ToolSelection = structured_llm.invoke(prompt_text)

    return {
        "success": True,
        "tool_name": response.tool_name,           # e.g. "query_network_path"
        "parameters": response.parameters.model_dump(),  # e.g. {"source": "10.0.0.1", ...}
        "needs_clarification": response.needs_clarification,
        "clarification_question": response.clarification_question,
    }
```

The prompt it receives lists every registered MCP tool with its name, one-line description, and parameter names — built dynamically from what the MCP server reports at runtime:

```python
# chat_service.py (lines 472-487)

tool_descriptions_list = [
    f"{tool_name}: {concise_description} | Params: {param_names}"
    for t in tools
]
tools_description = "\n".join([
    f"{i+1}. {desc}"
    for i, desc in enumerate(tool_descriptions_list)
])
# Example output handed to LLM:
# 1. query_network_path: Calculate the network path between two IPs | Params: source, destination, protocol, port, is_live
# 2. get_rack_details: Get rack details from NetBox | Params: rack_name, site_name
# ...
```

The `ToolSelection` Pydantic model that structures the LLM's response:

```python
# mcp_client_tool_selection.py (lines 29-56)

class ToolParameters(BaseModel):
    ip_address: Optional[str]       # for Panorama / Splunk tools
    device_name: Optional[str]      # for NetBox rack location
    rack_name: Optional[str]        # for NetBox rack details
    source: Optional[str]           # for NetBrain path queries
    destination: Optional[str]      # for NetBrain path queries
    protocol: Optional[str]
    port: Optional[str]
    limit: Optional[int]            # for Splunk (e.g. "latest 10")
    site_name: Optional[str]
    # ... more fields

class ToolSelection(BaseModel):
    entity_analysis: Optional[str]       # LLM's reasoning (logged for debug)
    tool_name: Optional[str]             # exact tool name, or null
    needs_clarification: bool
    clarification_question: Optional[str]
    parameters: ToolParameters
```

### 1.2 Scope Classification (client side — `chat_service.py`)

Before tool selection, the LLM is used as a lightweight binary classifier to check whether the user's query is within scope of the system at all — i.e. does it relate to network infrastructure?

```python
# chat_service.py (lines 338-392)

async def is_query_in_scope(prompt: str) -> Dict[str, Any]:
    scope_check_prompt = f"""You are a scope classifier for a network infrastructure assistant.
The assistant can ONLY handle queries related to:
• Network device locations and rack details (NetBox)
• Network path queries and traffic allowed checks (NetBrain)
• Panorama / Palo Alto firewall lookups
• Splunk deny event searches

Determine if the following query is IN SCOPE or OUT OF SCOPE.
Query: "{prompt}"
RESPOND WITH ONLY "IN_SCOPE" OR "OUT_OF_SCOPE".
"""
    response = await asyncio.wait_for(llm.ainvoke(scope_check_prompt), timeout=5.0)
    # Returns {"in_scope": True/False, "reason": str}
```

### 1.3 AI Enrichment of Tool Responses (server side — `tools/netbox_tools.py`)

Inside the MCP server itself, certain tool handlers optionally call the LLM to generate a human-readable summary of raw API data. This uses LangChain's `ChatPromptTemplate | llm` chain pattern:

```python
# tools/netbox_tools.py — inside a tool handler
from tools.shared import _get_llm, ChatPromptTemplate

llm = _get_llm()  # lazy-initialised singleton on mcp.llm
if llm:
    chain = ChatPromptTemplate.from_template(
        "Summarise the following rack configuration for a network engineer:\n{data}"
    ) | llm
    result["ai_summary"] = chain.invoke({"data": raw_rack_data}).content
# if llm is None → result returned without ai_summary, no crash
```

### 1.4 Final Answer Synthesis on Tool Failure (client side — `mcp_client_tool_selection.py`)

When the agent loop exhausts all retries after a failed tool call, rather than returning a raw error dict to the user, it asks the LLM to write a short, plain-English explanation of what went wrong and what the user can try next:

```python
# mcp_client_tool_selection.py (lines 359-408)

async def synthesize_final_answer(user_prompt, tool_name, error_or_result):
    prompt_text = f"""The user asked: "{user_prompt}"
The system tried to answer using "{tool_name}" but got: {err_msg}

Write a 2-4 sentence final answer:
1. State what was attempted.
2. Say why it failed in plain language.
3. Suggest one or two concrete things the user can try.
"""
    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)
    response = await asyncio.wait_for(llm.ainvoke(prompt_text), timeout=15.0)
    return response.content.strip()
```

---

## 2. What Happens When the LLM Is Down

Tool selection goes through four layers of defence in sequence. The MCP server and tool execution are entirely unaffected — the LLM is only involved on the client side, before and after the MCP call.

```
User query
    │
    ▼
Layer 1: Keyword pre-check  ──── no LLM ────▶  matched? → skip all LLM calls
    │ (not matched)
    ▼
Layer 2: Scope check (LLM, 5s timeout)
    │  timeout / error → assume in-scope, continue
    ▼
Layer 3: Socket probe (2s TCP check to Ollama port)
    │  port closed → immediate error returned, no hang
    ▼
LLM tool selection call
    │  fails → tool_name = None
    ▼
Layer 4: Regex safety net ── no LLM ──▶  pattern matched? → tool selected
    │ (no match)
    ▼
Agent retry loop (up to 3×)
    │  all fail
    ▼
synthesize_final_answer (LLM, 15s timeout)
    │  also fails → raw error string returned
    ▼
User sees error message
```

### Layer 1 — Keyword Pre-check (no LLM)

`_is_obviously_in_scope()` runs pure regex before any LLM call. If it matches, the scope check LLM call is **skipped entirely**:

```python
# chat_service.py (lines 294-335)

def _is_obviously_in_scope(prompt: str) -> bool:
    lower = prompt.lower()
    has_ip = bool(_IP_OR_CIDR_RE.search(prompt))

    panorama_kw = any(k in lower for k in ("object group", "address group", "panorama", ...))
    netbrain_kw = any(k in lower for k in ("network path", "path from", "traffic allowed", ...))
    netbox_kw   = any(k in lower for k in ("rack", "racked", "device location", ...))
    splunk_kw   = any(k in lower for k in ("splunk", "deny", "denied", "firewall log", ...))

    if has_ip and (panorama_kw or netbrain_kw or splunk_kw): return True
    if panorama_kw or netbrain_kw or netbox_kw or splunk_kw: return True
    if len(_IP_OR_CIDR_RE.findall(prompt)) >= 2:             return True
    return False
```

Examples of queries that bypass the LLM scope check entirely:
- `"path from 10.0.0.1 to 192.168.1.1"` — two IPs detected
- `"rack A4"` — `netbox_kw` matched
- `"recent denies for 10.0.0.5"` — `splunk_kw` matched

### Layer 2 — Scope Check Fails Open

If the keyword check misses and the LLM scope call fails (timeout or exception), the system **assumes in-scope** rather than blocking the user:

```python
# chat_service.py (lines 387-392)

except asyncio.TimeoutError:
    logger.warning("Scope check timed out, assuming in-scope")
    return {"in_scope": True, "reason": "Timeout - assuming in scope"}

except Exception as e:
    logger.warning(f"Scope check failed ({e}), assuming in-scope")
    return {"in_scope": True, "reason": f"Error during scope check: {e}"}
```

The timeout for the scope check LLM call is deliberately short — **5 seconds** — so a downed Ollama does not noticeably stall the request.

### Layer 3 — Socket Probe Before LLM Call

`select_tool_with_llm()` opens a raw TCP socket to the Ollama port before constructing the `ChatOllama` object. This detects a downed Ollama in **2 seconds** and returns a clear, actionable error immediately — no waiting for an HTTP timeout:

```python
# mcp_client_tool_selection.py (lines 230-257)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
result = sock.connect_ex((host, port))
sock.close()

if result != 0:
    return {
        "success": False,
        "error": "Ollama server is not running or not accessible at http://localhost:11434. "
                 "Please start Ollama with 'ollama serve' or check if it's running on a different port.",
        "tool_name": None,
        "parameters": {},
        "needs_clarification": None,
    }
```

### Layer 4 — Regex Safety Net

If the socket check passes but the LLM returns `tool_name: null` (hallucination or parse failure), `discover_tool()` falls back to a set of pattern-matching functions — no LLM involved:

```python
# chat_service.py (lines 578-628)

if tool_name is None and not needs_clarification:
    logger.warning("LLM returned tool_name=None, trying regex safety net...")

    # Pattern: device name (dashes) + location keyword
    device_rack_params = _parse_device_rack_query(prompt)
    if device_rack_params and "get_device_rack_location" in tool_names_list:
        return {"success": True, "tool_name": "get_device_rack_location",
                "parameters": device_rack_params, ...}

    # Pattern: two IPs + "allowed" / "can reach"
    path_allowed_params = _parse_path_allowed_query(prompt)
    if path_allowed_params and "check_path_allowed" in tool_names_list:
        return {"success": True, "tool_name": "check_path_allowed",
                "parameters": path_allowed_params, ...}

    # Pattern: "rack <name>"
    rack_details_params = _parse_rack_details_query(prompt)
    if rack_details_params and "get_rack_details" in tool_names_list:
        return {"success": True, "tool_name": "get_rack_details",
                "parameters": rack_details_params, ...}

    # Pattern: "list racks at <site>"
    list_racks_params = _parse_list_racks_query(prompt)
    if list_racks_params and "list_racks" in tool_names_list:
        return {"success": True, "tool_name": "list_racks",
                "parameters": list_racks_params, ...}
```

Note: this safety net is also triggered when `USE_REGEX_FALLBACK = True` is set, which makes it a fast-path **before** the LLM call rather than a fallback after.

### Agent Retry Loop and Final Degradation

If tool selection fails outright (`success: False`), the agent loop retries up to 3 times:

```python
# chat_service.py (lines 1093-1207)

for iteration in range(max_iterations):  # default max_iterations = 3
    selection = await discover_tool(prompt, history_so_far)

    if not selection.get("success"):
        if iteration == max_iterations - 1:
            # All retries exhausted — try to synthesize a human-readable error
            msg = await synthesize_final_answer(prompt, last_tool_name, last_error)
            return {"role": "assistant", "content": msg}
        history_so_far.append({"role": "assistant", "content": selection.get("error")})
        continue
```

`synthesize_final_answer()` also calls the LLM. If Ollama is still down, its own exception handler returns the raw error string directly:

```python
# mcp_client_tool_selection.py (lines 404-408)

except asyncio.TimeoutError:
    return f"The query could not be completed in time. {err_msg}"

except Exception as e:
    logger.error(f"synthesize_final_answer failed: {e}")
    return (
        f"{err_msg}\n\n"
        "(You can retry or rephrase your question; if the problem continues, "
        "check mcp_server.log and backend connectivity.)"
    )
```

---

## 3. Capability Matrix — LLM Up vs Down

| Capability | LLM up | LLM down |
|---|---|---|
| Scope check for obvious queries (IPs, rack, deny) | Keyword bypass (no LLM) | Same — unaffected |
| Scope check for ambiguous queries | LLM classifies | Assumes in-scope (5s timeout) |
| Tool selection for common patterns | LLM | Regex safety net (Layer 4) |
| Tool selection for ambiguous / complex queries | LLM | Fails — user sees error |
| Parameter extraction (IPs, device names, etc.) | LLM via Pydantic | Regex (limited patterns only) |
| Conversation context / follow-up queries | LLM reads history | Regex cannot handle follow-ups |
| MCP tool execution (NetBrain, NetBox, etc.) | Unaffected | Unaffected — server-side only |
| AI enrichment of tool responses | LLM generates summary | Raw API data returned, no summary |
| Human-readable error messages | LLM synthesizes | Raw error string shown |

---

## 4. Key Design Decisions

### Why `temperature=0.0` for tool selection

Tool selection must be **deterministic** — the same query should always map to the same tool and extract the same parameters. `temperature=0.0` disables sampling randomness in the LLM output.

`synthesize_final_answer` uses `temperature=0.3` because slightly varied, natural-sounding error messages are acceptable and preferable to robotic repetition.

### Why Pydantic structured output over raw JSON parsing

When the LLM is asked to return JSON as free text, it sometimes wraps it in markdown fences, adds comments, or uses Python `None`/`True`/`False` instead of JSON `null`/`true`/`false`. Pydantic structured output (`llm.with_structured_output(ToolSelection)`) bypasses this entirely — the LLM is constrained to produce output that directly populates the model fields. Raw JSON parsing is kept only as a fallback:

```python
# mcp_client_tool_selection.py (lines 262-298)

if PYDANTIC_AVAILABLE:
    try:
        structured_llm = llm.with_structured_output(ToolSelection)
        response = structured_llm.invoke(prompt_text)   # → typed ToolSelection object
        ...
    except Exception:
        pass  # fall through to manual JSON parsing

# Manual fallback: find first { ... last } and fix Python literals
json_str = re.sub(r'\bNone\b', 'null', json_str)
json_str = re.sub(r'\bTrue\b', 'true', json_str)
json_str = re.sub(r'\bFalse\b', 'false', json_str)
parsed = json.loads(json_str)
```

### Why the socket probe before the LLM call

`ChatOllama.invoke()` does not have a short connect timeout — it uses the underlying HTTP client default which can be tens of seconds. A 2-second TCP socket probe to `localhost:11434` detects a downed Ollama instantly and returns a clear, actionable error message without making the user wait.
