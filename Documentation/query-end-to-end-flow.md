# End-to-End Query Flow

This document traces exactly what happens, component by component, from the moment a user submits a query to the moment a response is returned. The example used throughout is a NetBrain network path query:

> **"Show network path from 10.0.0.1 to 192.168.1.1"**

---

## High-Level Map

```
Browser / Frontend
        │  POST /api/chat
        ▼
┌─────────────────────────────────────┐
│  FastAPI  (app_fastapi.py :8000)    │  ← auth check, request parsing
└──────────────┬──────────────────────┘
               │  process_message()
               ▼
┌─────────────────────────────────────┐
│  Chat Service  (chat_service.py)    │
│                                     │
│  1. discover_tool()                 │  ← scope check + LLM tool selection
│  2. RBAC check                      │  ← auth.py
│  3. execute_tool()                  │  ← dispatches to mcp_client.py
│  4. _normalize_result()             │  ← post-processes result
└──────────────┬──────────────────────┘
               │
        ┌──────┴──────────────────────────┐
        │                                 │
        ▼                                 ▼
┌───────────────────────┐    ┌─────────────────────────────┐
│  LangChain / Ollama   │    │  MCP Client (mcp_client.py) │
│                       │    │                             │
│  Scope classifier     │    │  get_mcp_session()          │
│  Tool selector        │    │  call_tool(...)             │
│  (qwen2.5:14b)        │    │  Normalise response shape   │
└───────────────────────┘    └──────────────┬──────────────┘
                                            │  HTTP :8765/mcp
                                            ▼
                             ┌─────────────────────────────┐
                             │  MCP Server (mcp_server.py) │
                             │                             │
                             │  query_network_path()       │  ← tool handler
                             │    → NetBrain REST API      │
                             │    → Panorama enrichment    │
                             └─────────────────────────────┘
```

---

## Step-by-Step Flow

### Step 1 — Browser sends HTTP request

The frontend POSTs to `/api/chat`:

```
POST http://localhost:8000/api/chat
Cookie: session=<session_id>

{
  "message": "Show network path from 10.0.0.1 to 192.168.1.1",
  "conversation_history": []
}
```

**Component:** Browser / frontend
**File:** none (HTTP client)

---

### Step 2 — FastAPI receives and authenticates the request

```python
# app_fastapi.py (lines 357-370)

@app.post("/api/chat")
async def api_chat(request: Request, body: ChatRequest):
    # 1. Extract session cookie and look up username
    username = get_current_username(request)   # reads session from auth.py

    if not username:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    # 2. Hand off to chat_service
    from netbrain.chat_service import process_message
    result = await process_message(
        body.message.strip(),              # "Show network path from 10.0.0.1 to 192.168.1.1"
        body.conversation_history or [],   # []
        default_live=True,
        username=username,                 # e.g. "admin"
    )
    return result
```

**Component:** FastAPI (`app_fastapi.py`)
**What happens:** Session cookie is validated against the in-memory session store (`auth.py`). If valid, `username` is resolved. The request body is deserialised into a `ChatRequest` Pydantic model. Control passes to `chat_service.process_message()`.

---

### Step 3 — Chat service starts the agent loop

```python
# chat_service.py (lines 1093-1163)

async def process_message(prompt, conversation_history, *, username, ...):
    history_so_far = list(conversation_history)  # []

    for iteration in range(max_iterations):      # default: 3 attempts
        selection = await discover_tool(prompt, history_so_far)
        ...
```

**Component:** `chat_service.py`
**What happens:** `process_message` initialises the agent retry loop. Up to 3 iterations of discover → execute are attempted. On each failure the error is appended to `history_so_far` so the next iteration's LLM call has the context.

---

### Step 4 — Scope check (LangChain)

Inside `discover_tool()`, before calling the LLM for tool selection, the query is first checked for scope.

#### 4a — Keyword pre-check (no LLM)

```python
# chat_service.py (lines 294-335)

def _is_obviously_in_scope(prompt: str) -> bool:
    ips = _IP_OR_CIDR_RE.findall(prompt)
    if len(ips) >= 2:
        return True   # ← two IPs detected → clearly in scope, skip LLM scope call
    ...
```

The query contains two IPs (`10.0.0.1` and `192.168.1.1`), so this returns `True` immediately. **The LLM scope-check call is skipped entirely.**

#### 4b — LLM scope check (only for ambiguous queries)

For queries that don't match the keyword check, `ChatOllama` is called with a 5-second timeout:

```python
# chat_service.py (lines 349-392)  — skipped for this example

from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:14b", base_url="http://localhost:11434", temperature=0.0)
response = await asyncio.wait_for(llm.ainvoke(scope_check_prompt), timeout=5.0)
# Returns {"in_scope": True/False}
# On timeout or error → defaults to {"in_scope": True}
```

**Component:** LangChain (`ChatOllama`), Ollama (`qwen2.5:14b`)

---

### Step 5 — MCP server queried for tool list

`discover_tool()` opens a session to the MCP server and fetches the live list of registered tools:

```python
# chat_service.py (lines 413-432)

async for client_or_session in get_mcp_session():
    # FastMCPClient("http://127.0.0.1:8765/mcp") or stdio fallback
    if isinstance(client_or_session, FastMCPClient):
        tools = await client_or_session.list_tools()
    else:
        tools_response = await client_or_session.list_tools()
        tools = tools_response.tools
```

**Component:** MCP Client (`mcp_client.py`), MCP Server (`mcp_server.py`)
**What returns:** A list of 8 tool objects, each with `name`, `description`, and `inputSchema` (parameter names + types).

The tool list is then formatted into a numbered string for the LLM:

```
1. query_network_path: Calculate the network path between two IPs | Params: source, destination, protocol, port, is_live
2. check_path_allowed: Check if traffic is allowed between two IPs | Params: source, destination, protocol, port
3. get_rack_details: Get rack details from NetBox | Params: rack_name, site_name
...
```

---

### Step 6 — LLM tool selection (LangChain + Pydantic)

`select_tool_with_llm()` is called with the user query and the tool list string.

#### 6a — Socket probe (before LLM call)

```python
# mcp_client_tool_selection.py (lines 230-257)

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(2)
result = sock.connect_ex(("localhost", 11434))   # probe Ollama port
sock.close()
# If port closed → return {"success": False, "error": "Ollama not running..."}
```

**Component:** Python `socket`
**What happens:** 2-second TCP connect to Ollama. If Ollama is down, an immediate error is returned without waiting for an HTTP timeout.

#### 6b — Build the selection prompt

```python
# mcp_client_tool_selection.py (lines 77-193)

def build_tool_selection_prompt(prompt, tools_description, conversation_history):
    return f"""You are a tool selection expert for network infrastructure queries.

DECISION RULES:
1. Contains DOTS (.)? → IP address
2. TWO IP ADDRESSES + "path/traffic" → check_path_allowed or query_network_path
...

Current user query: "Show network path from 10.0.0.1 to 192.168.1.1"

AVAILABLE TOOLS:
1. query_network_path: Calculate the network path ... | Params: source, destination, ...
2. check_path_allowed: Check if traffic is allowed ... | Params: source, destination, ...
...

Respond with a JSON object:
{{"tool_name": "...", "parameters": {{...}}, "needs_clarification": false, ...}}
"""
```

#### 6c — Pydantic structured output call (primary path)

```python
# mcp_client_tool_selection.py (lines 262-294)

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

class ToolParameters(BaseModel):
    source:      Optional[str]   # ← LLM fills "10.0.0.1"
    destination: Optional[str]   # ← LLM fills "192.168.1.1"
    protocol:    Optional[str]
    port:        Optional[str]
    ip_address:  Optional[str]
    device_name: Optional[str]
    rack_name:   Optional[str]
    limit:       Optional[int]
    # ... all possible params for all tools

class ToolSelection(BaseModel):
    entity_analysis:       Optional[str]   # LLM's reasoning (logged only)
    tool_name:             Optional[str]   # ← LLM fills "query_network_path"
    needs_clarification:   bool            # ← False
    clarification_question: Optional[str]
    parameters:            ToolParameters

llm = ChatOllama(model="qwen2.5:14b", base_url="http://localhost:11434", temperature=0.0)
structured_llm = llm.with_structured_output(ToolSelection)

response: ToolSelection = structured_llm.invoke(prompt_text)
# response is a typed Python object — no JSON parsing needed
```

**Component:** LangChain (`ChatOllama.with_structured_output`), Pydantic (`ToolSelection`, `ToolParameters`), Ollama (`qwen2.5:14b`)

**What the LLM returns (as a populated Pydantic object):**

```python
ToolSelection(
    entity_analysis = "Two IPs found (dots) + 'path' keyword → TWO IP ADDRESSES",
    tool_name        = "query_network_path",
    needs_clarification = False,
    clarification_question = None,
    parameters = ToolParameters(
        source      = "10.0.0.1",
        destination = "192.168.1.1",
        protocol    = None,    # not specified, defaults applied later
        port        = None,
        # all other fields = None
    )
)
```

#### 6d — Fallback if structured output fails

If `with_structured_output` raises (model doesn't support it), the code falls back to `llm.invoke()` and manually extracts JSON from the raw text:

```python
# mcp_client_tool_selection.py (lines 300-330)

response = llm.invoke(prompt_text)
content  = response.content   # raw string

first_brace = content.find('{')
last_brace  = content.rfind('}')
json_str    = content[first_brace:last_brace + 1]
json_str    = re.sub(r'\bNone\b', 'null', json_str)   # fix Python → JSON
parsed      = json.loads(json_str)
```

---

### Step 7 — RBAC check

Back in `chat_service.process_message()`, before executing the tool, the user's role is verified:

```python
# chat_service.py (lines 1177-1180)

access_err = _check_tool_access(username, tool_name)
# Looks up role for "admin" → None (unrestricted) → access_err = None

if access_err:
    return {"role": "assistant", "content": access_err}
```

```python
# auth.py (lines 57-65)

ROLE_ALLOWED_TOOLS = {
    "admin":    None,    # None = all tools allowed
    "netadmin": {"query_network_path", "check_path_allowed", ...},
}
```

**Component:** `auth.py`
**What happens:** `admin` role has `None` (unrestricted), so the check passes. A `netadmin` user would also pass here since `query_network_path` is in their allowed set.

---

### Step 8 — Parameter fixes applied

Before the MCP call, `_apply_tool_param_fixes()` patches up any missing or defaulted parameters:

```python
# chat_service.py (~line 1182)

_apply_tool_param_fixes(tool_name, tool_params, selection, prompt)
# For query_network_path: no special fixes needed
# tool_params = {"source": "10.0.0.1", "destination": "192.168.1.1"}
```

---

### Step 9 — MCP client calls the tool

`execute_tool()` dispatches to `execute_network_query()` in `mcp_client.py`:

```python
# chat_service.py (lines 811-835)

if tool_name == "query_network_path":
    source      = tool_params.get("source")       # "10.0.0.1"
    destination = tool_params.get("destination")  # "192.168.1.1"
    result = await asyncio.wait_for(
        execute_network_query(
            source, destination,
            protocol = tool_params.get("protocol") or "TCP",
            port     = tool_params.get("port") or "0",
            is_live  = default_live,   # True
        ),
        timeout=385.0,
    )
```

Inside `execute_network_query()`:

```python
# mcp_client.py (lines 220-290)

async def execute_network_query(source, destination, protocol, port, is_live):
    async for client_or_session in get_mcp_session():
        # get_mcp_session() tries HTTP first:
        #   FastMCPClient("http://127.0.0.1:8765/mcp", timeout=600)
        # Falls back to stdio if HTTP fails.

        tool_arguments = {
            "source":      "10.0.0.1",
            "destination": "192.168.1.1",
            "protocol":    "TCP",
            "port":        "0",
            "is_live":     1,
            "continue_on_policy_denial": True,
        }

        is_fastmcp = isinstance(client_or_session, FastMCPClient)

        try:
            # Standard MCP format
            result = await asyncio.wait_for(
                client_or_session.call_tool("query_network_path", arguments=tool_arguments),
                timeout=360.0,
            )
        except TypeError:
            # FastMCP kwargs format fallback
            result = await asyncio.wait_for(
                client_or_session.call_tool("query_network_path", **tool_arguments),
                timeout=360.0,
            )

        # Normalise response shape
        if isinstance(result, list):
            result_text = result[0].text       # FastMCP transport
        elif hasattr(result, "content"):
            result_text = result.content[0].text  # stdio transport

        return json.loads(result_text)
```

**Component:** MCP Client (`mcp_client.py`)
**Transport used:** `streamable-http` via `FastMCPClient` → `POST http://127.0.0.1:8765/mcp`

---

### Step 10 — MCP server executes the tool

The MCP server receives the `call_tool` request, deserialises it, and calls the registered Python function:

```python
# tools/netbrain_tools.py — query_network_path handler

@mcp.tool()
async def query_network_path(
    source: str,          # "10.0.0.1"
    destination: str,     # "192.168.1.1"
    protocol: str,        # "TCP"
    port: str,            # "0"
    is_live: int,         # 1
    continue_on_policy_denial: bool,  # True
) -> Dict[str, Any]:

    # 1. Authenticate with NetBrain
    token = await netbrainauth.get_token()

    # 2. Call NetBrain path calculation API
    response = await http.post(
        f"{NETBRAIN_URL}/ServicesAPI/API/V1/PathCalculation/PathCalculation",
        json={"sourceIP": source, "destIP": destination, "protocol": protocol, ...},
        headers={"token": token},
    )
    task_id = response["taskID"]

    # 3. Poll until path calculation completes (max 120 polls × 3s = 6 min)
    for _ in range(120):
        status = await http.get(f".../PathCalculation/{task_id}")
        if status["status"] == "complete":
            break
        await asyncio.sleep(3)

    # 4. Enrich firewall hops with Panorama zone + device group data (LangChain used here too)
    await _add_panorama_zones_to_hops(simplified_hops)
    await _add_panorama_device_groups_to_hops(simplified_hops)

    # 5. Return structured result
    return {
        "path_status": "complete",
        "path_hops": simplified_hops,
        "path_status_description": "...",
    }
```

**Component:** MCP Server (`mcp_server.py`, `tools/netbrain_tools.py`)
**External call:** NetBrain REST API
**Side calls:** Panorama API (for firewall zone enrichment on each firewall hop)

The result is serialised by FastMCP and sent back over HTTP as `text/event-stream` (streamable-http transport).

---

### Step 11 — MCP client receives and parses the response

```python
# mcp_client.py — back in execute_network_query()

# result from call_tool:
# FastMCPClient: result = [TextContent(text='{"path_status": "complete", "path_hops": [...]}')]
# stdio:         result = CallToolResult(content=[TextContent(text='...')])

if isinstance(result, list):
    result_text = result[0].text
elif hasattr(result, "content"):
    result_text = result.content[0].text

return json.loads(result_text)
# → {"path_status": "complete", "path_hops": [...], "path_status_description": "..."}
```

---

### Step 12 — Result normalised

Back in `chat_service.py`, `_normalize_result()` post-processes the raw dict:

```python
# chat_service.py (lines 951-980)

def _normalize_result(tool_name, result, prompt):
    if isinstance(result, dict) and result.get("path_hops"):
        result = dict(result)
        _strip_l2_noise(result)   # removes "l2 connections has not been discovered" noise
    return result
```

**Component:** `chat_service.py`

---

### Step 13 — FastAPI returns the response

```python
# chat_service.py (line 1200)
return {"role": "assistant", "content": result_dict}
```

FastAPI serialises this to JSON and sends it back to the browser:

```json
{
  "role": "assistant",
  "content": {
    "path_status": "complete",
    "path_hops": [
      {
        "from_device": "core-router-01",
        "to_device":   "dist-switch-01",
        "from_interface": "GigabitEthernet0/0",
        "to_interface":   "GigabitEthernet1/1",
        "in_zone":  "untrust",
        "out_zone": "trust"
      },
      ...
    ],
    "path_status_description": ""
  }
}
```

---

## Agent Retry Loop

If any step from 6 onwards fails, the loop retries (up to 3 times). Each retry adds the error to `history_so_far` so the LLM can correct its tool selection:

```python
# chat_service.py (lines 1163-1200)

for iteration in range(3):

    selection = await discover_tool(prompt, history_so_far)

    if not selection.get("success"):
        last_error = selection.get("error")
        history_so_far.append({"role": "assistant", "content": last_error})
        continue   # retry with error in history

    result = await execute_tool(tool_name, tool_params)

    if "error" in result:
        last_error = result
        history_so_far.append({"role": "assistant", "content": result["error"]})
        continue   # retry

    return {"role": "assistant", "content": _normalize_result(tool_name, result, prompt)}

# All retries exhausted → LLM synthesises a human-readable error message
msg = await synthesize_final_answer(prompt, last_tool_name, last_error)
return {"role": "assistant", "content": msg}
```

`synthesize_final_answer` itself uses `ChatOllama` (temperature=0.3, timeout=15s) to turn the raw error into a 2–4 sentence plain-English explanation.

---

## Component Responsibility Summary

| Step | Component | File | Technology |
|------|-----------|------|------------|
| 1 | HTTP transport | — | Browser fetch / axios |
| 2 | Request parsing + auth | `app_fastapi.py` | FastAPI, Pydantic (`ChatRequest`), session cookie |
| 3 | Agent loop orchestration | `chat_service.py` | Python asyncio |
| 4a | Keyword scope pre-check | `chat_service.py` | Regex (`re`) |
| 4b | LLM scope classification | `chat_service.py` | LangChain `ChatOllama.ainvoke`, Ollama |
| 5 | Fetch live tool list | `mcp_client.py` → `mcp_server.py` | MCP protocol, FastMCP |
| 6 | Tool name + param extraction | `mcp_client_tool_selection.py` | LangChain `with_structured_output`, Pydantic `ToolSelection` |
| 7 | Role-based access check | `chat_service.py` → `auth.py` | Python dict lookup |
| 8 | Parameter defaulting | `chat_service.py` | Python |
| 9 | MCP tool call (client side) | `mcp_client.py` | FastMCP `call_tool`, asyncio timeout |
| 10 | Tool execution (server side) | `tools/netbrain_tools.py` | FastMCP `@mcp.tool()`, aiohttp, NetBrain REST, Panorama API |
| 11 | Response shape normalisation | `mcp_client.py` | Python `json.loads` |
| 12 | Result post-processing | `chat_service.py` | Python (`_normalize_result`) |
| 13 | JSON response to browser | `app_fastapi.py` | FastAPI JSON serialisation |
| retry | Error synthesis | `mcp_client_tool_selection.py` | LangChain `ChatOllama.ainvoke`, Ollama |
