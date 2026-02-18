# MCP Server, MCP Client, and LangChain Integration

This document explains how the MCP (Model Context Protocol) server and clients are integrated, how they communicate, and how LangChain is used throughout the system.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Chat Interface                        │
│              (app_fastapi.py - Port 8000)                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │ HTTP POST /api/chat
                       ↓
        ┌──────────────────────────────────┐
        │   Chat Service (chat_service.py) │
        │  1. Tool Discovery (LLM)         │
        │  2. Tool Execution               │
        │  3. Response Synthesis           │
        └──────────┬───────────────────────┘
                   │
        ┌──────────┴──────────┐
        ↓                     ↓
   LLM (Ollama)          MCP Client
   Tool Selection        (mcp_client.py)
   (LangChain)                │
                              │ call_tool(...)
                              ↓
                    ┌─────────────────────────┐
                    │    MCP Server           │
                    │  (mcp_server.py :8765)  │
                    │  8 registered tools     │
                    └──────┬──────────────────┘
                           │
          ┌────────────────┼────────────────┐
          ↓                ↓                ↓
      NetBrain          NetBox         Panorama / Splunk
```

---

## 1. MCP Server Setup

The MCP server is built with **FastMCP**, a high-level library wrapping the Model Context Protocol. It runs as a standalone HTTP server on port `8765`.

### Entry Point — `mcp_server.py`

```python
# mcp_server.py (lines 24-31)
from tools.shared import mcp, MCP_SERVER_HOST, MCP_SERVER_PORT

# Domain modules self-register tools on import via @mcp.tool()
import tools.splunk_tools      # registers: get_splunk_recent_denies
import tools.netbox_tools      # registers: get_rack_details, list_racks, get_device_rack_location
import tools.panorama_tools    # registers: query_panorama_ip_object_group, query_panorama_address_group_members
import tools.netbrain_tools    # registers: query_network_path, check_path_allowed

# Health check endpoint
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    tools = await mcp.get_tools()
    return JSONResponse({
        "status": "ok",
        "server": mcp.name,
        "tools_registered": len(tools),
    })

# Starts server using streamable-http transport
mcp.run(transport="streamable-http", port=MCP_SERVER_PORT, host=MCP_SERVER_HOST)
```

### Shared FastMCP Instance — `tools/shared.py`

All domain tool modules import and decorate the same shared `FastMCP` instance, so they all register onto one server:

```python
# tools/shared.py (lines 62-68)
from fastmcp import FastMCP

mcp = FastMCP("netbrain-mcp-server")

# LLM state is lazily attached to the server instance
mcp.llm = None
mcp.llm_error = None
```

### Tool Registration Pattern

Each domain module registers tools using the `@mcp.tool()` decorator. The function signature (type annotations + docstring) is what MCP exposes as the tool's schema to clients:

```python
# tools/splunk_tools.py (lines 241-260)
@mcp.tool()
async def get_splunk_recent_denies(
    ip_address: str,
    limit: int = 100,
    earliest_time: str = "-24h"
) -> Dict[str, Any]:
    """Get the list of recent deny/denied events in Splunk for a given IP address."""
    return await _splunk_search_impl(ip_address, limit, earliest_time)
```

### All 8 Registered Tools

| Domain     | Tool Name                               |
|------------|-----------------------------------------|
| NetBrain   | `query_network_path`                    |
| NetBrain   | `check_path_allowed`                    |
| NetBox     | `get_rack_details`                      |
| NetBox     | `list_racks`                            |
| NetBox     | `get_device_rack_location`              |
| Panorama   | `query_panorama_ip_object_group`        |
| Panorama   | `query_panorama_address_group_members`  |
| Splunk     | `get_splunk_recent_denies`              |

---

## 2. MCP Client Connection and Communication

The MCP client (`mcp_client.py`) uses a **dual-transport strategy**: HTTP (FastMCPClient) as the primary transport, falling back to stdio if the server is unreachable.

### Session Management — `mcp_client.py`

```python
# mcp_client.py (lines 54-103)

def get_server_url():
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = os.getenv("MCP_SERVER_PORT", "8765")
    return f"http://{host}:{port}/mcp"

async def get_mcp_session():
    """Returns a live MCP session: HTTP primary, stdio fallback."""
    if FASTMCP_CLIENT_AVAILABLE:
        try:
            server_url = get_server_url()
            client = FastMCPClient(server_url, timeout=600)
            async with client:
                yield client          # HTTP session yielded
        except Exception:
            pass  # fall through to stdio

    # Stdio fallback
    server_params = StdioServerParameters(command="python", args=[server_path])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session             # stdio session yielded
```

### Tool Invocation with Timeout and Fallback — `mcp_client.py`

The client handles differences between the FastMCP client API and the standard MCP `ClientSession` API:

```python
# mcp_client.py (lines 220-280)
async def execute_network_query(source, destination, protocol, port, is_live):
    async for client_or_session in get_mcp_session():
        tool_arguments = {
            "source": source,
            "destination": destination,
            "protocol": protocol or "TCP",
            "port": port or "0",
            "is_live": 1 if is_live else 0,
        }

        is_fastmcp = isinstance(client_or_session, FastMCPClient)

        try:
            # Standard MCP format
            result = await asyncio.wait_for(
                client_or_session.call_tool("query_network_path", arguments=tool_arguments),
                timeout=360.0
            )
        except TypeError as e:
            if "unexpected keyword argument 'arguments'" in str(e) and is_fastmcp:
                # FastMCP kwargs format
                result = await asyncio.wait_for(
                    client_or_session.call_tool("query_network_path", **tool_arguments),
                    timeout=360.0
                )

        # Normalise result — both transports return different shapes
        if isinstance(result, list):
            result_text = result[0].text if result else None
        elif hasattr(result, 'content') and result.content:
            result_text = result.content[0].text
        else:
            result_text = str(result)

        return json.loads(result_text)
```

### Response Shape Differences by Transport

| Transport      | Result Type        | Access Pattern              |
|----------------|--------------------|-----------------------------|
| FastMCP HTTP   | `list[TextContent]`| `result[0].text`            |
| stdio (MCP SDK)| `CallToolResult`   | `result.content[0].text`    |

---

## 3. LangChain Integration

LangChain is used in two places:

1. **Tool selection** — an LLM determines which MCP tool to call and extracts parameters from the user's natural language query.
2. **AI enrichment** — MCP tool handlers use an LLM to generate human-readable summaries of structured API responses.

### 3.1 LLM Initialization — `tools/shared.py`

The LLM is lazily initialised to avoid startup failures if Ollama is unavailable:

```python
# tools/shared.py (lines 74-111)
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

def _get_llm():
    """Lazy singleton — returns None if Ollama is unavailable."""
    if mcp.llm is None:
        try:
            mcp.llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.0,
                base_url=OLLAMA_BASE_URL,
            )
        except Exception as e:
            mcp.llm = False          # False = tried and failed
            mcp.llm_error = {"error": str(e)}
    return mcp.llm if mcp.llm is not False else None
```

### 3.2 Structured Tool Selection — `mcp_client_tool_selection.py`

The tool selection module uses **Pydantic structured outputs** (`llm.with_structured_output(...)`) to get reliable, typed JSON from the LLM rather than parsing free-form text:

```python
# mcp_client_tool_selection.py (lines 197-356)
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

class ToolParameters(BaseModel):
    ip_address: Optional[str] = Field(None, description="IP address or CIDR")
    device_name: Optional[str] = Field(None, description="Network device name")
    rack_name: Optional[str]   = Field(None, description="Rack name")
    source: Optional[str]      = Field(None, description="Source IP for path query")
    destination: Optional[str] = Field(None, description="Destination IP for path query")
    # ... additional parameter fields

class ToolSelection(BaseModel):
    entity_analysis: Optional[str]        # LLM's reasoning about the query
    tool_name: Optional[str]              # Selected tool
    needs_clarification: bool             # Ask user for more info?
    clarification_question: Optional[str]
    parameters: ToolParameters            # Extracted call arguments

async def select_tool_with_llm(
    prompt: str,
    tools_description: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:

    llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)

    if PYDANTIC_AVAILABLE:
        structured_llm = llm.with_structured_output(ToolSelection)
        prompt_text = build_tool_selection_prompt(prompt, tools_description, conversation_history)
        response: ToolSelection = structured_llm.invoke(prompt_text)

        return {
            "success": True,
            "tool_name": response.tool_name,
            "parameters": response.parameters.model_dump(),
            "needs_clarification": response.needs_clarification,
            "clarification_question": response.clarification_question,
        }
    else:
        # Fallback: parse raw JSON from LLM text output
        ...
```

### 3.3 AI Enrichment inside MCP Tools — `tools/netbox_tools.py`

Inside the MCP server, tool handlers call `_get_llm()` to generate AI summaries of raw API data using `ChatPromptTemplate`:

```python
# tools/netbox_tools.py (lines 16-24)
from tools.shared import mcp, _get_llm, ChatPromptTemplate

# Inside a tool handler:
llm = _get_llm()
if llm:
    prompt = ChatPromptTemplate.from_template(
        "Summarise the following rack configuration for a network engineer:\n{data}"
    )
    chain = prompt | llm
    ai_summary = chain.invoke({"data": raw_rack_data})
```

---

## 4. Agent Loop with Retry Logic — `chat_service.py`

`chat_service.py` orchestrates the full request cycle. It runs an **agent loop** of up to 3 iterations: discover a tool, execute it, and on failure retry with the error added to conversation history so the LLM can self-correct.

```python
# chat_service.py (lines 1093-1207)
async def process_message(
    prompt: str,
    conversation_history: list,
    default_live: bool = True,
    max_iterations: int = 3,
    username: str | None = None,
) -> dict:

    history_so_far = list(conversation_history)

    for iteration in range(max_iterations):

        # --- Step 1: LLM selects tool ---
        selection = await discover_tool(prompt, history_so_far)

        if not selection.get("success"):
            if iteration == max_iterations - 1:
                # Final attempt failed → synthesise answer
                return {"role": "assistant",
                        "content": await synthesize_final_answer(prompt, last_tool_name, last_error)}
            history_so_far.append({"role": "assistant", "content": selection.get("error")})
            continue

        tool_name   = selection["tool_name"]
        tool_params = selection["parameters"]

        # --- Step 2: Execute tool via MCP client ---
        result = await execute_tool(tool_name, tool_params, default_live=default_live)

        if isinstance(result, dict) and "error" in result:
            if iteration == max_iterations - 1:
                return {"role": "assistant",
                        "content": await synthesize_final_answer(prompt, tool_name, result)}
            history_so_far.append({"role": "assistant", "content": result["error"]})
            continue

        # --- Step 3: Normalise and return ---
        return {"role": "assistant",
                "content": _normalize_result(tool_name, result, prompt)}
```

---

## 5. End-to-End Communication Flow

### Example: "Show network path from 10.0.0.1 to 192.168.1.1"

```
1. FastAPI receives POST /api/chat
   └─ Authenticates user (local or OIDC)
   └─ Calls chat_service.process_message(prompt, history)

2. chat_service.discover_tool()
   ├─ Keyword pre-check: 2 IPs + "path" keyword → in scope, skip LLM scope check
   └─ Calls select_tool_with_llm(prompt, tools_description)
       ├─ ChatOllama (qwen2.5:14b, temp=0.0) + structured output (Pydantic)
       └─ LLM returns:
           {
             "tool_name": "query_network_path",
             "parameters": {
               "source": "10.0.0.1",
               "destination": "192.168.1.1",
               "protocol": "TCP",
               "port": "0"
             }
           }

3. chat_service.execute_tool("query_network_path", params)
   └─ Calls mcp_client.execute_network_query(...)

4. mcp_client.execute_network_query()
   ├─ get_mcp_session() → FastMCPClient("http://127.0.0.1:8765/mcp")
   ├─ client.call_tool("query_network_path", **params)  [timeout: 360s]
   └─ Parses JSON result from response

5. MCP Server handles call_tool()
   ├─ Deserialises request (streamable-http transport)
   ├─ Dispatches to async def query_network_path(source, destination, ...)
   ├─ Function authenticates with NetBrain, calls path calculation API
   ├─ Polls until complete, enriches hops with Panorama data
   └─ Returns serialised JSON dict

6. MCP Client receives serialised result
   └─ Detects shape (list vs CallToolResult), extracts .text, parses JSON

7. chat_service normalises result
   └─ Strips L2 noise, formats hop list

8. FastAPI returns response to frontend
   {
     "role": "assistant",
     "content": {
       "path_status": "complete",
       "path_hops": [ { "from_device": "router1", "to_device": "switch1", ... }, ... ]
     }
   }
```

---

## 6. Configuration

All connection details are centralised in environment variables (`.env`), consumed in `tools/shared.py`:

```python
# tools/shared.py
NETBRAIN_URL     = os.getenv("NETBRAIN_URL",     "http://localhost")
NETBOX_URL       = os.getenv("NETBOX_URL",        "http://192.168.15.109:8080")
SPLUNK_HOST      = os.getenv("SPLUNK_HOST",       "192.168.15.110")
MCP_SERVER_HOST  = os.getenv("MCP_SERVER_HOST",   "127.0.0.1")
MCP_SERVER_PORT  = int(os.getenv("MCP_SERVER_PORT", "8765"))
OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",   "http://localhost:11434")
OLLAMA_MODEL     = os.getenv("OLLAMA_MODEL",      "qwen2.5:14b")
```

---

## 7. Role-Based Access Control (RBAC)

Tool access is enforced in `auth.py` before any MCP call is made:

```python
# auth.py (lines 57-71)
ROLE_ALLOWED_TOOLS = {
    "admin":    None,           # unrestricted
    "netadmin": {
        "query_network_path",
        "check_path_allowed",
        "query_panorama_ip_object_group",
        "query_panorama_address_group_members",
    },
    # netadmin cannot access Splunk or NetBox tools
}

def _check_tool_access(username: str, tool_name: str) -> str | None:
    role    = get_user_role(username)
    allowed = get_allowed_tools(role)
    if allowed is not None and tool_name not in allowed:
        return f"Your role ({role}) does not have access to {tool_name}"
    return None
```

---

## Summary

| Component                      | Role                                                         |
|-------------------------------|--------------------------------------------------------------|
| `mcp_server.py`               | FastMCP HTTP server, registers 8 tools, exposes `/mcp`      |
| `tools/*.py`                  | Domain logic; each registers tools via `@mcp.tool()`         |
| `tools/shared.py`             | Shared `FastMCP` instance, LLM singleton, config             |
| `mcp_client.py`               | HTTP-first MCP client; dual-transport (HTTP + stdio)         |
| `mcp_client_tool_selection.py`| LLM-driven tool selection with Pydantic structured output    |
| `chat_service.py`             | Agent loop: discover → execute → normalise, up to 3 retries  |
| `app_fastapi.py`              | FastAPI entry point, auth, routes                            |
| LangChain (`ChatOllama`)      | Tool selection inference + AI enrichment of tool responses   |

---

## 8. How to Integrate a New MCP Tool Domain

Adding a new backend (e.g. a CMDB, a ticketing system, a monitoring platform) involves touching **six files** in a fixed order. Each step below references the exact pattern used by the existing domains so you can follow by example.

### Overview of touch points

```
tools/your_tools.py              ← 1. New domain module (tool logic)
mcp_server.py                    ← 2. Import to register tools
mcp_client.py                    ← 3. Client-side execute function(s)
chat_service.py                  ← 4. Dispatch branch + optional normalisation
mcp_client_tool_selection.py     ← 5. ToolParameters fields for new params
auth.py                          ← 6. RBAC: add tool to role allow-lists
```

---

### Step 1 — Create `tools/your_tools.py`

Create a new file under `tools/`. Import the shared `mcp` instance and decorate every callable with `@mcp.tool()`. The decorator signature (type annotations + docstring) becomes the tool schema that both the MCP client and the LLM see.

```python
# tools/your_tools.py

from typing import Dict, Any
from tools.shared import mcp, setup_logging

logger = setup_logging(__name__)


@mcp.tool()
async def your_new_tool(
    query_param: str,
    optional_param: int = 10,
) -> Dict[str, Any]:
    """
    One-sentence description of what the tool does.

    Use this tool when the user asks about <topic>.
    Example queries:
      - "What is the status of <X>?"
      - "Show me <Y> for <Z>"

    Args:
        query_param:    The primary search value (required).
        optional_param: Controls result size (default 10).

    Returns a dict with keys: status, results, count.
    """
    # Call your backend API here
    raw = await _call_your_api(query_param, optional_param)
    return {
        "status": "ok",
        "results": raw,
        "count": len(raw),
    }
```

Key rules to follow (matching the existing pattern in e.g. `tools/panorama_tools.py` line 23):

- Import only `mcp` and `setup_logging` from `tools.shared`; add `_get_llm` / `ChatPromptTemplate` only if you need AI enrichment.
- Keep private helper functions prefixed with `_`; they are not MCP-exposed.
- Always return a plain `dict` (JSON-serialisable). Never raise — catch exceptions and return `{"error": "..."}`.

---

### Step 2 — Register the module in `mcp_server.py`

A bare `import` is enough. Importing the module triggers Python to execute the `@mcp.tool()` decorators, which registers the functions on the shared `FastMCP` instance.

```python
# mcp_server.py (lines 28-31 — add one line)

import tools.splunk_tools      # noqa: F401
import tools.netbox_tools      # noqa: F401
import tools.panorama_tools    # noqa: F401
import tools.netbrain_tools    # noqa: F401
import tools.your_tools        # noqa: F401  ← add this
```

After restarting the server, `GET /health` will show `tools_registered` incremented by however many tools your module decorates.

---

### Step 3 — Add an execute function in `mcp_client.py`

`chat_service.py` calls named helper functions in `mcp_client.py` for each tool. Add one async function that:

1. Opens a session via `get_mcp_session()`
2. Calls `client.call_tool()` with a timeout
3. Normalises the response shape (list vs `CallToolResult`)

```python
# mcp_client.py — add after the last execute_* function

async def execute_your_new_tool_query(query_param: str, optional_param: int = 10):
    """Execute your_new_tool via MCP."""
    try:
        async for client_or_session in get_mcp_session():
            tool_arguments = {
                "query_param": query_param,
                "optional_param": optional_param,
            }
            is_fastmcp = isinstance(client_or_session, FastMCPClient)

            try:
                result = await asyncio.wait_for(
                    client_or_session.call_tool("your_new_tool", arguments=tool_arguments),
                    timeout=90.0,
                )
            except TypeError as e:
                if "unexpected keyword argument 'arguments'" in str(e) and is_fastmcp:
                    result = await asyncio.wait_for(
                        client_or_session.call_tool("your_new_tool", **tool_arguments),
                        timeout=90.0,
                    )
                else:
                    raise

            # Normalise response shape (FastMCP returns list, stdio returns CallToolResult)
            if isinstance(result, list):
                result_text = result[0].text if result else None
            elif hasattr(result, "content") and result.content:
                result_text = result.content[0].text
            else:
                result_text = str(result)

            return json.loads(result_text)

    except asyncio.TimeoutError:
        return {"error": "your_new_tool timed out"}
    except Exception as e:
        return {"error": f"your_new_tool failed: {e}"}
```

Choose a timeout that suits your backend: `360.0` s is used for long-running NetBrain path traces, `90.0` s is used for Splunk/Panorama queries.

---

### Step 4 — Add a dispatch branch in `chat_service.py`

`execute_tool()` in `chat_service.py` is a large `if/elif` chain (starting around line 752). Add a branch for your new tool name:

```python
# chat_service.py — inside execute_tool(), after the last `if tool_name == ...` block

if tool_name == "your_new_tool":
    query_param = (tool_params.get("query_param") or "").strip()
    if not query_param:
        return {"error": "query_param not found in query."}
    result = await asyncio.wait_for(
        execute_your_new_tool_query(
            query_param,
            tool_params.get("optional_param", 10),
        ),
        timeout=95.0,
    )
    return result or {"error": "No result from your_new_tool."}
```

**Optional — add result normalisation** in `_normalize_result()` (around line 951) if you want to rewrite or annotate the raw dict before it reaches the frontend:

```python
# chat_service.py — inside _normalize_result()

if tool_name == "your_new_tool" and isinstance(result, dict) and "error" not in result:
    if result.get("count") == 0:
        result = dict(result)
        result["summary"] = "No results found."
    return result
```

---

### Step 5 — Add parameters to `ToolParameters` in `mcp_client_tool_selection.py`

The LLM uses the `ToolParameters` Pydantic model to extract call arguments from the user's natural language query. Add a field for each new parameter your tool needs:

```python
# mcp_client_tool_selection.py — inside class ToolParameters (lines 29-47)

class ToolParameters(BaseModel):
    # ... existing fields ...

    # Add your new fields:
    query_param: Optional[str] = Field(
        None,
        description="Primary search value for your_new_tool (e.g. 'ticket-1234', 'server-prod-01')"
    )
    optional_param: Optional[int] = Field(
        None,
        description="Result limit for your_new_tool (default 10)"
    )
```

Write the `description` string as if it were a prompt — the LLM reads it when deciding how to populate the field. Look at the existing `ip_address` field description (line 33) for the tone to use.

---

### Step 6 — Update RBAC in `auth.py`

Decide which roles should have access. The two structures to update are in `auth.py` (lines 57-71):

```python
# auth.py

ROLE_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "admin": None,                   # None = unrestricted
    "netadmin": {
        "query_network_path",
        "check_path_allowed",
        "query_panorama_ip_object_group",
        "query_panorama_address_group_members",
        "your_new_tool",             # ← add here if netadmin should have access
    },
}

ROLE_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin": None,
    "netadmin": ["netbrain", "panorama", "your_category"],  # ← add sidebar category slug
}
```

If a role is not in `ROLE_ALLOWED_TOOLS` or the tool name is absent from its set, `_check_tool_access()` will return an access-denied error before any MCP call is made.

---

### Checklist

| # | File | What to add |
|---|------|-------------|
| 1 | `tools/your_tools.py` | New file; `@mcp.tool()` decorated async functions |
| 2 | `mcp_server.py` | `import tools.your_tools` |
| 3 | `mcp_client.py` | `async def execute_your_new_tool_query(...)` |
| 4 | `chat_service.py` | `if tool_name == "your_new_tool":` dispatch branch; optional `_normalize_result` branch |
| 5 | `mcp_client_tool_selection.py` | New fields on `ToolParameters` |
| 6 | `auth.py` | Tool name in `ROLE_ALLOWED_TOOLS`; category slug in `ROLE_ALLOWED_CATEGORIES` |

### Verifying the integration

```bash
# 1. Restart the MCP server and confirm the tool is registered
curl http://127.0.0.1:8765/health
# → "tools_registered" should be 9 (or N+1)

# 2. Send a chat message that targets the new tool
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "your_new_tool trigger phrase", "session_id": "<sid>"}'

# 3. Check the MCP server log for the tool call
tail -f mcp_server.log
```

---

## Appendix

### A. What "LLM state is lazily attached to the server instance" means

#### Breaking it down

**"Lazily"** means: *not at startup — only on first use.*

**"Attached to the server instance"** means: the LLM object is stored directly on `mcp` (the `FastMCP` object) as a plain attribute, not in a separate module-level variable.

#### Why not initialise at startup?

The naive approach would be:

```python
# tools/shared.py — eager initialisation (what is NOT done)
mcp = FastMCP("netbrain-mcp-server")
mcp.llm = ChatOllama(model=OLLAMA_MODEL, ...)  # runs at import time
```

This fails if Ollama isn't running when the server starts — the import of `tools/shared.py` would crash, taking down the MCP server entirely, even though the LLM is only needed for AI-enrichment of certain tool responses (NetBox rack summaries etc.) and not for core network queries.

#### Three states of `mcp.llm`

```python
# tools/shared.py (lines 65-69)
mcp = FastMCP("netbrain-mcp-server")

mcp.llm = None        # initial state: "never tried"
mcp.llm_error = None
```

```python
# tools/shared.py (lines 82-111)
def _get_llm():
    if mcp.llm is None:       # State 1: never tried → attempt init
        try:
            mcp.llm = ChatOllama(
                model=OLLAMA_MODEL,
                temperature=0.0,
                base_url=OLLAMA_BASE_URL,
            )                 # State 2: success → mcp.llm holds the ChatOllama object
        except Exception as e:
            mcp.llm = False   # State 3: tried and failed → False sentinel, not None
            mcp.llm_error = {"error": str(e), ...}

    return mcp.llm if mcp.llm is not False else None
```

| `mcp.llm` value | Meaning | What `_get_llm()` returns |
|-----------------|---------|--------------------------|
| `None` | Never attempted | Tries to init, then returns result |
| `ChatOllama(...)` | Initialised successfully | Returns the cached LLM object |
| `False` | Tried and failed | Returns `None` — no retry |

The `False` sentinel is deliberate. Without it, every tool call on a downed Ollama would retry a new `ChatOllama()` connection and fail slowly, instead of failing fast after the first attempt.

#### Why attach it to `mcp` rather than a plain module variable?

A plain module-level variable would work functionally:

```python
# Alternative — plain module variable (not used)
_llm = None

def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(...)
    return _llm
```

But storing the LLM on `mcp` instead gives one concrete advantage: all domain tool modules (`netbox_tools.py`, `panorama_tools.py`, etc.) already import `mcp` from `tools/shared.py`. By piggy-backing on that same object, the cached LLM instance is **shared across all tool modules through a single well-known object** with no extra imports and no risk of each module holding its own separate instance.

#### Graceful degradation in tool handlers

Every tool handler that uses the LLM guards the call with `if llm:`, so the tool still returns raw API data if Ollama is unavailable — just without the AI-generated summary:

```python
# tools/netbox_tools.py — typical usage pattern
llm = _get_llm()
if llm:
    chain = ChatPromptTemplate.from_template("Summarise:\n{data}") | llm
    result["ai_summary"] = chain.invoke({"data": raw_data}).content
# if llm is None, result is returned without ai_summary — no crash
```
