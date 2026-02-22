# Panorama query end-to-end flow

This document traces a **single example query** from the moment the user types it to the response, and explains **why** the chat service and LLM do tool discovery, **when** authentication and authorization happen, **how** MCP tools are exposed, and **what** the client is and how it consumes Panorama tools. It also lists **all components involved**, including LangChain, Ollama, and Pydantic.

---

## Example query (used throughout)

We follow this one question from end to end:

> **"What address group is 11.0.0.1 part of?"**

The user expects an answer like: *"11.0.0.1 is in address groups: leander_web, dmz_servers."*

---

## Components involved (full stack)

| Layer | Component | Role |
|--------|-----------|------|
| **Frontend** | Browser (React SPA) | Sends message to `/api/chat`, displays response; never talks to MCP or Panorama. |
| **API** | FastAPI (`app.py`) | Authn (session cookie), routes `/api/chat`, persists chat history, returns JSON. |
| **Orchestration** | Chat service (`chat_service.py`) | Coordinates discovery → authz → execute; calls MCP client and tool-selection LLM. |
| **Tool selection** | `mcp_client_tool_selection.py` | **LangChain** + **ChatOllama** (Ollama) + **Pydantic** (`ToolSelection`, `ToolParameters`): maps natural language to one MCP tool name and parameters. |
| **LLM runtime** | Ollama | Local LLM server (e.g. `llama3.1:8b`); used by tool selection and by optional result summarization. |
| **MCP client** | `mcp_client.py` | Gets MCP session, calls `list_tools()` and `call_tool(name, arguments)` over HTTP to the MCP server. |
| **MCP server** | `mcp_server.py` | Serves FastMCP app; loads `tools/panorama_tools`, `tools/netbrain_tools`, etc.; exposes tools over streamable-http (:8765). |
| **Tools + LLM** | `tools/panorama_tools.py` | Implements `query_panorama_ip_object_group` (and address group members); calls Panorama REST API; optionally uses **LangChain** (`ChatPromptTemplate` + `_get_llm()` from `tools/shared.py`) to add a short **ai_analysis** narrative summary. |
| **Shared LLM** | `tools/shared.py` | Holds FastMCP instance, **OLLAMA_MODEL** / **OLLAMA_BASE_URL**; **`_get_llm()`** lazily creates **ChatOllama** (LangChain) for use by Panorama/NetBox tools for result summarization. |
| **Backend** | Panorama (Palo Alto) | REST API (XML); returns address objects/groups and policy usage. |

So for a Panorama query, the full chain is: **Browser → FastAPI → Chat service → (MCP client + list_tools) → Tool selection (LangChain + ChatOllama + Pydantic) → Chat service → authz → MCP client → call_tool → MCP server → tools/panorama_tools → Panorama API → (optional LangChain summarization in panorama_tools) → back up the stack.**

---

## High-level flow (same example)

```
User types: "What address group is 11.0.0.1 part of?"
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Browser (frontend)                                                │
│  POST /api/chat  { message: "What address group...", history: [] } │
│  Cookie: atlas_session=<signed>                                 │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼  [1] AUTHENTICATION (who are you?)
┌───────────────────────────────────────────────────────────────────┐
│  FastAPI (app.py :8000)                                    │
│  get_current_username(request) → session cookie → username         │
│  If no valid session → 401 + redirect to /login                     │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Chat service (chat_service.py)                                    │
│  discover_tool("What address group is 11.0.0.1 part of?", [])      │
│    → MCP list_tools() + LLM → tool_name + params                    │
│  [2] AUTHORIZATION (are you allowed this tool?)                     │
│  execute_tool("query_panorama_ip_object_group", { ip_address: ... })│
│  _normalize_result() → content for UI                               │
└─────────────────────────────┬─────────────────────────────────────┘
                              │
         discover_tool        │        execute_tool
         ▼                    │                    ▼
┌─────────────────────┐       │       ┌─────────────────────────────┐
│  MCP client         │       │       │  MCP client (mcp_client.py)  │
│  list_tools()       │       │       │  call_tool(...)             │
└──────────┬──────────┘       │       └──────────────┬──────────────┘
           │                   │                      │
           ▼                   │                      │
┌─────────────────────┐       │                      │
│  Tool selection     │       │                      │
│  (mcp_client_tool_  │       │                      │
│   selection.py)     │       │                      │
│  LangChain +        │       │                      │
│  ChatOllama (Ollama)│       │                      │
│  Pydantic           │       │                      │
│  (ToolSelection,     │       │                      │
│   ToolParameters)    │       │                      │
│  → tool_name +      │       │                      │
│    parameters        │       │                      │
└─────────────────────┘       │                      │
                              │                       │  HTTP POST to MCP server
                              │                       ▼
                              │       ┌─────────────────────────────┐
                              │       │  MCP server (mcp_server.py) │
                              │       │  :8765  transport: streamable-http
                              │       │  Tools registered by       │
                              │       │  import tools.panorama_tools │
                              │       └──────────────┬──────────────┘
                              │                      │
                              │                      ▼
                              │       ┌─────────────────────────────┐
                              │       │  tools/panorama_tools.py    │
                              │       │  query_panorama_ip_object_  │
                              │       │  group(ip_address="11.0.0.1")│
                              │       │  → panoramaauth → Panorama  │
                              │       │  → (optional) LangChain      │
                              │       │     ChatPromptTemplate +    │
                              │       │     _get_llm() → ai_analysis │
                              │       └──────────────┬──────────────┘
                              │                      │
                              │                      ▼
                              │       ┌─────────────────────────────┐
                              │       │  Panorama (Palo Alto)       │
                              │       │  /api/?type=config&action=   │
                              │       │  get&xpath=...&key=<key>     │
                              │       └─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Response: { content: { ip_address, address_objects, address_groups } }
│  Persist to chat history; return to browser → user sees answer.     │
└───────────────────────────────────────────────────────────────────┘
```

---

## 1. Example query step-by-step

### 1.1 User sends the message

- **User action:** Types *"What address group is 11.0.0.1 part of?"* and sends.
- **Frontend:** Builds `POST /api/chat` with body `{ message: "What address group is 11.0.0.1 part of?", conversation_history: [] }` and sends with the session cookie.

### 1.2 FastAPI: authentication (when it happens)

- **Route:** `POST /api/chat` in `app.py`.
- **Authentication (authn)** happens **here**, before any chat logic:
  - `get_current_username(request)` reads the session cookie (`SESSION_COOKIE`), deserializes and validates it (signed cookie, not expired).
  - If valid → returns **username** (and session carries **role**).
  - If missing or invalid → returns `None` → handler returns **401** and clears session / redirects to login.
- So: **the user is identified and proven once per request at the FastAPI boundary.** No MCP or Panorama code runs if authn fails.

### 1.3 Chat service: tool discovery (why chat_service + LLM do it; why it feels redundant)

`process_message()` first calls `discover_tool(prompt, conversation_history)`.

**What discovery does:**

1. **Get tool list from MCP**  
   The chat service obtains an MCP session (`get_mcp_session()`) and calls `list_tools()`. The MCP server returns **all** registered tools (NetBox, Atlas path, Panorama, Splunk, etc.) with names and descriptions.

2. **Decide if the query is in scope**  
   An LLM (or heuristics) decides whether the query is something the system can answer (e.g. not off-topic). If not, the service returns a clarification or error and **does not** run any tool.

3. **LLM selects one tool and its parameters**  
   The user prompt and the tool list are sent to **`select_tool_with_llm()`** in **`mcp_client_tool_selection.py`**. That module uses **LangChain** and **ChatOllama** to call the local **Ollama** server (e.g. `llama3.1:8b`) and **Pydantic** structured output (`ToolSelection`, `ToolParameters`) so the LLM returns:
   - **Tool name:** e.g. `query_panorama_ip_object_group`
   - **Parameters:** e.g. `{ "ip_address": "11.0.0.1", "vsys": "vsys1" }`

   For our example, the LLM sees the question and the description of `query_panorama_ip_object_group` (“user has one IP and asks which address group contains it”) and correctly picks this tool with `ip_address: "11.0.0.1"`. So **LangChain**, **ChatOllama**, **Pydantic**, and **Ollama** are the components that perform tool discovery.

**Why use the LLM at all?**

- The app has **many** tools (path, rack, device, Panorama IP group, Panorama group members, Splunk, etc.). The same words can mean different tools (e.g. “11.0.0.1” could be Panorama, NetBox device, or path source). The LLM disambiguates **intent** and maps natural language to **one** tool and its parameters.

**Why it can feel redundant:**

- After discovery, the chat service has `tool_name` and `parameters`. It then calls `execute_tool(tool_name, tool_params)`.
- `execute_tool()` is a **large if/elif** that maps `tool_name` to a **specific Python function** in `mcp_client.py` (e.g. `execute_panorama_ip_object_group_query(...)`). That function opens **another** MCP session and does nothing more than `call_tool("query_panorama_ip_object_group", arguments=tool_arguments)`.
- So in principle the app could:
  - Use **one** MCP session for both `list_tools()` and `call_tool(tool_name, arguments)`.
  - Have a **single** generic path: “call_tool(selection[‘tool_name’], selection[‘parameters'])” instead of one wrapper per tool.
- The current design keeps **per-tool** logic in the FastAPI app: validation (e.g. “IP address required”), timeouts (e.g. 65s for Panorama), and a clear place to hook **authorization** (see below). So there is some duplication between “what the LLM chose” and “what the app decides to run,” but it is intentional for control and safety.

### 1.4 Chat service: authorization (when it happens)

- **After** discovery and **before** running the tool, the chat service calls `_check_tool_access(username, tool_name, session_id)`.
- **Authorization (authz)** happens **here**:
  - The user’s **role** is resolved from the session (e.g. via `get_role_for_session(session_id)` in `auth.py`).
  - `get_allowed_tools(role)` returns the set of MCP tool names that role may use (e.g. for `netadmin`: `query_network_path`, `check_path_allowed`, `query_panorama_ip_object_group`, `query_panorama_address_group_members`). `None` means “all tools.”
  - If `tool_name` is not in that set, the service returns an error message to the user and **does not** call the MCP client.
- So: **authn** at the FastAPI boundary; **authz** in the chat service right before `execute_tool()`.

### 1.5 Execute tool → MCP client → MCP server → Panorama

- For our example, `execute_tool("query_panorama_ip_object_group", { "ip_address": "11.0.0.1", "vsys": "vsys1" })` runs.
- The chat service calls `execute_panorama_ip_object_group_query("11.0.0.1", None, "vsys1")` in **mcp_client.py**.
- The MCP client:
  - Calls `get_mcp_session()` (HTTP to `MCP_SERVER_HOST:MCP_SERVER_PORT`, e.g. `http://127.0.0.1:8765/mcp`).
  - Calls `client.call_tool("query_panorama_ip_object_group", arguments={"ip_address": "11.0.0.1", "vsys": "vsys1"})`.
- The MCP server receives the call and invokes **tools/panorama_tools.py** `query_panorama_ip_object_group(ip_address="11.0.0.1", ...)`, which uses **panoramaauth** to get an API key, calls the Panorama REST API, and parses XML into a result dict.
- **Optional LLM summarization (LangChain):** After the Panorama API returns, **panorama_tools** may call **`_get_llm()`** from **tools/shared.py** (which returns a **LangChain** **ChatOllama** instance). Using **ChatPromptTemplate** from **langchain_core**, it sends the raw result to the LLM and gets a short narrative summary, which is attached as **`result["ai_analysis"]["summary"]`**. This step uses the same Ollama model as tool selection; if Ollama is unavailable, a short fallback summary is used.
- The result (with or without `ai_analysis`) is returned over MCP to the client, then to the chat service, which may run `_normalize_result()` and then returns `{ role: "assistant", content: ... }` to FastAPI.
- FastAPI persists the exchange to chat history and returns the response to the browser; the user sees the answer.

---

## 2. When authentication and authorization happen

| Moment | What | Where |
|--------|------|--------|
| **Authentication** | Prove who the user is (valid session). | **FastAPI** before handling `POST /api/chat`: `get_current_username(request)`. No session → 401, no chat or MCP. |
| **Authorization** | Decide whether that user may run the chosen tool. | **Chat service** after `discover_tool()` and before `execute_tool()`: `_check_tool_access(username, tool_name, session_id)` using `get_allowed_tools(role)`. No access → error message, no MCP call. |

So: **authn at the API edge; authz right before executing the selected MCP tool.**

---

## 3. How MCP tools are exposed

- **Single MCP server process** (e.g. `python mcp_server.py`) runs the **FastMCP** app from `tools/shared.py` (`mcp = FastMCP("atlas-mcp-server")`).
- **Tool registration** is by **import**: `mcp_server.py` imports `tools.panorama_tools` (and other domain modules). Each domain module imports `mcp` from `tools.shared` and decorates functions with `@mcp.tool()`.
- In **tools/panorama_tools.py**:
  - `@mcp.tool()` on `query_panorama_ip_object_group(ip_address, device_group, vsys)` 
  - `@mcp.tool()` on `query_panorama_address_group_members(address_group_name, device_group, vsys)`
- FastMCP registers these with the MCP protocol (name, description, input schema from the function signature). The server listens over **streamable-http** (e.g. port 8765, path `/mcp`). So **tools are exposed** as MCP tools on that endpoint; any MCP client that can connect to that URL can call `list_tools()` and `call_tool(name, arguments)`.

---

## 4. What the client is and how it consumes Panorama tools

- **“Client” in this context** is **not** the browser. The browser is the **frontend**; it only talks to the FastAPI app (`POST /api/chat`).
- The **MCP client** is the **Python code in the Atlas process** that talks to the MCP server. It lives in **mcp_client.py** and is used only by the **chat service** (same process as FastAPI).
- **How it consumes Panorama tools:**
  1. **Chat service** calls `execute_panorama_ip_object_group_query(ip_address, device_group, vsys)` (or the “address group members” variant) in **mcp_client.py**.
  2. **mcp_client** gets a session to the MCP server via `get_mcp_session()` (HTTP to `http://<MCP_SERVER_HOST>:<MCP_SERVER_PORT>/mcp`).
  3. It calls `call_tool("query_panorama_ip_object_group", arguments={...})` (or `query_panorama_address_group_members` with the right arguments).
  4. The MCP server executes the corresponding `@mcp.tool()` function in **tools/panorama_tools.py**, which talks to Panorama and returns a result.
  5. The MCP client parses the tool result (e.g. JSON text in the MCP response) and returns a dict to the chat service.

So: **the only “consumer” of Panorama MCP tools is the MCP client in mcp_client.py, on behalf of the chat service.** The browser never talks to MCP or Panorama; it only talks to FastAPI.

---

## 5. Summary table (example query)

| Step | Component | What happens for “What address group is 11.0.0.1 part of?” |
|------|-----------|-------------------------------------------------------------|
| 1 | User / frontend | User sends message; frontend POSTs to /api/chat with cookie. |
| 2 | FastAPI | **Authn:** validate session → username; else 401. |
| 3 | Chat service | `discover_tool()`: MCP `list_tools()` then **mcp_client_tool_selection** (LangChain + ChatOllama + Pydantic + Ollama) → `query_panorama_ip_object_group`, `{ ip_address: "11.0.0.1" }`. |
| 4 | Chat service | **Authz:** `_check_tool_access()` → role allowed for this tool? Else error. |
| 5 | Chat service | `execute_tool()` → `execute_panorama_ip_object_group_query("11.0.0.1", ...)` in mcp_client. |
| 6 | MCP client | New MCP session; `call_tool("query_panorama_ip_object_group", arguments={...})` over HTTP to server. |
| 7 | MCP server | Runs `query_panorama_ip_object_group` in tools/panorama_tools.py. |
| 8 | Panorama tools | panoramaauth → Panorama REST API → parse XML → result dict; optionally **LangChain** (ChatPromptTemplate + _get_llm()) → **ai_analysis** summary. |
| 9 | Back up stack | Result → chat service → _normalize_result() → FastAPI → persist + response. |
| 10 | Frontend | User sees address groups containing 11.0.0.1. |

---

## Related docs

- General query flow (path query example): [query-end-to-end-flow.md](query-end-to-end-flow.md)
- Auth and RBAC: [auth-rbac.md](auth-rbac.md)
- Chat history storage and encryption: [chat-history-security.md](chat-history-security.md)
