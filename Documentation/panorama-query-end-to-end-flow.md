# Panorama query end-to-end flow

This document traces what happens from the moment a user types a Panorama-related query in the chat to the moment the response is shown. Example query:

> **"What address group is 11.0.0.1 part of?"**

---

## High-level flow

```
User types query in UI
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  Frontend (React or static)                                         │
│  POST /api/chat  { message, conversation_history, conversation_id }│
│  Credentials: session cookie                                        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  FastAPI  (app_fastapi.py :8000)                                   │
│  • Validate session → username                                      │
│  • POST /api/chat → process_message(message, history, username)    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│  Chat service  (chat_service.py)                                   │
│  1. discover_tool(prompt, history)                                  │
│  2. RBAC: user must have "panorama" category                        │
│  3. execute_tool(tool_name, tool_params)                            │
│  4. _normalize_result(tool_name, result, prompt)                     │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
         ┌────────────────────┴────────────────────┐
         │  discover_tool                           │  execute_tool
         ▼                                         ▼
┌─────────────────────┐                 ┌─────────────────────────────┐
│  MCP session        │                 │  MCP client (mcp_client.py)   │
│  list_tools()       │                 │  execute_panorama_ip_        │
│  LLM selects tool   │                 │  object_group_query(...)     │
│  + parameters       │                 │  or execute_panorama_        │
│  (Ollama)           │                 │  address_group_members_query  │
└─────────────────────┘                 └──────────────┬────────────────┘
                                                       │  HTTP to MCP server :8765
                                                       │  call_tool("query_panorama_...", arguments)
                                                       ▼
                                        ┌─────────────────────────────┐
                                        │  MCP server (mcp_server.py) │
                                        │  tools.panorama_tools       │
                                        │  query_panorama_ip_object_  │
                                        │  group(...)                 │
                                        └──────────────┬──────────────┘
                                                       │
                                                       ▼
                                        ┌─────────────────────────────┐
                                        │  Panorama tools             │
                                        │  (tools/panorama_tools.py)  │
                                        │  • panoramaauth.get_api_key │
                                        │  • Panorama REST API (XML)  │
                                        └──────────────┬──────────────┘
                                                       │
                                                       ▼
                                        ┌─────────────────────────────┐
                                        │  Panorama (Palo Alto)       │
                                        │  REST API :443              │
                                        │  /api/?type=config&action=  │
                                        │  get&xpath=...&key=...      │
                                        └─────────────────────────────┘
```

---

## Step-by-step

### 1. User types a Panorama query

The user enters something like:

- "What address group is 11.0.0.1 part of?"
- "What IPs are in address group leander_web?"

**Where:** Browser (React app at port 8000 or 5173, or static Jinja2 UI).

---

### 2. Frontend sends the message to the backend

The chat store (or static `app.js`) sends:

- **Endpoint:** `POST /api/chat`
- **Body:** `{ message: "<user text>", conversation_history: [...], conversation_id?: "<id>" }`
- **Credentials:** Session cookie (same-origin).

Optional: the frontend may first call `POST /api/discover` with the same message to get the tool name (e.g. for a "Querying Panorama" status). The actual run still goes through `POST /api/chat`.

**Files:** `frontend/src/stores/chatStore.js` → `sendChat()` in `frontend/src/utils/api.js`; or `static/js/app.js` for the static UI.

---

### 3. FastAPI receives the request

- **Route:** `POST /api/chat` in `app_fastapi.py`.
- **Auth:** `get_current_username(request)` resolves the user from the session cookie. If missing or invalid → 401 and redirect to login.
- **Action:** Call `process_message(body.message, body.conversation_history, username=..., session_id=...)`.
- **After response:** Persist the exchange via `append_to_conversation()` (or create new conversation) and return `{ content, ... }` to the client.

**File:** `netbrain/app_fastapi.py`.

---

### 4. Chat service: tool discovery

`process_message()` calls `discover_tool(prompt, conversation_history)`:

1. **MCP session**  
   The chat service obtains an MCP client session (`get_mcp_session()`) and calls `list_tools()` so it has the full list of tools (including Panorama tools).

2. **Scope check**  
   An LLM classifies whether the query is in scope. If not, the service returns a clarification or error and does not execute a tool.

3. **LLM tool selection**  
   The prompt and tool list (with short descriptions) are sent to the LLM (Ollama, via `mcp_client_tool_selection.select_tool_with_llm`). The LLM picks:
   - **Tool name**, e.g. `query_panorama_ip_object_group` or `query_panorama_address_group_members`.
   - **Parameters**, e.g. `{ "ip_address": "11.0.0.1" }` or `{ "address_group_name": "leander_web" }`.

4. **Special cases**  
   - If the user sent only an IP (e.g. "11.0.0.1") with no context, discovery can return a clarification question instead of a tool.
   - Rack follow-ups (e.g. "which site?") are handled before the LLM and can return `get_rack_details` with site.

**Files:** `netbrain/chat_service.py` (`discover_tool`), `netbrain/mcp_client_tool_selection.py`.

**Panorama tools:**

- **query_panorama_ip_object_group** — User has an **IP** and asks which address/object groups contain it. Input: `ip_address` (and optional `device_group`, `vsys`).
- **query_panorama_address_group_members** — User has a **group name** and asks for the list of IPs in that group. Input: `address_group_name` (and optional `device_group`, `vsys`).

---

### 5. Chat service: RBAC check

Before running the tool, the chat service checks that the user is allowed to use Panorama:

- It gets the role for the session (e.g. from auth / session).
- Allowed tool categories are defined in `auth.py` (e.g. `get_allowed_tools(role)`). Panorama tools map to the `"panorama"` category.
- If the role does not have access to `"panorama"`, the service returns an error message and does not call the MCP client.

**File:** `netbrain/chat_service.py` (`_check_tool_access`), `netbrain/auth.py`.

---

### 6. Chat service: execute tool

`process_message()` calls `execute_tool(tool_name, tool_params)`.

For Panorama:

- **query_panorama_ip_object_group**  
  `execute_tool` calls `execute_panorama_ip_object_group_query(ip_address, device_group, vsys)` from the MCP client (with a timeout).

- **query_panorama_address_group_members**  
  `execute_tool` calls `execute_panorama_address_group_members_query(address_group_name, device_group, vsys)`.

**File:** `netbrain/chat_service.py` (`execute_tool`).

---

### 7. MCP client calls the MCP server

- **Connection:** The MCP client uses `get_mcp_session()` to get a session to the MCP server (HTTP, typically `MCP_SERVER_HOST:MCP_SERVER_PORT`, e.g. `127.0.0.1:8765`).

- **Call:** It invokes the tool by name with the chosen arguments, e.g.  
  `call_tool("query_panorama_ip_object_group", arguments={"ip_address": "11.0.0.1", "vsys": "vsys1"})`.

- **Response:** The server returns a result (e.g. JSON text in the MCP response). The client parses it and returns a dict to the chat service.

**File:** `netbrain/mcp_client.py` (`execute_panorama_ip_object_group_query`, `execute_panorama_address_group_members_query`).

---

### 8. MCP server runs the Panorama tool

- The MCP server is the FastMCP app that registers all tools. It imports `tools.panorama_tools`, so the Panorama tools are registered (e.g. `query_panorama_ip_object_group`, `query_panorama_address_group_members`).

- When the client calls `query_panorama_ip_object_group`, the server invokes the async function `query_panorama_ip_object_group(ip_address, device_group, vsys)` in `tools/panorama_tools.py`.

**Files:** `netbrain/mcp_server.py` (imports `tools.panorama_tools`), `tools/panorama_tools.py` (tool implementations).

---

### 9. Panorama tool: auth and API calls

Inside `tools/panorama_tools.py`:

1. **API key**  
   The tool calls `await panoramaauth.get_api_key()`.  
   - **panoramaauth** loads Panorama username/password from env or Azure Key Vault (`PANORAMA-USERNAME`, `PANORAMA-PASSWORD`).  
   - It uses them to call Panorama’s keygen endpoint and caches the API key.  
   - If auth fails, the tool returns an error dict (e.g. "Failed to authenticate with Panorama...").

2. **Panorama URL**  
   Base URL comes from `panoramaauth.PANORAMA_URL` (env `PANORAMA_URL`).

3. **REST calls**  
   The tool builds Panorama REST URLs, e.g.:  
   `{PANORAMA_URL}/api/?type=config&action=get&xpath=...&key={api_key}`  
   and uses aiohttp to send GET requests. Panorama returns XML.

4. **Parsing**  
   The tool parses the XML (e.g. with `xml.etree.ElementTree`), finds address objects, address groups, and policies that reference the IP or the group, and builds a structured result dict (e.g. `ip_address`, `address_objects`, `address_groups`, `members`, policies, etc.).

5. **Return**  
   The tool returns that dict. The MCP server serializes it (e.g. to JSON) and sends it back to the MCP client.

**Files:** `tools/panorama_tools.py`, `netbrain/panoramaauth.py`.

---

### 10. Result back through chat service

- The MCP client returns the result dict to `execute_tool()` in the chat service.
- The chat service may pass it through `_normalize_result(tool_name, result, prompt)` to shape it for the UI (e.g. add a short summary for `query_panorama_ip_object_group`).
- `process_message()` then returns `{ "role": "assistant", "content": <result or normalized structure> }` to FastAPI.

**File:** `netbrain/chat_service.py` (`_normalize_result`, `process_message`).

---

### 11. FastAPI response and persistence

- FastAPI returns the assistant `content` (and any conversation metadata) in the HTTP response.
- It also appends the user message and assistant response to chat history via `chat_history.append_to_conversation()` (or creates a new conversation). Chat files are stored under `data/chats/` and are encrypted at rest when Key Vault is configured.

**File:** `netbrain/app_fastapi.py`.

---

### 12. Frontend displays the response

- The frontend receives the JSON response and updates the UI: it appends the assistant message (and may update the conversation list if a new conversation was created).
- The user sees the Panorama result (e.g. address groups containing 11.0.0.1, or the list of IPs in a group) in the chat.

**Files:** `frontend/src/stores/chatStore.js`, `frontend/src/components/chat/ChatMessages.jsx` (or static `app.js`).

---

## Summary table

| Step | Component | What happens |
|------|-----------|--------------|
| 1 | User | Types a Panorama query in the chat input |
| 2 | Frontend | POST /api/chat with message and history (and optional conversation_id) |
| 3 | FastAPI | Validates session, calls process_message() |
| 4 | Chat service | discover_tool() → MCP list_tools() + LLM → selects Panorama tool + params |
| 5 | Chat service | RBAC: user must have "panorama" category |
| 6 | Chat service | execute_tool() → MCP client (execute_panorama_*_query) |
| 7 | MCP client | HTTP to MCP server :8765, call_tool("query_panorama_...", arguments) |
| 8 | MCP server | Invokes tools.panorama_tools.query_panorama_* |
| 9 | Panorama tools | get_api_key() from panoramaauth → Panorama REST API (XML) → parse → result dict |
| 10 | Chat service | _normalize_result() → return { content } to FastAPI |
| 11 | FastAPI | Return response; append to chat_history (encrypted at rest) |
| 12 | Frontend | Render assistant message in the chat |

---

## Related docs

- General query flow (NetBrain path example): [query-end-to-end-flow.md](query-end-to-end-flow.md)
- Auth and RBAC: [auth-rbac.md](auth-rbac.md)
- Chat history storage and encryption: [chat-history-security.md](chat-history-security.md)
