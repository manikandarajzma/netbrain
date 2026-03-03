# Splunk Query — End-to-End Flow

This document traces the complete lifecycle of a Splunk query through Atlas, from the moment a user types a message in the browser to the rendered response. One Splunk tool is covered:

- **`get_splunk_recent_denies`** — "Recent deny events for 10.0.0.1"

---

## Architecture Overview

```
Browser (React + Zustand)
    │  POST /api/discover  (tool discovery)
    │  POST /api/chat      (full query)
    ▼
FastAPI (app.py)
    │  Session cookie validation (auth.py)
    │  RBAC check (auth.py)
    ▼
chat_service.py  process_message()
    │  Fetches MCP tool schemas
    │  LangChain ChatOllama + bind_tools()
    │  LLM selects tool + extracts args
    ▼
MCP Server (mcp_server.py — separate process)
    │  FastMCP over streamable-http
    │  Dispatches to splunk_tools.py
    ▼
splunk_tools.py  _splunk_search_impl()
    │  Step 1: POST /services/auth/login          (get session key)
    │  Step 2: POST /services/search/jobs         (create search job)
    │  Step 3: GET  /services/search/jobs/{sid}   (poll until done)
    │  Step 4: GET  /services/search/jobs/{sid}/results  (fetch results)
    │  Step 5: Normalize events (extract fields from _raw)
    ▼
chat_service.py  _normalize_result()
    ▼
FastAPI → JSON response → React
```

---

## Step 1: User Types a Query (Frontend)

**File:** [frontend/src/components/chat/ChatInput.jsx](../../frontend/src/components/chat/ChatInput.jsx)
**Store:** [frontend/src/stores/chatStore.js](../../frontend/src/stores/chatStore.js)

The user types `"Recent deny events for 10.0.0.1"` and presses Enter.

`chatStore.sendMessage(text)` runs:

1. **Disambiguation check** — inspects the last assistant message in `conversationHistory`. If it had `requires_site: true`, it would combine site + reply. For Splunk queries this path is never triggered.

2. **UI state** — `isLoading: true`, `currentStatus: 'Identifying query'`.

3. **History is intentionally empty** — `historySlice = []`. Each query is stateless; prior context is not sent to the LLM.

---

## Step 2: Tool Discovery (Frontend → FastAPI)

**File:** [frontend/src/utils/api.js](../../frontend/src/utils/api.js) → `discoverTool()`

```
POST /api/discover
Content-Type: application/json
Cookie: atlas_session=<signed-cookie>

{ "message": "Recent deny events for 10.0.0.1", "conversation_history": [] }
```

- `fetch` includes `credentials: 'include'` (cookies sent automatically).
- A 2 min client-side timeout is applied via `AbortSignal`.
- On 401, the browser is redirected to `/login`.

The response contains `tool_display_name: "Splunk"`. The UI updates `currentStatus` to `"Querying Splunk"`.

---

## Step 3: Session & RBAC Check (FastAPI)

**File:** [app.py](../../app.py), [auth.py](../../auth.py)

### Session validation

```python
# app.py
def get_current_username(request: Request) -> str | None:
    sid = request.cookies.get("atlas_session")
    return get_username_for_session(sid)
```

`get_username_for_session` calls `get_session(sid)` which uses `itsdangerous.URLSafeTimedSerializer` to verify the cookie signature and TTL (30 minutes). The cookie contains `{username, role, auth_mode, created_at}` — no server-side session store; the cookie IS the session.

If the cookie is missing, invalid, or expired → 401 + `{"redirect": "/login"}`.

### RBAC check

The role is resolved from the session cookie. RBAC is enforced later inside `chat_service.process_message()`. The `get_splunk_recent_denies` tool is accessible to `admin` only (not in the `netadmin` allowed set):

```python
# auth.py
ROLE_ALLOWED_TOOLS = {
    "admin":    None,       # all tools — includes get_splunk_recent_denies
    "netadmin": {           # Splunk not listed here
        "query_network_path", "check_path_allowed",
        "query_panorama_ip_object_group", "query_panorama_address_group_members"
    },
    "guest":    set(),      # no tools
}
```

### How roles are assigned (OIDC login flow)

When a user logs in via Microsoft (`GET /auth/microsoft` → Microsoft login page → `GET /auth/callback`):

1. `authlib` exchanges the authorization code for tokens (id_token + access_token).
2. `extract_role_from_token(userinfo)` in [auth.py](../../auth.py) checks, in priority order:
   - **Azure app roles** — `roles` claim in the token (e.g. `"admin"`, `"netadmin"`).
   - **Azure security group → role map** — `OIDC_GROUP_ROLE_MAP` env var (group object ID → role).
3. If no role resolves → 302 redirect to `/login?error=norole`.
4. A signed session cookie is set (`atlas_session`, `HttpOnly`, `SameSite=Lax`, 30 min TTL).

---

## Step 4: FastAPI Routes to chat_service

**File:** [app.py](../../app.py)

```python
@app.post("/api/discover")
async def api_discover(request: Request, body: ChatRequest):
    result = await process_message(
        body.message.strip(),
        body.conversation_history or [],
        discover_only=True,
        username=username,
        session_id=get_session_id(request),
    )
    return result
```

The `ChatRequest` Pydantic model validates the request body:

```python
class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict[str, Any]] = []
    conversation_id: str | None = None
    parent_conversation_id: str | None = None
```

Pydantic enforces types and provides defaults. If `message` is missing or not a string, FastAPI returns a 422 Unprocessable Entity automatically.

---

## Step 5: Tool Discovery in chat_service

**File:** [chat_service.py](../../chat_service.py)

### Fetching MCP tool schemas

```python
mcp_tools = await _fetch_mcp_tools()
```

`_fetch_mcp_tools()` connects to the MCP server at `http://127.0.0.1:8765` using the MCP streamable-http transport and calls `list_tools()`. The result is cached for the process lifetime. Each tool's schema includes its name, docstring-derived description, and JSON Schema for parameters.

### Building tool descriptions for the LLM

```python
def _to_openai_tool(t) -> dict:
    # Strips everything from "Args:" onward, keeps up to 600 chars
    if desc:
        args_idx = desc.find("\n    Args:")
        if args_idx > 0:
            desc = desc[:args_idx].strip()
        desc = desc[:600]
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}
```

The description passed to the LLM for `get_splunk_recent_denies` includes:

```
Search Splunk for recent firewall deny events involving a given IP address.

Use for: queries asking for deny/denial events for an IP — "recent denies for X",
"deny events for X", "Splunk denies for X", "firewall denials for X".
Do NOT use for: path queries, device/rack lookups, address group lookups.

Examples:
- "recent denies for 10.0.0.1" → ip_address="10.0.0.1"
- "latest 10 denies for 10.0.0.250" → ip_address="10.0.0.250", limit=10
```

### LLM tool selection (LangChain + Ollama)

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434", temperature=0.0)
llm_with_tools = llm.bind_tools(openai_tools, tool_choice="required")
messages = _build_llm_messages(prompt, conversation_history)
ai_msg = await asyncio.wait_for(llm_with_tools.ainvoke(messages), timeout=90.0)
```

`tool_choice="required"` forces the LLM to always output a structured tool call. The LLM also extracts optional parameters — if the user specifies a limit (`"latest 10 denies"`) or time range (`"last 7 days"`), those values are extracted as `limit` and `earliest_time`.

The LLM responds with a structured `tool_calls` entry:
```json
{
  "name": "get_splunk_recent_denies",
  "args": { "ip_address": "10.0.0.1", "limit": 100, "earliest_time": "-24h" }
}
```

### RBAC enforcement (before tool execution)

```python
def _check_tool_access(username, tool_name, session_id):
    role = get_role_for_session(session_id)
    allowed = get_allowed_tools(role)
    if allowed is not None and tool_name not in allowed:
        return f"Your role ({role}) does not have access to this tool."
    return None
```

If the user's role forbids the tool (e.g. `netadmin`), an error message is returned immediately — the tool is never called.

### discover_only mode

When called from `/api/discover`, the function returns immediately after tool selection:

```python
if discover_only:
    return {
        "tool_name": "get_splunk_recent_denies",
        "parameters": {"ip_address": "10.0.0.1"},
        "tool_display_name": "Splunk",
        "format": "table",
    }
```

---

## Step 6: Full Chat Request (Frontend → FastAPI)

After discovery, the frontend fires the actual chat request:

```
POST /api/chat
{ "message": "Recent deny events for 10.0.0.1", "conversation_history": [] }
```

`process_message()` runs again from scratch (not reusing the discover result), this time with `discover_only=False`. The LLM is invoked again, makes the same tool selection, and proceeds to execution.

---

## Step 7: MCP Tool Execution

**File:** [chat_service.py](../../chat_service.py) → `call_mcp_tool()`
**File:** [mcp_client.py](../../mcp_client.py)

```python
result = await call_mcp_tool(
    "get_splunk_recent_denies",
    {"ip_address": "10.0.0.1", "limit": 100, "earliest_time": "-24h"},
    timeout=65.0
)
```

The MCP client sends a JSON-RPC `tools/call` message over the streamable-http transport to `http://127.0.0.1:8765`. The MCP server (running as a separate process) receives it and dispatches to the registered `@mcp.tool()` handler.

---

## Step 8: splunk_tools.py — Tool Execution

**File:** [tools/splunk_tools.py](../../tools/splunk_tools.py) → `_splunk_search_impl()`

Splunk credentials (`SPLUNK_USER`, `SPLUNK_PASSWORD`) are loaded at MCP server startup from [tools/shared.py](../../tools/shared.py). The loading priority is:

1. Environment variables (`SPLUNK_USER`, `SPLUNK_PASSWORD` in `.env`)
2. Azure Key Vault (secrets `SPLUNK-USER`, `SPLUNK-PASSWORD` via `DefaultAzureCredential`)

All API calls use port **8089** — the Splunk management/REST API (HTTPS). Port 8000 is the web UI and does not expose the REST endpoints. SSL certificate verification is disabled (`ssl.CERT_NONE`) for self-signed certificates.

```python
base_url = f"https://{SPLUNK_HOST}:{SPLUNK_PORT}"   # e.g. https://192.168.15.110:8089
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
```

### Step 8a: Authenticate — get Splunk session key

```
POST https://192.168.15.110:8089/services/auth/login
Content-Type: application/x-www-form-urlencoded

username=<SPLUNK_USER>&password=<SPLUNK_PASSWORD>
```

Splunk returns XML (not JSON):

```xml
<response>
  <sessionKey>VtBVLwYGRwYKGqzlCzMuYg==...</sessionKey>
</response>
```

The session key is parsed with `xml.etree.ElementTree` and used in all subsequent requests:

```python
headers = {"Authorization": f"Splunk {session_key}"}
```

> **Note:** Unlike NetBrain and Panorama, Splunk does not cache the session key — a fresh login is performed for every query. Splunk session keys are short-lived and tied to a single search context.

### Step 8b: Build the SPL search query

The search is constructed in Python before submission:

```python
search = (
    f'search index=* (deny OR denied) '
    f'(src_ip="{ip_address}" OR dest_ip="{ip_address}" '
    f'OR src="{ip_address}" OR dst="{ip_address}" '
    f'OR src_ip={ip_address} OR dest_ip={ip_address}) '
    f'earliest={earliest_time} | head {limit}'
)
```

**Breaking down the query:**

| Clause | Purpose |
|---|---|
| `index=*` | Search across all Splunk indexes |
| `deny OR denied` | Filter to firewall deny/denied events |
| `src_ip="..." OR dest_ip="..."` | Match quoted IP in standard CIM fields |
| `OR src="..." OR dst="..."` | Match quoted IP in alternative field names |
| `OR src_ip=... OR dest_ip=...` | Match unquoted IP (some Splunk configs omit quotes) |
| `earliest={earliest_time}` | Time window (default `-24h` = last 24 hours) |
| `\| head {limit}` | Cap results (default 100) |

The query intentionally uses six different field name variants (`src_ip`, `dest_ip`, `src`, `dst` with and without quotes) to handle variations across different log sources and Splunk configurations (Palo Alto, Cisco ASA, generic syslog).

### Step 8c: Create the search job

```
POST https://192.168.15.110:8089/services/search/jobs
Authorization: Splunk <session_key>
Content-Type: application/x-www-form-urlencoded

search=<SPL query>&output_mode=json
```

Splunk responds with a job ID (`sid`):

```json
{ "sid": "1706123456.12345" }
```

### Step 8d: Poll until the job completes

```
GET https://192.168.15.110:8089/services/search/jobs/{sid}?output_mode=json
Authorization: Splunk <session_key>
```

The tool polls every 2 seconds for up to 120 seconds (60 attempts):

```python
for _ in range(60):
    await asyncio.sleep(2)
    async with session.get(status_url, ...) as resp:
        data = await resp.json()
        content = data["entry"][0]["content"]
        if content.get("isDone") is True:
            break
```

`isDone: true` in the `content` object signals the job has finished.

### Step 8e: Fetch results

```
GET https://192.168.15.110:8089/services/search/jobs/{sid}/results?output_mode=json
Authorization: Splunk <session_key>
```

Splunk returns the results as a JSON array:

```json
{
  "results": [
    {
      "_time": "2024-01-15T10:23:45.000+00:00",
      "_raw": "Jan 15 10:23:45 leander-dc-fw1 1,2024/01/15 10:23:45,...,vsys1,trust,untrust,...,deny",
      "src_ip": "10.0.0.1",
      "dest_ip": "192.168.100.5",
      "action": "deny",
      "host": "leander-dc-fw1",
      ...
    }
  ]
}
```

### Step 8f: Normalize events

Each raw Splunk event is normalized into a consistent display dict. Splunk field names and formats vary by vendor and log source, so the tool uses several extraction strategies:

**Standard field lookup (case-insensitive):**

```python
def _get(e, *keys):
    for k in keys:
        if k in e and e[k] not in (None, ""):
            return e[k]
    # Also try case-insensitive match
    for k, v in e.items():
        if k.lower() in {x.lower() for x in keys} and v not in (None, ""):
            return v
    return ""
```

**Protocol extraction — also parses `_raw`:**

Some Splunk sources do not expose `protocol` as a top-level field; it only appears inside the raw log line. The tool searches several patterns:

```python
def _get_protocol(e):
    # 1. Try named fields: protocol, proto, transport
    v = _get(e, "protocol", "Protocol", "proto", "transport")
    if v:
        return v
    # 2. Parse _raw: "protocol=icmp" or "Protocol: icmp"
    raw = e.get("_raw", "")
    for m in re.finditer(r"(?:protocol|proto)\s*[=:]\s*(\w+)", raw, re.IGNORECASE):
        return m.group(1).strip()
    # 3. Match common protocol keywords in _raw
    for m in re.finditer(r"\b(icmp|tcp|udp|gre|esp|ip)\b", raw, re.IGNORECASE):
        return m.group(1).lower()
    return ""
```

**Zone extraction — Palo Alto CSV format:**

Palo Alto TRAFFIC logs are comma-separated. The tool recognizes the `vsys` field as an anchor to extract `from_zone` and `to_zone`:

```python
def _get_palo_alto_zones(raw):
    # Palo Alto TRAFFIC log CSV: ...,vsys1,from_zone,to_zone,...
    m = re.search(r",vsys\d*\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,", raw)
    if m:
        return (m.group(1).strip(), m.group(2).strip())
    return ("", "")
```

For non-Palo Alto sources, the tool falls back to key=value pattern matching (`from_zone=...`, `to_zone=...`) and JSON patterns within `_raw`.

**Firewall/device name extraction:**

```python
def _get_firewall(e):
    # Try standard fields first
    v = _get(e, "dvc_name", "host", "device", "firewall", "hostname", "DeviceName")
    if v:
        return v
    # Palo Alto syslog format: "timestamp hostname 1,CSV..."
    raw = e.get("_raw", "")
    m = re.search(r"\d{1,2}:\d{2}:\d{2}\s+(\S+)\s+\d,", raw)
    if m:
        return m.group(1).strip()
    return ""
```

**Final normalized event:**

```python
{
    "time":           "2024-01-15T10:23:45.000+00:00",
    "firewall":       "leander-dc-fw1",
    "vendor_product": "Palo Alto Networks Firewall",
    "src_ip":         "10.0.0.1",
    "dst_ip":         "192.168.100.5",
    "src_zone":       "trust",
    "dest_zone":      "untrust",
    "protocol":       "tcp",
    "port":           "443",
    "action":         "deny"
}
```

### Return value

```python
{
    "ip_address": "10.0.0.1",
    "events": [
        {
            "time": "2024-01-15T10:23:45.000+00:00",
            "firewall": "leander-dc-fw1",
            "vendor_product": "Palo Alto Networks Firewall",
            "src_ip": "10.0.0.1",
            "dst_ip": "192.168.100.5",
            "src_zone": "trust",
            "dest_zone": "untrust",
            "protocol": "tcp",
            "port": "443",
            "action": "deny"
        },
        ...
    ],
    "count": 15,
    "sample_keys": ["_time", "_raw", "src_ip", "dest_ip", ...],  # debug only
    "sample_raw": "Jan 15 10:23:45 leander-dc-fw1 1,..."         # debug only
}
```

---

## Step 9: Result Normalization (chat_service.py)

**File:** [chat_service.py](../../chat_service.py) → `_normalize_result()`

For `get_splunk_recent_denies`, normalization adds a user-facing message when no events are found:

```python
if tool_name == "get_splunk_recent_denies" and isinstance(result, dict):
    if result.get("count") == 0 and "error" not in result:
        ip = result.get("ip_address", "this IP")
        result["message"] = (
            f"No deny events found for {ip} in the last 24 hours. "
            "Try a different IP or time range, or check that Splunk has "
            "Palo Alto logs for that period."
        )
```

When events are found, the result is passed through unchanged. No `direct_answer` or `yes_no_answer` is added — the events table provides the full answer.

---

## Step 10: Conversation History Persistence (FastAPI)

**File:** [app.py](../../app.py), [chat_history.py](../../chat_history.py)

After `process_message()` returns:

```python
if conversation_id:
    append_to_conversation(APP_DIR, username, conversation_id, user_msg, assistant_content)
else:
    title = user_msg[:60] + "…"
    conv_id = create_conversation(APP_DIR, username, title)
    append_to_conversation(APP_DIR, username, conv_id, user_msg, assistant_content)
    result["conversation_id"] = conv_id
```

Conversations are stored per-user on disk, encrypted at rest (AES-256-GCM). The `conversation_id` is returned in the response so the frontend can track it.

---

## Step 11: Response Rendering (Frontend)

**File:** [frontend/src/stores/chatStore.js](../../frontend/src/stores/chatStore.js)

```js
const content = data.content ?? data.message ?? 'No response'
addMessage('assistant', content)
```

### Response classification

**File:** [frontend/src/utils/responseClassifier.js](../../frontend/src/utils/responseClassifier.js)

The result has an `events` array of objects → classified as `'table'`:

```js
const arrayKeys = Object.keys(content).filter(k => Array.isArray(content[k]) && ...)
// → arrayKeys = ["events"]
return { type: 'table', content }
```

If `count === 0` and a `message` was added by normalization:

```js
if (content.message) {
    return { type: 'structured', content }
}
```

### AssistantMessage rendering

**File:** [frontend/src/components/messages/AssistantMessage.jsx](../../frontend/src/components/messages/AssistantMessage.jsx)

**Events found** → rendered as a horizontal `DataTable`:

```jsx
{classified.type === 'table' && (
    tableGroups.map((group, gi) => (
        <div key={gi}>
            {group.heading && <p>{group.heading}</p>}
            <DataTable rows={group.rows} columns={group.columns || null} />
        </div>
    ))
)}
```

The `events` array maps directly to table rows. Column order follows the normalized field order: `time`, `firewall`, `vendor_product`, `src_ip`, `dst_ip`, `src_zone`, `dest_zone`, `protocol`, `port`, `action`.

**No events found** → rendered as a message:

```jsx
{classified.type === 'structured' && structuredText && (
    <p>{structuredText}</p>
)}
// "No deny events found for 10.0.0.1 in the last 24 hours. Try a different IP..."
```

---

## Error Paths

| Scenario | Where caught | Response |
|---|---|---|
| Unauthenticated request | `app.py` | 401 + `{"redirect": "/login"}` |
| Expired session cookie | `auth.py` `get_session()` | 401 |
| No Azure role assigned | `auth.py` `auth_callback` | 302 `/login?error=norole` |
| Role lacks Splunk access (`netadmin`) | `chat_service._check_tool_access` | `"Your role (netadmin) does not have access to this tool."` |
| Splunk unreachable | `_splunk_search_impl` | `{"ip_address": ..., "events": [], "error": "..."}` |
| Splunk login failure (bad credentials) | `_splunk_search_impl` | `{"error": "Splunk login failed: 401 - ..."}` |
| No `sessionKey` in login response | `_splunk_search_impl` | `{"error": "Splunk login response had no sessionKey"}` |
| Job creation failure | `_splunk_search_impl` | `{"error": "Splunk create job failed: ..."}` |
| No `sid` returned | `_splunk_search_impl` | `{"error": "Splunk job created but no sid returned"}` |
| Poll timeout (job never completes) | `_splunk_search_impl` | Proceeds to fetch partial/empty results |
| Results fetch failure | `_splunk_search_impl` | `{"error": "Splunk results failed: ..."}` |
| Request timeout | `asyncio.TimeoutError` | `{"error": "Splunk request timed out"}` |
| No deny events found | `_normalize_result` | `{"count": 0, "events": [], "message": "No deny events found..."}` |
| LLM narrates instead of calling tool | `chat_service` | `"I could not determine how to answer that."` |
| LLM timeout (>90s) | `chat_service` | `"Tool selection timed out. Please try again."` |
| Tool timeout (>65s) | `mcp_client.call_mcp_tool` | Error propagated → `synthesize_final_answer()` |
| MCP server unreachable | `_fetch_mcp_tools()` | `"Could not connect to MCP server."` |

---

## Configuration Reference

| Variable | Source | Purpose |
|---|---|---|
| `SPLUNK_HOST` | `.env` | Splunk server hostname or IP |
| `SPLUNK_PORT` | `.env` | Splunk REST API port (default `8089`) |
| `SPLUNK_USER` | `.env` or Key Vault (`SPLUNK-USER`) | Splunk username |
| `SPLUNK_PASSWORD` | `.env` or Key Vault (`SPLUNK-PASSWORD`) | Splunk password |
| `AZURE_KEYVAULT_URL` | `.env` | Key Vault URL for secret retrieval |
| `AZURE_TENANT_ID` | `.env` | Azure AD tenant for OIDC login |
| `AZURE_CLIENT_ID` | `.env` | App registration client ID |
| `AZURE_CLIENT_SECRET` | `.env` | App registration client secret |
| `OLLAMA_MODEL` | `.env` | LLM for tool selection (currently `llama3.1:8b`) |
| `OLLAMA_BASE_URL` | `.env` | Ollama server address |
| `MCP_SERVER_HOST/PORT` | `.env` | MCP server address (default `127.0.0.1:8765`) |
| `CORS_ALLOWED_ORIGINS` | `.env` | Browser origins allowed by CORS middleware |

---

## Sequence Diagram

```
User          Browser           FastAPI          chat_service       MCP Server       splunk_tools      Splunk
 │  type query  │                  │                   │                 │                 │               │
 │─────────────►│                  │                   │                 │                 │               │
 │              │ POST /api/discover│                   │                 │                 │               │
 │              │─────────────────►│                   │                 │                 │               │
 │              │                  │ validate session   │                 │                 │               │
 │              │                  │ check RBAC         │                 │                 │               │
 │              │                  │──process_message──►│                 │                 │               │
 │              │                  │                   │ list_tools()───►│                 │               │
 │              │                  │                   │◄────────────────│                 │               │
 │              │                  │                   │ LLM: select tool│                 │               │
 │              │                  │◄─ tool_display_name─│               │                 │               │
 │              │◄─ {tool:"Splunk"} │                  │                 │                 │               │
 │              │ POST /api/chat   │                   │                 │                 │               │
 │              │─────────────────►│                   │                 │                 │               │
 │              │                  │──process_message──►│                 │                 │               │
 │              │                  │                   │ LLM: select tool│                 │               │
 │              │                  │                   │ call_mcp_tool()─►                │               │
 │              │                  │                   │                 │ POST /auth/login │               │
 │              │                  │                   │                 │─────────────────────────────── ►│
 │              │                  │                   │                 │◄──────────────── session key ───│
 │              │                  │                   │                 │ build SPL query  │               │
 │              │                  │                   │                 │ POST /search/jobs│               │
 │              │                  │                   │                 │─────────────────────────────── ►│
 │              │                  │                   │                 │◄──────────────── sid ───────────│
 │              │                  │                   │                 │ GET /jobs/{sid} (poll)           │
 │              │                  │                   │                 │─────────────────────────────── ►│
 │              │                  │                   │                 │◄──────────────── isDone:true ───│
 │              │                  │                   │                 │ GET /jobs/{sid}/results          │
 │              │                  │                   │                 │─────────────────────────────── ►│
 │              │                  │                   │                 │◄──────────────── JSON results ──│
 │              │                  │                   │◄── normalized ──│                 │               │
 │              │                  │                   │    events dict  │                 │               │
 │              │                  │                   │ _normalize_result│                │               │
 │              │                  │◄── normalized JSON─│                │                 │               │
 │              │◄─ {content: {...}}│                  │                 │                 │               │
 │              │ render events    │                   │                 │                 │               │
 │              │ table            │                   │                 │                 │               │
 │◄─────────────│                  │                   │                 │                 │               │
```

---
