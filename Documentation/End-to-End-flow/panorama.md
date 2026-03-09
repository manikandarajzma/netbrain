# Panorama Query — End-to-End Flow

This document traces the complete lifecycle of a Panorama query through Atlas, from the moment a user types a message in the browser to the rendered response. Two Panorama tools are covered:

- **`query_panorama_ip_object_group`** — "What address group is 11.0.0.1 part of?"
- **`query_panorama_address_group_members`** — "What IPs are in address group leander_web?"

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
    │  Dispatches to panorama_tools.py
    ▼
panoramaauth.py
    │  Gets/caches Panorama API key (username+password → keygen)
    │  SSL context (certificate verification disabled)
    ▼
Panorama REST/XML API (https://192.168.15.247)
    │  Returns XML
    ▼
panorama_tools.py  (result processing)
    ▼
chat_service.py  _normalize_result()
    ▼
FastAPI → JSON response → React
```

---

## Step 1: User Types a Query (Frontend)

**File:** [frontend/src/components/chat/ChatInput.jsx](../../frontend/src/components/chat/ChatInput.jsx)
**Store:** [frontend/src/stores/chatStore.js](../../frontend/src/stores/chatStore.js)

The user types `"What address group is 11.0.0.1 part of?"` and presses Enter.

`chatStore.sendMessage(text)` runs:

1. **Disambiguation check** — inspects the last assistant message in `conversationHistory`. If it had `requires_site: true` and `rack`, it would combine the rack name + user reply into a single query. For Panorama queries this path is never triggered; `textToSend` remains the original text.

2. **UI state** — `isLoading: true`, `currentStatus: 'Identifying query'`.

3. **History is intentionally empty** — `historySlice = []`. Each query is stateless; prior context is not sent to the LLM to prevent pollution.

---

## Step 2: Tool Pre-selection (Frontend → FastAPI)

**File:** [frontend/src/utils/api.js](../../frontend/src/utils/api.js) → `discoverTool()`

```
POST /api/discover
Content-Type: application/json
Cookie: atlas_session=<signed-cookie>

{ "message": "What address group is 11.0.0.1 part of?", "conversation_history": [] }
```

- The `atlas_session` cookie is sent automatically by the browser — it identifies who is making the request and drives RBAC. See [FAQ](#faq).
- An `AbortController` is created per message send; its `signal` is shared by both `/api/discover` and `/api/chat`. The stop button calls `ctrl.abort()` to cancel both in-flight requests. See [FAQ](#faq).
- On 401, the browser is immediately redirected to `/login`. See [FAQ](#faq).

> **What `/api/discover` actually does:** Despite the name, this is not MCP tool list discovery. It invokes `process_message(..., discover_only=True)` in `chat_service.py`, which runs a full LLM call — the LLM selects the appropriate tool and extracts arguments from the prompt — but stops before executing the tool. The response is `{ tool_name, parameters, tool_display_name, intent }`. No backend system (Panorama) is contacted at this point.
>
> The sole purpose is **UI feedback**. The calls are sequential: `/api/discover` is awaited first → `tool_display_name: "Panorama"` is returned → `currentStatus` updates to `"Querying Panorama"` → only then does `/api/chat` fire to actually execute the query. If `/api/discover` fails, `currentStatus` falls back to `"Processing"` but `/api/chat` still runs.
>
> The cost of this UX feature is **one full redundant LLM call per query** — the tool selection that `/api/discover` performs is repeated from scratch by `/api/chat`.

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

The RBAC check for Panorama tools happens later inside `chat_service.process_message()`, but the role is resolved from the session here:

```python
# auth.py
ROLE_ALLOWED_TOOLS = {
    "admin":    None,                                    # all tools
    "netadmin": {"query_network_path", "check_path_allowed",
                 "query_panorama_ip_object_group",
                 "query_panorama_address_group_members"},
    "guest":    set(),                                   # no tools
}
```

Both Panorama tools are accessible to `admin` and `netadmin` roles. `guest` is denied.

### How roles are assigned (OIDC login flow)

When a user logs in via Microsoft (`GET /auth/microsoft` → Microsoft login page → `GET /auth/callback`):

1. Atlas redirects the browser to `https://login.microsoftonline.com/{tenant_id}/v2.0/authorize`. Azure handles all credential verification (password, MFA) — Atlas never sees the password. Azure redirects back to `/auth/callback` with a short-lived one-time authorization code.

2. `authlib` makes a server-to-server POST to Azure's token endpoint, exchanging the code for tokens. Azure responds with an `id_token`, `access_token`, and `refresh_token`.

   > **What is a JWT?** A JSON Web Token is a string of three base64url-encoded segments separated by dots: `header.payload.signature`. The **payload** is a plain JSON object containing claims (`"email"`, `"groups"`, etc.) — readable by anyone, but not encrypted. The **signature** is a cryptographic hash signed with Azure's private key — it proves the token came from Azure and has not been tampered with. authlib verifies the signature against Azure's public keys before Atlas reads any claims.

3. Atlas manually decodes the `id_token` JWT payload to extract the `groups` claim. The `groups` claim is a Microsoft extension that only appears in the `id_token` — it is never returned by the standard `/userinfo` endpoint.

4. `extract_group_from_token()` in [auth.py](../../auth.py) iterates the `groups` claim and matches each value directly against the keys of `GROUP_ALLOWED_TOOLS` (e.g. `"admin"`, `"netadmin"`). This works because on-prem synced groups emit the `sAMAccountName` in the token — the group name is the same string Atlas uses internally, so no mapping is needed. If no group matches → 302 redirect to `/login?error=norole`.

5. A signed session cookie is set (`atlas_session`, `HttpOnly`, `SameSite=Lax`, 30 min TTL) containing `{ username, group, auth_mode, created_at }`. No PIM, no app roles — auth is entirely group-based.

For full detail on the login flow see [auth-rbac.md](../Security/auth-rbac.md).

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

`_fetch_mcp_tools()` connects to the MCP server at `http://127.0.0.1:8765` using the MCP streamable-http transport and calls `list_tools()`. The result is cached for the process lifetime (reset on restart). Each tool's schema includes its name, docstring-derived description, and JSON Schema for parameters.

### Building tool descriptions for the LLM

> **Why is it called `_to_openai_tool`?**
> The name refers to the output format, not the provider. OpenAI introduced the function-calling JSON schema (`{"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}`), which has since become the de facto standard adopted by LangChain's `bind_tools()`. Even though Atlas uses Ollama/Llama as the LLM, LangChain still expects tools in this OpenAI-compatible format.

```python
def _to_openai_tool(t) -> dict:
    # Strips everything from "Args:" onward, keeps up to 600 chars
    # This preserves "Use for / Do NOT use for / Examples" sections
    if desc:
        args_idx = desc.find("\n    Args:")
        if args_idx > 0:
            desc = desc[:args_idx].strip()
        desc = desc[:600]
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": schema}}
```

The description passed to the LLM for `query_panorama_ip_object_group` includes:

```
Find which Panorama address groups contain a given IP address.

Use for: queries with an IP address (has dots, e.g. "10.0.0.1") asking which address group/object group it belongs to.
Do NOT use for: device names (have dashes, use get_device_rack_location), ...

Examples:
- "what address group is 10.0.0.1 in?" → ip_address="10.0.0.1"
...
```

This rich context guides the LLM to select the correct tool.

### LLM tool selection (LangChain + Ollama)

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434", temperature=0.0)
llm_with_tools = llm.bind_tools(openai_tools, tool_choice="required")
messages = _build_llm_messages(prompt, conversation_history)
ai_msg = await asyncio.wait_for(llm_with_tools.ainvoke(messages), timeout=90.0)
```

**System prompt** tells the LLM:
- Always call a tool — never answer from memory.
- IP addresses → `query_panorama_ip_object_group` or `get_splunk_recent_denies`.
- Address group names → `query_panorama_address_group_members`.

`tool_choice="required"` forces the LLM to always output a structured tool call rather than a text response.

The LLM responds with a structured `tool_calls` entry:
```json
{
  "name": "query_panorama_ip_object_group",
  "args": { "ip_address": "11.0.0.1" }
}
```

### RBAC enforcement (before tool execution)

```python
def _check_tool_access(username, tool_name, session_id):
    role = get_role_for_session(session_id)
    allowed = get_allowed_tools(role)
    if allowed is not None and tool_name not in allowed:
        return f"Your role ({role}) does not have access to Panorama queries."
    return None
```

If the user's role forbids the tool, an error message is returned immediately — the tool is never called.

### discover_only mode

When called from `/api/discover`, the function returns immediately after tool selection:

```python
if discover_only:
    return {
        "tool_name": "query_panorama_ip_object_group",
        "parameters": {"ip_address": "11.0.0.1"},
        "tool_display_name": "Panorama",
        "format": "table",
    }
```

---

## Step 6: Full Chat Request (Frontend → FastAPI)

After discovery, the frontend fires the actual chat request:

```
POST /api/chat
{ "message": "What address group is 11.0.0.1 part of?", "conversation_history": [] }
```

`process_message()` runs again from scratch (not reusing the discover result), this time with `discover_only=False`. The LLM is invoked again, makes the same tool selection, and proceeds to execution.

---

## Step 7: MCP Tool Execution

**File:** [chat_service.py](../../chat_service.py) → `call_mcp_tool()`
**File:** [mcp_client.py](../../mcp_client.py)

```python
result = await call_mcp_tool(
    "query_panorama_ip_object_group",
    {"ip_address": "11.0.0.1"},
    timeout=65.0
)
```

The MCP client sends a JSON-RPC `tools/call` message over the streamable-http transport to `http://127.0.0.1:8765`. The MCP server (running as a separate process) receives it and dispatches to the registered `@mcp.tool()` handler.

---

## Step 8: panorama_tools.py — Tool Execution

**File:** [tools/panorama_tools.py](../../tools/panorama_tools.py)

### Panorama API key retrieval (panoramaauth.py)

**File:** [panoramaauth.py](../../panoramaauth.py)

```python
api_key = await panoramaauth.get_api_key()
```

`get_api_key()` checks for a module-level cached key first. If none:

1. Loads `PANORAMA_USERNAME` and `PANORAMA_PASSWORD` exclusively from **Azure Key Vault** using `DefaultAzureCredential`:
   ```python
   from azure.identity import DefaultAzureCredential
   from azure.keyvault.secrets import SecretClient
   _client = SecretClient(vault_url=AZURE_KEYVAULT_URL, credential=_cred)
   PANORAMA_USERNAME = _client.get_secret("PANORAMA-USERNAME").value
   PANORAMA_PASSWORD = _client.get_secret("PANORAMA-PASSWORD").value
   ```
   If `AZURE_KEYVAULT_URL` is not set, credentials are unavailable and authentication fails.

2. Calls the Panorama XML API keygen endpoint:
   ```
   GET https://192.168.15.247/api/?type=keygen&user=<user>&password=<encoded>
   ```
   SSL certificate verification is disabled (`ssl.CERT_NONE`) because Panorama uses a self-signed certificate.
   > **Production note:** `PANORAMA_VERIFY_SSL=false` should be validated before production deployment. If Panorama has a valid CA-signed certificate, remove the SSL bypass (`ssl.CERT_NONE` / `check_hostname=False`) in `panoramaauth.py` and set `PANORAMA_VERIFY_SSL=true` in `.env`.

3. Parses the XML response:
   ```xml
   <response status="success"><result><key>LUFRPT14...</key></result></response>
   ```
   Extracts and caches the key in the module-level `_api_key` variable.

---

### Caching and parallel processing

To avoid hammering the Panorama appliance on every query, the toolbox includes two layers of caching plus concurrent fetches.

#### Device‑group / address‑object cache
`panorama_tools._get_address_objects_cached()` and `_get_address_groups_for_location()` each maintain an in‑memory cache keyed by location (e.g. `"device-group:leander-dg"` or `"shared"`).

```python
now = _time.monotonic()
cached = _addr_obj_cache.get(key)
if cached and (now - cached[1]) < _CACHE_TTL:
    logger.debug("Address objects: cache hit for %s (%d objects)", key, len(cached[0]))
    return cached[0]
# fetch from Panorama and on success store:
_addr_obj_cache[key] = (objects, now)
```

- TTL is five minutes (`_CACHE_TTL`).
- After a cache miss the first query pays the HTTP cost; subsequent requests serve the cached value until expiry.
- If Panorama credentials expire the code clears both the API key cache and the location cache to force a re‑auth on the next request.
- Cache entries do **not** auto‑refresh; they only update when a fetch is performed. Manual invalidation requires restarting the MCP server.

Log messages at DEBUG level indicate hits/misses and object counts:
```
Device groups: cache hit (2 groups)
Address objects: fetched 6001 from device-group:leander
```

#### API key
A fresh API key is fetched from Panorama's `keygen` endpoint on every tool invocation — no caching. This ensures a stale or expired key is never reused.

#### Parallel HTTP requests
Once the list of locations (shared + device groups) is determined, requests for objects/groups are dispatched concurrently with `asyncio.gather()`: this turns an N‑round‑trip operation into a single parallel batch.

```python
addr_obj_results = await asyncio.gather(
    *[_get_address_objects_cached(session, panorama_url, api_key, ssl_context, lt, ln)
      for lt, ln in locations],
    return_exceptions=True,
)
```

The same pattern appears later when fetching address groups. `return_exceptions=True` ensures a failure on one location doesn’t abort the whole query; errors are logged and skipped.

| Phase | Before | After |
|-------|--------|-------|
| DG list | 1 HTTP call | cached (0–1) |
| Objects | serial N calls | N parallel + cache |
| Groups | serial N calls | N parallel |

On a warm cache with two groups, a typical IP lookup now needs **2–4 total HTTP calls** instead of 20+.  

These optimizations are what keep Panorama queries snappy even when the configuration contains thousands of address objects.

---

### API retrieval details

Every interaction with Panorama is a simple GET to the device’s `/api/` endpoint. URLs are built dynamically using the configured `PANORAMA_URL`, the current API key, and a URL‑quoted XPath expression. For example, to fetch address objects from a device group:

```python
xpath = (
    f"/config/devices/entry[@name='localhost.localdomain']"
    f"/device-group/entry[@name='{location_name}']/address"
)
url = (
    f"{panorama_url}/api/?type=config&action=get"
    f"&xpath={urllib.parse.quote(xpath)}&key={api_key}"
)
async with session.get(url, ssl=ssl_context,
                       timeout=aiohttp.ClientTimeout(total=45)) as resp:
    resp_text = await resp.text()
    # xml parsing follows here…
```

The client uses `aiohttp` with a 45‑second timeout and an SSL context constructed from `PANORAMA_VERIFY_SSL`. The raw XML response is parsed with `xml.etree.ElementTree` and inspected for `status="success"` before further processing.


### IP validation

```python
query_ip = ipaddress.ip_address("11.0.0.1")   # validates format
```

CIDR notation is also supported: `ipaddress.ip_network(ip, strict=False)`.

### Step 8a: Discover device groups to search

When no `device_group` is specified, the tool searches **shared** objects and **all device groups**:

```
GET /api/?type=config&action=get
    &xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry
    &key=<api_key>
```

Returns XML listing all device group names (e.g., `leander-dg`, `roundrock-dg`).

### Step 8b: Find address objects containing the IP

For each location (shared + each device group):

```
GET /api/?type=config&action=get
    &xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='leander-dg']/address
    &key=<api_key>
```

The XML response contains address object entries with `<ip-netmask>`, `<ip-range>`, or `<fqdn>`. The tool uses Python's `ipaddress` module to check containment:

- `ip-netmask` with CIDR → `ip in ipaddress.ip_network(...)`
- `ip-netmask` without CIDR → exact IP comparison
- `ip-range` → `start <= ip <= end`
- FQDN → skipped (cannot match IP)

Matching objects are collected: e.g., `{"name": "leander_web_server", "type": "ip-netmask", "value": "11.0.0.0/24"}`.

### Step 8c: Find address groups containing those objects

```
GET /api/?type=config&action=get
    &xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='leander-dg']/address-group
    &key=<api_key>
```

Each address group entry has `<static><member>` children. The tool checks whether any matching address object name appears as a member. Nested groups (groups containing groups) are resolved recursively.

### Return value

```python
{
    "ip_address": "11.0.0.1",
    "address_objects": [
        {"name": "leander_web_server", "type": "ip-netmask", "value": "11.0.0.0/24",
         "location": "device-group", "device_group": "leander-dg"}
    ],
    "address_groups": [
        {"name": "leander_web", "location": "device-group", "device_group": "leander-dg",
         "members": ["leander_web_server"]}
    ],
    "device_group": null,
    "vsys": "vsys1"
}
```

---

## Step 9: Result Normalization (chat_service.py)

**File:** [chat_service.py](../../chat_service.py) → `_normalize_result()`

For `query_panorama_ip_object_group`, normalization generates a human-readable `direct_answer`:

```python
if tool_name == "query_panorama_ip_object_group" and result.get("address_groups"):
    group_names = [ag.get("name") for ag in address_groups]
    # → ["leander_web"]
    direct_answer = "11.0.0.1 is part of address group 'leander_web'"
    # If resolved via address objects:
    direct_answer += " (via leander_web_server)"
    result["direct_answer"] = direct_answer
```

For `query_panorama_address_group_members`:

```python
if tool_name == "query_panorama_address_group_members":
    count = len(members)
    direct_answer = f"Address group 'leander_web' contains {count} members"
    result["direct_answer"] = direct_answer
```

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

Conversations are stored per-user on disk. The `conversation_id` is returned in the response so the frontend can track it.

---

## Step 11: Response Rendering (Frontend)

**File:** [frontend/src/stores/chatStore.js](../../frontend/src/stores/chatStore.js)

```js
const content = data.content ?? data.message ?? 'No response'
addMessage('assistant', content)
```

`content` is the normalized result dict from the backend.

### Response classification

**File:** [frontend/src/utils/responseClassifier.js](../../frontend/src/utils/responseClassifier.js)

`classifyResponse(content)` examines the object shape:

```js
// Panorama result has address_groups array → classified as 'table'
const arrayKeys = Object.keys(c).filter(k => Array.isArray(v) && v.every(...))
// → arrayKeys = ["address_objects", "address_groups"]
return { type: 'table', content }
```

### AssistantMessage rendering

**File:** [frontend/src/components/messages/AssistantMessage.jsx](../../frontend/src/components/messages/AssistantMessage.jsx)

```jsx
// Direct answer badge shown at top (e.g. "11.0.0.1 is part of address group 'leander_web'")
{hasDirectAnswer && <DirectAnswerBadge text={content.direct_answer} />}

// Tables rendered in order: members → address_objects → address_groups → policies
const tableOrder = ['members', 'address_objects', 'address_groups', 'policies']
for (const key of tableOrder) {
    groups.push({ type: 'horizontal', rows: arr, columns: colOrder, heading })
}
```

Panorama-specific column ordering is defined in `formatters.js` via `PANORAMA_COLUMN_ORDER` and `PANORAMA_TABLE_LABELS`, ensuring consistent column display.

Hidden fields (not rendered in tables): `vsys`, `queried_ip`, `intent`, `format`, `desc_units`, etc.

---

## Error Paths

| Scenario | Where caught | Response |
|---|---|---|
| Unauthenticated request | `app.py` | 401 + `{"redirect": "/login"}` |
| Expired session cookie | `auth.py` `get_session()` | 401 (via 401 handler) |
| No Azure role assigned | `auth.py` `auth_callback` | 302 `/login?error=norole` |
| Role lacks Panorama access | `chat_service._check_tool_access` | `"Your role (guest) does not have access to Panorama queries."` |
| Panorama unreachable | `panoramaauth.get_api_key()` | `{"error": "Failed to authenticate with Panorama..."}` |
| Panorama auth failure (bad credentials) | `panoramaauth.get_api_key()` | `{"error": "Failed to authenticate..."}` |
| IP validation failure | `query_panorama_ip_object_group` | `{"error": "Invalid IP address or CIDR format: ..."}` |
| LLM narrates instead of calling tool | `chat_service` | `"I could not determine how to answer that."` |
| LLM timeout (>90s) | `chat_service` | `"Tool selection timed out. Please try again."` |
| Tool timeout (>65s) | `mcp_client.call_mcp_tool` | Error propagated → `synthesize_final_answer()` |
| MCP server unreachable | `_fetch_mcp_tools()` | `"Could not connect to MCP server."` |

---

## Configuration Reference

| Variable | Source | Purpose |
|---|---|---|
| `PANORAMA_URL` | `.env` | Panorama appliance URL |
| `PANORAMA_VERIFY_SSL` | `.env` | SSL verification (set `false` for self-signed) |
| `PANORAMA_USERNAME` | Key Vault (`PANORAMA-USERNAME`) | API username |
| `PANORAMA_PASSWORD` | Key Vault (`PANORAMA-PASSWORD`) | API password |
| `AZURE_KEYVAULT_URL` | `.env` | Key Vault URL for secret retrieval |
| `AZURE_TENANT_ID` | `.env` | Azure AD tenant for OIDC login |
| `AZURE_CLIENT_ID` | `.env` | App registration client ID |
| `AZURE_CLIENT_SECRET` | `.env` | App registration client secret |
| `OLLAMA_MODEL` | `.env` | LLM for tool selection (currently `llama3.1:8b`) |
| `OLLAMA_BASE_URL` | `.env` | Ollama server address |
| `MCP_SERVER_HOST/PORT` | `.env` | MCP server address (default `127.0.0.1:8765`) |
| `CORS_ALLOWED_ORIGINS` | `.env` | Browser origins allowed by CORS middleware |
| `OAUTH_STATE_SECRET` | `.env` | Starlette session secret for OIDC state cookie |

---

## FAQ

### Why does Atlas use a session cookie?

HTTP is stateless — every request arrives at the server with no memory of who made previous requests. Without a cookie, FastAPI would have to demand credentials on every single request.

After OIDC login, FastAPI signs a session payload (`{ username, group, auth_mode, created_at }`) using `itsdangerous.URLSafeTimedSerializer` and sets it as the `atlas_session` cookie. On every subsequent request the browser sends this cookie automatically — FastAPI verifies the signature and reads the payload to know who you are and what you're allowed to do.

The `group` field drives RBAC: `_check_tool_access()` in `chat_service.py` reads it to decide which tools the user can call. There is no server-side session store — the signed payload *is* the session, so sessions survive app restarts with no shared cache needed.

The cookie is `HttpOnly` (JavaScript cannot read or steal it) and `SameSite=Lax` (blocks cross-site request forgery). Because the frontend and backend share the same origin (same scheme, host, and port), the browser attaches the cookie automatically — no explicit `credentials` setting is needed in the frontend fetch calls.

---

### What is AbortController / signal?

A browser `fetch()` call, once started, runs until the server responds or the network fails — there is no built-in way to cancel it from code. `AbortController` is the browser API that adds cancellation.

`new AbortController()` gives you a `controller` object and a `controller.signal`. Passing the `signal` into `fetch({ signal })` links the request to the controller. Calling `controller.abort()` immediately cancels the in-flight request and `fetch` throws an `AbortError`.

In Atlas, `chatStore.sendMessage` creates one `AbortController` per message send and stores it in state. The stop button calls `ctrl.abort()`, which cancels both the `/api/discover` and `/api/chat` requests simultaneously since they share the same signal. This is a **user-abort only** — there is no automatic timeout on `/api/discover`.

---

### What happens on a 401 response?

`checkAuthRedirect` immediately sets `window.location.href = '/login'` — the page navigates away. It also throws `'Not authenticated'`, which is caught by the inner try-catch in `chatStore` and falls back to `currentStatus: 'Processing'` — but the navigation has already happened so this is moot.

---

## Sequence Diagram

```
User          Browser           FastAPI          chat_service       MCP Server      panoramaauth     Panorama
 │  type query  │                  │                   │                 │                │              │
 │─────────────►│                  │                   │                 │                │              │
 │              │ POST /api/discover│                   │                 │                │              │
 │              │─────────────────►│                   │                 │                │              │
 │              │                  │ validate session   │                 │                │              │
 │              │                  │ check RBAC         │                 │                │              │
 │              │                  │──process_message──►│                 │                │              │
 │              │                  │                   │ list_tools()───►│                 │              │
 │              │                  │                   │◄────────────────│                 │              │
 │              │                  │                   │ LLM: select tool│                 │              │
 │              │                  │                   │ (llama3.1:8b)   │                 │              │
 │              │                  │◄─ tool_display_name─│               │                 │              │
 │              │◄─ {tool: "Panorama"}│                 │                │                 │              │
 │              │ POST /api/chat   │                   │                 │                │              │
 │              │─────────────────►│                   │                 │                │              │
 │              │                  │──process_message──►│                 │                │              │
 │              │                  │                   │ LLM: select tool│                 │              │
 │              │                  │                   │ call_mcp_tool()─►                │              │
 │              │                  │                   │                 │ get_api_key()──►│              │
 │              │                  │                   │                 │                │ GET keygen   │
 │              │                  │                   │                 │                │─────────────►│
 │              │                  │                   │                 │                │◄── API key ──│
 │              │                  │                   │                 │ query address  │              │
 │              │                  │                   │                 │ objects/groups │              │
 │              │                  │                   │                 │────────────────────────────► │
 │              │                  │                   │                 │◄─────────────── XML result ──│
 │              │                  │                   │◄── parsed dict ─│                │              │
 │              │                  │                   │ _normalize_result│               │              │
 │              │                  │◄── normalized JSON─│                │                │              │
 │              │◄─ {content: {...}}│                  │                 │                │              │
 │              │ render tables    │                   │                 │                │              │
 │◄─────────────│                  │                   │                 │                │              │
```

---
