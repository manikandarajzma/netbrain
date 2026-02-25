# NetBrain Query — End-to-End Flow

This document traces the complete lifecycle of a NetBrain query through Atlas, from the moment a user types a message in the browser to the rendered response. Two NetBrain tools are covered:

- **`query_network_path`** — "Find path from 10.0.0.1 to 10.0.1.1"
- **`check_path_allowed`** — "Is traffic from 10.0.0.1 to 10.0.1.1 on TCP 443 allowed?"

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
    │  Dispatches to netbrain_tools.py
    ▼
netbrainauth.py
    │  Gets/caches NetBrain session token (username+password → Session API)
    │  Token TTL: 30 minutes; auto-refreshes on expiry or 401
    ▼
NetBrain REST API (NETBRAIN_URL)
    │  Step 1: GET  /V1/CMDB/Path/Gateways         (resolve source gateway)
    │  Step 2: POST /V1/CMDB/Path/Calculation       (submit path task)
    │  Step 3: GET  /V1/CMDB/Path/Calculation/{id}/OverView  (poll for results)
    ▼
netbrain_tools.py  (hop extraction, device type mapping, firewall detection)
    ▼
panorama_tools.py  (zone + device group enrichment)
    ▼
chat_service.py  _normalize_result()
    ▼
FastAPI → JSON response → React
```

---

## Step 1: User Types a Query (Frontend)

**File:** [frontend/src/components/chat/ChatInput.jsx](../../frontend/src/components/chat/ChatInput.jsx)
**Store:** [frontend/src/stores/chatStore.js](../../frontend/src/stores/chatStore.js)

The user types `"Find path from 10.0.0.1 to 10.0.1.1"` and presses Enter.

`chatStore.sendMessage(text)` runs:

1. **Disambiguation check** — inspects the last assistant message in `conversationHistory`. If it had `requires_site: true`, it would combine the site name + user reply. For NetBrain queries this path is never triggered.

2. **UI state** — `isLoading: true`, `currentStatus: 'Identifying query'`.

3. **History is intentionally empty** — `historySlice = []`. Each query is stateless; prior context is not sent to the LLM.

---

## Step 2: Tool Discovery (Frontend → FastAPI)

**File:** [frontend/src/utils/api.js](../../frontend/src/utils/api.js) → `discoverTool()`

```
POST /api/discover
Content-Type: application/json
Cookie: atlas_session=<signed-cookie>

{ "message": "Find path from 10.0.0.1 to 10.0.1.1", "conversation_history": [] }
```

- `fetch` includes `credentials: 'include'` (cookies sent automatically).
- A 2 min client-side timeout is applied via `AbortSignal`.
- On 401, the browser is redirected to `/login`.

The response contains `tool_display_name: "NetBrain"`. The UI updates `currentStatus` to `"Querying NetBrain"`.

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

The role is resolved from the session cookie. RBAC is enforced later inside `chat_service.process_message()`:

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

Both NetBrain tools are accessible to `admin` and `netadmin` roles. `guest` is denied.

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

### Scope check (fast-path, no LLM)

```python
def _is_obviously_in_scope(prompt: str) -> bool:
    path_kw = any(k in lower for k in ("path", "route", "trace", "hop", "allowed", "reachable", ...))
    has_two_ips = len(_IP_OR_CIDR_RE.findall(prompt)) >= 2
    if has_two_ips and path_kw:
        return True
    ...
```

A prompt like `"Find path from 10.0.0.1 to 10.0.1.1"` matches `has_two_ips=True` + `path_kw=True` → scope check passes immediately without an LLM call.

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

The description passed to the LLM for `query_network_path` includes:

```
Trace the hop-by-hop network path between two IP addresses using NetBrain.

Use for: queries asking to SEE the path/route/hops between two IPs — "find path from X to Y",
"show path", "trace path", "network path between X and Y".
Do NOT use for: "is path allowed?" / "is traffic allowed?" (use check_path_allowed), device/rack lookups.

Examples:
- "find path from 10.0.0.1 to 10.0.1.1" → source="10.0.0.1", destination="10.0.1.1"
```

### LLM tool selection (LangChain + Ollama)

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b", base_url="http://localhost:11434", temperature=0.0)
llm_with_tools = llm.bind_tools(openai_tools, tool_choice="required")
messages = _build_llm_messages(prompt, conversation_history)
ai_msg = await asyncio.wait_for(llm_with_tools.ainvoke(messages), timeout=90.0)
```

`tool_choice="required"` forces the LLM to always output a structured tool call. The LLM distinguishes between the two NetBrain tools based on intent:

- "find path / show hops / trace route" → `query_network_path`
- "is traffic allowed? / can X reach Y?" → `check_path_allowed`

The LLM responds with a structured `tool_calls` entry:
```json
{
  "name": "query_network_path",
  "args": { "source": "10.0.0.1", "destination": "10.0.1.1", "protocol": "TCP", "port": "443" }
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

If the user's role forbids the tool, an error message is returned immediately — the tool is never called.

### discover_only mode

When called from `/api/discover`, the function returns immediately after tool selection:

```python
if discover_only:
    return {
        "tool_name": "query_network_path",
        "parameters": {"source": "10.0.0.1", "destination": "10.0.1.1", "protocol": "TCP", "port": "443"},
        "tool_display_name": "NetBrain",
        "format": "path",
    }
```

---

## Step 6: Full Chat Request (Frontend → FastAPI)

After discovery, the frontend fires the actual chat request:

```
POST /api/chat
{ "message": "Find path from 10.0.0.1 to 10.0.1.1", "conversation_history": [] }
```

`process_message()` runs again from scratch (not reusing the discover result), this time with `discover_only=False`. The LLM is invoked again, makes the same tool selection, and proceeds to execution.

---

## Step 7: MCP Tool Execution

**File:** [chat_service.py](../../chat_service.py) → `call_mcp_tool()`
**File:** [mcp_client.py](../../mcp_client.py)

```python
result = await call_mcp_tool(
    "query_network_path",
    {"source": "10.0.0.1", "destination": "10.0.1.1", "protocol": "TCP", "port": "443"},
    timeout=65.0
)
```

The MCP client sends a JSON-RPC `tools/call` message over the streamable-http transport to `http://127.0.0.1:8765`. The MCP server (running as a separate process) receives it and dispatches to the registered `@mcp.tool()` handler.

---

## Step 8: netbrain_tools.py — Tool Execution

**File:** [tools/netbrain_tools.py](../../tools/netbrain_tools.py)

### Protocol and port mapping

Before submitting the path calculation, the tool converts the protocol string to a numeric code:

```python
protocol_map = {"TCP": 6, "UDP": 17, "IP": 4, "IPv4": 4}
protocol_num = protocol_map.get(protocol.upper(), 4)   # default: IPv4
port_num = int(port) if port and port.isdigit() else 0  # 0 = all ports
```

### Step 8a: Resolve source gateway

```
GET {NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Gateways
    ?ipOrHost=10.0.0.1
Token: <auth_token>
```

NetBrain returns the gateway device that the source IP is connected to:
```json
{
  "statusCode": 790200,
  "gatewayList": [
    { "gatewayName": "leander-dc-leaf1", "type": "Cisco Nexus Switch" }
  ]
}
```

If no gateway is found (`statusCode: 792040` or empty list), a placeholder gateway is created from the source IP. NetBrain's path fix-up rules can override placeholders during calculation.

### Step 8b: Submit path calculation

```
POST {NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Calculation
Token: <auth_token>

{
  "sourceIP": "10.0.0.1",
  "sourcePort": 0,
  "destIP": "10.0.1.1",
  "destPort": 443,
  "pathAnalysisSet": 1,       // 1=L3 Path
  "protocol": 6,              // TCP
  "isLive": 1,                // Live data (vs. Baseline)
  "sourceGateway": { ... },   // From Step 8a
  "advanced": {
    "calcWhenDeniedByACL": true,
    "calcWhenDeniedByPolicy": true,   // continue_on_policy_denial
    "enablePathFixup": true,
    "enablePathIPAndGatewayFixup": true
  }
}
```

For `check_path_allowed`, `calcWhenDeniedByPolicy` is set to `false` — this causes NetBrain to stop at the denying firewall and return a failed status, which the tool uses to determine the denied verdict.

The response contains a `taskID`:
```json
{ "statusCode": 790200, "taskID": "550e8400-e29b-41d4-a716-446655440000" }
```

### Step 8c: Poll for path results

```
GET {NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Calculation/{taskID}/OverView
Token: <auth_token>
```

The path calculation is asynchronous. The tool polls until the result is ready:

- **Live data**: Up to 120 attempts at 2–5 second intervals (max ~4 minutes).
- **Baseline**: Up to 30 attempts at 2–3 second intervals (max ~60 seconds).
- `statusCode: 794007` means the task is still processing — continue polling.
- `statusCode: 790200` with path data → stop polling.

### Step 8d: Extract path hops

The response has a nested structure:

```
path_overview[]
  └── path_list[]
        └── branch_list[]
              └── hop_detail_list[]
                    ├── fromDev { devName, devType, ip }
                    ├── toDev   { devName, devType, ip }
                    ├── inInterface
                    └── outInterface
```

For each hop, the tool builds a simplified dict:

```python
{
    "hop_sequence": 1,
    "from_device": "leander-dc-leaf1",
    "to_device": "leander-dc-fw1",
    "from_device_type": "Cisco Nexus Switch",
    "to_device_type": "Palo Alto",
    "status": "Success",
    "failure_reason": "",
    "in_interface": "Ethernet1/1",   # firewall hops only
    "out_interface": "Ethernet1/2",  # firewall hops only
    "is_firewall": True,
    "firewall_device": "leander-dc-fw1"
}
```

**Firewall detection** checks device type and name patterns:
- Type contains: `"firewall"`, `"fw"`, `"palo alto"`, `"fortinet"`, `"checkpoint"`, `"asa"`
- Name contains: `"fw"`, `"palo"`, `"fortinet"`, `"checkpoint"`, `"asa"`

**Interface normalization** — because firewall devices appear as both `fromDev` and `toDev` in consecutive hops, the tool merges interface data and infers missing ingress/egress interfaces where possible.

### Step 8e: Device type mapping

NetBrain returns numeric device type codes in path hops. The tool maps these to readable names by calling two endpoints at startup (cached):

1. `GET /ServicesAPI/SystemModel/getAllDisplayDeviceTypes` — numeric code → name
2. `GET /ServicesAPI/API/V1/CMDB/Devices` — device name → type name (fallback)

### Step 8f: Panorama enrichment

After extracting hops, the tool enriches firewall hops with Panorama data:

```python
if simplified_hops:
    await _add_panorama_zones_to_hops(simplified_hops)      # Adds security_zone field
    await _add_panorama_device_groups_to_hops(simplified_hops)  # Adds device_group field
```

These functions are imported from [tools/panorama_tools.py](../../tools/panorama_tools.py) and modify hops in-place, adding:
- `security_zone` — the Panorama security zone (e.g., `"trust"`, `"untrust"`)
- `device_group` — the Panorama device group the firewall belongs to

### Return value — query_network_path

```python
{
    "source": "10.0.0.1",
    "destination": "10.0.1.1",
    "protocol": "TCP",
    "port": "443",
    "taskID": "550e8400-...",
    "statusCode": 790200,
    "gateway_used": "leander-dc-leaf1",
    "path_hops": [
        {
            "hop_sequence": 1,
            "from_device": "leander-dc-leaf1",
            "to_device": "leander-dc-fw1",
            "from_device_type": "Cisco Nexus Switch",
            "to_device_type": "Palo Alto",
            "status": "Success",
            "is_firewall": True,
            "firewall_device": "leander-dc-fw1",
            "in_interface": "Ethernet1/1",
            "out_interface": "Ethernet1/2",
            "security_zone": "trust",
            "device_group": "leander-dg"
        }
    ],
    "path_hops_count": 3,
    "path_status": "Success",
    "path_status_description": "",
    "path_failure_reason": ""
}
```

### Return value — check_path_allowed

`check_path_allowed` calls `_query_network_path_impl` internally (with `continue_on_policy_denial=False`) and interprets the result:

```python
# Allowed
if path_status == "Success" and path_hops:
    status = "allowed"
    reason = "Path exists and traffic is allowed by policy"

# Denied — policy denial
elif path_status == "Failed" and "policy" in failure_reason.lower():
    status = "denied"
    firewall_denied_by = _denying_firewall_from_hops(path_hops)
    reason = f"Traffic is denied by policy: {failure_reason}"

# Unknown — failure but not policy-related
else:
    status = "unknown"
    reason = failure_reason
```

```python
{
    "source": "10.0.0.1",
    "destination": "10.0.1.1",
    "protocol": "TCP",
    "port": "443",
    "status": "denied",
    "reason": "Traffic is denied by policy: Policy denied by firewall leander-dc-fw1",
    "path_exists": True,
    "path_hops_count": 2,
    "path_hops": [ ... ],
    "firewall_denied_by": "leander-dc-fw1",
    "policy_details": "Policy denied by firewall leander-dc-fw1"
}
```

---

## Step 9: Result Normalization (chat_service.py)

**File:** [chat_service.py](../../chat_service.py) → `_normalize_result()`

For both NetBrain tools, normalization strips noisy L2 status messages:

```python
if isinstance(result, dict) and result.get("path_hops"):
    result = dict(result)
    _strip_l2_noise(result)
```

```python
def _strip_l2_noise(result):
    noise = ["l2 connections has not been discovered", "l2 connection has not been discovered"]
    for key in ("path_status_description", "statusDescription"):
        val = result.get(key)
        if isinstance(val, str) and any(p in val.lower() for p in noise):
            result[key] = ""
    return result
```

No `direct_answer` or `yes_no_answer` is added — the `status` field on `check_path_allowed` (`"allowed"` / `"denied"` / `"unknown"`) already provides the verdict, and the frontend renders it using the `path-summary` or `path` classifier.

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

```js
// query_network_path: has path_hops array → classified as 'path'
if (content.path_hops && Array.isArray(content.path_hops) && content.path_hops.length > 0) {
    return { type: 'path', content }
}

// check_path_allowed: has source + destination but no hops (or denied) → classified as 'path-summary'
if (content.source && content.destination && !content.error) {
    return { type: 'path-summary', content }
}
```

### AssistantMessage rendering

**File:** [frontend/src/components/messages/AssistantMessage.jsx](../../frontend/src/components/messages/AssistantMessage.jsx)

**`query_network_path`** → rendered by `<PathVisualization>`:

```jsx
{classified.type === 'path' && <PathVisualization content={content} />}
```

`PathVisualization` renders the hop-by-hop path as an interactive diagram — each hop shows device name, type, interfaces, and Panorama security zone.

**`check_path_allowed`** → rendered inline:

```jsx
{classified.type === 'path-summary' && (
    <div>
        <p>Path: {content.source} → {content.destination}</p>
        {content.status === 'denied' && (
            <>
                <p>Denied by firewall: {content.firewall_denied_by}</p>
                <p>Policy: {content.policy_details}</p>
            </>
        )}
    </div>
)}
```

---

## Error Paths

| Scenario | Where caught | Response |
|---|---|---|
| Unauthenticated request | `app.py` | 401 + `{"redirect": "/login"}` |
| Expired session cookie | `auth.py` `get_session()` | 401 |
| No Azure role assigned | `auth.py` `auth_callback` | 302 `/login?error=norole` |
| Role lacks NetBrain access | `chat_service._check_tool_access` | `"Your role (guest) does not have access to this tool."` |
| NetBrain unreachable | `netbrainauth.get_auth_token()` | `{"error": "Failed to get authentication token"}` |
| NetBrain auth failure (bad credentials) | `netbrainauth.get_auth_token()` | `{"error": "Failed to get authentication token"}` |
| Gateway resolution fails | `_query_network_path_impl` | Placeholder gateway created; path calculation continues |
| Path calculation API error | `_query_network_path_impl` | `{"error": "Path calculation failed: ..."}` |
| Poll timeout (task never completes) | `_query_network_path_impl` | Returns partial result with `message` field |
| No hops extractable | `_query_network_path_impl` | Returns result with `message` + `note` instead of `path_hops` |
| LLM narrates instead of calling tool | `chat_service` | `"I could not determine how to answer that."` |
| LLM timeout (>90s) | `chat_service` | `"Tool selection timed out. Please try again."` |
| Tool timeout (>65s) | `mcp_client.call_mcp_tool` | Error propagated → `synthesize_final_answer()` |
| MCP server unreachable | `_fetch_mcp_tools()` | `"Could not connect to MCP server."` |

---

## Configuration Reference

| Variable | Source | Purpose |
|---|---|---|
| `NETBRAIN_URL` | `.env` | NetBrain server base URL |
| `NETBRAIN_USERNAME` | `.env` | API username |
| `NETBRAIN_PASSWORD` | `.env` | API password |
| `NETBRAIN_AUTH_ID` | `.env` | Optional auth ID for LDAP/AD/TACACS users |
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
User          Browser           FastAPI          chat_service       MCP Server      netbrainauth     NetBrain
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
 │              │                  │◄─ tool_display_name─│               │                 │              │
 │              │◄─ {tool:"NetBrain"}│                  │                │                 │              │
 │              │ POST /api/chat   │                   │                 │                │              │
 │              │─────────────────►│                   │                 │                │              │
 │              │                  │──process_message──►│                 │                │              │
 │              │                  │                   │ LLM: select tool│                 │              │
 │              │                  │                   │ call_mcp_tool()─►                │              │
 │              │                  │                   │                 │ get_auth_token()►              │
 │              │                  │                   │                 │                │ POST /Session │
 │              │                  │                   │                 │                │─────────────►│
 │              │                  │                   │                 │                │◄─── token ───│
 │              │                  │                   │                 │ GET /Gateways  │              │
 │              │                  │                   │                 │────────────────────────────► │
 │              │                  │                   │                 │◄──────────────── gateway ────│
 │              │                  │                   │                 │ POST /Calculation             │
 │              │                  │                   │                 │────────────────────────────► │
 │              │                  │                   │                 │◄─────────────── taskID ──────│
 │              │                  │                   │                 │ GET /OverView (poll)          │
 │              │                  │                   │                 │────────────────────────────► │
 │              │                  │                   │                 │◄─────────────── path data ───│
 │              │                  │                   │◄── path hops ───│                │              │
 │              │                  │                   │ _normalize_result│               │              │
 │              │                  │◄── normalized JSON─│                │                │              │
 │              │◄─ {content: {...}}│                  │                 │                │              │
 │              │ render path viz  │                   │                 │                │              │
 │◄─────────────│                  │                   │                 │                │              │
```

---
