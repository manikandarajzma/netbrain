# End-to-End Query Flow

This document traces exactly what happens, component by component, from the moment a user submits a query to the moment a response is returned. The example used throughout is a NetBrain network path query:

> **"Show network path from 10.0.0.1 to 192.168.1.1"**

---

## High-Level Map

```
Browser / Frontend
        │
        │  (first visit) GET /auth/microsoft → redirect to Azure
        │  (callback)    GET /auth/callback  ← Azure returns with auth code
        │  (subsequent)  POST /api/chat with session cookie
        ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI  (app_fastapi.py :8000)                            │
│                                                             │
│  AUTH_MODE=local  → username/password vs LOCAL_USERS dict   │
│  AUTH_MODE=oidc   → Authlib OAuth flow with Azure AD        │
│    ├─ /auth/microsoft  → redirect to Microsoft login        │
│    ├─ /auth/callback   → exchange code, decode JWT, set     │
│    │                      session cookie (30 min TTL)       │
│    └─ /api/chat        → validate session cookie → username │
└──────────────┬──────────────────────────────────────────────┘
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

### Step 2 — Authentication

There are two auth modes, controlled by the `AUTH_MODE` environment variable. Both produce the same end result: a `session_id` cookie that maps to a `{username, role}` entry in the in-memory session store (`auth.py`).

---

#### Auth Mode A — Local (`AUTH_MODE=local`)

Used for development or environments without Azure. Credentials are defined in `.env` as `NETBRAIN_USERS=admin:secret:admin,viewer:pass:netadmin`.

```python
# auth.py (lines 40-49)
_default_users_env = os.getenv("NETBRAIN_USERS", "admin:admin:admin")
LOCAL_USERS = {}
for part in _default_users_env.strip().split(","):
    u, p, r = part.split(":")
    LOCAL_USERS[u] = {"password": p, "role": r}
```

Login flow:

```python
# app_fastapi.py (lines 148-170)

@app.post("/login")
async def login_post(username: str = Form(...), password: str = Form(...)):

    # 1. Compare plaintext password against LOCAL_USERS dict
    if not verify_local_user(username, password):
        return RedirectResponse(url="/login?error=invalid", status_code=302)

    # 2. Look up role from LOCAL_USERS
    role = get_user_role(username)           # e.g. "admin"

    # 3. Create server-side session (random 32-byte token)
    session_id = create_session(username, role=role, auth_mode="local")

    # 4. Set httponly session cookie (7-day TTL for local auth)
    r = RedirectResponse(url="/", status_code=302)
    r.set_cookie(
        key="netbrain_session",
        value=session_id,
        max_age=86400 * 7,
        httponly=True,
        samesite="lax",
    )
    return r
```

---

#### Auth Mode B — Microsoft OIDC (`AUTH_MODE=oidc`)

Uses Microsoft Entra ID (Azure AD) via the **Authlib** library. Requires `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, and `AZURE_TENANT_ID` in `.env`.

**Sub-step B1 — App startup: OAuth client registered**

```python
# app_fastapi.py (lines 62-76)

from authlib.integrations.starlette_client import OAuth

oauth = OAuth()
oauth.register(
    name="microsoft",
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET,
    # Fetches JWKS, token endpoint, etc. from Microsoft's OIDC discovery URL
    server_metadata_url=(
        f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0"
        "/.well-known/openid-configuration"
    ),
    client_kwargs={"scope": "openid profile email offline_access"},
)

# SessionMiddleware stores the OAuth state parameter (CSRF protection)
app.add_middleware(SessionMiddleware, secret_key=..., session_cookie="netbrain_oauth_state", max_age=600)
```

**Sub-step B2 — User clicks "Sign in with Microsoft"**

```
Browser → GET /auth/microsoft
```

```python
# app_fastapi.py (lines 175-182)

@app.get("/auth/microsoft")
async def auth_microsoft(request: Request):
    redirect_uri = str(request.url_for("auth_callback"))
    # Redirects browser to:
    # https://login.microsoftonline.com/<tenant>/oauth2/v2.0/authorize
    #   ?client_id=<client_id>
    #   &response_type=code
    #   &scope=openid profile email offline_access
    #   &state=<random CSRF token>          ← stored in netbrain_oauth_state cookie
    #   &prompt=select_account
    return await oauth.microsoft.authorize_redirect(request, redirect_uri, prompt="select_account")
```

**Sub-step B3 — Microsoft authenticates user, redirects back**

```
Microsoft → GET /auth/callback?code=<auth_code>&state=<csrf_token>
```

```python
# app_fastapi.py (lines 185-240)

@app.get("/auth/callback")
async def auth_callback(request: Request):

    # 1. Exchange auth code for tokens (validates state/CSRF automatically)
    token = await oauth.microsoft.authorize_access_token(request)
    # token contains: access_token, refresh_token, id_token (JWT)

    # 2. Extract claims from the ID token
    userinfo = token.get("userinfo") or {}
    if not userinfo:
        # Manually decode JWT payload (base64url, no signature verification needed
        # here — Authlib already verified the signature in step 1)
        id_token = token.get("id_token")
        payload = id_token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)   # re-pad base64
        userinfo = json.loads(base64.urlsafe_b64decode(payload))

    # userinfo now contains Azure JWT claims, e.g.:
    # {
    #   "preferred_username": "user@company.com",
    #   "name": "John Smith",
    #   "roles": ["admin"],          ← Azure app roles (if configured)
    #   "groups": ["<group-id>"],    ← Azure security groups (if configured)
    #   "oid": "...",
    # }

    # 3. Extract display username
    username = extract_username_from_token(userinfo)
    # Tries: preferred_username → email → name → sub

    # 4. Resolve role from token claims (three-priority cascade)
    role = extract_role_from_token(userinfo)
    # Returns None if no role found → redirect to /login?error=norole
```

**Sub-step B4 — Role resolution from JWT claims (3-level priority)**

```python
# auth.py (lines 196-229)

def extract_role_from_token(token_claims: dict) -> Optional[str]:

    # Priority 1: Azure app roles (most explicit)
    # Set in Azure portal: Enterprise Application → App roles → Assign to users
    roles = token_claims.get("roles", [])
    for r in roles:
        if r.lower() in ROLE_ALLOWED_TOOLS:   # "admin" or "netadmin"
            return r.lower()

    # Priority 2: Azure security group → role mapping
    # Configured via: OIDC_GROUP_ROLE_MAP=<group-object-id>:netadmin
    # Requires "groups" claim added in Azure App Registration → Token configuration
    if OIDC_GROUP_ROLE_MAP:
        groups = token_claims.get("groups", [])
        for gid in groups:
            role = OIDC_GROUP_ROLE_MAP.get(str(gid).lower())
            if role and role in ROLE_ALLOWED_TOOLS:
                return role

    # Priority 3: Per-email override (for testing or exceptions)
    # Configured via: OIDC_ROLE_MAP=user@company.com:admin
    if OIDC_ROLE_MAP:
        for key in ("preferred_username", "email", "upn"):
            email = token_claims.get(key, "").strip().lower()
            if email and email in OIDC_ROLE_MAP:
                return OIDC_ROLE_MAP[email]

    return None   # → user has no access, redirect to /login?error=norole
```

**Sub-step B5 — Session created, cookie set**

```python
# app_fastapi.py (lines 221-240)

    # 5. Create server-side session with OIDC tokens stored for later refresh
    session_id = create_session(
        username,
        role=role,
        auth_mode="oidc",
        tokens={
            "access_token":  token.get("access_token"),
            "refresh_token": token.get("refresh_token"),
            "id_token":      token.get("id_token"),
        },
    )
    # Session store entry:
    # _sessions[session_id] = {
    #   "username":   "user@company.com",
    #   "role":       "netadmin",
    #   "auth_mode":  "oidc",
    #   "created_at": 1234567890.0,
    #   "tokens":     { ... }       ← stored for potential token refresh
    # }

    # 6. Set httponly session cookie (30-minute TTL for OIDC — shorter than local)
    r = RedirectResponse(url="/", status_code=302)
    r.set_cookie(
        key="netbrain_session",
        value=session_id,
        max_age=1800,       # 30 minutes — OIDC_SESSION_TTL
        httponly=True,
        samesite="lax",
    )
    return r
```

---

#### Session Validation on Every Request

Once a session cookie exists, every protected endpoint resolves it the same way regardless of how it was created:

```python
# app_fastapi.py (lines 110-116)
def get_current_username(request: Request) -> str | None:
    sid = request.cookies.get("netbrain_session")
    return get_username_for_session(sid)   # from auth.py

# auth.py (lines 154-175)
def get_session(session_id) -> Optional[dict]:
    sess = _sessions.get(session_id)
    if sess is None:
        return None
    # OIDC sessions expire after 30 minutes; local sessions never expire server-side
    if sess.get("auth_mode") == "oidc":
        elapsed = time.time() - sess.get("created_at", 0)
        if elapsed > OIDC_SESSION_TTL:       # 1800s
            del _sessions[session_id]        # evict expired session
            return None
    return sess
```

**OIDC sessions expire after 30 minutes** and are deleted from the store on the next request. Local sessions have a 7-day cookie TTL but no server-side expiry check.

---

#### FastAPI hands off to `chat_service`

```python
# app_fastapi.py (lines 357-370)

@app.post("/api/chat")
async def api_chat(request: Request, body: ChatRequest):
    username = get_current_username(request)   # "user@company.com" or "admin"
    if not username:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    from netbrain.chat_service import process_message
    result = await process_message(
        body.message.strip(),
        body.conversation_history or [],
        default_live=True,
        username=username,    # passed through for RBAC check inside chat_service
    )
    return result
```

**Component:** `app_fastapi.py`, `auth.py`
**What happens:** Session cookie `netbrain_session` is read, looked up in the in-memory `_sessions` dict, expiry is checked (OIDC only), and `username` is resolved. `username` is forwarded to `process_message` so the per-tool RBAC check (Step 7) can use it.

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
| 2 (local) | Credential check | `app_fastapi.py` → `auth.py` | `verify_local_user()`, plaintext compare against `LOCAL_USERS` dict |
| 2 (OIDC) | OAuth redirect | `app_fastapi.py` | Authlib `authorize_redirect` → Microsoft Entra ID |
| 2 (OIDC) | Auth code exchange | `app_fastapi.py` | Authlib `authorize_access_token`, JWT decode, `extract_role_from_token` |
| 2 (both) | Session creation | `auth.py` | `create_session()`, `secrets.token_urlsafe(32)`, httponly cookie |
| 2 (both) | Session validation per request | `app_fastapi.py` → `auth.py` | `get_session()`, OIDC 30-min TTL check, session dict lookup |
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
