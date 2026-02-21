# FastAPI — Role and Functionality

This document covers what FastAPI does in this codebase, how the application is structured, all endpoints, middleware, request/response models, and how FastAPI ties together authentication (`auth.py`) and the chat pipeline (`chat_service.py`).

---

## 1. What FastAPI Does Here

FastAPI is the **HTTP entry point** for the entire application. It runs on port `8000` and is responsible for:

- Serving the login UI (Jinja2 or React SPA)
- Handling local username/password authentication and Microsoft OIDC login
- Protecting every API endpoint with session-cookie authentication
- Delegating all AI/chat work to `chat_service.process_message()`
- Parsing uploaded spreadsheets and running batch tool queries
- Proxying health checks to the MCP server

The MCP server (`mcp_server.py`) runs separately on port `8765` — FastAPI never calls it directly. All MCP interaction goes through `chat_service.py`.

---

## 2. Application Setup

```python
# app_fastapi.py (lines 46-56)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NetAssist", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

The `title` and `version` appear in the auto-generated OpenAPI docs at `/docs`.

CORS is configured to **allow all origins** — appropriate for an internal tool where the React frontend may be served from a different port during development.

---

## 3. Middleware

Two middleware layers are registered, depending on `AUTH_MODE`.

### 3.1 CORS Middleware (always active)

Always registered. Allows cross-origin requests from any origin so that a React dev server on `localhost:5173` can call the FastAPI backend on `localhost:8000`.

### 3.2 Session Middleware (OIDC only)

```python
# app_fastapi.py (lines 81-89)
if AUTH_MODE == "oidc":
    from starlette.middleware.sessions import SessionMiddleware
    app.add_middleware(
        SessionMiddleware,
        secret_key=secrets.token_urlsafe(32),  # random on every startup
        session_cookie="netbrain_oauth_state",
        max_age=600,  # 10 minutes
    )
```

`SessionMiddleware` stores the OAuth2 state parameter in a short-lived, server-signed cookie. This is only needed during the OIDC redirect flow (steps B2–B3 in the auth flow). It is not the same as the session cookie that identifies logged-in users — that is a separate cookie (`netbrain_session`) managed by `auth.py`.

The `secret_key` is randomly generated each startup, which means the OAuth state cookie is invalidated on server restart (acceptable — the user just needs to log in again).

---

## 4. OIDC Client Registration

```python
# app_fastapi.py (lines 61-76)
oauth = None
if AUTH_MODE == "oidc" and AZURE_CLIENT_ID and AZURE_TENANT_ID:
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register(
        name="microsoft",
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
        server_metadata_url=(
            "https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0"
            "/.well-known/openid-configuration"
        ),
        client_kwargs={
            "scope": "openid profile email offline_access",
        },
    )
```

`authlib` is used as the OIDC client library. It fetches Azure's OIDC discovery document (`/.well-known/openid-configuration`) at startup to retrieve token/auth endpoints automatically. The `oauth` object is `None` if `AUTH_MODE` is not `"oidc"` — all OIDC routes check for this and redirect to `/login?error=oidc` if so.

**Scopes requested:**
| Scope | Purpose |
|---|---|
| `openid` | OIDC protocol — enables ID token |
| `profile` | Includes `name`, `preferred_username` in token |
| `email` | Includes email address in token |
| `offline_access` | Requests a refresh token for session extension |

---

## 5. Session Cookie Model

FastAPI manages two separate cookies:

| Cookie | Name | Set by | Max-age | Purpose |
|---|---|---|---|---|
| Application session | `netbrain_session` | `POST /login`, `GET /auth/callback` | 7 days (local) / 30 min (OIDC) | Identifies logged-in user across requests |
| OAuth state | `netbrain_oauth_state` | `SessionMiddleware` | 10 min | Carries the OAuth2 `state` parameter during OIDC redirect |

The application session cookie is `httponly=True` (not accessible to JavaScript) and `samesite="lax"` (protects against CSRF while still allowing redirect-based login flows).

---

## 6. Auth Helper Functions

Three small functions bridge the `Request` object to the `auth.py` session store:

```python
# app_fastapi.py (lines 110-124)

def get_session_id(request: Request) -> str | None:
    """Read netbrain_session cookie from the request."""
    return request.cookies.get(SESSION_COOKIE)

def get_current_username(request: Request) -> str | None:
    """Return username if the session cookie maps to a valid session."""
    sid = get_session_id(request)
    return get_username_for_session(sid)  # from auth.py — checks TTL

def require_auth(request: Request) -> str:
    """FastAPI Depends() guard — raises 302 redirect if not authenticated."""
    username = get_current_username(request)
    if not username:
        raise RedirectResponse(url="/login", status_code=302)
    return username
```

`require_auth` is a FastAPI dependency function — routes that inject it with `Depends(require_auth)` are automatically protected (though most routes in this app call `get_current_username()` directly and return a 401/302 manually for finer control).

---

## 7. Routes

### Route Summary Table

| Method | Path | Auth required | Purpose |
|---|---|---|---|
| `GET` | `/login` | No | Serve login page (HTML) |
| `POST` | `/login` | No | Validate local credentials, set session cookie |
| `GET` | `/auth/microsoft` | No | Redirect to Microsoft OAuth2 login |
| `GET` | `/auth/callback` | No | Handle OAuth2 callback, create session |
| `GET` | `/logout` | No | Destroy session, redirect to `/login` |
| `GET` | `/api/me` | Yes | Return current user info (for React SPA) |
| `GET` | `/` | Yes | Serve main chat UI (React SPA or Jinja2) |
| `GET` | `/health` | No | App + MCP server health check |
| `POST` | `/api/discover` | Yes | Tool discovery only (no execution) |
| `POST` | `/api/chat` | Yes | Full chat query — tool selection + execution |
| `POST` | `/api/batch-upload` | Yes | Upload CSV/XLSX and run tool per row |

---

### 7.1 `GET /login`

Serves `templates/login.html` via Jinja2. Passes three boolean flags derived from the `?error=` query parameter:

| `?error=` value | Template variable | Meaning |
|---|---|---|
| `oidc` | `error_oidc=True` | OIDC redirect or token exchange failed, or OIDC not configured |
| `norole` | `error_norole=True` | OIDC login succeeded but no role assigned |

Also passes `oidc_configured` so the template shows "Sign in with Microsoft" only when Azure is configured. There is no local password auth; all credentials are in Azure Key Vault.

If the user is already logged in, immediately redirects to `/`.

---

### 7.2 `GET /auth/microsoft`

Initiates the Microsoft OIDC flow:

```python
# app_fastapi.py
@app.get("/auth/microsoft")
async def auth_microsoft(request: Request):
    redirect_uri = str(request.url_for("auth_callback")).replace("://127.0.0.1", "://localhost")
    return await oauth.microsoft.authorize_redirect(request, redirect_uri, prompt="select_account")
```

`authorize_redirect()` generates a URL to `https://login.microsoftonline.com/...` with the OAuth2 state parameter (stored in the `netbrain_oauth_state` cookie by SessionMiddleware) and redirects the browser there.

The `127.0.0.1` → `localhost` replacement ensures the callback URL matches the **Redirect URI** configured in Azure App Registration exactly — Azure requires `localhost` not `127.0.0.1`.

`prompt="select_account"` forces Microsoft's account picker even if the user has an active browser session, useful for multi-account environments.

---

### 7.3 `GET /auth/callback`

Handles the token exchange after Microsoft redirects back:

```python
# app_fastapi.py (lines 185-240)
@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.microsoft.authorize_access_token(request)  # exchange code for tokens

    userinfo = token.get("userinfo") or {}
    if not userinfo:
        # Manual JWT decode fallback if authlib didn't populate userinfo
        id_token = token.get("id_token")
        payload = id_token.split(".")[1]
        payload += "=" * (4 - len(payload) % 4)  # add base64 padding
        userinfo = json.loads(base64.urlsafe_b64decode(payload))

    username = extract_username_from_token(userinfo)   # from auth.py
    role = extract_role_from_token(userinfo)            # from auth.py — 3-priority resolution
    if role is None:
        return RedirectResponse(url="/login?error=norole", status_code=302)

    session_id = create_session(username, role=role, auth_mode="oidc",
                                tokens={"access_token": ..., "refresh_token": ..., "id_token": ...})
    r = RedirectResponse(url="/", status_code=302)
    r.set_cookie(SESSION_COOKIE, session_id, max_age=SESSION_MAX_AGE_OIDC, ...)
    return r
```

**JWT decode fallback:** `authlib` normally populates `token["userinfo"]` automatically. The manual decode path (`id_token.split(".")[1]`) is a defensive fallback for environments where `authlib` can't validate the signature (e.g. a misconfigured tenant). It simply base64-decodes the payload segment without signature verification — this is acceptable because the OIDC state parameter (validated by authlib before reaching this line) already proves the code came from Microsoft.

**Role resolution:** `extract_role_from_token()` checks three sources in priority order:
1. Azure app `roles` claim (Enterprise Application role assignments)
2. `groups` claim mapped via `OIDC_GROUP_ROLE_MAP` env var
3. Per-email override via `OIDC_ROLE_MAP` env var

If none match → `role=None` → user is redirected to `/login?error=norole`.

---

### 7.4 `GET /logout`

```python
@app.get("/logout")
async def logout(request: Request, response: Response):
    sid = get_session_id(request)
    destroy_session(sid)           # removes from auth._sessions
    r = RedirectResponse(url="/login", status_code=302)
    r.delete_cookie(SESSION_COOKIE)
    return r
```

Destroys the server-side session and deletes the browser cookie. Does **not** hit Microsoft's logout endpoint — the user's Microsoft SSO session remains active. This is intentional for internal tools (re-login via OIDC will be instant).

---

### 7.5 `GET /api/me`

Returns the current user's context as JSON, consumed by the React SPA on load:

```python
@app.get("/api/me")
async def api_me(request: Request):
    username = get_current_username(request)
    if not username:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    sid = get_session_id(request)
    role = get_role_for_session(sid)
    categories = get_allowed_categories(role)
    return {
        "username": username,
        "role": role,
        "allowed_categories": categories,   # None = all, or list of slugs
    }
```

The React SPA calls this on startup to decide which sidebar categories to render. `allowed_categories` comes from `auth.ROLE_ALLOWED_CATEGORIES` — `None` means all categories are visible (admin), a list like `["netbrain", "panorama"]` restricts the UI for `netadmin`.

---

### 7.6 `GET /` — Main Chat UI

```python
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    username = get_current_username(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)

    react_index = APP_DIR / "frontend" / "dist" / "index.html"
    if react_index.exists():
        return HTMLResponse(react_index.read_text(encoding="utf-8"))  # React SPA

    # Fallback to Jinja2 server-rendered template
    sid = get_session_id(request)
    role = get_role_for_session(sid)
    categories = get_allowed_categories(role)
    return templates.TemplateResponse("index.html", {
        "request": request, "username": username, "role": role,
        "allowed_categories": categories,
    })
```

**Two rendering modes:**
1. **React SPA** (production) — if `frontend/dist/index.html` exists, FastAPI serves it as a raw HTML string. The React app then calls `/api/me` to get user context and manages its own routing.
2. **Jinja2 fallback** (development / no frontend build) — FastAPI renders `templates/index.html` server-side, injecting `username`, `role`, and `allowed_categories` directly into the template context.

---

### 7.7 `GET /health`

```python
@app.get("/health")
async def health_check():
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = os.getenv("MCP_SERVER_PORT", "8765")
    mcp_url = f"http://{host}:{port}/health"
    async with aiohttp.ClientSession() as session:
        async with session.get(mcp_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
            data = await resp.json()
            mcp_status = data.get("status", "ok")
            mcp_tools = data.get("tools_registered")
    return {
        "status": "ok",
        "auth_mode": AUTH_MODE,
        "mcp_server": mcp_status,       # "ok" / "unreachable" / "error (N)"
        "mcp_tools_registered": mcp_tools,
    }
```

Probes the MCP server's `/health` endpoint via `aiohttp` with a 3-second timeout. Returns a combined status object. No authentication required — useful for load balancer and monitoring checks.

---

### 7.8 `POST /api/discover`

```python
@app.post("/api/discover")
async def api_discover(request: Request, body: ChatRequest):
    username = get_current_username(request)
    if not username:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    result = await process_message(
        body.message.strip(),
        body.conversation_history or [],
        discover_only=True,         # ← key difference from /api/chat
        username=username,
    )
    return result
```

Calls `process_message()` with `discover_only=True`. Tool selection (LLM + regex) runs, but the selected tool is **not executed** against NetBrain/NetBox/Panorama. Returns the tool name and extracted parameters only — used by the React frontend to show a "preview" before committing to a full query.

---

### 7.9 `POST /api/chat`

```python
@app.post("/api/chat")
async def api_chat(request: Request, body: ChatRequest):
    username = get_current_username(request)
    if not username:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    result = await process_message(
        body.message.strip(),
        body.conversation_history or [],
        default_live=True,
        username=username,
    )
    # Strip noisy L2 messages before returning
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, dict) and content.get("path_hops"):
            _strip_l2_noise(content)
    return result
```

The primary chat endpoint. Calls `process_message()` with the full agent loop (tool selection → RBAC check → execution → response normalisation). `default_live=True` means NetBrain path queries default to live mode (real traffic capture).

**`_strip_l2_noise()`** post-processes the result before it reaches the frontend:

```python
# app_fastapi.py (lines 330-337)
def _strip_l2_noise(d: dict) -> dict:
    """Remove noisy NetBrain L2 status messages from path results."""
    noise = (
        "l2 connections has not been discovered",
        "l2 connection has not been discovered",
    )
    for key in ("path_status_description", "statusDescription"):
        val = d.get(key)
        if isinstance(val, str) and any(p in val.lower() for p in noise):
            d[key] = ""
    return d
```

NetBrain sometimes populates `path_status_description` with L2 discovery warnings that are irrelevant to L3 path queries. This function blanks those strings out so they don't clutter the UI.

---

### 7.10 `POST /api/batch-upload`

Accepts a multipart form upload (`.csv`, `.xlsx`, `.xls`) plus an optional natural language `message` describing the intent:

```python
@app.post("/api/batch-upload")
async def batch_upload(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(""),
):
```

**Step 1 — Parse the file:**

```python
raw = await file.read()
if filename.lower().endswith(".csv"):
    df = pd.read_csv(io.BytesIO(raw))
else:
    df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
```

`pandas` reads the file into a `DataFrame`. `openpyxl` is the engine for `.xlsx`/`.xls` files.

**Step 2 — Column alias resolution:**

```python
_COL_ALIASES = {
    "source":      ("source ip", "source_ip", "src", "src ip", ...),
    "destination": ("destination ip", "destination_ip", "dest", "dst", ...),
    "protocol":    ("protocol", "proto"),
    "port":        ("port", "dst port", "dst_port", ...),
}

def _match_columns(columns: list[str]) -> dict[str, str]:
    """Map spreadsheet columns to expected field names (case-insensitive)."""
    col_map = {}
    lower_cols = {c.strip().lower(): c for c in columns}
    for field, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in lower_cols:
                col_map[field] = lower_cols[alias]
                break
    return col_map
```

This allows users to upload spreadsheets with any of the common column name variants (e.g. "Src IP", "source", "SRC_IP" all map to `source`). `source` and `destination` are **required**; `protocol` and `port` are optional and default to `"tcp"` and `"0"` respectively.

**Step 3 — Detect tool from message:**

```python
def _detect_batch_tool(message: str) -> str:
    path_kw = ("network path", "trace", "hop", "route", "show path", ...)
    if any(kw in msg for kw in path_kw):
        return "query_network_path"
    return "check_path_allowed"   # default
```

The user's optional `message` (e.g. `"show network paths"` vs `"check if traffic is allowed"`) determines whether to run `query_network_path` or `check_path_allowed` for every row. Defaults to `check_path_allowed`.

**Step 4 — Row iteration:**

```python
for idx, row in df.iterrows():
    src = str(row[col_map["source"]]).strip()
    dst = str(row[col_map["destination"]]).strip()
    proto = ...
    port_str = ...
    prompt = f"Is path from {src} to {dst} on {proto} port {port_str} allowed?"

    result = await process_message(
        prompt, [],
        tool_name=tool_name,
        parameters={"source": src, "destination": dst, "protocol": proto, "port": port_str},
        username=username,
    )
```

`process_message()` is called with `tool_name` and `parameters` pre-specified — this bypasses the LLM tool selection step and goes straight to execution. Each row runs independently, and errors are caught per-row so one failure doesn't abort the whole batch.

**Response format:**

```json
{
  "role": "assistant",
  "content": {
    "batch_results": [
      {
        "source": "10.0.0.1",
        "destination": "192.168.1.1",
        "protocol": "tcp",
        "port": "443",
        "status": "allowed",
        "reason": "...",
        "firewall_denied_by": ""
      },
      ...
    ],
    "tool": "check_path_allowed"
  }
}
```

For `query_network_path` rows, the result also includes `path_summary` (device names joined with `→`) and `path_hops`.

---

## 8. Request Model — `ChatRequest`

```python
# app_fastapi.py (lines 325-327)
from pydantic import BaseModel
from typing import Any

class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict[str, Any]] = []
```

Used by both `/api/discover` and `/api/chat`. FastAPI automatically validates and deserialises the JSON request body against this model. `conversation_history` defaults to an empty list, so clients can omit it for single-turn queries.

Each element in `conversation_history` is a dict with `{"role": "user"|"assistant", "content": str}`, matching the OpenAI chat message format that `chat_service.py` and the LLM prompt builder expect.

---

## 9. Static File Mounts

```python
# app_fastapi.py (lines 534-547)

VUE_ASSETS = APP_DIR / "frontend" / "dist" / "assets"
if VUE_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_ASSETS)), name="react-assets")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

ICONS_DIR = APP_DIR / "icons"
if ICONS_DIR.exists():
    app.mount("/icons", StaticFiles(directory=str(ICONS_DIR)), name="icons")
```

| Mount | Directory | Purpose |
|---|---|---|
| `/assets` | `frontend/dist/assets/` | React SPA compiled JS/CSS chunks |
| `/static` | `static/` | Shared CSS/JS for Jinja2 fallback templates |
| `/icons` | `icons/` | Device type icons for path visualisation |

Mounts are conditional — they are only registered if the directories exist, so the app starts cleanly even without a React build.

---

## 10. Error Handler

```python
# app_fastapi.py (lines 92-102)
@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    tb = traceback.format_exc()
    print(tb, flush=True)
    return HTMLResponse(
        content=f"<h1>Internal Server Error</h1><pre>{tb}</pre>",
        status_code=500,
    )
```

A catch-all exception handler returns a plain HTML page with the full Python traceback. This is useful during development but would need to be replaced with a generic error page in production to avoid leaking implementation details.

---

## 11. Startup and Server

```python
# app_fastapi.py (lines 550-561)
def main():
    import uvicorn
    uvicorn.run(
        "netbrain.app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
```

The app is started via `uvicorn`, an ASGI server. `reload=True` enables hot-reload for development — uvicorn watches for file changes and restarts automatically. In production this would be set to `False`.

The `app` object is the ASGI callable: `netbrain.app_fastapi:app` is the module path used by uvicorn to locate it.

---

## 12. Data Flow — How FastAPI Connects Everything

```
Browser / API client
        │
        │  HTTP (port 8000)
        ▼
┌─────────────────────────────────────┐
│           FastAPI (app_fastapi.py)  │
│                                     │
│  Session cookie  ─────────►  auth.py│
│  (netbrain_session)         │  _sessions{}
│                             │  OIDC TTL check
│                             │
│  POST /api/chat  ──────────►  chat_service.process_message()
│  POST /api/discover                 │
│  POST /api/batch-upload             │
│                                     │  ┌─────────────────────┐
│                             MCP client│◄─┤  mcp_server.py      │
│                                     │  │  (port 8765)        │
└─────────────────────────────────────┘  │  tools/ domain mods │
                                         └─────────────────────┘
```

- **Auth boundary:** Every protected route calls `get_current_username()` → `get_username_for_session()` in `auth.py`. Username is then passed into `process_message()` so RBAC is enforced per user.
- **Chat boundary:** FastAPI does zero AI or tool logic. It strips the L2 noise from results, but all intelligence lives in `chat_service.py`.
- **Batch bypass:** Batch upload passes `tool_name` and `parameters` directly to `process_message()`, skipping LLM tool selection for speed and reliability.
