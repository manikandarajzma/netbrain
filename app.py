"""
FastAPI application with Microsoft Entra ID (OIDC) authentication.
All credentials are in Azure Key Vault; no local passwords.
"""
import hashlib
import logging
import os
from pathlib import Path

import asyncio
import json as _json

_sse_log = logging.getLogger("atlas.app")

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any

try:
    from fastapi.templating import Jinja2Templates
except ImportError:
    from starlette.templating import Jinja2Templates

from atlas.auth import (
    AUTH_MODE,
    AZURE_AUTHORITY,
    AZURE_CLIENT_ID,
    AZURE_CLIENT_SECRET,
    AZURE_TENANT_ID,
    OIDC_SESSION_TTL,
    create_session,
    destroy_session,
    extract_group_from_token,
    extract_username_from_token,
    get_allowed_categories,
    get_group_for_session,
    get_user_group,
    get_username_for_session,
)

# Paths
APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

async def _midnight_sync_loop():
    """Background task: sync closed ServiceNow incidents into memory daily at midnight."""
    from datetime import datetime, timedelta
    while True:
        try:
            now = datetime.now()
            next_midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            sleep_secs = (next_midnight - now).total_seconds()
            _sse_log.info("servicenow_memory_sync: next run in %.0fs (at midnight)", sleep_secs)
            await asyncio.sleep(sleep_secs)
            from servicenow_memory_sync import sync_closed_incidents
            await sync_closed_incidents()
        except asyncio.CancelledError:
            break
        except Exception as exc:
            _sse_log.warning("servicenow_memory_sync: error in midnight loop: %s", exc)
            await asyncio.sleep(3600)  # retry in 1 hour on failure


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    # Run an initial sync on startup so memory is populated immediately
    try:
        from servicenow_memory_sync import sync_closed_incidents
        asyncio.create_task(sync_closed_incidents())
    except Exception as exc:
        _sse_log.warning("servicenow_memory_sync: startup sync failed: %s", exc)
    # Schedule daily midnight syncs
    sync_task = asyncio.create_task(_midnight_sync_loop())
    yield
    sync_task.cancel()


app = FastAPI(title="Atlas", version="0.1.0", lifespan=lifespan)

# CORS: require CORS_ALLOWED_ORIGINS (no default "*"). Set to your origin(s), e.g. https://atlas.company.com or localhost for dev.
_cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] if _cors_origins_env else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# OIDC setup (authlib)
# ---------------------------------------------------------------------------
oauth = None
if AUTH_MODE == "oidc" and AZURE_CLIENT_ID and AZURE_TENANT_ID:
    from authlib.integrations.starlette_client import OAuth
    oauth = OAuth()
    oauth.register(
        name="microsoft",
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
        server_metadata_url=(
            f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0"
            "/.well-known/openid-configuration"
        ),
        client_kwargs={
            "scope": "openid profile email offline_access",
        },
    )

# ---------------------------------------------------------------------------
# Middleware: add session secret for authlib state parameter
# ---------------------------------------------------------------------------
if AUTH_MODE == "oidc":
    from starlette.middleware.sessions import SessionMiddleware
    import secrets as _secrets
    # Use fixed secret from env so multiple instances share it; else random per process (single-instance only).
    _oauth_secret = os.getenv("OAUTH_STATE_SECRET", "").strip() or _secrets.token_urlsafe(32)
    app.add_middleware(
        SessionMiddleware,
        secret_key=_oauth_secret,
        session_cookie="atlas_oauth_state",
        max_age=600,  # 10 min for OAuth state
    )


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    """Log traceback server-side only; return generic error to client (no info disclosure)."""
    import logging
    logging.exception("Unhandled exception")
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            {"detail": "Something went wrong. Please try again."},
            status_code=500,
        )
    return HTMLResponse(
        content="<h1>Internal Server Error</h1><p>Something went wrong. Please try again.</p>",
        status_code=500,
    )

# Session cookie name and settings
SESSION_COOKIE = "atlas_session"
SESSION_MAX_AGE_OIDC = OIDC_SESSION_TTL  # 30 min


def get_session_id(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)


def response_401_clear_session(request: Request):
    """Return 401 with redirect to /login; clear server session and cookie so user can sign in again."""
    sid = get_session_id(request)
    destroy_session(sid)
    r = JSONResponse(
        {"detail": "Not authenticated", "redirect": "/login"},
        status_code=401,
    )
    r.delete_cookie(SESSION_COOKIE)
    return r


def get_current_username(request: Request) -> str | None:
    sid = get_session_id(request)
    return get_username_for_session(sid)


def require_auth(request: Request) -> str:
    """Dependency: returns username if authenticated."""
    username = get_current_username(request)
    if not username:
        raise RedirectResponse(url="/login", status_code=302)
    return username


# --- Routes ---


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page."""
    if get_current_username(request):
        return RedirectResponse(url="/", status_code=302)
    error_param = request.query_params.get("error", "")
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "error_oidc": error_param == "oidc",
            "error_norole": error_param == "norole",
            "oidc_configured": AUTH_MODE == "oidc" and bool(AZURE_CLIENT_ID and AZURE_TENANT_ID),
        },
    )


# --- OIDC routes ---

@app.get("/auth/microsoft")
async def auth_microsoft(request: Request):
    """Redirect to Microsoft login."""
    if oauth is None:
        return RedirectResponse(url="/login?error=oidc", status_code=302)
    # Build callback URL, ensuring we use "localhost" (must match Azure portal)
    redirect_uri = str(request.url_for("auth_callback")).replace("://127.0.0.1", "://localhost")
    return await oauth.microsoft.authorize_redirect(request, redirect_uri, prompt="select_account")


@app.get("/auth/callback")
async def auth_callback(request: Request):
    """Handle the Microsoft OIDC redirect after the user authenticates.

    Flow:
      1. Exchange the authorisation code for tokens (authlib handles PKCE/state).
      2. Decode the id_token JWT payload to get the user's claims, including the
         `groups` claim containing Entra ID group Object IDs.
         We decode the JWT ourselves rather than relying on authlib's userinfo
         because the `groups` claim is only present in the id_token, not in the
         /userinfo endpoint response (Microsoft does not include group memberships
         in the userinfo endpoint).
      3. Resolve the group from the claims (auth.py). Reject if not in a known group.
      4. Create a signed session cookie and redirect to the main app.
    """
    if oauth is None:
        return RedirectResponse(url="/login?error=oidc", status_code=302)

    try:
        token = await oauth.microsoft.authorize_access_token(request)
    except Exception as exc:
        print(f"OIDC token error: {exc}", flush=True)
        return RedirectResponse(url="/login?error=oidc", status_code=302)

    # Decode the id_token JWT payload (middle segment, base64url-encoded JSON).
    # We do not verify the signature here — authlib already verified it during
    # authorize_access_token(). We just need the claims dict.
    import json as _json
    import base64 as _base64
    userinfo = {}
    id_token = token.get("id_token")
    if id_token:
        try:
            payload = id_token.split(".")[1]
            # JWT base64url omits padding — restore it before decoding.
            payload += "=" * (4 - len(payload) % 4)
            userinfo = _json.loads(_base64.urlsafe_b64decode(payload))
        except Exception:
            pass
    # Fall back to authlib's parsed userinfo if id_token decoding failed.
    if not userinfo:
        userinfo = token.get("userinfo") or {}

    if not userinfo:
        return RedirectResponse(url="/login?error=oidc", status_code=302)

    username = extract_username_from_token(userinfo)
    # Resolve group from the `groups` claim in the token. Returns None if the
    # user is not a member of any recognised group — reject the login.
    group = extract_group_from_token(userinfo)
    if group is None:
        return RedirectResponse(url="/login?error=norole", status_code=302)

    # Bake username + group into a signed cookie. No server-side session store.
    session_id = create_session(
        username,
        group=group,
        auth_mode="oidc",
        tokens={
            "access_token": token.get("access_token"),
            "refresh_token": token.get("refresh_token"),
            "id_token": token.get("id_token"),
        },
    )

    r = RedirectResponse(url="/", status_code=302)
    r.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=SESSION_MAX_AGE_OIDC,
        httponly=True,   # not accessible from JavaScript
        samesite="lax",  # sent on top-level navigations; blocks CSRF
    )
    return r


@app.get("/logout")
async def logout(request: Request, response: Response):
    """Clear session and redirect to login (or Microsoft logout)."""
    sid = get_session_id(request)
    destroy_session(sid)
    r = RedirectResponse(url="/login", status_code=302)
    r.delete_cookie(SESSION_COOKIE)
    return r


@app.get("/api/me")
async def api_me(request: Request):
    """Return current user context for the React SPA."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    sid = get_session_id(request)
    group = get_group_for_session(sid)
    categories = get_allowed_categories(group)
    return {
        "username": username,
        "group": group,
        "allowed_categories": categories,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the React SPA."""
    username = get_current_username(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)

    react_index = APP_DIR / "frontend" / "dist" / "index.html"
    if not react_index.exists():
        return HTMLResponse(
            "<h2>Frontend not built.</h2><p>Run <code>npm run build</code> inside <code>frontend/</code>.</p>",
            status_code=503,
        )
    return HTMLResponse(react_index.read_text(encoding="utf-8"))


# --- Health check ---

@app.get("/health")
async def health_check():
    """Return app status, MCP server reachability, and Ollama status."""
    import os, aiohttp
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = os.getenv("MCP_SERVER_PORT", "8765")
    mcp_url = f"http://{host}:{port}/health"
    mcp_status = "unknown"
    mcp_tools = None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(mcp_url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    mcp_status = data.get("status", "ok")
                    mcp_tools = data.get("tools_registered")
                else:
                    mcp_status = f"error ({resp.status})"
    except Exception:
        mcp_status = "unreachable"

    # Ollama status: check reachability and whether the configured model is available
    from atlas.tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL
    ollama_status = "unknown"
    ollama_model_available = None
    try:
        # Use OpenAI-compatible /models endpoint (works with Docker Model Runner and Ollama)
        base = OLLAMA_BASE_URL.rstrip("/")
        models_url = f"{base}/models"
        async with aiohttp.ClientSession() as session:
            async with session.get(
                models_url,
                timeout=aiohttp.ClientTimeout(total=3),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    models = [m.get("id", "") for m in data.get("data", [])]
                    ollama_model_available = any(
                        m == OLLAMA_MODEL or m.split(":")[0] == OLLAMA_MODEL.split(":")[0]
                        for m in models
                    )
                    ollama_status = "ok" if ollama_model_available else "model_not_found"
                else:
                    ollama_status = f"error ({resp.status})"
    except Exception:
        ollama_status = "unreachable"

    return {
        "status": "ok",
        "auth_mode": AUTH_MODE,
        "mcp_server": mcp_status,
        "mcp_tools_registered": mcp_tools,
        "ollama": {
            "status": ollama_status,
            "url": OLLAMA_BASE_URL,
            "model": OLLAMA_MODEL,
            "model_available": ollama_model_available,
        },
    }


# --- Chat API ---
class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict[str, Any]] = []
    conversation_id: str | None = None
    parent_conversation_id: str | None = None


def _strip_l2_noise(d: dict) -> dict:
    """Remove noisy path L2 status messages from path results."""
    noise = ("l2 connections has not been discovered", "l2 connection has not been discovered")
    for key in ("path_status_description", "statusDescription"):
        val = d.get(key)
        if isinstance(val, str) and any(p in val.lower() for p in noise):
            d[key] = ""
    return d


@app.get("/api/topology")
async def api_topology(request: Request):
    """Return network topology: devices, links (derived from /31 subnets), and per-device stats."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from db import fetch
    devices_rows = await fetch(
        "SELECT hostname, host(mgmt_ip) AS mgmt_ip, platform, site, role FROM devices ORDER BY hostname"
    )
    links_rows = await fetch(
        """
        SELECT a.device AS device_a, a.interface AS iface_a, host(a.ip) AS ip_a,
               b.device AS device_b, b.interface AS iface_b, host(b.ip) AS ip_b
        FROM interface_ips a
        JOIN interface_ips b ON
          a.prefix_len = 31 AND b.prefix_len = 31
          AND a.device < b.device
          AND network(set_masklen(a.ip, 31)) = network(set_masklen(b.ip, 31))
        """
    )
    stats_rows = await fetch(
        """
        SELECT r.device,
               COUNT(r.prefix) AS route_count,
               (SELECT COUNT(*) FROM ospf_neighbors n WHERE n.device = r.device) AS ospf_neighbor_count,
               (SELECT COUNT(*) FROM ospf_neighbors n WHERE n.device = r.device AND n.state = 'full') AS ospf_full_count,
               (SELECT array_agg(n.router_id) FROM ospf_neighbors n WHERE n.device = r.device AND n.state = 'full') AS ospf_neighbors
        FROM routing_table r
        GROUP BY r.device
        """
    )
    # Also include devices with no routes in stats
    stat_map = {r["device"]: dict(r) for r in stats_rows}
    devices = []
    for d in devices_rows:
        hostname = d["hostname"]
        stat = stat_map.get(hostname, {})
        devices.append({
            "hostname": hostname,
            "mgmt_ip": d["mgmt_ip"],
            "platform": d["platform"],
            "site": d["site"],
            "role": d["role"],
            "route_count": stat.get("route_count", 0),
            "ospf_neighbor_count": stat.get("ospf_neighbor_count", 0),
            "ospf_full_count": stat.get("ospf_full_count", 0),
            "ospf_neighbors": stat.get("ospf_neighbors") or [],
        })
    links = [
        {
            "device_a": r["device_a"], "iface_a": r["iface_a"], "ip_a": r["ip_a"],
            "device_b": r["device_b"], "iface_b": r["iface_b"], "ip_b": r["ip_b"],
        }
        for r in links_rows
    ]
    return {"devices": devices, "links": links}


@app.post("/api/discover")
async def api_discover(request: Request, body: ChatRequest):
    """Lightweight tool discovery — returns display name without running the graph."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    # Return immediately without any LLM call — the label is always the same
    # for this app (single troubleshooting flow).
    return {"tool_display_name": "Network troubleshooter"}


@app.get("/api/chat/history")
async def api_chat_history(request: Request):
    """Return messages from the most recent conversation (backward compat)."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from atlas.chat_history import load_history
    messages = load_history(APP_DIR, username)
    return {"messages": messages}


@app.delete("/api/chat/history")
async def api_chat_history_clear(request: Request):
    """Clear all conversations for the current user (backward compat)."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from atlas.chat_history import clear_history
    clear_history(APP_DIR, username)
    return {"ok": True}


@app.get("/api/chat/conversations")
async def api_chat_conversations(request: Request):
    """List conversations for the current user (id, title, created_at), newest first."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from atlas.chat_history import list_conversations
    convs = list_conversations(APP_DIR, username)
    return {"conversations": convs}


@app.get("/api/chat/conversations/{conversation_id}")
async def api_chat_conversation(request: Request, conversation_id: str):
    """Get messages for a single conversation."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from atlas.chat_history import get_conversation
    messages = get_conversation(APP_DIR, username, conversation_id)
    if messages is None:
        return JSONResponse({"detail": "Not found"}, status_code=404)
    return {"messages": messages}


@app.delete("/api/chat/conversations/{conversation_id}")
async def api_chat_conversation_delete(request: Request, conversation_id: str):
    """Delete a conversation."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from atlas.chat_history import delete_conversation
    delete_conversation(APP_DIR, username, conversation_id)
    return {"ok": True}


@app.post("/api/chat")
async def api_chat(request: Request, body: ChatRequest):
    """Process a chat message and stream SSE progress events followed by the final result."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)

    import atlas.status_bus as status_bus
    from atlas.chat_service import process_message
    from atlas.chat_history import create_conversation, append_to_conversation

    sid = get_session_id(request)
    conversation_id = (body.conversation_id or "").strip() or None
    user_msg = body.message.strip()
    history = body.conversation_history or []

    _WRITE_RE = __import__('re').compile(
        r'\b(create|update|close|resolve|assign|delete|submit|open an? incident|add note)\b',
        __import__('re').IGNORECASE,
    )
    parent_id = (body.parent_conversation_id or "").strip() or None

    async def event_generator():
        queue = status_bus.register(sid)
        task = asyncio.create_task(
            process_message(
                user_msg,
                history,
                default_live=True,
                username=username,
                session_id=sid,
            )
        )
        try:
            while not task.done():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {_json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send SSE comment as heartbeat so the proxy doesn't drop idle connections
                    yield ": keep-alive\n\n"

            # Drain any remaining status events pushed before task completed
            while not queue.empty():
                event = queue.get_nowait()
                yield f"data: {_json.dumps(event)}\n\n"

            _sse_log.info("SSE: task completed, reading result")
            result = task.result()
            _sse_log.info("SSE: result obtained, role=%s", result.get("role") if isinstance(result, dict) else type(result))
            if _WRITE_RE.search(user_msg):
                # After any write (create/update/close), flush tool-level caches so the
                # next troubleshoot or list query reflects the new state.
                try:
                    import redis as _r
                    _rc = _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
                    for _pattern in ("atlas:snow:*", "atlas:snow:resp:*", "atlas:ts_cache:*"):
                        _keys = _rc.keys(_pattern)
                        if _keys:
                            _rc.delete(*_keys)
                            _sse_log.info("Write op: flushed %d keys matching %s", len(_keys), _pattern)
                except Exception as _ce:
                    _sse_log.warning("Write op: cache flush failed: %s", _ce)

            # Strip noisy L2 messages before sending to frontend
            if isinstance(result, dict):
                content = result.get("content")
                if isinstance(content, dict) and content.get("path_hops"):
                    _strip_l2_noise(content)

            assistant_content = result.get("content") if isinstance(result, dict) else None
            if assistant_content is None and isinstance(result, dict):
                assistant_content = result.get("message") or "No response"
            elif assistant_content is None:
                assistant_content = "No response"

            try:
                if conversation_id:
                    append_to_conversation(APP_DIR, username, conversation_id, user_msg, assistant_content)
                    if isinstance(result, dict):
                        result["conversation_id"] = conversation_id
                else:
                    title = (user_msg[:60] + "…") if len(user_msg) > 60 else user_msg or "New chat"
                    conv_id = create_conversation(APP_DIR, username, title, parent_id=parent_id)
                    append_to_conversation(APP_DIR, username, conv_id, user_msg, assistant_content)
                    if isinstance(result, dict):
                        result["conversation_id"] = conv_id
                        result["conversation_title"] = title
                        if parent_id:
                            result["parent_id"] = parent_id
            except Exception as hist_exc:
                _sse_log.warning("Chat history save failed (non-fatal): %s", hist_exc)

            done_event = {"type": "done", "result": result}
            _sse_log.info("SSE: sending done event, content_type=%s", type(result.get("content")).__name__ if isinstance(result, dict) else "?")
            yield f"data: {_json.dumps(done_event)}\n\n"
            _sse_log.info("SSE: done event sent successfully")
        except Exception as exc:
            _sse_log.exception("SSE generator error")
            if not task.done():
                task.cancel()
            error_event = {"type": "done", "result": {"role": "assistant", "content": f"Error: {exc}"}}
            yield f"data: {_json.dumps(error_event)}\n\n"
        finally:
            status_bus.deregister(sid)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Mount React build assets (must come before /static to take priority)
REACT_DIST = APP_DIR / "frontend" / "dist"
REACT_ASSETS = REACT_DIST / "assets"
if REACT_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_ASSETS)), name="react-assets")

# Mount icons for path visualization
ICONS_DIR = APP_DIR / "icons"
if ICONS_DIR.exists():
    app.mount("/icons", StaticFiles(directory=str(ICONS_DIR)), name="icons")


def main():
    import uvicorn
    uvicorn.run(
        "atlas.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
