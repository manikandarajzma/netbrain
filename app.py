"""
FastAPI application with Microsoft Entra ID (OIDC) authentication.
All credentials are in Azure Key Vault; no local passwords.
"""
import os
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
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
STATIC_DIR = APP_DIR / "static"

# Ensure dirs exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Atlas", version="0.1.0")

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
    """Serve main chat app — React SPA if built, else Jinja2 fallback."""
    username = get_current_username(request)
    if not username:
        return RedirectResponse(url="/login", status_code=302)

    # Serve React build if available
    react_index = APP_DIR / "frontend" / "dist" / "index.html"
    if react_index.exists():
        return HTMLResponse(react_index.read_text(encoding="utf-8"))

    # Fallback to Jinja2 template
    sid = get_session_id(request)
    group = get_group_for_session(sid)
    categories = get_allowed_categories(group)
    return templates.TemplateResponse(request, "index.html", {
        "username": username,
        "group": group,
        "allowed_categories": categories,
    })


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


@app.post("/api/discover")
async def api_discover(request: Request, body: ChatRequest):
    """Lightweight tool discovery — returns tool name without executing."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from atlas.chat_service import process_message
    result = await process_message(
        body.message.strip(),
        body.conversation_history or [],
        discover_only=True,
        username=username,
        session_id=get_session_id(request),
    )
    return result


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
    """Process a chat message and return assistant response. Uses conversation_id when provided."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)
    from atlas.chat_service import process_message
    from atlas.chat_history import create_conversation, append_to_conversation
    conversation_id = (body.conversation_id or "").strip() or None
    history = body.conversation_history or []
    result = await process_message(
        body.message.strip(),
        history,
        default_live=True,
        username=username,
        session_id=get_session_id(request),
    )
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
    user_msg = body.message.strip()
    if conversation_id:
        # Append to existing conversation
        append_to_conversation(APP_DIR, username, conversation_id, user_msg, assistant_content)
        if isinstance(result, dict):
            result["conversation_id"] = conversation_id
    else:
        # New conversation: create and append (optionally as follow-up under parent)
        parent_id = (body.parent_conversation_id or "").strip() or None
        title = (user_msg[:60] + "…") if len(user_msg) > 60 else user_msg or "New chat"
        conv_id = create_conversation(APP_DIR, username, title, parent_id=parent_id)
        append_to_conversation(APP_DIR, username, conv_id, user_msg, assistant_content)
        if isinstance(result, dict):
            result["conversation_id"] = conv_id
            result["conversation_title"] = title
            if parent_id:
                result["parent_id"] = parent_id
    return result


# --- Batch Upload API ---

_COL_ALIASES = {
    "source": ("source ip", "source_ip", "src", "src ip", "src_ip", "source"),
    "destination": ("destination ip", "destination_ip", "dest", "dest ip", "dest_ip", "destination", "dst", "dst ip", "dst_ip"),
    "protocol": ("protocol", "proto"),
    "port": ("port", "dst port", "dst_port", "dest port", "dest_port"),
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


def _detect_batch_tool(message: str) -> str:
    """Determine which MCP tool to run based on the user's natural language message."""
    msg = message.lower()
    path_kw = ("network path", "trace", "hop", "route", "show path", "find path",
               "path between", "traceroute", "path map", "path query", "path from")
    if any(kw in msg for kw in path_kw):
        return "query_network_path"
    return "check_path_allowed"


@app.post("/api/batch-upload")
async def batch_upload(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(""),
):
    """Parse an uploaded spreadsheet and run the appropriate tool for each row."""
    username = get_current_username(request)
    if not username:
        return response_401_clear_session(request)

    filename = file.filename or ""
    if not filename.lower().endswith((".xlsx", ".xls", ".csv")):
        return JSONResponse(
            {"error": "Unsupported file type. Upload .xlsx or .csv"},
            status_code=400,
        )

    import pandas as pd
    import io

    raw = await file.read()
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
        else:
            df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except Exception as exc:
        return JSONResponse({"error": f"Failed to parse file: {exc}"}, status_code=400)

    if df.empty:
        return JSONResponse({"error": "Spreadsheet is empty"}, status_code=400)

    col_map = _match_columns(list(df.columns))
    missing = [f for f in ("source", "destination") if f not in col_map]
    if missing:
        return JSONResponse(
            {"error": f"Missing required columns: {', '.join(missing)}. Found: {', '.join(df.columns)}"},
            status_code=400,
        )

    tool_name = _detect_batch_tool(message)

    from atlas.chat_service import process_message

    results = []
    for idx, row in df.iterrows():
        src = str(row[col_map["source"]]).strip()
        dst = str(row[col_map["destination"]]).strip()
        proto = str(row.get(col_map.get("protocol", ""), "tcp")).strip().lower() or "tcp"
        port_val = row.get(col_map.get("port", ""), 0)
        port_str = str(int(port_val)) if pd.notna(port_val) and port_val else "0"

        prompt = (
            f"Is path from {src} to {dst} on {proto} port {port_str} allowed?"
            if tool_name == "check_path_allowed"
            else f"Show network path from {src} to {dst} on {proto} port {port_str}"
        )

        try:
            result = await process_message(
                prompt,
                [],
                tool_name=tool_name,
                parameters={"source": src, "destination": dst, "protocol": proto, "port": port_str},
                username=username,
                session_id=get_session_id(request),
            )
            content = result.get("content", {}) if isinstance(result, dict) else {}
            if isinstance(content, dict):
                _strip_l2_noise(content)
                if tool_name == "check_path_allowed":
                    results.append({
                        "source": src,
                        "destination": dst,
                        "protocol": proto,
                        "port": port_str,
                        "status": content.get("status", "unknown"),
                        "reason": content.get("reason", ""),
                        "firewall_denied_by": content.get("firewall_denied_by", ""),
                        "policy_details": content.get("policy_details", ""),
                    })
                else:
                    hops = content.get("path_hops", [])
                    hop_names = []
                    if hops:
                        first = hops[0].get("from_device")
                        if first:
                            hop_names.append(first)
                        for h in hops:
                            td = h.get("to_device")
                            if td and td not in hop_names:
                                hop_names.append(td)
                    raw_status = (content.get("path_status") or "unknown").lower()
                    effective_status = "success" if hops else raw_status
                    row_result = {
                        "source": src,
                        "destination": dst,
                        "protocol": proto,
                        "port": port_str,
                        "status": effective_status,
                        "reason": content.get("path_status_description", ""),
                        "path_summary": " → ".join(hop_names) if hop_names else "",
                        "path_hops": hops,
                        "path_status": raw_status,
                        "path_failure_reason": content.get("path_failure_reason", ""),
                    }
                    results.append(row_result)
            else:
                results.append({
                    "source": src, "destination": dst, "protocol": proto,
                    "port": port_str, "status": "error", "reason": str(content),
                })
        except Exception as exc:
            results.append({
                "source": src, "destination": dst, "protocol": proto,
                "port": port_str, "status": "error", "reason": str(exc),
            })

    return {"role": "assistant", "content": {"batch_results": results, "tool": tool_name}}


# Mount React build assets (must come before /static to take priority)
REACT_DIST = APP_DIR / "frontend" / "dist"
REACT_ASSETS = REACT_DIST / "assets"
if REACT_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=str(REACT_ASSETS)), name="react-assets")

# Mount static files (CSS/JS) if present
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

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
