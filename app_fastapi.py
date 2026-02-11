"""
FastAPI application with local authentication.
Replace Streamlit as the main web entry point; SAML can be added later.
"""
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any

try:
    from fastapi.templating import Jinja2Templates
except ImportError:
    from starlette.templating import Jinja2Templates

from netbrain.auth import (
    create_session,
    destroy_session,
    get_username_for_session,
    verify_local_user,
)

# Paths
APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"

# Ensure dirs exist
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="NetAssist", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.exception_handler(Exception)
async def catch_all(request: Request, exc: Exception):
    """Log and return a simple 500 page so we can see the error."""
    import traceback
    tb = traceback.format_exc()
    print(tb, flush=True)
    from fastapi.responses import HTMLResponse
    return HTMLResponse(
        content=f"<h1>Internal Server Error</h1><pre>{tb}</pre>",
        status_code=500,
    )

# Session cookie name and settings
SESSION_COOKIE = "netbrain_session"
SESSION_MAX_AGE = 86400 * 7  # 7 days


def get_session_id(request: Request) -> str | None:
    return request.cookies.get(SESSION_COOKIE)


def get_current_username(request: Request) -> str | None:
    sid = get_session_id(request)
    return get_username_for_session(sid)


def require_auth(request: Request) -> str:
    """Dependency: returns username if authenticated. For redirect, return from route instead of raising."""
    username = get_current_username(request)
    if not username:
        raise RedirectResponse(url="/login", status_code=302)  # FastAPI handles this
    return username


# --- Routes ---


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve login page."""
    if get_current_username(request):
        return RedirectResponse(url="/", status_code=302)
    error_invalid = request.query_params.get("error") == "invalid"
    return templates.TemplateResponse(
        "login.html",
        {"request": request, "error_invalid": error_invalid},
    )


@app.post("/login")
async def login_post(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
):
    """Validate credentials and set session cookie."""
    if not verify_local_user(username, password):
        return RedirectResponse(
            url="/login?error=invalid",
            status_code=302,
        )
    session_id = create_session(username)
    r = RedirectResponse(url="/", status_code=302)
    r.set_cookie(
        key=SESSION_COOKIE,
        value=session_id,
        max_age=SESSION_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return r


@app.get("/logout")
async def logout(request: Request, response: Response):
    """Clear session and redirect to login."""
    sid = get_session_id(request)
    destroy_session(sid)
    r = RedirectResponse(url="/login", status_code=302)
    r.delete_cookie(SESSION_COOKIE)
    return r


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve main chat app (protected)."""
    if not get_current_username(request):
        return RedirectResponse(url="/login", status_code=302)
    return templates.TemplateResponse("index.html", {"request": request})


# --- Chat API ---
class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict[str, Any]] = []  # content may be str or dict (e.g. requires_site)


def _strip_l2_noise(d: dict) -> dict:
    """Remove noisy NetBrain L2 status messages from path results."""
    noise = ("l2 connections has not been discovered", "l2 connection has not been discovered")
    for key in ("path_status_description", "statusDescription"):
        val = d.get(key)
        if isinstance(val, str) and any(p in val.lower() for p in noise):
            d[key] = ""
    return d


@app.post("/api/chat")
async def api_chat(request: Request, body: ChatRequest):
    """Process a chat message and return assistant response."""
    if not get_current_username(request):
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    from netbrain.chat_service import process_message
    result = await process_message(
        body.message.strip(),
        body.conversation_history or [],
        default_live=True,
    )
    # Strip noisy L2 messages before sending to frontend
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, dict) and content.get("path_hops"):
            _strip_l2_noise(content)
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
    # Path query keywords → query_network_path (hop-by-hop trace)
    path_kw = ("network path", "trace", "hop", "route", "show path", "find path",
               "path between", "traceroute", "path map", "path query", "path from")
    if any(kw in msg for kw in path_kw):
        return "query_network_path"
    # Default → check_path_allowed (firewall allow/deny check)
    return "check_path_allowed"


@app.post("/api/batch-upload")
async def batch_upload(
    request: Request,
    file: UploadFile = File(...),
    message: str = Form(""),
):
    """Parse an uploaded spreadsheet and run the appropriate tool for each row."""
    from fastapi.responses import JSONResponse

    if not get_current_username(request):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

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

    from netbrain.chat_service import process_message

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
                    # query_network_path – extract hop summary
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
                    # If hops were found, path was resolved → "success"
                    # NetBrain may report "Failed" for L2/policy reasons even when hops exist
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
                        # Include full path data for graphic rendering
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


# Mount static files (CSS/JS) if present
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount icons for path visualization (NetBrain-style)
ICONS_DIR = APP_DIR / "icons"
if ICONS_DIR.exists():
    app.mount("/icons", StaticFiles(directory=str(ICONS_DIR)), name="icons")


def main():
    import uvicorn
    # reload=True: watch source files and restart on change (development)
    uvicorn.run(
        "netbrain.app_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
