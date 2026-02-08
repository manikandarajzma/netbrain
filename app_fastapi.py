"""
FastAPI application with local authentication.
Replace Streamlit as the main web entry point; SAML can be added later.
"""
from pathlib import Path

from fastapi import Depends, FastAPI, Form, Request, Response
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

app = FastAPI(title="NetBrain Network Query", version="0.1.0")

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
    return result


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
