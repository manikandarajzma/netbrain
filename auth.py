"""
Local authentication for FastAPI app.
Designed to be swapped for SAML later.
"""
import os
import secrets
from typing import Optional

# In-memory session store: session_id -> username
# For production with multiple workers, use Redis or signed cookies
_sessions: dict[str, str] = {}

# Default local users (override via env); format: USER1:PASS1,USER2:PASS2
_default_users_env = os.getenv("NETBRAIN_USERS", "admin:admin")
LOCAL_USERS: dict[str, str] = {}
for part in _default_users_env.strip().split(","):
    part = part.strip()
    if ":" in part:
        u, p = part.split(":", 1)
        LOCAL_USERS[u.strip()] = p.strip()


def verify_local_user(username: str, password: str) -> bool:
    """Verify username/password against local user store."""
    if not username or not password:
        return False
    return LOCAL_USERS.get(username) == password


def create_session(username: str) -> str:
    """Create a new session for username; returns session_id."""
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = username
    return session_id


def get_username_for_session(session_id: Optional[str]) -> Optional[str]:
    """Return username for session_id, or None if invalid."""
    if not session_id:
        return None
    return _sessions.get(session_id)


def destroy_session(session_id: Optional[str]) -> None:
    """Remove session."""
    if session_id and session_id in _sessions:
        del _sessions[session_id]
