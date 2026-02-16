"""
Local authentication for FastAPI app.
Designed to be swapped for SAML later.
"""
import os
import secrets
from typing import Optional

# Load .env file if available
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    load_dotenv(_env_path)

# In-memory session store: session_id -> username
# For production with multiple workers, use Redis or signed cookies
_sessions: dict[str, str] = {}

# Default local users (override via env); format: USER1:PASS1:ROLE1,USER2:PASS2:ROLE2
# Role is optional and defaults to "admin" for backward compatibility (USER:PASS works)
_default_users_env = os.getenv("NETBRAIN_USERS", "admin:admin:admin")
LOCAL_USERS: dict[str, dict[str, str]] = {}
for part in _default_users_env.strip().split(","):
    part = part.strip()
    if ":" in part:
        pieces = part.split(":")
        u = pieces[0].strip()
        p = pieces[1].strip() if len(pieces) > 1 else ""
        r = pieces[2].strip() if len(pieces) > 2 else "admin"
        LOCAL_USERS[u] = {"password": p, "role": r}


# ---------------------------------------------------------------------------
# Role-based access control
# ---------------------------------------------------------------------------

# Maps role → set of allowed MCP tool names.  None = all tools allowed.
ROLE_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "admin": None,
    "netadmin": {
        "query_network_path",
        "check_path_allowed",
        "query_panorama_ip_object_group",
        "query_panorama_address_group_members",
    },
}

# Maps role → list of sidebar category slugs shown in the UI.  None = all.
ROLE_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin": None,
    "netadmin": ["netbrain", "panorama"],
}


def verify_local_user(username: str, password: str) -> bool:
    """Verify username/password against local user store."""
    if not username or not password:
        return False
    entry = LOCAL_USERS.get(username)
    if entry is None:
        return False
    return entry["password"] == password


def get_user_role(username: str) -> str:
    """Return the role for *username*, defaulting to 'admin'."""
    entry = LOCAL_USERS.get(username)
    if entry is None:
        return "admin"
    return entry.get("role", "admin")


def get_allowed_tools(role: str) -> set[str] | None:
    """Return the set of MCP tool names allowed for *role*, or None if all are allowed."""
    return ROLE_ALLOWED_TOOLS.get(role)


def get_allowed_categories(role: str) -> list[str] | None:
    """Return the sidebar categories visible for *role*, or None if all are visible."""
    return ROLE_ALLOWED_CATEGORIES.get(role)


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
