"""
Authentication for FastAPI app.
Supports local username/password and Microsoft Entra ID (OIDC).
"""
import os
import secrets
import time
from typing import Optional

# Load .env file if available
from dotenv import load_dotenv
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    load_dotenv(_env_path)

# ---------------------------------------------------------------------------
# Auth mode
# ---------------------------------------------------------------------------
AUTH_MODE = os.getenv("AUTH_MODE", "local").strip().lower()  # "local" or "oidc"

# ---------------------------------------------------------------------------
# Microsoft OIDC configuration
# ---------------------------------------------------------------------------
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
AZURE_AUTHORITY = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0"

# Session TTL for OIDC (30 minutes); local sessions last 7 days
OIDC_SESSION_TTL = 1800

# ---------------------------------------------------------------------------
# In-memory session store: session_id -> {username, role, ...}
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Local users (used when AUTH_MODE == "local")
# ---------------------------------------------------------------------------
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

# Maps role -> set of allowed MCP tool names.  None = all tools allowed.
ROLE_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "admin": None,
    "netadmin": {
        "query_network_path",
        "check_path_allowed",
        "query_panorama_ip_object_group",
        "query_panorama_address_group_members",
    },
}

# Maps role -> list of sidebar category slugs shown in the UI.  None = all.
ROLE_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin": None,
    "netadmin": ["netbrain", "panorama"],
}

# ---------------------------------------------------------------------------
# OIDC role resolution fallbacks (when Azure app roles aren't in the token)
# ---------------------------------------------------------------------------

# 1. Per-user email override: OIDC_ROLE_MAP=user@example.com:netadmin
OIDC_ROLE_MAP: dict[str, str] = {}
_oidc_role_map_env = os.getenv("OIDC_ROLE_MAP", "")
for _part in _oidc_role_map_env.strip().split(","):
    _part = _part.strip()
    if ":" in _part:
        _email, _role = _part.rsplit(":", 1)
        OIDC_ROLE_MAP[_email.strip().lower()] = _role.strip().lower()

# 2. Azure security group -> role mapping (scalable):
#    OIDC_GROUP_ROLE_MAP=<group-object-id>:netadmin,<group-object-id>:admin
#    Requires: App Registration > Token configuration > Add groups claim
OIDC_GROUP_ROLE_MAP: dict[str, str] = {}
_oidc_group_map_env = os.getenv("OIDC_GROUP_ROLE_MAP", "")
for _part in _oidc_group_map_env.strip().split(","):
    _part = _part.strip()
    if ":" in _part:
        _gid, _role = _part.rsplit(":", 1)
        OIDC_GROUP_ROLE_MAP[_gid.strip().lower()] = _role.strip().lower()


def verify_local_user(username: str, password: str) -> bool:
    """Verify username/password against local user store."""
    if not username or not password:
        return False
    entry = LOCAL_USERS.get(username)
    if entry is None:
        return False
    return entry["password"] == password


def get_user_role(username: str) -> str:
    """Return the role for *username*.

    For OIDC users the role is stored in the session, so we check there first.
    For local users we look up LOCAL_USERS.  Defaults to 'admin'.
    """
    # Check if any active session has this username with a stored role
    for sess in _sessions.values():
        if sess.get("username") == username and "role" in sess:
            return sess["role"]
    # Fall back to local users dict
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


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def create_session(username: str, *, role: str = "admin",
                   auth_mode: str = "local",
                   tokens: dict | None = None) -> str:
    """Create a new session; returns session_id."""
    session_id = secrets.token_urlsafe(32)
    _sessions[session_id] = {
        "username": username,
        "role": role,
        "auth_mode": auth_mode,
        "created_at": time.time(),
        "tokens": tokens,  # OIDC tokens for refresh
    }
    return session_id


def get_session(session_id: Optional[str]) -> Optional[dict]:
    """Return full session dict, or None."""
    if not session_id:
        return None
    sess = _sessions.get(session_id)
    if sess is None:
        return None
    # Check OIDC session expiry
    if sess.get("auth_mode") == "oidc":
        elapsed = time.time() - sess.get("created_at", 0)
        if elapsed > OIDC_SESSION_TTL:
            del _sessions[session_id]
            return None
    return sess


def get_username_for_session(session_id: Optional[str]) -> Optional[str]:
    """Return username for session_id, or None if invalid."""
    sess = get_session(session_id)
    if sess is None:
        return None
    return sess.get("username")


def get_role_for_session(session_id: Optional[str]) -> str:
    """Return role for session_id, defaulting to 'admin'."""
    sess = get_session(session_id)
    if sess is None:
        return "admin"
    return sess.get("role", "admin")


def destroy_session(session_id: Optional[str]) -> None:
    """Remove session."""
    if session_id and session_id in _sessions:
        del _sessions[session_id]


# ---------------------------------------------------------------------------
# OIDC helpers
# ---------------------------------------------------------------------------

def extract_role_from_token(token_claims: dict) -> Optional[str]:
    """Extract the NetAssist role from Azure token claims.

    Priority order:
    1. Azure 'roles' claim (app roles assigned via Enterprise Application)
    2. OIDC_GROUP_ROLE_MAP (Azure security group -> role)
    3. OIDC_ROLE_MAP (per-email override)

    Returns None if no role could be resolved (user has no access).
    """
    # 1. Check Azure app roles
    roles = token_claims.get("roles", [])
    for r in roles:
        r_lower = r.lower().strip()
        if r_lower in ROLE_ALLOWED_TOOLS:
            return r_lower

    # 2. Check Azure security groups
    if OIDC_GROUP_ROLE_MAP:
        groups = token_claims.get("groups", [])
        for gid in groups:
            role = OIDC_GROUP_ROLE_MAP.get(str(gid).lower())
            if role and role in ROLE_ALLOWED_TOOLS:
                return role

    # 3. Check per-email override
    if OIDC_ROLE_MAP:
        for key in ("preferred_username", "email", "upn"):
            email = token_claims.get(key, "").strip().lower()
            if email and email in OIDC_ROLE_MAP:
                return OIDC_ROLE_MAP[email]

    # No role found — user has no access
    return None


def extract_username_from_token(token_claims: dict) -> str:
    """Extract display username from Azure token claims."""
    return (
        token_claims.get("preferred_username")
        or token_claims.get("email")
        or token_claims.get("name")
        or token_claims.get("sub", "unknown")
    )
