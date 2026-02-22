"""
Authentication for FastAPI app.
Uses Microsoft Entra ID (OIDC) only. No local passwords; credentials are in Azure Key Vault.
Sessions are stored in signed cookies (no server-side store), so they survive restarts and work across instances.
"""
import os
import secrets
import time
from typing import Optional

from itsdangerous import BadSignature, URLSafeTimedSerializer

# Load .env from atlas/, project root, then cwd (first file wins per variable)
from dotenv import load_dotenv
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _path in (
    os.path.join(_this_dir, ".env"),
    os.path.join(os.path.dirname(_this_dir), ".env"),
    os.path.join(os.getcwd(), ".env"),
):
    if os.path.isfile(_path):
        load_dotenv(_path)

# ---------------------------------------------------------------------------
# Auth mode (OIDC only; no local password auth)
# ---------------------------------------------------------------------------
AUTH_MODE = os.getenv("AUTH_MODE", "oidc").strip().lower()

# ---------------------------------------------------------------------------
# Microsoft OIDC configuration
# ---------------------------------------------------------------------------
AZURE_CLIENT_ID = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET", "")
AZURE_TENANT_ID = os.getenv("AZURE_TENANT_ID", "")
AZURE_AUTHORITY = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/v2.0"

# Session TTL for OIDC (30 minutes)
OIDC_SESSION_TTL = 1800

# Signed cookie sessions: same secret across instances so sessions work after restart and with multiple app instances
_SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip() or secrets.token_urlsafe(32)
_session_serializer = URLSafeTimedSerializer(_SESSION_SECRET, salt="netassist-session")

# ---------------------------------------------------------------------------
# Role-based access control
# ---------------------------------------------------------------------------

# Maps role -> set of allowed MCP tool names.  None = all tools allowed.
# "guest" = no tools (used for unknown/invalid users; do not default to admin).
ROLE_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "admin": None,
    "netadmin": {
        "query_network_path",
        "check_path_allowed",
        "query_panorama_ip_object_group",
        "query_panorama_address_group_members",
    },
    "guest": set(),  # least privilege: no tool access
}

# Maps role -> list of sidebar category slugs shown in the UI.  None = all.
ROLE_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin": None,
    "netadmin": ["atlas", "panorama"],
    "guest": [],
}

# ---------------------------------------------------------------------------
# OIDC role resolution fallbacks (when Azure app roles aren't in the token)
# ---------------------------------------------------------------------------

# Azure security group -> role mapping (scalable):
#    OIDC_GROUP_ROLE_MAP=<group-object-id>:netadmin,<group-object-id>:admin
#    Requires: App Registration > Token configuration > Add groups claim
OIDC_GROUP_ROLE_MAP: dict[str, str] = {}
_oidc_group_map_env = os.getenv("OIDC_GROUP_ROLE_MAP", "")
for _part in _oidc_group_map_env.strip().split(","):
    _part = _part.strip()
    if ":" in _part:
        _gid, _role = _part.rsplit(":", 1)
        OIDC_GROUP_ROLE_MAP[_gid.strip().lower()] = _role.strip().lower()


def get_user_role(username: str) -> str:
    """Return the role for *username*. With signed-cookie sessions there is no server-side store; use get_role_for_session(session_id) when you have the session. This returns 'guest' for any username."""
    return "guest"


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
                   auth_mode: str = "oidc",
                   tokens: dict | None = None) -> str:
    """Create a session; returns signed cookie value (no server-side store). tokens are not stored (signed cookie size limit)."""
    payload = {
        "username": username,
        "role": role,
        "auth_mode": auth_mode,
        "created_at": time.time(),
    }
    return _session_serializer.dumps(payload)


def get_session(session_id: Optional[str]) -> Optional[dict]:
    """Return session dict from signed cookie value, or None if invalid/expired."""
    if not session_id:
        return None
    try:
        payload = _session_serializer.loads(session_id, max_age=OIDC_SESSION_TTL)
        if isinstance(payload, dict):
            return payload
    except (BadSignature, Exception):
        pass
    return None


def get_username_for_session(session_id: Optional[str]) -> Optional[str]:
    """Return username for session_id, or None if invalid."""
    sess = get_session(session_id)
    if sess is None:
        return None
    return sess.get("username")


def get_role_for_session(session_id: Optional[str]) -> str:
    """Return role for session_id. Invalid/missing session returns 'guest' (no access)."""
    sess = get_session(session_id)
    if sess is None:
        return "guest"
    return sess.get("role", "guest")


def destroy_session(session_id: Optional[str]) -> None:
    """No-op: sessions are in the cookie; logout clears the cookie in the response."""
    pass


# ---------------------------------------------------------------------------
# OIDC helpers
# ---------------------------------------------------------------------------

def extract_role_from_token(token_claims: dict) -> Optional[str]:
    """Extract the Atlas role from Azure token claims.

    Priority order:
    1. Azure 'roles' claim (app roles assigned via Enterprise Application)
    2. OIDC_GROUP_ROLE_MAP (Azure security group -> role)

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
