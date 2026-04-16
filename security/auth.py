"""
Authentication for Atlas (FastAPI).

Identity provider: Microsoft Entra ID via OIDC. No local passwords.
All backend credentials (ServiceNow, etc.) are stored in Azure Key Vault.

Session storage: signed cookies (itsdangerous URLSafeTimedSerializer).
There is no server-side session store — the session payload is the cookie itself.
This means sessions survive app restarts and work across multiple instances without
a shared cache, as long as SESSION_SECRET is the same on all instances.

Access control: group membership from the OIDC id_token `groups` claim.
Recognised groups and their tool/category permissions are defined in
GROUP_ALLOWED_TOOLS and GROUP_ALLOWED_CATEGORIES.
Any session that does not resolve to a recognised group is treated as the
least-privilege group.
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

# SESSION_SECRET should be set in .env for production so sessions survive restarts
# and work across multiple app instances. If unset, a random secret is generated
# per process, which means all existing sessions are invalidated on every restart.
_SESSION_SECRET = os.getenv("SESSION_SECRET", "").strip() or secrets.token_urlsafe(32)
# salt scopes the serializer to this app — prevents cookies from one app being
# replayed against another app that shares the same secret.
_session_serializer = URLSafeTimedSerializer(_SESSION_SECRET, salt="atlas-session")

# ---------------------------------------------------------------------------
# Group-based access control
# ---------------------------------------------------------------------------

# Maps group -> set of allowed MCP tool names.  None = all tools allowed.
# The least-privilege group must map to an empty set (no tool access).
GROUP_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "admin": None,
    "netadmin": {
        "search_servicenow_incidents",
        "list_servicenow_change_requests",
        "get_servicenow_incident",
    },
    "guest": set(),  # least privilege: no tool access
}

# Maps group -> list of sidebar category slugs shown in the UI.  None = all.
GROUP_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin": None,
    "netadmin": ["troubleshoot", "servicenow"],
    "guest": [],
}


def get_user_group(username: str) -> str:
    """Fallback group lookup by username.

    Because sessions are stored in signed cookies (no server-side store), the group
    cannot be looked up from a username alone. Callers that have a session cookie value
    should use get_group_for_session(session_id) instead.
    Always returns 'guest' — the safest default when the session is unavailable.
    """
    return "guest"


def get_allowed_tools(group: str) -> set[str] | None:
    """Return the set of MCP tool names allowed for *group*, or None if all are allowed."""
    return GROUP_ALLOWED_TOOLS.get(group)


def get_allowed_categories(group: str) -> list[str] | None:
    """Return the sidebar categories visible for *group*, or None if all are visible."""
    return GROUP_ALLOWED_CATEGORIES.get(group)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def create_session(username: str, *, group: str = "admin",
                   auth_mode: str = "oidc",
                   tokens: dict | None = None) -> str:
    """Serialise a session payload into a signed cookie value.

    The returned string is stored directly as the cookie value — there is no
    server-side session entry. The payload is signed (not encrypted) so it can
    be read by the client but cannot be tampered with.

    tokens is accepted for interface compatibility but is intentionally not
    stored — OAuth tokens are too large for a cookie and are not needed after
    the group is resolved at sign-in.
    """
    payload = {
        "username": username,
        "group": group,
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


def get_group_for_session(session_id: Optional[str]) -> str:
    """Return group for session_id. Invalid/missing session returns 'guest' (no access)."""
    sess = get_session(session_id)
    if sess is None:
        return "guest"
    return sess.get("group", "guest")


def destroy_session(session_id: Optional[str]) -> None:
    """No-op: sessions are in the cookie; logout clears the cookie in the response."""
    pass


# ---------------------------------------------------------------------------
# Group Object ID mapping (for cloud-only Entra ID groups)
# ---------------------------------------------------------------------------
# On-prem synced groups: leave these unset — sAMAccountName in token matches directly.
# Cloud-only groups: set these to the group Object IDs from Entra ID > Groups > <group> > Overview.
_ADMIN_GROUP_ID = os.getenv("ATLAS_ADMIN_GROUP_ID", "").strip().lower()
_NETADMIN_GROUP_ID = os.getenv("ATLAS_NETADMIN_GROUP_ID", "").strip().lower()

# Build Object ID -> group name lookup (only populated when env vars are set)
_GROUP_ID_MAP: dict[str, str] = {}
if _ADMIN_GROUP_ID:
    _GROUP_ID_MAP[_ADMIN_GROUP_ID] = "admin"
if _NETADMIN_GROUP_ID:
    _GROUP_ID_MAP[_NETADMIN_GROUP_ID] = "netadmin"


# ---------------------------------------------------------------------------
# OIDC helpers
# ---------------------------------------------------------------------------

def extract_group_from_token(token_claims: dict) -> Optional[str]:
    """Resolve access level from the groups claim in the OIDC id_token.

    Two modes:
    - On-prem synced groups: configure token config to emit sAMAccountName.
      The groups claim will contain the group name directly.
    - Cloud-only groups: set ATLAS_ADMIN_GROUP_ID / ATLAS_NETADMIN_GROUP_ID env
      vars to the group Object IDs. The groups claim contains GUIDs which are
      mapped to group names here.

    Returns None if the user is not a member of any recognised group.
    """
    for group in token_claims.get("groups", []):
        g_lower = str(group).lower().strip()
        # Cloud-only: match by Object ID
        if _GROUP_ID_MAP:
            name = _GROUP_ID_MAP.get(g_lower)
            if name:
                return name
        # On-prem synced: match by sAMAccountName
        if g_lower in GROUP_ALLOWED_TOOLS:
            return g_lower
    return None


def extract_username_from_token(token_claims: dict) -> str:
    """Extract display username from Azure token claims."""
    return (
        token_claims.get("preferred_username")
        or token_claims.get("email")
        or token_claims.get("name")
        or token_claims.get("sub", "unknown")
    )
