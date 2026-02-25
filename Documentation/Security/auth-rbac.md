# Atlas Authentication & Access Control

## Overview

Atlas uses Microsoft Entra ID (OIDC) for user authentication. Access is controlled by Entra ID group membership — no local passwords, no PIM, no Azure app roles.

When a user signs in, their group memberships are read from the OIDC `id_token` and mapped to one of two access levels (`admin` or `netadmin`). The resolved group is baked into a signed session cookie and checked on every tool call.

**All backend credentials** (NetBrain, Panorama, Splunk) are stored in Azure Key Vault — not in the session or in user-facing tokens.

---

## How Sign-In Works (app.py and auth.py)

The sign-in flow is handled by two routes in `app.py` and helper functions in `auth.py`.

**Step 1 — Redirect to Microsoft**

`GET /auth/microsoft` redirects the browser to Microsoft's login page via authlib:

```python
# app.py
@app.get("/auth/microsoft")
async def auth_microsoft(request: Request):
    redirect_uri = str(request.url_for("auth_callback")).replace("://127.0.0.1", "://localhost")
    return await oauth.microsoft.authorize_redirect(request, redirect_uri, prompt="select_account")
```

**Step 2 — Exchange code for tokens and decode the id_token**

After login, Microsoft redirects to `GET /auth/callback` with an authorization code. `auth_callback()` in `app.py` exchanges it for tokens, then manually decodes the JWT payload to extract the `groups` claim. The `groups` claim is only in the `id_token` — Microsoft does not include group memberships in the `/userinfo` endpoint:

```python
# app.py
@app.get("/auth/callback")
async def auth_callback(request: Request):
    token = await oauth.microsoft.authorize_access_token(request)  # authlib verifies signature

    # Decode id_token JWT to get the groups claim
    id_token = token.get("id_token")
    payload = id_token.split(".")[1]
    payload += "=" * (4 - len(payload) % 4)   # restore base64url padding
    userinfo = json.loads(base64.urlsafe_b64decode(payload))
    # userinfo["groups"] contains Entra ID group Object IDs, e.g.:
    # ["d1cc53dd-b663-48d3-a711-214ba91188c7", "47fc0125-81f8-4244-956c-4897200598cf"]
```

**Step 3 — Resolve the group from token claims**

`extract_group_from_token()` in `auth.py` maps the group Object IDs to an access level using the env vars `ATLAS_ADMIN_GROUP_ID` and `ATLAS_NETADMIN_GROUP_ID`:

```python
# auth.py
_GROUP_ID_MAP = {}
if os.getenv("ATLAS_ADMIN_GROUP_ID"):    _GROUP_ID_MAP[os.getenv("ATLAS_ADMIN_GROUP_ID").lower()]    = "admin"
if os.getenv("ATLAS_NETADMIN_GROUP_ID"): _GROUP_ID_MAP[os.getenv("ATLAS_NETADMIN_GROUP_ID").lower()] = "netadmin"

def extract_group_from_token(token_claims: dict) -> str | None:
    for group in token_claims.get("groups", []):
        g_lower = str(group).lower().strip()
        # Cloud-only groups: match by Object ID
        if _GROUP_ID_MAP:
            name = _GROUP_ID_MAP.get(g_lower)
            if name:
                return name
        # On-prem synced groups: match by sAMAccountName (e.g. "admin", "netadmin")
        if g_lower in GROUP_ALLOWED_TOOLS:
            return g_lower
    return None  # user is not in any recognised group
```

Back in `app.py`, if no group resolves, the login is rejected:

```python
# app.py
group = extract_group_from_token(userinfo)
if group is None:
    return RedirectResponse(url="/login?error=norole", status_code=302)
```

**Step 4 — Create signed session cookie**

The resolved group is baked into a signed cookie by `create_session()` in `auth.py` using `itsdangerous.URLSafeTimedSerializer`. There is no server-side session store — the cookie itself is the session:

```python
# auth.py
_session_serializer = URLSafeTimedSerializer(SESSION_SECRET, salt="netassist-session")

def create_session(username, *, group="admin", auth_mode="oidc", tokens=None) -> str:
    payload = {"username": username, "group": group, "auth_mode": auth_mode, "created_at": time.time()}
    return _session_serializer.dumps(payload)
```

`app.py` sets the cookie with `HttpOnly=True` (blocks JavaScript access) and `SameSite=lax` (blocks CSRF):

```python
# app.py
session_id = create_session(username, group=group, auth_mode="oidc")
r = RedirectResponse(url="/", status_code=302)
r.set_cookie(key="atlas_session", value=session_id, max_age=1800, httponly=True, samesite="lax")
```

---

## How Sessions Work (auth.py)

Sessions are stateless signed cookies — no database, no Redis. Every request verifies the cookie locally:

```python
# auth.py
def get_session(session_id) -> dict | None:
    try:
        # Verifies HMAC signature and enforces 30-minute TTL
        payload = _session_serializer.loads(session_id, max_age=1800)
        if isinstance(payload, dict):
            return payload
    except (BadSignature, Exception):
        pass
    return None  # tampered, expired, or missing

def get_group_for_session(session_id) -> str:
    sess = get_session(session_id)
    if sess is None:
        return "guest"   # no access
    return sess.get("group", "guest")
```

`app.py` reads the cookie on every request:

```python
# app.py
def get_current_username(request: Request) -> str | None:
    sid = request.cookies.get("atlas_session")
    return get_username_for_session(sid)  # calls auth.get_session() internally
```

If the cookie is missing, tampered, or older than 30 minutes → 401 + redirect to `/login`. Because sessions survive app restarts and work across multiple instances, `SESSION_SECRET` must be set to the same value on all instances.

---

## How Tool Access is Enforced (auth.py and chat_service.py)

Group membership is enforced twice: at sign-in (group is locked into the signed cookie) and on every tool call.

**Access levels defined in auth.py:**

```python
# auth.py
GROUP_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "admin": None,   # None = all tools allowed
    "netadmin": {
        "query_network_path",
        "check_path_allowed",
        "query_panorama_ip_object_group",
        "query_panorama_address_group_members",
        "find_unused_panorama_objects",
        "search_documentation",
    },
    "guest": set(),  # no tools
}

GROUP_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin": None,                              # all sidebar categories
    "netadmin": ["netbrain", "panorama", "docs"],
    "guest": [],
}
```

**RBAC check in chat_service.py — runs before every tool call:**

```python
# chat_service.py
def _check_tool_access(username, tool_name, session_id=None) -> str | None:
    # Group is read from the signed session cookie, NOT from user input.
    # This means prompt injection cannot bypass the check.
    group = get_group_for_session(session_id) if session_id else get_user_group(username)
    allowed = get_allowed_tools(group)   # calls auth.GROUP_ALLOWED_TOOLS
    if allowed is not None and tool_name not in allowed:
        display = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        return f"Your group ({group}) does not have access to {display} queries."
    return None  # access granted

# Called before executing any tool:
access_err = _check_tool_access(username, sel_tool_name, session_id)
if access_err:
    return {"role": "assistant", "content": access_err}
```

The group is read from the **server-signed cookie**, not from the prompt. A user cannot escalate by saying "pretend you are admin" — the RBAC check runs in Python code, independently of the LLM.

---

## Environment Variables

| Variable                  | Required | Description                                                          |
|---------------------------|----------|----------------------------------------------------------------------|
| `AZURE_TENANT_ID`         | Yes      | Azure tenant ID                                                      |
| `AZURE_CLIENT_ID`         | Yes      | App Registration client ID                                           |
| `AZURE_CLIENT_SECRET`     | Yes      | App Registration client secret                                       |
| `ATLAS_ADMIN_GROUP_ID`    | Yes      | Object ID of the full-access Entra ID group                         |
| `ATLAS_NETADMIN_GROUP_ID` | Yes      | Object ID of the limited-access Entra ID group                      |
| `SESSION_SECRET`          | No       | Fixed signing key for session cookies — required for multi-instance deployments |
| `OAUTH_STATE_SECRET`      | No       | Fixed signing key for the OAuth state cookie — required for multi-instance deployments |

---

## On-Prem Synced vs Cloud-Only Groups

**Cloud-only groups** (default): The `groups` claim contains Object IDs (GUIDs). Set `ATLAS_ADMIN_GROUP_ID` and `ATLAS_NETADMIN_GROUP_ID` to the Object IDs from Entra ID → Groups → Overview.

**On-prem synced groups**: Configure the Entra ID token to emit `sAMAccountName` in the `groups` claim. The group name (`"admin"`, `"netadmin"`) is matched directly — no env vars needed.

Both modes can coexist: Object ID lookup runs first, then falls back to sAMAccountName matching.
