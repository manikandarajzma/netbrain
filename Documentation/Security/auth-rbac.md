# Atlas Authentication & Access Control

## Overview

Atlas uses Microsoft Entra ID (OIDC) for user authentication. Access is controlled by Entra ID group membership — no local passwords, no PIM, no Azure app roles.

When a user signs in, their group memberships are read from the OIDC `id_token` and mapped to a configured access level. The resolved group is baked into a signed session cookie and checked on every tool call.

**All backend credentials** (NetBrain, Panorama, Splunk) are stored in Azure Key Vault — not in the session or in user-facing tokens.

---

## How Sign-In Works (app.py and auth.py)

The sign-in flow is handled by two routes in `app.py` and helper functions in `auth.py`.

**Step 1 — Redirect to Microsoft**

`GET /auth/microsoft` redirects the browser to Microsoft's authorization endpoint via authlib:

```python
# app.py
@app.get("/auth/microsoft")
async def auth_microsoft(request: Request):
    redirect_uri = str(request.url_for("auth_callback")).replace("://127.0.0.1", "://localhost")
    return await oauth.microsoft.authorize_redirect(request, redirect_uri, prompt="select_account")
```

authlib builds and redirects to `https://login.microsoftonline.com/{tenant_id}/v2.0/authorize` with these query parameters:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `client_id` | Atlas App Registration client ID | Tells Azure which application is requesting login |
| `redirect_uri` | `https://<host>/auth/callback` | Where Azure redirects after the user authenticates — must match exactly what's registered in the Azure App Registration |
| `response_type` | `code` | Authorization Code flow — Azure returns a short-lived code, not tokens directly. Tokens are only exchanged server-side, never exposed to the browser |
| `scope` | `openid profile email offline_access` | `openid` triggers OIDC and makes Azure return an `id_token`; `profile` adds `name`/`preferred_username`; `email` adds email address; `offline_access` adds a `refresh_token` |
| `state` | random value (stored in session) | CSRF protection — Azure returns it unchanged and authlib verifies it matches, so an attacker cannot forge a callback |
| `nonce` | random value (stored in session) | Replay protection — embedded in the `id_token`, authlib verifies it to ensure the token was issued for this specific request |
| `prompt` | `select_account` | Forces the account picker even if the user is already signed in to Microsoft |

**Azure then handles all authentication itself** — the user enters their corporate credentials, MFA, etc. on Microsoft's login page. Atlas never sees the password or MFA interaction. Once authenticated, Azure redirects to `/auth/callback?code=XXXX&state=YYYY` with a short-lived (≈10 min), one-time authorization code.

**Step 2 — Exchange code for tokens and decode the id_token**

Atlas calls `oauth.microsoft.authorize_access_token(request)`, which makes a **server-to-server POST** (not browser) to `https://login.microsoftonline.com/{tenant_id}/v2.0/token`:

| Sent to Azure | Value |
|---------------|-------|
| `grant_type` | `authorization_code` |
| `code` | the one-time code from the callback query parameter |
| `redirect_uri` | must match Step 1 exactly |
| `client_id` + `client_secret` | Atlas authenticates itself to Azure — the secret is server-side only, never exposed to the browser |

Azure responds with:

| Field | What it is |
|-------|-----------|
| `id_token` | JWT containing the user's identity claims: `sub`, `name`, `preferred_username`, `email`, `groups`, `iss`, `aud`, `exp`, `iat`, `nonce` |
| `access_token` | JWT for calling Microsoft Graph API — Atlas does not use this |
| `refresh_token` | For getting new tokens without re-login — Atlas does not use this; session TTL (30 min) handles expiry |
| `token_type` | `Bearer` |
| `expires_in` | Seconds until the access_token expires |

> **What is a JWT?** A JSON Web Token is a string made of three base64url-encoded segments separated by dots: `header.payload.signature`. The **header** says which signing algorithm was used. The **payload** is a plain JSON object containing claims (key-value pairs like `"email": "user@corp.com"`, `"groups": [...]`). The **signature** is a cryptographic hash of the header + payload, signed with Azure's private key. Anyone can base64-decode and read the payload — it is not encrypted. But without Azure's private key, no one can forge a valid signature. This is what authlib verifies: it fetches Azure's public keys from `/.well-known/openid-configuration` and checks that the signature on the `id_token` is valid — proving the token genuinely came from Azure and has not been tampered with.

authlib verifies the `id_token` signature against Azure's public keys (fetched from `/.well-known/openid-configuration`). Atlas then **manually decodes the JWT payload** (the middle segment) to read the claims:

```python
# app.py
token = await oauth.microsoft.authorize_access_token(request)  # authlib verifies signature

id_token = token.get("id_token")
payload = id_token.split(".")[1]
payload += "=" * (4 - len(payload) % 4)   # restore base64url padding
userinfo = json.loads(base64.urlsafe_b64decode(payload))
# userinfo["groups"] contains Entra ID group Object IDs, e.g.:
# ["d1cc53dd-b663-48d3-a711-214ba91188c7", "47fc0125-81f8-4244-956c-4897200598cf"]
```

> **Why decode the JWT manually instead of calling `/userinfo`?** Microsoft's `/userinfo` endpoint only returns standard OIDC claims. The `groups` claim — which contains the user's Entra ID group memberships — is a Microsoft extension that only appears inside the `id_token` JWT itself. It is never returned by `/userinfo`. So Atlas must decode the JWT payload directly to get it.

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

Group names and their permitted tools are configured in `GROUP_ALLOWED_TOOLS` and `GROUP_ALLOWED_CATEGORIES` in `auth.py`. The structure is:

```python
# auth.py — group names are configurable; add, rename, or remove groups here
GROUP_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "full_access_group": None,          # None = all tools allowed
    "limited_access_group": {           # explicit allowlist of tool names
        "tool_a",
        "tool_b",
    },
    "no_access_group": set(),           # empty set = no tool access (least privilege)
}

GROUP_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "full_access_group": None,          # None = all sidebar categories
    "limited_access_group": ["cat_a", "cat_b"],
    "no_access_group": [],
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

The group is read from the **server-signed cookie**, not from the prompt. A user cannot escalate privileges through prompt injection — the RBAC check runs in Python code, independently of the LLM.

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

**Cloud-only groups** (default): The `groups` claim contains Object IDs (GUIDs). Set the `ATLAS_*_GROUP_ID` env vars to the Object IDs from Entra ID → Groups → Overview. Each env var maps a GUID to a group name defined in `GROUP_ALLOWED_TOOLS`.

**On-prem synced groups**: Configure the Entra ID token to emit `sAMAccountName` in the `groups` claim. The group name is matched directly against the keys in `GROUP_ALLOWED_TOOLS` — no env vars needed.

Both modes can coexist: Object ID lookup runs first, then falls back to sAMAccountName matching.
