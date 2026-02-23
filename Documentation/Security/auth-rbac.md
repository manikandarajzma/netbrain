# Atlas Authentication & RBAC

This guide covers how to set up Microsoft Entra ID (OIDC) authentication with Privileged Identity Management (PIM) and role-based access control (RBAC) for Atlas.

**No local passwords.** Authentication is OIDC only. All credentials (Azure app registration, NetBrain API, Panorama, Splunk, etc.) should be sourced from **Azure Key Vault** (or equivalent) and injected into the environment; do not store secrets in `.env` in production.

---

## "Sign-in is not configured" — Quick fix

1. Create a `.env` file (copy from `.env.example`) in the **project root** or in the **atlas/** directory.
2. Set these (get values from Azure App Registration or Key Vault):
   - `AUTH_MODE=oidc`
   - `AZURE_CLIENT_ID=<your-app-client-id>`
   - `AZURE_TENANT_ID=<your-tenant-id>`
   - `AZURE_CLIENT_SECRET=<your-client-secret>`
3. Restart the app. The **Sign in with Microsoft** button should appear.

The app loads `.env` from `atlas/`, then the project root, then the current working directory.

---

## 10-Step OIDC + PIM Setup

### Step 1: Register the Application in Azure

1. Go to **Microsoft Entra ID > App registrations > New registration**
2. Name: `Atlas`
3. Supported account types: **Single tenant**
4. Redirect URI: **Web** — `http://localhost:8000/auth/callback`
5. Click **Register**
6. Copy the **Application (client) ID** and **Directory (tenant) ID**

### Step 2: Create a Client Secret

1. In the App Registration, go to **Certificates & secrets > Client secrets > New client secret**
2. Add a description (e.g., `netassist-secret`) and set an expiry
3. Copy the secret **Value** immediately (it won't be shown again)
4. Store the client secret in **Azure Key Vault** and inject into the app (e.g. via env or Key Vault references). For local dev you can use `.env`:

```env
AZURE_CLIENT_ID=<application-client-id>
AZURE_CLIENT_SECRET=<client-secret-value>
AZURE_TENANT_ID=<directory-tenant-id>
AUTH_MODE=oidc
```

### Step 3: Configure API Permissions

1. In the App Registration, go to **API permissions > Add a permission**
2. Select **Microsoft Graph > Delegated permissions**
3. Add: `openid`, `profile`, `email`, `offline_access`
4. Click **Grant admin consent** for the tenant

### Step 4: Define App Roles

App roles map directly to Atlas roles (`admin`, `netadmin`). When assigned, they appear in the token's `roles` claim.

1. In the App Registration, go to **App roles > Create app role**
2. Create the `admin` role:
   - Display name: `Admin`
   - Allowed member types: **Users/Groups**
   - Value: `admin` (must match exactly — this is what appears in the token)
   - Description: `Full access to all tools`
3. Create the `netadmin` role:
   - Display name: `NetAdmin`
   - Allowed member types: **Users/Groups**
   - Value: `netadmin`
   - Description: `Access to Atlas path and Panorama tools only`

### Step 5: Create Azure Security Groups for PIM

Create one security group per role. These groups will be PIM-enabled so users must activate membership before getting the role.

1. Go to **Microsoft Entra ID > Groups > New group**
2. Create group:
   - Group type: **Security**
   - Name: `Atlas-Admins`
   - Membership type: **Assigned**
   - Check **Microsoft Entra roles can be assigned to the group** (required for PIM)
3. Repeat for `Atlas-NetAdmins`

### Step 6: Assign App Roles to the Security Groups

1. Go to **Enterprise Applications > Atlas > Users and groups > Add user/group**
2. Select the **Atlas-Admins** group and assign the **Admin** role
3. Select the **Atlas-NetAdmins** group and assign the **NetAdmin** role

Now any member of `Atlas-Admins` will receive `"roles": ["admin"]` in their token, and members of `Atlas-NetAdmins` will receive `"roles": ["netadmin"]`.

### Step 7: Enable PIM for the Groups

1. Go to **Microsoft Entra ID > Privileged Identity Management > Groups**
2. Select **Atlas-NetAdmins** > **Assignments > Add assignments**
3. Role: **Member**
4. Select the users who should be eligible
5. Assignment type: **Eligible** (not Active)
6. Set duration (e.g., 8 hours max activation, 365 days eligibility)
7. Repeat for `Atlas-Admins` if you want PIM-gated admin access

With Eligible assignments, users must explicitly activate their membership before the role appears in the token.

### Step 8: User Activates PIM Membership (Per-Session)

When a user needs access, they activate their PIM membership:

1. Go to **Microsoft Entra ID > Privileged Identity Management > My roles**
2. Switch to the **Groups** tab (not Microsoft Entra roles)
3. Find the eligible group (e.g., `Atlas-NetAdmins`)
4. Click **Activate**
5. Provide justification and duration
6. After approval/activation, sign in to Atlas — the role will now be in the token

**Important:** The self-service activation view is under **My roles > Groups**. The admin management view (which shows Remove/Update/Extend) is a different page.

### Step 9: Configure OIDC

Provide these in the environment (from **Azure Key Vault** in production, or in `.env` for local development):

```env
AUTH_MODE=oidc
AZURE_CLIENT_ID=<your-client-id>
AZURE_CLIENT_SECRET=<your-client-secret>
AZURE_TENANT_ID=<your-tenant-id>
```

If any of these are missing, the login page will show “Sign-in is not configured” until they are set.

#### Optional: Fallback Role Resolution

If you cannot use App Roles (Step 4-6), you can use Azure security group to role mapping:

```env
# Get group Object IDs from Entra ID > Groups
OIDC_GROUP_ROLE_MAP=<group-object-id>:admin,<group-object-id>:netadmin
```
This requires adding a **groups claim** to the token: App Registration > Token configuration > Add groups claim > Security groups.

### Step 10: Start the Server

```bash
cd atlas
python app.py
```

Open `http://localhost:8000`. The login page will show a **Sign in with Microsoft** button. Users without a resolved role will see: _"Your account does not have an assigned role. Contact your administrator."_

---

## How RBAC Works

### Roles

Three roles are defined in `auth.py`:

| Role       | Tools Allowed                          | Sidebar Categories         |
|------------|----------------------------------------|----------------------------|
| `admin`    | All tools                              | All categories             |
| `netadmin` | Atlas (path, allow) + Panorama only | Atlas, Panorama         |
| `guest`    | None (no access)                       | None                       |

The `guest` role is used internally for unknown usernames or invalid/expired sessions; users with this role have no tool access and see no sidebar categories.

### Role Definition (auth.py)

```python
# Maps role -> set of allowed MCP tool names.  None = all tools allowed.
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

# Maps role -> list of sidebar categories shown in the UI.  None = all.
ROLE_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin": None,
    "netadmin": ["atlas", "panorama"],
    "guest": [],
}
```

### Role Resolution from OIDC Tokens (auth.py)

When a user signs in via Microsoft, the role is extracted from the token in priority order:

```python
def extract_role_from_token(token_claims: dict) -> Optional[str]:
    # 1. Azure 'roles' claim (app roles via Enterprise Application)
    roles = token_claims.get("roles", [])
    for r in roles:
        r_lower = r.lower().strip()
        if r_lower in ROLE_ALLOWED_TOOLS:
            return r_lower

    # 2. OIDC_GROUP_ROLE_MAP (Azure security group -> role)
    if OIDC_GROUP_ROLE_MAP:
        groups = token_claims.get("groups", [])
        for gid in groups:
            role = OIDC_GROUP_ROLE_MAP.get(str(gid).lower())
            if role and role in ROLE_ALLOWED_TOOLS:
                return role

    # No role found — user denied access
    return None
```

If no role is resolved, the user is redirected to the login page with an error.

### Session Creation (app.py)

After OIDC callback, the role is stored in the session:

```python
username = extract_username_from_token(userinfo)
role = extract_role_from_token(userinfo)
if role is None:
    return RedirectResponse(url="/login?error=norole", status_code=302)

session_id = create_session(username, role=role, auth_mode="oidc", tokens={...})
```

OIDC sessions expire after 30 minutes. Local sessions last 7 days.

### Server-Side Tool Enforcement (chat_service.py)

Every tool execution is checked against the user's role before running:

```python
def _check_tool_access(username: str | None, tool_name: str) -> str | None:
    """Return an error message if the user's role forbids tool_name, else None."""
    if username is None:
        return None
    from atlas.auth import get_user_role, get_allowed_tools
    role = get_user_role(username)
    allowed = get_allowed_tools(role)
    if allowed is not None and tool_name not in allowed:
        display = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
        return f"Your role ({role}) does not have access to {display} queries."
    return None
```

This check runs in `process_message()` both for pre-filled tool calls and for the agent discovery loop. If denied, the user sees: _"Your role (netadmin) does not have access to Splunk queries."_

### UI-Side Category Filtering (index.html)

The sidebar only renders categories the user's role permits, using Jinja2 conditionals:

```html
{% if allowed_categories is none or "atlas" in allowed_categories %}
<div class="example-category" data-category="atlas">
    <div class="example-category-label">Atlas</div>
    ...
</div>
{% endif %}

```

`admin` users see all categories (`allowed_categories = None`). `netadmin` users only see Atlas and Panorama.

### User Display

The header shows the logged-in user's email and role:

```html
<span class="user-info">{{ username }} <span class="user-role">{{ role }}</span></span>
```

---

## Environment Variable Reference

| Variable              | Required | Description                                                                 |
|-----------------------|----------|-----------------------------------------------------------------------------|
| `AUTH_MODE`           | Yes      | `oidc` (only supported mode; default)                                      |
| `AZURE_CLIENT_ID`     | OIDC     | App Registration client ID (from Key Vault in production)                  |
| `AZURE_CLIENT_SECRET` | OIDC     | App Registration client secret (from Key Vault in production)               |
| `AZURE_TENANT_ID`     | OIDC     | Azure tenant ID                                                             |
| `OIDC_GROUP_ROLE_MAP` | No       | Group ID to role: `<object-id>:admin,<object-id>:netadmin`                  |
| `OAUTH_STATE_SECRET`  | No       | Fixed secret for OAuth state cookie. Set when running multiple app instances so they share the same secret; otherwise a random per-process secret is used. |
| `SESSION_SECRET`      | No       | Secret used to sign session cookies. Set to a fixed value so sessions survive restarts and work across multiple app instances; otherwise a random per-process secret is used (sessions then only valid for that process). |

Sessions are stored in **signed cookies** (no server-side store), so they survive app restarts and work across multiple instances when `SESSION_SECRET` is set.

Backend credentials (NetBrain, Panorama, Splunk, etc.) are also typically sourced from Azure Key Vault. The same Azure identity (e.g. `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`) is used for Key Vault access.
