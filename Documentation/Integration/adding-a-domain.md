# Adding a New Domain / Integration

This guide covers connecting a brand new external domain or system to Atlas — for example, adding ServiceNow, Cisco DNA Center, or any other API that doesn't have an existing tool file.

Once the integration is wired up, follow [adding-a-tool.md](./adding-a-tool.md) to add individual tools within it.

---

## Checklist

- [ ] 1. Add credentials to `tools/shared.py`
- [ ] 2. Create the service credential module (`newdomainauth.py`)
- [ ] 3. Create the tool module (`tools/newdomain_tools.py`)
- [ ] 4. Register the module in `mcp_server.py`
- [ ] 5. Add `.env` variables
- [ ] 6. Follow [adding-a-tool.md](./adding-a-tool.md) for each tool

---

## File layout of existing integrations

Each integration follows the same three-file pattern:

```
atlas/
├── netbrainauth.py          ← authentication (token/session management)
├── servicenowauth.py        ← authentication (credential helper example)
├── tools/
│   ├── shared.py            ← credentials loaded here at startup
│   ├── netbrain_tools.py    ← @mcp.tool() definitions
│   ├── servicenow_tools.py
│   └── newdomain_tools.py
└── mcp_server.py            ← imports tool modules to trigger registration
```

`tools/shared.py` is the single place where credentials are loaded (from `.env` or Azure Key Vault) and exposed as module-level constants. Tool modules import these constants — they never load credentials themselves.

---

## Step 1: Add credentials to tools/shared.py

**File:** [tools/shared.py](../../tools/shared.py)

Add a credential-loading block at the bottom of the configuration constants section. Follow the existing pattern: try `.env` first, then fall back to Azure Key Vault.

```python
# tools/shared.py

# ServiceNow
SERVICENOW_URL      = os.getenv("SERVICENOW_URL", "").rstrip("/")
SERVICENOW_USER     = os.getenv("SERVICENOW_USER", "")
SERVICENOW_PASSWORD = os.getenv("SERVICENOW_PASSWORD", "")
if not SERVICENOW_USER or not SERVICENOW_PASSWORD:
    _vault_url = os.getenv("AZURE_KEYVAULT_URL", "").strip().rstrip("/")
    if _vault_url:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.secrets import SecretClient
            _credential = DefaultAzureCredential()
            _client = SecretClient(vault_url=_vault_url, credential=_credential)
            if not SERVICENOW_USER:
                _s = _client.get_secret(os.getenv("SERVICENOW_USER_KEYVAULT_SECRET_NAME", "SERVICENOW-USER"))
                if _s and _s.value:
                    SERVICENOW_USER = _s.value
            if not SERVICENOW_PASSWORD:
                _s = _client.get_secret(os.getenv("SERVICENOW_PASSWORD_KEYVAULT_SECRET_NAME", "SERVICENOW-PASSWORD"))
                if _s and _s.value:
                    SERVICENOW_PASSWORD = _s.value
        except Exception as e:
            logger.warning("Key Vault: failed to load ServiceNow credentials: %s", e)
```

**Why here:** All credential loading is centralised in `shared.py` so tool modules stay clean, and so credentials are loaded once at MCP server startup — not on every tool call.

---

## Step 2: Create the backend service credential module

Create `newdomainauth.py` in the atlas root (alongside `netbrainauth.py` and other auth helpers). Its job is to manage session/token state and expose a single function the tool module calls to get a valid credential.

Choose the pattern that matches how the external API authenticates:

### Credential pattern A — session token with TTL (e.g., NetBrain)

The external API requires a login call that returns a short-lived session token. Cache the token in a module-level variable and re-fetch it when it expires.

See [`netbrainauth.py`](../../netbrainauth.py) for a complete working example. Key elements: a module-level `_token` variable, a `TOKEN_TTL_SECONDS` constant, a `get_token()` function that returns the cached token or fetches a new one, and a `clear_token_cache()` function used when a tool receives an unexpected 401 response.

### Credential pattern B — API key cached indefinitely

The external API uses an API key that does not expire. Fetch it once (from Key Vault or environment) and cache it for the process lifetime.

Use the same module-level caching pattern as the other auth helpers in the repo.

### Credential pattern C — credentials passed directly

Some APIs accept credentials on every request — no session management needed. Skip the credential module entirely and import the username/password constants directly from `tools/shared.py` in the tool module.

---

## Step 3: Create the tool module

Create `tools/newdomain_tools.py`. The module must:

1. Import `mcp` from `tools.shared` — this is the shared FastMCP instance all tools register on.
2. Import credentials from `tools.shared`.
3. Optionally import the auth module.
4. Define `@mcp.tool()` functions following the pattern in [adding-a-tool.md](./adding-a-tool.md).

```python
# tools/servicenow_tools.py
"""
ServiceNow MCP tools.

Exposes:
  - get_incident        – look up an incident by number
  - list_open_incidents – list open incidents for a CI
"""
import asyncio
import aiohttp
import ssl
from typing import Optional, Dict, Any

from tools.shared import mcp, SERVICENOW_URL, SERVICENOW_USER, SERVICENOW_PASSWORD, setup_logging

logger = setup_logging(__name__)
import servicenowauth


# ---------------------------------------------------------------------------
# Implementation functions
# ---------------------------------------------------------------------------

async def _get_incident_impl(incident_number: str) -> Dict[str, Any]:
    token = servicenowauth.get_token()
    if not token:
        return {"error": "Failed to authenticate with ServiceNow"}

    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE

    url = f"{SERVICENOW_URL}/api/now/table/incident"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {"sysparm_query": f"number={incident_number}", "sysparm_limit": 1}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params,
                                   ssl=ssl_ctx, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return {"error": f"ServiceNow API returned {resp.status}"}
                data = await resp.json()
    except asyncio.TimeoutError:
        return {"incident_number": incident_number, "error": "Request timed out"}
    except Exception as e:
        logger.debug("get_incident error: %s", e)
        return {"incident_number": incident_number, "error": str(e)}

    records = data.get("result", [])
    if not records:
        return {"incident_number": incident_number, "error": f"No incident found: {incident_number}"}

    r = records[0]
    return {
        "incident_number": incident_number,
        "short_description": r.get("short_description"),
        "state": r.get("state", {}).get("display_value"),
        "priority": r.get("priority", {}).get("display_value"),
        "assigned_to": r.get("assigned_to", {}).get("display_value"),
        "opened_at": r.get("opened_at"),
    }


# ---------------------------------------------------------------------------
# MCP tool registrations
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_incident(incident_number: str) -> Dict[str, Any]:
    """
    Look up a ServiceNow incident by incident number.

    Use for: queries asking for incident details — "look up INC0012345",
    "what is incident INC0012345?", "details for INC0012345".
    Do NOT use for: path queries, address group lookups, rack queries.

    Examples:
    - "look up INC0012345" → incident_number="INC0012345"
    - "what is INC0012345?" → incident_number="INC0012345"

    Args:
        incident_number: ServiceNow incident number (e.g., "INC0012345")

    Returns:
        dict: incident_number, short_description, state, priority, assigned_to, opened_at, or error
    """
    return await _get_incident_impl(incident_number)
```

---

## Step 4: Register in mcp_server.py

**File:** [mcp_server.py](../../mcp_server.py)

Add one import line. The act of importing the module triggers all `@mcp.tool()` decorators and registers the tools on the shared FastMCP instance.

```python
# mcp_server.py

# Import domain modules – the act of importing triggers @mcp.tool() registration
import tools.servicenow_tools  # noqa: F401
import tools.netbrain_tools    # noqa: F401
import tools.netbrain_tools    # noqa: F401
import tools.servicenow_tools  # noqa: F401   ← add this line
```

No other changes are needed in `mcp_server.py`. The shared `mcp` instance accumulates all registered tools automatically.

---

## Step 5: Add `.env` variables

Add the new variables to your `.env` file:

```bash
# ServiceNow
SERVICENOW_URL=https://yourinstance.service-now.com
SERVICENOW_USER=atlas-api-user          # or leave blank to use Key Vault
SERVICENOW_PASSWORD=                    # or leave blank to use Key Vault

# Key Vault secret names (if using Azure Key Vault)
SERVICENOW_USER_KEYVAULT_SECRET_NAME=SERVICENOW-USER
SERVICENOW_PASSWORD_KEYVAULT_SECRET_NAME=SERVICENOW-PASSWORD
```

For production, leave `SERVICENOW_USER` and `SERVICENOW_PASSWORD` blank and store them exclusively in Azure Key Vault. `tools/shared.py` will load them automatically if `AZURE_KEYVAULT_URL` is set.

---

## Step 6: Add individual tools

Follow [adding-a-tool.md](./adding-a-tool.md) for each tool, starting at Step 3 (`TOOL_DISPLAY_NAMES`). Steps 1 and 2 (impl + wrapper) are already done above.

---

## HTTPS TLS settings for API calls

Most internal systems use self-signed certificates. Bypass TLS verification with:

```python
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE
```

> **Production note:** Remove the SSL bypass if the system has a valid CA-signed certificate, or pin the CA certificate explicitly.

---

## File layout summary

```
atlas/
├── servicenowauth.py            ← new: token/session management
├── tools/
│   ├── shared.py                ← modified: add credential loading block
│   └── servicenow_tools.py      ← new: @mcp.tool() definitions
└── mcp_server.py                ← modified: add one import line
```

After all steps are complete, restart the MCP server and verify the new tools appear in `GET http://127.0.0.1:8765/health` → `tools_registered`.
