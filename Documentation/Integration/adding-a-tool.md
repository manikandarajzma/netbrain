# Adding a New MCP Tool

This guide covers adding a new MCP tool to an existing integration module — for example, adding a new NetBrain query to [tools/netbrain_tools.py](../../tools/netbrain_tools.py), a new Splunk search to [tools/splunk_tools.py](../../tools/splunk_tools.py), and so on.

To add an entirely new integration (new file, new auth, new credentials), see [adding-a-domain.md](./adding-a-domain.md) first, then return here.

---

## Checklist

- [ ] 1. Write the implementation function in the integration tool file
- [ ] 2. Write the `@mcp.tool()` wrapper
- [ ] 3. Update `TOOL_DISPLAY_NAMES` in `chat_service.py`
- [ ] 4. Update `_TOOL_TIMEOUTS` in `chat_service.py`
- [ ] 5. Update `_is_obviously_in_scope()` keywords in `chat_service.py`
- [ ] 6. Update the LLM system prompt in `_build_llm_messages()` in `chat_service.py`
- [ ] 7. Update `_normalize_result()` in `chat_service.py` (if needed)
- [ ] 8. Update `ROLE_ALLOWED_TOOLS` in `auth.py`
- [ ] 9. Update the frontend (if the response shape is new)

---

## Step 1: Write the Implementation Function

Add a private `async def _my_tool_impl(...)` function in the integration tool file. Keep business logic here and away from the `@mcp.tool()` wrapper.

```python
# tools/netbrain_tools.py  (example: get_device_interfaces)

async def _get_device_interfaces_impl(device_name: str) -> dict:
    """Internal implementation — called by the MCP tool wrapper."""
    auth_token = netbrainauth.get_auth_token()
    if not auth_token:
        return {"error": "Failed to get authentication token"}

    url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Interfaces"
    headers = {"Token": auth_token, "Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params={"hostname": device_name},
                                   ssl=False, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    return {"error": f"NetBrain API returned {resp.status}"}
                data = await resp.json()
    except asyncio.TimeoutError:
        return {"device_name": device_name, "interfaces": [], "error": "Request timed out"}
    except Exception as e:
        logger.debug("get_device_interfaces error: %s", e)
        return {"device_name": device_name, "interfaces": [], "error": str(e)}

    interfaces = data.get("interfaces", [])
    return {
        "device_name": device_name,
        "interfaces": [
            {"name": i.get("name"), "ip": i.get("ip"), "type": i.get("interfaceType")}
            for i in interfaces
        ],
        "count": len(interfaces),
    }
```

**Rules:**
- Always return a `dict`.
- On failure, return `{"error": "..."}`. Never raise exceptions out of the impl.
- Include the main input key(s) in error responses (`"device_name": device_name`) so the frontend can still display context.
- Use `async`/`await` with `aiohttp` for all HTTP calls.

---

## Step 2: Write the `@mcp.tool()` Wrapper

The wrapper is a thin function — it only validates inputs and delegates to the impl. All logic stays in the impl.

```python
@mcp.tool()
async def get_device_interfaces(
    device_name: str,
    include_inactive: bool = False,
) -> dict:
    """
    Retrieve all interfaces for a given network device from NetBrain.

    Use for: queries asking for device interfaces — "interfaces on leander-dc-leaf1",
    "what interfaces does router-X have?", "show ports on device Y".
    Do NOT use for: path queries, address group lookups, rack/location queries.

    Examples:
    - "interfaces on leander-dc-leaf1" → device_name="leander-dc-leaf1"
    - "show ports on router-X" → device_name="router-X"

    Args:
        device_name: Hostname or name of the device (e.g., "leander-dc-leaf1")
        include_inactive: Include administratively down interfaces (default False)

    Returns:
        dict: device_name, interfaces (list), count, or error
    """
    return await _get_device_interfaces_impl(device_name, include_inactive)
```

### Docstring format — critical for LLM tool selection

FastMCP uses the docstring as the tool description sent to the LLM. `chat_service._to_openai_tool()` trims everything from `Args:` onward and caps at 600 chars, so everything the LLM needs to make the right choice must appear **before** `Args:`.

| Section | Purpose |
|---|---|
| First sentence | What the tool does (shown in full) |
| `Use for:` | Trigger phrases — the LLM matches these to the user query |
| `Do NOT use for:` | Disambiguation — guides the LLM away from wrong tool choices |
| `Examples:` | Concrete query → argument mappings (most effective guidance) |
| `Args:` | Parameter descriptions (used by FastMCP for schema, not sent to LLM) |
| `Returns:` | Response shape (for developers) |

### Type hints → JSON Schema

FastMCP auto-generates the tool's `inputSchema` from type hints:

| Python type | JSON Schema result |
|---|---|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `bool` | `{"type": "boolean"}` |
| `Optional[str]` or `str = None` | `{"type": "string"}`, marked optional |
| `str = "default"` | `{"type": "string", "default": "default"}` |

Parameters without defaults are `required` in the schema. Parameters with defaults are optional.

---

## Step 3: `TOOL_DISPLAY_NAMES` in chat_service.py

**File:** [chat_service.py](../../chat_service.py)

Add the tool name → UI label mapping. This label appears in the status bar while the query runs (`"Querying NetBrain"`).

```python
TOOL_DISPLAY_NAMES: dict[str, str] = {
    "check_path_allowed":                 "Atlas",
    "query_network_path":                 "Atlas",
    "query_panorama_ip_object_group":     "Panorama",
    "query_panorama_address_group_members": "Panorama",
    "get_splunk_recent_denies":           "Splunk",
    "get_device_interfaces":              "NetBrain",   # ← add this
}
```

Use the system name (e.g., `"NetBrain"`, `"Panorama"`, `"Splunk"`). This is what the user sees.

---

## Step 4: `_TOOL_TIMEOUTS` in chat_service.py

Set a per-tool timeout in seconds. This is the maximum time `call_mcp_tool()` will wait before giving up.

```python
_TOOL_TIMEOUTS: dict[str, float] = {
    "query_network_path":        385.0,   # NetBrain path polling can be slow
    "check_path_allowed":        370.0,
    "get_splunk_recent_denies":   95.0,   # Includes Splunk job polling
    "query_panorama_ip_object_group":       65.0,
    "query_panorama_address_group_members": 65.0,
    "get_device_interfaces":      30.0,   # ← add this (fast API call)
}
```

**Guidelines:**
- Simple single API calls: `30–65s`
- Calls that require polling (Splunk job, NetBrain path calculation): `90–400s`
- If a tool is not listed, a default timeout applies — always add it explicitly.

---

## Step 5: `_is_obviously_in_scope()` in chat_service.py

Add keywords that unambiguously identify your tool's queries. This fast-path check avoids an LLM call when the query obviously matches.

```python
def _is_obviously_in_scope(prompt: str) -> bool:
    lower = (prompt or "").lower()
    has_ip = bool(_IP_OR_CIDR_RE.search(prompt or ""))
    # ... existing keyword groups ...
    netbrain_kw = any(k in lower for k in (
        "network path", "path from", "path to", "traffic allowed",
        "path allowed", "can reach", "connectivity", "path",
        "interface", "interfaces", "ports on",   # ← add for get_device_interfaces
    ))
    ...
```

Only add keywords that are unambiguous — a keyword appearing in a completely unrelated query would incorrectly mark it as in-scope.

---

## Step 6: System prompt in `_build_llm_messages()`

Add a routing rule so the LLM knows when to select the new tool over others.

```python
SystemMessage(content=(
    "You are a network infrastructure assistant. "
    "Always call a tool — never answer from memory or prior context. "
    "Tool selection rules: "
    "short rack IDs like 'A4', 'B2' → get_rack_details; "
    "device names with dashes like 'leander-dc-leaf1' → get_device_rack_location; "
    "IP addresses → query_panorama_ip_object_group or get_splunk_recent_denies; "
    "address group names → query_panorama_address_group_members; "
    "list/all racks → list_racks; "
    "device interfaces/ports → get_device_interfaces. "   # ← add this
    ...
))
```

**Tips:**
- Phrase rules as `<trigger pattern> → <tool_name>`.
- If your tool overlaps with an existing one, add a disambiguation rule: `"interfaces on a device → get_device_interfaces (not get_device_rack_location)"`.
- Keep rules concise — the system prompt is sent on every query.

---

## Step 7: `_normalize_result()` in chat_service.py (optional)

Add normalization only if you want to inject a `direct_answer`, `yes_no_answer`, or `metric_answer` badge in the UI, or to clean up the result before rendering.

```python
def _normalize_result(tool_name, result, prompt=""):
    ...
    # Example: add a direct_answer summary for get_device_interfaces
    if tool_name == "get_device_interfaces" and isinstance(result, dict) and "error" not in result:
        count = result.get("count", 0)
        device = result.get("device_name", "the device")
        if count > 0:
            result = dict(result)
            result["direct_answer"] = f"{device} has {count} interface{'s' if count != 1 else ''}"
    ...
    return result
```

**Badge types:**

| Key | Rendered as | When to use |
|---|---|---|
| `direct_answer` | Blue info badge above the table | Plain summary sentence (counts, names) |
| `yes_no_answer` | Green/red Yes/No badge | Binary verdicts only |
| `metric_answer` | Metric badge | Single numeric metric (utilization %, height) |

If your tool result renders fine as a table with no extra context needed, skip this step.

---

## Step 8: RBAC in auth.py

**File:** [auth.py](../../auth.py)

Add the tool to `ROLE_ALLOWED_TOOLS` for the roles that should have access:

```python
ROLE_ALLOWED_TOOLS: dict[str, set[str] | None] = {
    "admin":    None,   # all tools always allowed
    "netadmin": {
        "query_network_path",
        "check_path_allowed",
        "query_panorama_ip_object_group",
        "query_panorama_address_group_members",
        "get_device_interfaces",    # ← add for netadmin if appropriate
    },
    "guest":    set(),  # no tools
}
```

If `admin` should be the only role with access, do not add it to `netadmin` — `admin` always gets all tools (`None` = unrestricted).

Also update `ROLE_ALLOWED_CATEGORIES` if the tool belongs to a sidebar category that should be shown/hidden per role:

```python
ROLE_ALLOWED_CATEGORIES: dict[str, list[str] | None] = {
    "admin":    None,
    "netadmin": ["atlas", "panorama"],   # add a category slug here if needed
    "guest":    [],
}
```

---

## Step 9: Frontend updates (if needed)

Frontend changes are only needed if the response shape is new. If your tool returns a standard dict with array values, it renders as a table automatically.

### responseClassifier.js — new response types only

**File:** [frontend/src/utils/responseClassifier.js](../../frontend/src/utils/responseClassifier.js)

Only modify if your tool returns a shape that doesn't fit the existing classifiers (`text`, `batch`, `error`, `path`, `path-summary`, `structured`, `table`, `json`). Most tools return a dict with array keys → classified as `'table'` automatically.

### formatters.js — column ordering and table labels

**File:** [frontend/src/utils/formatters.js](../../frontend/src/utils/formatters.js)

If the table column order matters, add an entry to `PANORAMA_COLUMN_ORDER` (despite the name, this works for any tool):

```js
export const PANORAMA_COLUMN_ORDER = {
  address_objects: ['name', 'type', 'value', 'location', 'device_group'],
  address_groups:  ['name', 'contains_address_object', 'members', 'location', 'device_group'],
  // ↓ add your tool's array key and preferred column order
  interfaces: ['name', 'ip', 'type'],
}

export const PANORAMA_TABLE_LABELS = {
  address_objects: 'Address objects',
  address_groups:  'Address groups',
  interfaces: 'Device interfaces',   // ← human-readable heading
}
```

To hide fields from the table (debug fields, internal IDs), add them to `_HIDDEN_KEYS`:

```js
const _HIDDEN_KEYS = new Set([
  'desc_units', 'outer_width', 'outer_unit', 'outer_depth',
  'intent', 'format', 'vsys', 'queried_ip',
  'raw_response',   // ← add fields you want hidden
])
```

### AssistantMessage.jsx — new render types only

**File:** [frontend/src/components/messages/AssistantMessage.jsx](../../frontend/src/components/messages/AssistantMessage.jsx)

Only modify if your tool uses a new classifier type that needs custom JSX. For standard table/structured responses, no changes are needed.

---

## Verification

After making the changes, verify end-to-end:

1. **MCP server** — restart `mcp_server.py`. Check `mcp_server.log` confirms the new tool is registered (`tools_registered: N+1`).
2. **Tool schema** — hit `GET http://127.0.0.1:8765/health` — confirm tool count increased.
3. **RBAC** — log in as a `netadmin` user and confirm the tool is allowed/denied as configured.
4. **LLM selection** — type a query matching your new tool's "Use for" examples and confirm `tool_display_name` in the discover response matches.
5. **End-to-end** — run the full query and confirm the response renders correctly in the UI.
