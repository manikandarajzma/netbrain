# NetBrain MCP Tools

Atlas exposes two MCP tools for querying NetBrain for hop-by-hop network paths and traffic policy verdicts between two IP addresses.

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [MCP Tools](#mcp-tools)
4. [Query Pipeline](#query-pipeline)
5. [Caching Mechanisms](#caching-mechanisms)
6. [Panorama Integration](#panorama-integration)
7. [Known Pitfalls](#known-pitfalls)
8. [FAQs](#faqs)

---

## Overview

```
User question
     │
     ▼
MCP Client (AI assistant)
     │
     ▼
netbrain_tools.py  ──►  NetBrain REST API (NETBRAIN_URL/ServicesAPI/...)
     │                       │
     │   ◄────────────────────
     │
     ├──►  panorama_tools.py  ──►  Panorama XML API
     │         (zones + device groups)
     │
     ▼
Hop-by-hop path + policy verdict  ──►  Atlas UI
```

NetBrain is queried via its **REST API** using a session token obtained from `netbrainauth.py`. After the path is calculated, Panorama is called in parallel to enrich firewall hops with security zone and device group information.

---

## Authentication

Authentication is handled by `netbrainauth.py` using NetBrain's Session API.

### Credential Source

Credentials are read from environment variables (or `.env` file):

| Variable | Description |
|----------|-------------|
| `NETBRAIN_URL` | Base URL of the NetBrain server (e.g. `https://netbrain.example.com`) |
| `NETBRAIN_USERNAME` | NetBrain username |
| `NETBRAIN_PASSWORD` | NetBrain password |
| `NETBRAIN_AUTH_ID` | Authentication ID for external users (LDAP/AD/TACACS); optional |

### Token cache with TTL

Unlike Panorama's indefinite API key cache, NetBrain uses a **30-minute TTL** on the cached session token:

```python
TOKEN_TTL_SECONDS = 30 * 60  # 30 minutes

if _access_token and (time.time() - _token_obtained_at) < TOKEN_TTL_SECONDS:
    return _access_token  # cache hit

# token missing or expired — fetch a new one
```

The token is obtained via:

```
POST /ServicesAPI/API/V1/Session
  body: { username, password [, authentication_id] }
  ──►  token (string)
```

The token is passed as a `Token: <value>` header on all subsequent requests.

### Automatic retry on 401

If a NetBrain API call returns HTTP 401 (token expired mid-session), the tools automatically clear the cache and re-authenticate before retrying the failed request once:

```
HTTP 401 received
     │
     ▼
clear_token_cache()
get_auth_token()  ──►  new token
     │
     ▼
retry original request with new token
```

This handles cases where the 30-minute TTL is optimistic and NetBrain invalidates the token earlier.

---

## MCP Tools

### `query_network_path`

**Purpose:** Calculate the hop-by-hop network path between two IP addresses and return details about each hop including device name, type, interfaces, security zones (from Panorama), and ACL/policy decisions.

**When to use:** "What is the path from `10.0.0.1` to `192.168.1.5`?" / "Show me the network route between these IPs."

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | str | — | Source IP address or hostname |
| `destination` | str | — | Destination IP address or hostname |
| `protocol` | str | `"TCP"` | Protocol: `TCP`, `UDP`, or `IP` |
| `port` | str | `"0"` | Destination port number |
| `is_live` | int | `1` | `1` = live network access; `0` = baseline/cached data |

**Returns:** `hops` (list of hop dicts), `path_found` (bool), `source`, `destination`, and optional `error`.

Each hop contains: `hop_number`, `device_name`, `device_type`, `in_interface`, `out_interface`, `in_zone`, `out_zone` (from Panorama), `device_group` (from Panorama), `is_firewall`, `acl_decisions`.

---

### `check_path_allowed`

**Purpose:** Return a simple **allowed / denied** verdict for traffic between two IPs, along with the reason (which firewall and which policy caused a denial if applicable).

**When to use:** "Can `10.0.0.1` reach `192.168.1.5` on TCP/443?" / "Is traffic allowed between these hosts?"

**Parameters:** Same as `query_network_path`.

**Returns:** `allowed` (bool), `verdict` (string summary), `blocking_device` (if denied), `blocking_reason`, full `hops` list, and optional `error`.

Both tools share the same underlying path calculation implementation (`_query_network_path_impl`). `check_path_allowed` calls `_query_network_path_impl` with `continue_on_policy_denial=True` to ensure the full path is computed even when a firewall denies the traffic.

---

## Query Pipeline

All path queries follow a 3-step process against the NetBrain API, followed by Panorama enrichment:

```
Step 1: Resolve gateway
        GET /V1/CMDB/Path/Gateways?ipOrHost=<source>
        ──►  sourceGateway object
        │
        │    (if gateway not found: use source IP as placeholder,
        │     Path Fix-up rules will resolve it during calculation)
        │
Step 2: Calculate path
        POST /V1/CMDB/Path/Calculation
          body: {
            sourceIP, sourcePort, destIP, destPort,
            protocol, pathAnalysisSet=1 (L3),
            isLive=1,
            advanced: {
              calcWhenDeniedByACL: true,
              calcWhenDeniedByPolicy: true,
              enablePathFixup: true,
              enablePathIPAndGatewayFixup: true
            }
          }
        ──►  taskID
        │
Step 3: Fetch result
        GET /V1/CMDB/Path/Calculation/Result?taskID=<id>
        ──►  hop list with device info and ACL decisions
        │
Step 4: Enrich with Panorama (parallel)
        ├──►  _add_panorama_zones_to_hops()
        │     (security zones for firewall interfaces)
        │
        └──►  _add_panorama_device_groups_to_hops()
              (device group membership for firewalls)
```

### Protocol mapping

NetBrain's path calculation API uses numeric protocol codes:

| Protocol string | Numeric code |
|-----------------|-------------|
| `TCP` | `6` |
| `UDP` | `17` |
| `IP` / `IPv4` | `4` (default) |

### Path Fix-up rules

If the source IP's gateway cannot be resolved (NetBrain statusCode `792040`), Atlas still proceeds with path calculation using a placeholder gateway object. NetBrain's **Path Fix-up rules** (`enablePathFixup: true`, `enablePathIPAndGatewayFixup: true`) handle gateway resolution during the calculation itself. This allows paths to be computed for hosts that are not directly registered in NetBrain's CMDB.

---

## Caching Mechanisms

NetBrain tools use two module-level caches with **no TTL** (persist for the lifetime of the MCP server process):

### 1. Device type numeric cache (`_device_type_cache`)

```python
_device_type_cache: Optional[Dict[int, str]] = None
```

- **What:** Maps numeric device type codes (e.g. `1`) to descriptive names (e.g. `"Cisco Router"`).
- **Source:** NetBrain's `/ServicesAPI/SystemModel/getAllDisplayDeviceTypes` endpoint.
- **TTL:** None — populated once on first use, reused for the server lifetime.
- **Fallback:** If the device type endpoint is unavailable, falls back to the Devices API.

### 2. Device name-to-type cache (`_device_name_to_type_cache`)

```python
_device_name_to_type_cache: Optional[Dict[str, str]] = None
```

- **What:** Maps device hostnames (e.g. `"leander-fw1"`) to their type name (e.g. `"Palo Alto Firewall"`).
- **Source:** NetBrain's `/ServicesAPI/API/V1/CMDB/Devices` endpoint (fetches up to 100 devices).
- **TTL:** None — populated once, reused for the server lifetime.
- **Purpose:** Name-based lookup is the **preferred** method for resolving device types in path hops. Numeric codes are used as a fallback.

### Cache population sequence

On the first path query, `get_device_type_mapping()` is called to pre-build both caches before processing hop results:

```
First query arrives
     │
     ▼
get_device_type_mapping()
     │
     ├──►  Try /SystemModel/getAllDisplayDeviceTypes  ──►  _device_type_cache
     │
     └──►  Try /CMDB/Devices (fallback)
               ──►  _device_name_to_type_cache
               ──►  _device_type_cache (if numeric IDs found)
```

Subsequent queries skip this step entirely since the cache is already populated.

### Token cache

The session token is cached with a 30-minute TTL in `netbrainauth.py` (see [Authentication](#authentication)).

---

## Panorama Integration

After NetBrain returns the hop list, Atlas enriches firewall hops with data from Panorama. This is done via two helper functions imported from `panorama_tools.py`:

### `_add_panorama_zones_to_hops()`

For each firewall hop, queries Panorama's REST API to find which security zone each interface belongs to:

```
GET /restapi/<version>/network/zones
    ?location=template&template=Global
──►  zone list with interface members
```

The function tries multiple REST API versions (`v10.2`, `v10.1`, `v10.0`, `v9.1`, `v9.0`) until one succeeds. Interface matching is case-insensitive.

The result is written back into the hop dict: `hop["in_zone"]` and `hop["out_zone"]`.

### `_add_panorama_device_groups_to_hops()`

For each firewall hop, queries Panorama to find which device group the firewall belongs to. The firewall is matched to a Panorama-managed device using hostname comparison (case-insensitive, partial match). The device group is then written into the hop dict: `hop["device_group"]`.

Both Panorama enrichment calls run **after** the NetBrain path result is received. They are not parallelised with each other but are called sequentially in the post-processing phase.

---

## Known Pitfalls

### Device type cache covers only the first 100 devices

The fallback Devices API fetch is limited to 100 devices (`limit=100`, the maximum allowed by the API). In large environments, devices beyond the first 100 will have their type resolved via numeric code only, or shown as `"Device Type <N>"` if the numeric mapping is also missing.

### Panorama zone lookup requires the "Global" template

Zone queries are hardcoded to use template name `"Global"`. If the Panorama deployment uses a different template name for interface-to-zone mappings, zones will not be populated on firewall hops.

### `is_live=1` makes real-time calls to network devices

With `is_live=1` (the default), NetBrain queries the actual network devices for up-to-date routing and interface state. This is slower than baseline mode (`is_live=0`) and may time out if devices are unreachable. Use `is_live=0` for faster queries against NetBrain's cached topology data.

### SSL certificate verification disabled

All NetBrain and Panorama API calls in the path tools disable SSL verification. This is appropriate for internal lab/enterprise environments with self-signed certificates.

---

## FAQs

**Q: What is the difference between `query_network_path` and `check_path_allowed`?**

`query_network_path` returns the full hop-by-hop path with all details (interfaces, zones, ACL decisions). `check_path_allowed` returns the same data but leads with a simple allowed/denied verdict and the blocking device/reason, making it easier to answer yes/no traffic questions. Both use the same underlying path calculation.

**Q: Why does NetBrain use a 30-minute token TTL while Panorama has no TTL?**

NetBrain session tokens are explicitly time-limited server-side. The 30-minute TTL in Atlas is a proactive refresh to avoid using tokens that are about to expire. Panorama API keys are long-lived by design (they only expire when the account password changes), so a TTL on the Atlas side would not improve security — the old key remains valid on Panorama regardless.

**Q: What happens if the gateway cannot be resolved?**

Atlas uses a placeholder gateway and enables NetBrain's Path Fix-up rules (`enablePathFixup`, `enablePathIPAndGatewayFixup`). These rules allow NetBrain to resolve the gateway during path calculation using its own topology data. Most paths complete successfully even without an explicit gateway.

**Q: How are device types determined for each hop?**

Atlas first tries to match the hop's device hostname against the `_device_name_to_type_cache` (built from the Devices API). If no match is found, it falls back to the numeric device type code mapped via `_device_type_cache`. If neither works, the type is shown as `"Device Type <N>"` or left blank.

**Q: Why are security zones fetched from Panorama rather than NetBrain?**

NetBrain's path calculation returns interface names but not security zone assignments. Zone information lives in Panorama (or on the firewall directly). Atlas queries Panorama's REST API to map each firewall interface to its zone, enriching the hop data with context that NetBrain alone cannot provide.

**Q: How quickly does a path query complete?**

A typical query takes 5–15 seconds: ~1s for gateway resolution, ~3–10s for path calculation (live mode), ~1–3s for Panorama zone enrichment. Queries in baseline mode (`is_live=0`) are faster since NetBrain does not query live devices.
