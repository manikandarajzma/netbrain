# Panorama MCP Tools

Atlas exposes two MCP tools for querying Palo Alto Panorama's XML API for address objects, address groups, and security/NAT policies. This document explains how the tools work, how queries are routed, and the performance mechanisms that keep responses fast even at scale (6,000+ objects and groups).

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [MCP Tools](#mcp-tools)
4. [Query Pipeline](#query-pipeline)
5. [Caching Mechanisms](#caching-mechanisms)
6. [Parallel Fetching](#parallel-fetching)
7. [N+1 Elimination](#n1-elimination)
8. [Policy Resolution](#policy-resolution)
9. [LLM Analysis](#llm-analysis)
10. [Known Pitfalls](#known-pitfalls)

---

## Overview

```
User question
     │
     ▼
MCP Client (AI assistant)
     │
     ▼
panorama_tools.py  ──►  Panorama XML API (https://PANORAMA_URL/api/)
     │                       │
     │   ◄────────────────────
     │
     ▼
LLM summary + structured JSON  ──►  Atlas UI
```

Panorama uses a **candidate config model** — the XML API `action=get` always reads the *candidate* config, not the running config. Changes written by API calls are not live until a commit job completes.

---

## Authentication

Authentication is handled by `panoramaauth.py`.

### Credential Source

Credentials are loaded **exclusively from Azure Key Vault** at module import time:

| Secret name (Key Vault) | Environment variable override |
|-------------------------|-------------------------------|
| `PANORAMA-USERNAME`     | `PANORAMA_USERNAME_KEYVAULT_SECRET_NAME` |
| `PANORAMA-PASSWORD`     | `PANORAMA_PASSWORD_KEYVAULT_SECRET_NAME` |

The vault URL is read from `AZURE_KEYVAULT_URL`. Authentication to the vault uses `DefaultAzureCredential` (supports managed identity, environment, CLI, etc.).

### API Key Cache

Panorama's XML API uses **session-based API keys** rather than per-request Basic auth. The key is obtained once via the `keygen` endpoint:

```
GET /api/?type=keygen&user=<user>&password=<password>
```

The returned key is stored in the module-level variable `_api_key` in `panoramaauth.py`. All subsequent calls reuse this key without re-authenticating, saving one round-trip per query.

The key is **not TTL-expired** — it persists for the lifetime of the MCP server process. If Panorama invalidates the session (e.g. server restart, session timeout), the next API call will receive an `unauth` error. In that case, `clear_api_key_cache()` resets `_api_key = None` so the next call re-authenticates.

---

## MCP Tools

### `query_panorama_ip_object_group`

**Purpose:** Given an IP address or CIDR, find every address object that contains it, every address group those objects belong to, and every security/NAT policy that references those groups or objects.

**When to use:** "What address group is `10.0.0.1` in?" / "Which group contains `11.0.0.0/24`?"

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ip_address` | str | IP or CIDR to search for (e.g. `10.0.0.1`, `10.0.0.0/24`) |
| `device_group` | str \| None | Restrict search to one device group; if None, searches shared + all device groups |
| `vsys` | str | VSYS name (default: `vsys1`) |

**Returns:** `address_objects`, `address_groups` (with full member lists), `policies` (security + NAT), `ai_analysis`.

---

### `query_panorama_address_group_members`

**Purpose:** Given an address group name, list all member IP addresses/objects it contains and every policy that uses the group.

**When to use:** "What IPs are in `leander_web`?" / "List members of `dmz_hosts`."

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `address_group_name` | str | Exact group name to look up |
| `device_group` | str \| None | Restrict search; if None, searches shared + all device groups |
| `vsys` | str | VSYS name (default: `vsys1`) |

**Returns:** `members` (with name, type, IP value), `policies`, `ai_analysis`.

---

## Query Pipeline

### `query_panorama_ip_object_group` — 3-step pipeline

```
Step 1: Discover locations → fetch all address objects in parallel
        ┌─────────────┐  ┌──────────────┐  ┌──────────────┐
        │   shared    │  │ device-group │  │ device-group │  ...
        │  (cached)   │  │  leander     │  │  roundrock   │
        │             │  │  (cached)    │  │  (cached)    │
        └──────┬──────┘  └──────┬───────┘  └──────┬───────┘
               └────────────────┴──────────────────┘
                         asyncio.gather()
                               │
                        IP matching logic
                               │
Step 2: For locations that have matching objects →
        fetch all address groups in parallel
               │
        Look up member details from already-cached objects (no extra HTTP calls)
               │
Step 3: For each location with matching groups/objects →
        fetch pre-rulebase + post-rulebase security rules
        fetch pre-rulebase + post-rulebase NAT rules
```

### `query_panorama_address_group_members` — 2-step pipeline

```
Step 1: Search locations sequentially until the named group is found
        (stops at first match — no need to search all locations)
               │
        Fetch all address objects for that location (cached)
        Resolve member details from cache — no per-member HTTP call
               │
Step 2: Fetch pre-rulebase + post-rulebase security rules
        Fetch pre-rulebase + post-rulebase NAT rules
        Filter for rules referencing this group
```

---

## Caching Mechanisms

All caches live at **module level** in `panoramaauth.py` and `panorama_tools.py`, so they are shared across all MCP tool invocations for the lifetime of the server process.

### 1. API Key Cache (`panoramaauth._api_key`)

- **What:** The Panorama session API key string.
- **TTL:** No automatic expiry — persists until the server restarts or `clear_api_key_cache()` is called.
- **Benefit:** Eliminates one HTTP round-trip (`/api/?type=keygen`) on every query.

### 2. Device Group List Cache (`_dg_cache`)

Defined in `panorama_tools.py`:

```python
_CACHE_TTL = 300.0  # 5 minutes

_dg_cache: tuple[list, float] | None = None
```

- **What:** The list of device group names returned by:
  ```
  GET /api/?type=config&action=get
      &xpath=/config/devices/entry[@name='localhost.localdomain']/device-group/entry
  ```
- **TTL:** 5 minutes (checked with `time.monotonic()`).
- **Why it matters:** Every query that doesn't specify a `device_group` must first know all device groups so it can fan-out queries. Without caching, this would be one extra HTTP call on every query.
- **Cache key:** Global singleton (there is only one Panorama instance).

> **Important — XPath correctness:** The device group list query returns the *full subtree* of every device group, including all nested address objects inside them. Using `.//entry` (recursive search) would return thousands of entries — the device group names *plus* every address object name inside every group. The correct XPath is `./result/entry`, which selects only the **direct children** of the result element (i.e., the actual device group entries).

### 3. Address Object Cache (`_addr_obj_cache`)

```python
_addr_obj_cache: dict[str, tuple[dict, float]] = {}
```

- **What:** A dictionary mapping each location key (`"device-group:leander"`, `"shared:None"`) to a dict of all address objects in that location: `{name: {type, value}}`.
- **TTL:** 5 minutes per location entry.
- **Why it matters:** Address objects are the raw data that groups and policies reference. Without caching, every query for a group's members would require fetching all objects again. With caching, the second query to the same location is a pure in-memory dict lookup.
- **Cache key:** `f"{location_type}:{location_name or 'shared'}"` — one entry per Panorama location.

### Cache hit/miss logging

All cache decisions are logged at DEBUG level:

```
Device groups: cache hit (2 groups)
Address objects: cache hit for device-group:leander (6001 objects)
Address objects: fetched 6001 from device-group:leander
```

---

## Parallel Fetching

Before caching was introduced, queries fetched each location **sequentially** — each HTTP call had to complete before the next started. With N device groups, this meant N serial round-trips.

Now, after the device group list is known, **all address object fetches are dispatched simultaneously** using `asyncio.gather()`:

```python
addr_obj_results = await asyncio.gather(
    *[_get_address_objects_cached(session, panorama_url, api_key, ssl_context, lt, ln)
      for lt, ln in locations],
    return_exceptions=True,
)
```

The same pattern is used for address group fetches in step 2:

```python
addr_grp_results = await asyncio.gather(
    *[_fetch_address_groups_for_location(session, panorama_url, api_key, ssl_context, lt, ln)
      for lt, ln in relevant_locations],
    return_exceptions=True,
)
```

`return_exceptions=True` ensures that a timeout or error in one location doesn't abort the entire query — failed locations are skipped and logged.

### HTTP request count comparison

| Approach | Requests per query (N device groups, M matching members) |
|----------|----------------------------------------------------------|
| Original (serial, no cache) | `1 (DG list) + N (objects) + N (groups) + M (member details) + policies` |
| Current (parallel + cached) | `0–1 (DG list, cached) + 0–N (objects, cached, parallel) + N (groups, parallel) + 0 (member details from cache) + policies` |

On a warm cache with 2 device groups, a typical IP lookup completes in **2–4 HTTP calls** (address groups + policies) instead of 20+.

---

## N+1 Elimination

The original implementation looked up each group member's IP details with a **separate HTTP call per member**:

```python
# Old pattern — N+1 problem
for member_name in group.members:
    member_details = await fetch_address_object(member_name)  # HTTP call!
```

With 1,000 groups × 3 members average = 3,000 HTTP calls just for member resolution.

The current implementation resolves member details entirely **from the already-fetched address object cache**:

```python
# Current pattern — zero extra HTTP calls
loc_objs = location_objects.get((lt, ln), {})  # dict already in memory
for member_name in member_names:
    det = loc_objs.get(member_name, {})  # pure dict lookup
    group_members.append({
        "name": member_name,
        "type": det.get("type"),
        "value": det.get("value"),
        ...
    })
```

Member detail resolution is now O(1) per member with zero network overhead.

---

## Policy Resolution

After finding matching address objects and groups, both tools query **all four rulebases** for policies that reference them:

| Rulebase | Query XPath |
|----------|-------------|
| Security pre-rules | `.../pre-rulebase/security/rules/entry` |
| Security post-rules | `.../post-rulebase/security/rules/entry` |
| NAT pre-rules | `.../pre-rulebase/nat/rules/entry` |
| NAT post-rules | `.../post-rulebase/nat/rules/entry` |

For each rule entry, the tool checks:
- **Security rules:** `source/member` and `destination/member` lists for matching group or object names.
- **NAT rules:** Same `source`/`destination` members, plus `source-translation/static-ip/translated-address` and `destination-translation/static-ip/translated-address` for group references in translation targets.

Each matching policy is returned with: name, type (security/nat), rulebase (pre/post), action or NAT type, source list, destination list, services, and the matched address groups/objects.

---

## LLM Analysis

After the raw Panorama data is assembled, it is passed to the configured LLM (`_get_llm()`) for a natural-language summary. The LLM receives the full JSON result and a system prompt instructing it to:

- For `query_panorama_ip_object_group`: write a 2–4 sentence narrative (no tables) stating which group(s) the IP is in, the address object name, and how many policies reference them.
- For `query_panorama_address_group_members`: produce two markdown tables — one for group members (name, type, IP, location) and one for policies (group, policy name, type, rulebase, action, source, destination, services).

The LLM call has a **30-second timeout**. If it fails, a basic fallback summary is generated from the raw counts.

The analysis is returned in `result["ai_analysis"]["summary"]`.

---

## Known Pitfalls

### Session timeouts during long-running scripts

Panorama invalidates API keys after extended inactivity. If a bulk operation runs for 90+ minutes, individual API calls may start returning `code="22" Session timed out`. The generate script logs these as warnings and continues; the MCP server would need `clear_api_key_cache()` to recover.

### Candidate config vs running config

`action=get` reads the **candidate config**, not the running config. Objects written by API calls but not yet committed appear in query results. If the candidate config is reverted (`load running-config.xml`), those objects disappear. This can cause confusing query results if a commit is in progress.

### Large candidate configs slow API responses

Panorama's XML API response time degrades as the candidate config grows. With 6,000+ objects in a device group, individual `action=set` calls can slow from ~100ms to several seconds. The caching and parallel fetching in Atlas mitigate this for read queries, but write-heavy scripts will always be bounded by Panorama's processing capacity.

### SSL certificate verification

All API calls disable SSL certificate verification (`ssl.CERT_NONE`, `check_hostname=False`) because Panorama typically uses a self-signed certificate. This is intentional and appropriate for internal lab environments.
