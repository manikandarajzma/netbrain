# Splunk MCP Tool

Atlas exposes one MCP tool for querying Splunk for firewall deny events involving a given IP address.

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [MCP Tool](#mcp-tool)
4. [Query Pipeline](#query-pipeline)
5. [Event Normalisation](#event-normalisation)
6. [Known Pitfalls](#known-pitfalls)
7. [FAQs](#faqs)

---

## Overview

```
User question
     │
     ▼
MCP Client (AI assistant)
     │
     ▼
splunk_tools.py  ──►  Splunk REST API (https://SPLUNK_HOST:SPLUNK_PORT)
     │                     │
     │   ◄──────────────────
     │
     ▼
Normalised event list  ──►  Atlas UI
```

Splunk is queried via its **REST API** (port 8089, the management port) using a session key obtained at the start of each tool call. The query is submitted as a search job, polled until complete, and results are normalised before being returned.

---

## Authentication

Unlike Panorama and NetBrain, the Splunk tool does **not** cache its session key. A fresh login is performed on every tool invocation.

### Credential Source

Credentials are read from environment variables (or `.env` file) at startup:

| Variable | Description |
|----------|-------------|
| `SPLUNK_HOST` | Hostname or IP of the Splunk instance |
| `SPLUNK_PORT` | Management port (default: `8089`) |
| `SPLUNK_USER` | Splunk username |
| `SPLUNK_PASSWORD` | Splunk password |

### Login flow

Each call to `get_splunk_recent_denies` performs a fresh login:

```
POST /services/auth/login
  body: { username, password }
  ──►  sessionKey (XML response)
```

The session key is used as a `Authorization: Splunk <key>` header on all subsequent requests within the same call. It is discarded after the call completes.

**Why no caching?** Splunk session keys are short-lived and tied to the REST API session. Reusing a key across calls risks stale-session errors. The login is fast (~100ms) relative to the search job itself, so the overhead is acceptable.

---

## MCP Tool

### `get_splunk_recent_denies`

**Purpose:** Search Splunk for recent firewall deny/denied events involving a given IP address (as source or destination).

**When to use:** "Recent denies for `10.0.0.1`" / "Show deny events for `192.168.1.5`" / "Firewall denials for `10.0.0.250`".

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ip_address` | str | — | IP to search for in deny events |
| `limit` | int | `100` | Maximum number of events to return |
| `earliest_time` | str | `"-24h"` | Splunk time range specifier (e.g. `-24h`, `-7d`, `-30m`) |

**Returns:** `ip_address`, `events` (list of normalised event dicts), `count`, and optional `error`.

Each event contains: `time`, `firewall`, `vendor_product`, `src_ip`, `dst_ip`, `src_zone`, `dest_zone`, `protocol`, `port`, `action`.

---

## Query Pipeline

```
1. Login  ──►  POST /services/auth/login  ──►  sessionKey
               │
2. Create job  ──►  POST /services/search/jobs
               │    SPL: search index=* (deny OR denied)
               │         (src_ip="<ip>" OR dest_ip="<ip>" ...)
               │         earliest=<time> | head <limit>
               │    ──►  sid (search job ID)
               │
3. Poll    ──►  GET /services/search/jobs/<sid>
               │    repeat every 2s, up to 120s (60 iterations)
               │    until isDone == true
               │
4. Fetch   ──►  GET /services/search/jobs/<sid>/results
               │    output_mode=json
               │    ──►  raw event list
               │
5. Normalise  ──►  extract fields, return to caller
```

### Search query

The SPL search covers the most common field names used by different firewall vendors in Splunk:

```
search index=* (deny OR denied)
  (src_ip="<ip>" OR dest_ip="<ip>"
   OR src="<ip>"  OR dst="<ip>"
   OR src_ip=<ip> OR dest_ip=<ip>)
earliest=<time> | head <limit>
```

The query searches **all indexes** (`index=*`) since deny events may land in different indexes depending on the Splunk deployment.

### Polling

The tool polls the job status every **2 seconds** for up to **120 seconds** (60 iterations). If the job does not complete within 120 seconds, the tool returns a timeout error.

---

## Event Normalisation

Splunk events vary significantly in structure depending on the firewall vendor. The tool normalises all events to a consistent set of fields.

### Field extraction

Fields are extracted with a case-insensitive fallback — if `src_ip` is not present, it tries `src`, then `SRC_IP`, etc. This handles differences between Palo Alto, Cisco ASA, Fortinet, and other vendor log formats.

### Zone extraction

Firewall zones are frequently not exposed as top-level Splunk fields — they only appear in the raw log string (`_raw`). The tool uses multiple strategies to extract them:

1. **Palo Alto CSV format** — Palo Alto TRAFFIC logs use a fixed CSV layout where source and destination zones appear in a predictable position after the vsys field:
   ```
   ...,vsys1,<src_zone>,<dst_zone>,...
   ```
   Regex: `,vsys\d*\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,`

2. **Key=value format** — common in syslog: `from_zone=trust` / `to_zone=untrust`

3. **Natural language format** — e.g. `from zone trust` / `to zone untrust`

4. **JSON embedded** — `"from_zone": "trust"` inside a JSON blob within `_raw`

### Protocol extraction

Protocol is similarly extracted from multiple sources: top-level fields (`protocol`, `proto`, `transport`), any field whose name contains "protocol", and finally regex patterns in `_raw` (`protocol=icmp`, `\b(icmp|tcp|udp|gre|esp|ip)\b`).

### Normalised output fields

| Field | Source fields tried |
|-------|-------------------|
| `time` | `_time`, `time` |
| `firewall` | `dvc_name`, `host`, `device`, `firewall`, `hostname`, `DeviceName`; fallback: regex on `_raw` for Palo Alto syslog hostname |
| `vendor_product` | `vendor_product`, `product`, `vendor`; fallback: `"Palo Alto Networks Firewall"` |
| `src_ip` | `src_ip`, `src` |
| `dst_ip` | `dest_ip`, `dst_ip`, `dst` |
| `src_zone` | `src_zone`, `source_zone`, `from_zone`, `src_zone_name`; fallback: `_raw` parsing |
| `dest_zone` | `dest_zone`, `destination_zone`, `to_zone`, `dest_zone_name`; fallback: `_raw` parsing |
| `protocol` | `protocol`, `Protocol`, `proto`, `transport`; fallback: `_raw` regex |
| `port` | `port`, `dest_port`, `dport` |
| `action` | `action`, `Action`; fallback: `"drop"` |

---

## Known Pitfalls

### Port 8089, not 8000

The Splunk REST API runs on port **8089** (management port). Port 8000 is the web UI and does not expose `/services/auth/login`. Using the wrong port will result in a login failure.

### `index=*` performance

Searching all indexes can be slow on large Splunk deployments. If deny events are consistently in a known index (e.g. `index=firewall`), the search query could be scoped to that index to improve performance. This is currently hardcoded in `_splunk_search_impl`.

### SSL certificate verification disabled

All Splunk API calls disable SSL verification (`ssl.CERT_NONE`) because Splunk commonly uses a self-signed certificate on the management port. This is appropriate for internal environments.

### No result caching

Every tool invocation runs a full Splunk search job. There is no caching of results. Repeated queries for the same IP and time range will submit separate search jobs each time.

---

## FAQs

**Q: Why is a new login performed on every query instead of caching the session key?**

Splunk session keys are short-lived REST API session tokens. Caching them risks using an expired key on a subsequent call, which would fail with a 401 and require a re-login anyway. Since the login itself is fast relative to the search job, not caching it keeps the code simpler without meaningful performance impact.

**Q: How far back does the search go by default?**

The default `earliest_time` is `-24h` (last 24 hours). This can be overridden per call — e.g. `-7d` for 7 days, `-1h` for the last hour.

**Q: Why does the search use `index=*`?**

Deny events may be in different indexes depending on how the Splunk deployment is configured. Using `index=*` ensures all indexes are searched. In a production deployment with known index names, scoping the query would improve performance.

**Q: How does Atlas handle firewalls that log zones differently?**

The zone extraction logic tries multiple patterns (Palo Alto CSV, key=value, natural language, JSON). If none match, the zone fields are returned empty. Zones are informational and their absence does not prevent the event from being returned.

**Q: What happens if the Splunk job takes longer than 120 seconds?**

The polling loop runs for a maximum of 60 iterations × 2 seconds = 120 seconds. If the job is not done by then, the tool returns a timeout error: `"Splunk request timed out"`.
