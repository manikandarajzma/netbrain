"""
Splunk MCP tools – get_splunk_recent_denies.

Fully isolated domain module; no cross-domain dependencies.
"""

import sys
import ssl
import asyncio
import aiohttp
from typing import Dict, Any

from tools.shared import mcp, SPLUNK_HOST, SPLUNK_PORT, SPLUNK_USER, SPLUNK_PASSWORD


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

async def _splunk_search_impl(
    ip_address: str,
    limit: int = 100,
    earliest_time: str = "-24h"
) -> Dict[str, Any]:
    """
    Run a Splunk search for recent deny events involving the given IP.
    Uses Splunk REST API: login, create job, poll until done, fetch results.
    """
    # Port 8089 is Splunk management/REST API (HTTPS); 8000 is web UI and does not expose /services/auth/login
    base_url = f"https://{SPLUNK_HOST}:{SPLUNK_PORT}"
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    session_key = None
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            # Login
            login_url = f"{base_url}/services/auth/login"
            payload = {"username": SPLUNK_USER, "password": SPLUNK_PASSWORD}
            async with session.post(login_url, data=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {"ip_address": ip_address, "events": [], "error": f"Splunk login failed: {resp.status} - {text[:500]}"}
                text = await resp.text()
                # Parse session key from XML response
                import xml.etree.ElementTree as ET
                root = ET.fromstring(text)
                session_key = root.find(".//sessionKey")
                if session_key is None or session_key.text is None:
                    return {"ip_address": ip_address, "events": [], "error": "Splunk login response had no sessionKey"}
                session_key = session_key.text
            headers = {"Authorization": f"Splunk {session_key}"}
            # Build search: deny/denied and IP in common fields
            search = (
                f'search index=* (deny OR denied) (src_ip="{ip_address}" OR dest_ip="{ip_address}" '
                f'OR src="{ip_address}" OR dst="{ip_address}" OR src_ip={ip_address} OR dest_ip={ip_address}) '
                f'earliest={earliest_time} | head {limit}'
            )
            create_url = f"{base_url}/services/search/jobs"
            async with session.post(
                create_url,
                headers=headers,
                data={"search": search, "output_mode": "json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    return {"ip_address": ip_address, "events": [], "error": f"Splunk create job failed: {resp.status} - {text[:500]}"}
                data = await resp.json()
                sid = data.get("sid")
                if not sid:
                    return {"ip_address": ip_address, "events": [], "error": "Splunk job created but no sid returned"}
            # Poll until done (max 120 seconds); request JSON so Splunk does not return XML
            status_url = f"{base_url}/services/search/jobs/{sid}"
            for _ in range(60):
                await asyncio.sleep(2)
                async with session.get(
                    status_url, headers=headers, params={"output_mode": "json"},
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                    entry = data.get("entry", [{}])[0] if isinstance(data.get("entry"), list) else {}
                    content = entry.get("content", {}) if isinstance(entry, dict) else {}
                    if content.get("isDone") is True:
                        break
            # Get results
            results_url = f"{base_url}/services/search/jobs/{sid}/results"
            async with session.get(
                results_url,
                headers=headers,
                params={"output_mode": "json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {"ip_address": ip_address, "events": [], "error": f"Splunk results failed: {resp.status} - {text[:500]}"}
                data = await resp.json()
            raw_results = data.get("results", []) if isinstance(data, dict) else []
            # Log first event keys and _raw sample so we can see Splunk format
            if raw_results and isinstance(raw_results[0], dict):
                first_keys = list(raw_results[0].keys())
                print(f"DEBUG: Splunk first event keys: {first_keys}", file=sys.stderr, flush=True)
                raw_sample = raw_results[0].get("_raw", "")
                if raw_sample:
                    print(f"DEBUG: Splunk first event _raw (first 1000 chars): {raw_sample[:1000]!r}", file=sys.stderr, flush=True)
            # Get value by key (case-insensitive); Splunk may return "Protocol" etc.
            def _get(e, *keys):
                if not isinstance(e, dict):
                    return ""
                for k in keys:
                    if k in e and e[k] not in (None, ""):
                        return e[k]
                for k, v in e.items():
                    if k and v not in (None, "") and k.lower() in {x.lower() for x in keys}:
                        return v
                return ""
            # Find protocol: Splunk may not expose it as a field (only in _raw)
            def _get_protocol(e):
                v = _get(e, "protocol", "Protocol", "proto", "transport")
                if v:
                    return v
                if isinstance(e, dict):
                    for k, val in e.items():
                        if k and "protocol" in k.lower() and val not in (None, ""):
                            return val
                # Extract from _raw (e.g. "protocol=icmp" or "Protocol: icmp")
                raw = e.get("_raw") if isinstance(e, dict) else ""
                if isinstance(raw, str) and raw:
                    import re
                    for m in re.finditer(r"(?:protocol|proto)\s*[=:]\s*(\w+)", raw, re.IGNORECASE):
                        return m.group(1).strip()
                    for m in re.finditer(r"\b(icmp|tcp|udp|gre|esp|ip)\b", raw, re.IGNORECASE):
                        return m.group(1).lower()
                return ""
            # Find zone from fields or _raw (zones often only in _raw; format varies by vendor)
            import re as _re
            # Palo Alto TRAFFIC log CSV: ...,vsys1,from_zone,to_zone,... (e.g. ,vsys1,inside,outside,)
            def _get_palo_alto_zones(raw):
                if not isinstance(raw, str) or not raw:
                    return ("", "")
                m = _re.search(r",vsys\d*\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,", raw)
                if m:
                    return (m.group(1).strip(), m.group(2).strip())
                return ("", "")
            def _get_zone_from_raw(raw, src_side=True):
                if not isinstance(raw, str) or not raw:
                    return ""
                # Key=value or key: value
                key_pats = (
                    r"(?:from_zone|src_zone|source_zone|from zone|source zone)\s*[=:]\s*[\"']?([\w\-]+)[\"']?",
                    r"(?:to_zone|dest_zone|destination_zone|to zone|dest zone|destination zone)\s*[=:]\s*[\"']?([\w\-]+)[\"']?",
                )
                zone_word_pats = (
                    r"\bfrom\s+zone\s+[\"']?([\w\-]+)[\"']?",
                    r"\bto\s+zone\s+[\"']?([\w\-]+)[\"']?",
                )
                json_pats = (
                    r"[\"']from_zone[\"']\s*:\s*[\"']([^\"']+)[\"']",
                    r"[\"']to_zone[\"']\s*:\s*[\"']([^\"']+)[\"']",
                )
                if src_side:
                    for pat in (key_pats[0], zone_word_pats[0], json_pats[0]):
                        m = _re.search(pat, raw, _re.IGNORECASE)
                        if m:
                            return m.group(1).strip()
                else:
                    for pat in (key_pats[1], zone_word_pats[1], json_pats[1]):
                        m = _re.search(pat, raw, _re.IGNORECASE)
                        if m:
                            return m.group(1).strip()
                return ""
            def _get_src_zone(e, palo_src=""):
                if palo_src:
                    return palo_src
                v = _get(e, "src_zone", "source_zone", "from_zone", "src_zone_name")
                if v:
                    return v
                return _get_zone_from_raw(e.get("_raw") if isinstance(e, dict) else "", src_side=True)
            def _get_dest_zone(e, palo_dest=""):
                if palo_dest:
                    return palo_dest
                v = _get(e, "dest_zone", "destination_zone", "to_zone", "dest_zone_name")
                if v:
                    return v
                return _get_zone_from_raw(e.get("_raw") if isinstance(e, dict) else "", src_side=False)
            # Firewall/device name from fields or _raw (Palo Alto syslog: "timestamp hostname 1,CSV...")
            def _get_firewall(e):
                v = _get(e, "dvc_name", "host", "device", "firewall", "hostname", "DeviceName")
                if v:
                    return v
                raw = e.get("_raw") if isinstance(e, dict) else ""
                if isinstance(raw, str) and raw:
                    m = _re.search(r"\d{1,2}:\d{2}:\d{2}\s+(\S+)\s+\d,", raw)
                    if m:
                        return m.group(1).strip()
                return ""
            # Normalize to display fields only: time, firewall, vendor_product, src_ip, dst_ip, src_zone, dest_zone, protocol, port, action
            def row(e):
                raw = e.get("_raw") if isinstance(e, dict) else ""
                palo_src, palo_dest = _get_palo_alto_zones(raw) if raw else ("", "")
                return {
                    "time": _get(e, "_time", "time") or "",
                    "firewall": _get_firewall(e) or "",
                    "vendor_product": _get(e, "vendor_product", "product", "vendor") or "Palo Alto Networks Firewall",
                    "src_ip": _get(e, "src_ip", "src") or "",
                    "dst_ip": _get(e, "dest_ip", "dst_ip", "dst") or "",
                    "src_zone": _get_src_zone(e, palo_src) or "",
                    "dest_zone": _get_dest_zone(e, palo_dest) or "",
                    "protocol": _get_protocol(e) or "",
                    "port": _get(e, "port", "dest_port", "dport") or "",
                    "action": _get(e, "action", "Action") or "drop",
                }
            results = [row(e) for e in raw_results]
            out = {"ip_address": ip_address, "events": results, "count": len(results)}
            if raw_results and isinstance(raw_results[0], dict):
                out["sample_keys"] = list(raw_results[0].keys())
                raw_sample = raw_results[0].get("_raw", "")
                if isinstance(raw_sample, str) and raw_sample:
                    out["sample_raw"] = raw_sample[:800]
            return out
    except asyncio.TimeoutError:
        return {"ip_address": ip_address, "events": [], "error": "Splunk request timed out"}
    except Exception as e:
        import traceback
        print(f"DEBUG: Splunk search error: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        return {"ip_address": ip_address, "events": [], "error": str(e)}
    finally:
        await connector.close()


# ---------------------------------------------------------------------------
# MCP Tool
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_splunk_recent_denies(
    ip_address: str,
    limit: int = 100,
    earliest_time: str = "-24h"
) -> Dict[str, Any]:
    """
    Get the list of recent deny/denied events in Splunk for a given IP address.

    Use this tool when the user asks for "recent denies for an IP", "list of denies for [IP]",
    "deny events for IP", "Splunk denies for [IP]", or similar. Extract the IP address from the query.

    **Query variations (all → get_splunk_recent_denies; need one IP address):**
    - "recent denies for 10.0.0.1" / "list denies for 10.0.0.1" / "deny events for 10.0.0.1"
    - "Splunk denies for 192.168.1.1" / "show me denies for 10.0.0.250"
    - "what denials for 10.0.0.5?" / "firewall denies for 10.0.0.1"
    - "list all denies for IP 10.0.0.1" / "recent deny events 10.0.0.1"

    Args:
        ip_address: IP address to search for in deny events (e.g. "192.168.1.1", "10.0.0.5")
        limit: Maximum number of events to return (default 100)
        earliest_time: Splunk time range (default "-24h" for last 24 hours)

    Returns:
        dict: ip_address, events (list of Splunk event dicts), count, and optional error
    """
    print(f"DEBUG: get_splunk_recent_denies called with ip_address={ip_address}, limit={limit}, earliest_time={earliest_time}", file=sys.stderr, flush=True)
    return await _splunk_search_impl(ip_address, limit, earliest_time)
