"""
Splunk MCP tools – get_splunk_recent_denies, get_splunk_traffic_summary,
get_splunk_unique_destinations.

Fully isolated domain module; no cross-domain dependencies.
"""

import ssl
import asyncio
import aiohttp
from typing import Dict, Any, List

from tools.shared import mcp, SPLUNK_HOST, SPLUNK_PORT, SPLUNK_USER, SPLUNK_PASSWORD
from tools.shared import setup_logging

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Shared Splunk session helper
# ---------------------------------------------------------------------------

async def _run_spl(spl: str, ip_address: str) -> Dict[str, Any]:
    """
    Run arbitrary SPL against Splunk REST API.
    Returns {'results': [list of raw dicts]} or {'error': str}.
    """
    base_url = f"https://{SPLUNK_HOST}:{SPLUNK_PORT}"
    ssl_ctx = ssl.create_default_context()
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            # Login
            login_url = f"{base_url}/services/auth/login"
            async with session.post(
                login_url,
                data={"username": SPLUNK_USER, "password": SPLUNK_PASSWORD},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {"error": f"Splunk login failed: {resp.status} - {text[:500]}"}
                text = await resp.text()
                import xml.etree.ElementTree as ET
                root = ET.fromstring(text)
                sk_el = root.find(".//sessionKey")
                if sk_el is None or sk_el.text is None:
                    return {"error": "Splunk login response had no sessionKey"}
                session_key = sk_el.text

            headers = {"Authorization": f"Splunk {session_key}"}

            # Create search job
            create_url = f"{base_url}/services/search/jobs"
            async with session.post(
                create_url,
                headers=headers,
                data={"search": spl, "output_mode": "json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status not in (200, 201):
                    text = await resp.text()
                    return {"error": f"Splunk create job failed: {resp.status} - {text[:500]}"}
                data = await resp.json()
                sid = data.get("sid")
                if not sid:
                    return {"error": "Splunk job created but no sid returned"}

            # Poll until done (max 120 seconds)
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

            # Fetch results
            results_url = f"{base_url}/services/search/jobs/{sid}/results"
            async with session.get(
                results_url,
                headers=headers,
                params={"output_mode": "json"},
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    return {"error": f"Splunk results failed: {resp.status} - {text[:500]}"}
                data = await resp.json()

            return {"results": data.get("results", []) if isinstance(data, dict) else []}

    except asyncio.TimeoutError:
        return {"error": "Splunk request timed out"}
    except Exception as e:
        import traceback
        logger.debug(f"Splunk SPL error: {e}\n{traceback.format_exc()}")
        return {"error": str(e)}
    finally:
        await connector.close()


# ---------------------------------------------------------------------------
# Field normalization helpers (used by get_splunk_recent_denies)
# ---------------------------------------------------------------------------

import re as _re


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


def _get_protocol(e):
    v = _get(e, "protocol", "Protocol", "proto", "transport")
    if v:
        return v
    if isinstance(e, dict):
        for k, val in e.items():
            if k and "protocol" in k.lower() and val not in (None, ""):
                return val
    raw = e.get("_raw") if isinstance(e, dict) else ""
    if isinstance(raw, str) and raw:
        for m in _re.finditer(r"(?:protocol|proto)\s*[=:]\s*(\w+)", raw, _re.IGNORECASE):
            return m.group(1).strip()
        for m in _re.finditer(r"\b(icmp|tcp|udp|gre|esp|ip)\b", raw, _re.IGNORECASE):
            return m.group(1).lower()
    return ""


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


def _normalize_event(e):
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


# ---------------------------------------------------------------------------
# MCP Tool 1: recent deny events
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_splunk_recent_denies(
    ip_address: str,
    limit: int = 100,
    earliest_time: str = "-24h"
) -> Dict[str, Any]:
    """
    Search Splunk for recent firewall deny events involving a given IP address.

    Use for: queries asking for deny/denial events for an IP — "recent denies for X", "deny events for X".
    Do NOT use for: path queries, device/rack lookups, address group lookups.

    Args:
        ip_address: IP address to search for in deny events
        limit: Maximum number of events to return (default 100)
        earliest_time: Splunk time range (default "-24h" for last 24 hours)

    Returns:
        dict: ip_address, events (list), count, and optional error
    """
    logger.debug(f"get_splunk_recent_denies: ip={ip_address}")
    spl = (
        f'search index=* (deny OR denied) '
        f'(src_ip="{ip_address}" OR dest_ip="{ip_address}" OR src="{ip_address}" OR dst="{ip_address}" '
        f'OR src_ip={ip_address} OR dest_ip={ip_address}) '
        f'earliest={earliest_time} | head {limit}'
    )
    out = await _run_spl(spl, ip_address)
    if "error" in out:
        return {"ip_address": ip_address, "events": [], "count": 0, "error": out["error"]}

    raw_results = out["results"]
    results = [_normalize_event(e) for e in raw_results]
    ret = {"ip_address": ip_address, "events": results, "count": len(results)}
    if raw_results and isinstance(raw_results[0], dict):
        ret["sample_keys"] = list(raw_results[0].keys())
        raw_sample = raw_results[0].get("_raw", "")
        if isinstance(raw_sample, str) and raw_sample:
            ret["sample_raw"] = raw_sample[:800]
    return ret


# ---------------------------------------------------------------------------
# MCP Tool 2: traffic summary (allow vs deny counts by action)
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_splunk_traffic_summary(
    ip_address: str,
    earliest_time: str = "-24h"
) -> Dict[str, Any]:
    """
    Summarize all firewall traffic for a given IP by action (allow/deny/drop) over the time window.
    Useful for understanding whether an IP is actively communicating or just being blocked.

    Args:
        ip_address: IP address to summarize traffic for
        earliest_time: Splunk time range (default "-24h")

    Returns:
        dict: ip_address, by_action (list of {action, count}), total_events, and optional error
    """
    logger.debug(f"get_splunk_traffic_summary: ip={ip_address}")
    spl = (
        f'search index=* '
        f'(src_ip="{ip_address}" OR dest_ip="{ip_address}" OR src="{ip_address}" OR dst="{ip_address}") '
        f'earliest={earliest_time} '
        f'| stats count by action '
        f'| sort -count'
    )
    out = await _run_spl(spl, ip_address)
    if "error" in out:
        return {"ip_address": ip_address, "by_action": [], "total_events": 0, "error": out["error"]}

    raw = out["results"]
    by_action = [
        {"action": r.get("action", "unknown"), "count": int(r.get("count", 0))}
        for r in raw if isinstance(r, dict)
    ]
    total = sum(a["count"] for a in by_action)
    return {"ip_address": ip_address, "by_action": by_action, "total_events": total}


# ---------------------------------------------------------------------------
# MCP Tool 3: unique destination spread (scan / lateral movement indicator)
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_splunk_unique_destinations(
    ip_address: str,
    earliest_time: str = "-24h"
) -> Dict[str, Any]:
    """
    Count unique destination IPs and ports contacted by the given IP.
    High unique port counts indicate port scanning; high unique IP counts indicate lateral movement.

    Args:
        ip_address: Source IP to analyse outbound spread for
        earliest_time: Splunk time range (default "-24h")

    Returns:
        dict: ip_address, unique_dest_ips, unique_dest_ports, top_ports (list), and optional error
    """
    logger.debug(f"get_splunk_unique_destinations: ip={ip_address}")
    spl = (
        f'search index=* (src_ip="{ip_address}" OR src="{ip_address}") '
        f'earliest={earliest_time} '
        f'| stats dc(dest_ip) as unique_dest_ips dc(dest_port) as unique_dest_ports '
        f'values(dest_port) as ports by src_ip'
    )
    out = await _run_spl(spl, ip_address)
    if "error" in out:
        return {"ip_address": ip_address, "unique_dest_ips": 0, "unique_dest_ports": 0, "top_ports": [], "error": out["error"]}

    raw = out["results"]
    if not raw or not isinstance(raw[0], dict):
        return {"ip_address": ip_address, "unique_dest_ips": 0, "unique_dest_ports": 0, "top_ports": []}

    r = raw[0]
    ports_val = r.get("ports", "")
    top_ports = [p.strip() for p in ports_val.split(",") if p.strip()][:20] if isinstance(ports_val, str) else []

    return {
        "ip_address": ip_address,
        "unique_dest_ips": int(r.get("unique_dest_ips", 0) or 0),
        "unique_dest_ports": int(r.get("unique_dest_ports", 0) or 0),
        "top_ports": top_ports,
    }
