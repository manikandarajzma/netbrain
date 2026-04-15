"""
Atlas tool registry — every LangChain @tool in one place.

All tools follow the same contract:
  - Take well-typed arguments the LLM can fill directly.
  - Accept an injected RunnableConfig (not shown to LLM) for session_id.
  - Write structured side-effect data to the per-session store so the caller
    can attach path_hops / interface_counters to the final API response.
  - Return a human-readable string for the LLM to reason about.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from datetime import date, datetime
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.tools.resilience import retry_async, CircuitBreaker, CircuitOpenError
    from atlas.mcp_client import call_mcp_tool
    from atlas.services.memory_manager import memory_manager
except ImportError:
    from tools.resilience import retry_async, CircuitBreaker, CircuitOpenError
    from mcp_client import call_mcp_tool
    from services.memory_manager import memory_manager  # type: ignore

logger = logging.getLogger("atlas.tools")

NORNIR_AGENT_URL   = "http://localhost:8006"
PANORAMA_AGENT_URL = "http://localhost:8003"
SPLUNK_AGENT_URL   = "http://localhost:8002"

_STUB = os.getenv("STUB_UNREACHABLE_AGENTS", "").lower() in ("1", "true", "yes")

# ---------------------------------------------------------------------------
# Per-session structured data store
# Keyed by session_id — safe under async concurrency because each request
# has a unique session_id.  Cleaned up by pop_session_data() after the
# agent finishes.
# ---------------------------------------------------------------------------

_session_store: dict[str, dict[str, Any]] = {}
_REDIS_CLIENT = None
_REDIS_CHECKED = False
_RUN_CACHE_TTL = 600


def _store(session_id: str) -> dict[str, Any]:
    return _session_store.setdefault(session_id, {
        "path_hops":          [],
        "reverse_path_hops":  [],
        "interface_counters": [],
        "routing_history":    {},
        "ping_results":       [],
        "peering_inspections": [],
        "all_interfaces":     {},
        "interface_details":  {},
        "syslog":             {},
        "protocol_discovery": {},
        "connectivity_snapshot": {},
        "ip_owners":          {},
        "servicenow_summary": "",
    })


def _set_servicenow_summary(session_id: str, summary: str) -> str:
    _store(session_id)["servicenow_summary"] = summary
    return summary


def _backend_unavailable(backend: str, action: str, detail: Any, *, subject: str = "") -> str:
    reason = str(detail or "unknown error").strip() or "unknown error"
    context = f" for {subject}" if subject else ""
    return f"{backend} unavailable during {action}{context}: {reason}"


def pop_session_data(session_id: str) -> dict[str, Any]:
    """Read and remove all side-effect data after the agent completes."""
    return _session_store.pop(session_id, {})


def _sid(config: RunnableConfig) -> str:
    return (config or {}).get("configurable", {}).get("session_id", "default")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _get_redis():
    global _REDIS_CLIENT, _REDIS_CHECKED
    if _REDIS_CHECKED:
        return _REDIS_CLIENT
    _REDIS_CHECKED = True
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis
        client = redis.from_url(url, decode_responses=True)
        client.ping()
        _REDIS_CLIENT = client
    except Exception:
        _REDIS_CLIENT = None
    return _REDIS_CLIENT


def _run_cache_key(session_id: str, endpoint: str, payload: dict[str, Any]) -> str:
    blob = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True, default=str)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"atlas:run_cache:{session_id}:{digest}"


def _run_cache_index_key(session_id: str) -> str:
    return f"atlas:run_cache:{session_id}:keys"


def clear_session_cache(session_id: str) -> None:
    r = _get_redis()
    if not r or not session_id:
        return
    try:
        index_key = _run_cache_index_key(session_id)
        keys = list(r.smembers(index_key) or [])
        if keys:
            r.delete(*keys)
        r.delete(index_key)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers — not exposed to the LLM
# ---------------------------------------------------------------------------

async def _push_status(session_id: str, msg: str) -> None:
    try:
        try:
            import atlas.status_bus as sb
        except ImportError:
            import status_bus as sb  # type: ignore
        await sb.push(session_id, msg)
    except Exception:
        pass


async def _nornir_post_once(endpoint: str, payload: dict, timeout: float = 30.0) -> dict:
    """
    Single-shot POST to the Nornir HTTP backend with no retries.
    Use this for latency-sensitive live path tracing where a miss should fail fast.
    """
    url = f"{NORNIR_AGENT_URL}/{endpoint.lstrip('/')}"
    async with httpx.AsyncClient(timeout=timeout) as c:
        r = await c.post(url, json=payload)
        r.raise_for_status()
        return r.json()


async def _nornir_post(endpoint: str, payload: dict, timeout: float = 30.0) -> dict:
    """
    POST to the Nornir HTTP backend with circuit-breaker + retry.
    Every Nornir tool call should go through here — never call httpx directly.
    Raises on failure; callers catch and return an error string.
    """
    url = f"{NORNIR_AGENT_URL}/{endpoint.lstrip('/')}"
    cb  = CircuitBreaker.for_endpoint(url)

    async def _do() -> dict:
        return await _nornir_post_once(endpoint, payload, timeout=timeout)

    return await retry_async(
        cb, _do,
        retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError),
    )


async def _cached_nornir_post(
    session_id: str,
    endpoint: str,
    payload: dict,
    timeout: float = 30.0,
    retries: bool = True,
) -> dict:
    """
    Redis-backed per-session cache for read-only Nornir calls.
    Cache is scoped to the current Atlas session_id and explicitly cleared
    when the graph run completes.
    """
    r = _get_redis()
    post_fn = _nornir_post if retries else _nornir_post_once

    if not r or not session_id:
        return await post_fn(endpoint, payload, timeout=timeout)

    cache_key = _run_cache_key(session_id, endpoint, payload)
    try:
        cached = r.get(cache_key)
        if cached:
            data = json.loads(cached)
            if isinstance(data, dict):
                return data
    except Exception:
        pass

    data = await post_fn(endpoint, payload, timeout=timeout)
    try:
        r.setex(cache_key, _RUN_CACHE_TTL, json.dumps(_json_safe(data)))
        r.sadd(_run_cache_index_key(session_id), cache_key)
        r.expire(_run_cache_index_key(session_id), _RUN_CACHE_TTL)
    except Exception:
        pass
    return data


_PLATFORM_TO_TYPE = {
    "arista_eos":  "arista switch",
    "cisco_ios":   "cisco router",
    "cisco_nxos":  "cisco switch",
    "cisco_iosxr": "cisco router",
}

_HOSTNAME_RE = re.compile(r'^[A-Za-z0-9]([A-Za-z0-9\-_\.]*[A-Za-z0-9])?$')


async def _live_path_trace(src_ip: str, dst_ip: str, session_id: str = "") -> tuple[str, list[dict], dict]:
    """
    Hop-by-hop live path trace via SSH.  No database reads.

    Returns:
        text_summary   — human-readable for the LLM
        structured_hops — list of hop dicts for PathVisualization
        flags           — anomaly flags (mgmt_routing_detected, no_route_device, …)
    """
    flags: dict[str, Any] = {}

    store = _store(session_id) if session_id else {}

    async def _get_ip_owners() -> dict[str, Any]:
        owners = store.get("ip_owners") if isinstance(store, dict) else None
        if owners:
            return owners
        try:
            data = await _cached_nornir_post(
                session_id,
                "/ip-owners",
                {"devices": []},
                timeout=8.0,
                retries=False,
            )
            owners = data.get("owners") if isinstance(data, dict) else {}
        except Exception:
            owners = {}
        if isinstance(store, dict):
            store["ip_owners"] = owners or {}
        return owners or {}

    async def _find_device(ip: str) -> dict:
        owners = await _get_ip_owners()
        owner = owners.get(ip) if isinstance(owners, dict) else None
        if isinstance(owner, dict) and owner.get("device"):
            return {"found": True, **owner}
        try:
            result = await _cached_nornir_post(
                session_id,
                "/find-device",
                {"ip": ip},
                timeout=5.0,
                retries=False,
            )
            if isinstance(result, dict) and result.get("found") and result.get("device"):
                if isinstance(store, dict):
                    cached = store.setdefault("ip_owners", {})
                    if isinstance(cached, dict):
                        cached[ip] = {
                            "device": result.get("device"),
                            "interface": result.get("interface"),
                            "host": result.get("host"),
                            "port": result.get("port"),
                            "ip": ip,
                        }
                return result
        except Exception:
            pass
        return {"found": False, "ip": ip}

    async def _find_first_hop(src: str) -> tuple[str, str | None]:
        try:
            async with httpx.AsyncClient(timeout=25.0) as c:
                resp = await c.get(f"{NORNIR_AGENT_URL}/devices")
                devices = resp.json().get("devices", []) if resp.status_code == 200 else []
        except Exception:
            devices = []
        if not devices:
            try:
                import yaml
                from pathlib import Path
                hf = Path(__file__).resolve().parent.parent / "nornir" / "inventory" / "hosts.yaml"
                with open(hf) as f:
                    devices = list(yaml.safe_load(f).keys())
            except Exception:
                pass
        for device in devices:
            try:
                route = await _cached_nornir_post(
                    session_id,
                    "/route",
                    {"device": device, "destination": src},
                    timeout=5.0,
                    retries=False,
                )
                if route.get("found") and route.get("protocol", "").lower() == "connected":
                    return device, route.get("egress_interface")
            except Exception:
                continue
        return "", None

    current_device, gw_iface = await _find_first_hop(src_ip)
    if not current_device:
        return (
            f"Could not find a network device with a connected route to {src_ip} — check inventory.",
            [], flags,
        )

    await _get_ip_owners()

    text_hops: list[str] = []
    structured_hops: list[dict] = []
    seen: set[str] = set()
    MAX_HOPS = 15

    structured_hops.append({
        "from_device": src_ip, "from_device_type": "host",
        "out_interface": None, "out_zone": None, "device_group": None,
        "to_device": current_device, "to_device_type": "switch",
        "in_interface": gw_iface, "in_zone": None,
    })

    for _ in range(MAX_HOPS):
        if current_device in seen:
            text_hops.append(f"  !! Routing loop at {current_device}")
            break
        seen.add(current_device)

        try:
            route = await _cached_nornir_post(
                session_id,
                "/route",
                {"device": current_device, "destination": dst_ip},
                timeout=5.0,
                retries=False,
            )
        except Exception as exc:
            text_hops.append(f"  Hop {len(text_hops)+1}: {current_device} — SSH error: {exc}")
            break

        if not route.get("found"):
            text_hops.append(f"  ⚠️  Hop {len(text_hops)+1}: {current_device} — no route to {dst_ip}")
            flags["no_route_device"] = current_device
            break

        egress   = route.get("egress_interface") or ""
        next_hop = route.get("next_hop")
        protocol = route.get("protocol") or ""
        prefix   = route.get("prefix") or ""

        if egress.lower().startswith("management"):
            text_hops.append(
                f"  ⚠️  Hop {len(text_hops)+1}: **{current_device}** routing {dst_ip} via "
                f"**{egress}** (default 0.0.0.0/0) — data-plane is likely DOWN."
            )
            structured_hops.append({
                "from_device": current_device, "from_device_type": "switch",
                "out_interface": egress, "out_zone": None, "device_group": None,
                "to_device": f"⚠️ Mgmt fallback ({egress})", "to_device_type": "host",
                "in_interface": None, "in_zone": None,
            })
            flags["mgmt_routing_detected"]  = True
            flags["mgmt_routing_device"]    = current_device
            break

        text_hops.append(
            f"  Hop {len(text_hops)+1}: {current_device} | Egress: {egress} | "
            f"Protocol: {protocol} | Prefix: {prefix} | Next-hop: {next_hop or 'directly connected'}"
        )

        if not next_hop and str(protocol).lower() in {"connected", "local"}:
            try:
                async with httpx.AsyncClient(timeout=10.0) as c:
                    r = await c.post(f"{NORNIR_AGENT_URL}/arp",
                                     json={"device": current_device, "ip": dst_ip})
                    arp = r.json()
            except Exception:
                arp = {}
            in_iface = arp.get("interface") if arp.get("found") else None
            text_hops.append(f"  Destination {dst_ip} reachable via ARP on {current_device}")
            structured_hops.append({
                "from_device": current_device, "from_device_type": "switch",
                "out_interface": egress, "out_zone": None, "device_group": None,
                "to_device": dst_ip, "to_device_type": "host",
                "in_interface": in_iface, "in_zone": None,
            })
            break

        if not next_hop:
            text_hops.append(
                f"  ⚠️  Hop {len(text_hops)+1}: {current_device} returned route {prefix} via {egress} "
                f"({protocol}) but no resolvable next-hop was provided"
            )
            flags["missing_next_hop_device"] = current_device
            flags["missing_next_hop_prefix"] = prefix
            break

        next_dev = await _find_device(next_hop)
        if not next_dev.get("found"):
            text_hops.append(f"  Next-hop {next_hop} not found — path ends here")
            flags["next_hop_resolution_failed"] = next_hop
            break

        next_device = next_dev["device"]
        in_iface    = next_dev["interface"]
        structured_hops.append({
            "from_device": current_device, "from_device_type": "switch",
            "out_interface": egress, "out_zone": None, "device_group": None,
            "to_device": next_device, "to_device_type": "switch",
            "in_interface": in_iface, "in_zone": None,
        })
        current_device = next_device

    devices_in_path = list(seen)
    text = (
        f"Path from {src_ip} to {dst_ip} (live SSH):\n"
        + "\n".join(text_hops)
        + f"\n\nAll devices in path: {', '.join(devices_in_path)}"
    )
    return text, structured_hops, flags


def _extract_path_metadata(hops: list[dict]) -> dict:
    """
    Extract first_hop, last_hop, path_devices, has_firewalls from structured hops.
    Returns a dict stored in the session store under 'path_meta'.
    """
    meta: dict[str, Any] = {
        "path_devices":              [],
        "first_hop_device":          "",
        "first_hop_lan_interface":   "",
        "first_hop_egress_interface": "",
        "last_hop_device":           "",
        "last_hop_egress_interface": "",
        "has_firewalls":             False,
        "firewall_hostnames":        [],
        "path_hops_for_counters":    [],
    }
    if not hops:
        return meta

    # Clean hostname list
    seen_devs: set[str] = set()
    devices: list[str] = []
    for h in hops:
        for key in ("from_device", "to_device"):
            d = h.get(key, "")
            if d and d not in seen_devs and _HOSTNAME_RE.match(d):
                seen_devs.add(d)
                devices.append(d)
    meta["path_devices"] = devices

    # First hop: structured_hops[0] is src_ip → first_router
    first_h = hops[0]
    meta["first_hop_device"]          = first_h.get("to_device", "")
    meta["first_hop_lan_interface"]   = first_h.get("in_interface", "") or ""
    first_dev = meta["first_hop_device"]
    for h in hops:
        if h.get("from_device") == first_dev and h.get("out_interface"):
            meta["first_hop_egress_interface"] = h["out_interface"]
            break

    # Last hop: from_device of the last hop where to_device is an IP
    _ip_re = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')
    for h in reversed(hops):
        if _ip_re.match(h.get("to_device", "")) and not _ip_re.match(h.get("from_device", "")):
            meta["last_hop_device"]           = h.get("from_device", "")
            meta["last_hop_egress_interface"] = h.get("out_interface", "")
            break

    # Interfaces for counter polling
    dev_intfs: dict[str, set[str]] = {}
    for h in hops:
        for dev_key, intf_key in (("from_device", "out_interface"), ("to_device", "in_interface")):
            d = h.get(dev_key, "")
            i = h.get(intf_key, "")
            if d and not _ip_re.match(d) and i:
                dev_intfs.setdefault(d, set()).add(i)
    meta["path_hops_for_counters"] = [
        {"device": d, "interfaces": sorted(intfs)}
        for d, intfs in dev_intfs.items()
        if intfs
    ]

    return meta


def _extract_reverse_path_metadata(hops: list[dict]) -> dict:
    """
    Extract reverse-side metadata from a reverse trace where structured_hops[0]
    is dest_ip -> first_router_on_return_path.
    """
    meta: dict[str, Any] = {
        "reverse_first_hop_device": "",
        "reverse_first_hop_lan_interface": "",
        "reverse_first_hop_egress_interface": "",
        "reverse_last_hop_device": "",
        "reverse_last_hop_egress_interface": "",
    }
    if not hops:
        return meta

    _ip_re = re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$')

    first_h = hops[0]
    meta["reverse_first_hop_device"] = first_h.get("to_device", "")
    meta["reverse_first_hop_lan_interface"] = first_h.get("in_interface", "") or ""
    first_dev = meta["reverse_first_hop_device"]
    for h in hops:
        if h.get("from_device") == first_dev and h.get("out_interface"):
            meta["reverse_first_hop_egress_interface"] = h["out_interface"]
            break

    for h in reversed(hops):
        if _ip_re.match(h.get("to_device", "")) and not _ip_re.match(h.get("from_device", "")):
            meta["reverse_last_hop_device"] = h.get("from_device", "")
            meta["reverse_last_hop_egress_interface"] = h.get("out_interface", "")
            break

    return meta


async def _infer_vrf(src_ip: str, device: str) -> str:
    """Query DB for the VRF containing src_ip on the given device."""
    if not src_ip or not device:
        return "default"
    try:
        try:
            from atlas.db import fetchrow
        except ImportError:
            from db import fetchrow  # type: ignore
        row = await fetchrow(
            "SELECT vrf FROM routing_table WHERE device=$1 AND $2::inet << prefix "
            "ORDER BY masklen(prefix) DESC LIMIT 1",
            device, src_ip,
        )
        if row and row["vrf"]:
            return row["vrf"]
    except Exception:
        pass
    return "default"


# ---------------------------------------------------------------------------
# Path tracing tools
# ---------------------------------------------------------------------------

@tool
async def trace_path(source_ip: str, dest_ip: str, config: RunnableConfig) -> str:
    """
    Trace the hop-by-hop network path from source_ip to dest_ip via live SSH.
    Always call this FIRST for any connectivity troubleshooting query.
    Returns the path text including all device names, egress interfaces, and
    any anomalies (management-routing fallback, interface down, no route).

    The result tells you:
    - All device hostnames in the path (use these for ServiceNow, OSPF checks, etc.)
    - The first-hop device (use for ping source)
    - The last-hop device (use for TCP test source)
    - Any firewalls in the path (use for Panorama check)
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Tracing path {source_ip} → {dest_ip} via live SSH...")

    text, hops, flags = await _live_path_trace(source_ip, dest_ip, session_id=session_id)

    store = _store(session_id)
    store["path_hops"] = hops
    store["path_flags"] = flags
    meta = _extract_path_metadata(hops)
    store["path_meta"] = meta

    # Infer source VRF
    meta["src_vrf"] = await _infer_vrf(source_ip, meta.get("first_hop_device", ""))

    # Append structured metadata to text so LLM can use it
    if meta["first_hop_device"]:
        text += f"\n\nfirst_hop_device: {meta['first_hop_device']}"
        text += f"\nfirst_hop_lan_interface: {meta['first_hop_lan_interface']}"
        text += f"\nfirst_hop_egress_interface: {meta['first_hop_egress_interface']}"
    if meta["last_hop_device"]:
        text += f"\nlast_hop_device: {meta['last_hop_device']}"
        text += f"\nlast_hop_egress_interface: {meta['last_hop_egress_interface']}"
    text += f"\npath_devices: {', '.join(meta['path_devices'])}"
    text += f"\nsrc_vrf: {meta['src_vrf']}"

    return text


@tool
async def trace_reverse_path(source_ip: str, dest_ip: str, config: RunnableConfig) -> str:
    """
    Trace the return path from dest_ip back to source_ip via live SSH.
    Call in parallel with search_servicenow and get_interface_counters after trace_path.
    Reveals asymmetric routing.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Tracing return path {dest_ip} → {source_ip}...")

    text, hops, _ = await _live_path_trace(dest_ip, source_ip, session_id=session_id)

    if hops:
        store = _store(session_id)
        store["reverse_path_hops"] = hops
        meta = _extract_reverse_path_metadata(hops)
        store["reverse_path_meta"] = meta
        if meta["reverse_first_hop_device"]:
            text += f"\n\nreverse_first_hop_device: {meta['reverse_first_hop_device']}"
            text += f"\nreverse_first_hop_lan_interface: {meta['reverse_first_hop_lan_interface']}"
            text += f"\nreverse_first_hop_egress_interface: {meta['reverse_first_hop_egress_interface']}"
        if meta["reverse_last_hop_device"]:
            text += f"\nreverse_last_hop_device: {meta['reverse_last_hop_device']}"
            text += f"\nreverse_last_hop_egress_interface: {meta['reverse_last_hop_egress_interface']}"

    return text


# ---------------------------------------------------------------------------
# Active test tools
# ---------------------------------------------------------------------------

@tool
async def ping_device(
    device: str,
    destination: str,
    config: RunnableConfig,
    source_interface: str = "",
    vrf: str = "",
) -> str:
    """
    Send ICMP pings from a network device to a destination IP via live SSH.

    Forward ping:  device=first_hop_device, destination=dest_ip, source_interface=first_hop_lan_interface, vrf=src_vrf
    Reverse ping:  device=reverse_first_hop_device,  destination=src_ip,  source_interface=reverse_first_hop_lan_interface,  vrf=dst_vrf

    Returns: success/failure, packet loss %, RTT.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Pinging {device} → {destination}...")

    store = _store(session_id)
    path_meta = store.get("path_meta") or {}
    reverse_meta = store.get("reverse_path_meta") or {}
    resolved_source_interface = source_interface
    if not resolved_source_interface:
        if device == path_meta.get("first_hop_device"):
            resolved_source_interface = str(path_meta.get("first_hop_lan_interface") or "").strip()
        elif device == reverse_meta.get("reverse_first_hop_device"):
            resolved_source_interface = str(reverse_meta.get("reverse_first_hop_lan_interface") or "").strip()
        elif device == path_meta.get("last_hop_device"):
            resolved_source_interface = str(path_meta.get("last_hop_egress_interface") or "").strip()

    cb = CircuitBreaker.for_endpoint(NORNIR_AGENT_URL + "/ping")
    async def _do():
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{NORNIR_AGENT_URL}/ping",
                             json={"device": device, "destination": destination,
                                   "source_interface": resolved_source_interface, "vrf": vrf})
            r.raise_for_status()
            return r.json()
    try:
        result = await retry_async(cb, _do,
            retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError))
    except Exception as exc:
        result = {"success": False, "error": str(exc)}

    store["ping_result"] = result
    store["ping_results"].append({
        "device": device,
        "destination": destination,
        "source_interface": resolved_source_interface,
        "vrf": vrf or "default",
        **result,
    })

    src_note = f", source {resolved_source_interface}" if resolved_source_interface else ""
    if result.get("success"):
        rtt = result.get("rtt_avg_ms")
        rtt_str = f", RTT avg {rtt}ms" if rtt else ""
        return f"✓ Ping {device} → {destination} (VRF: {vrf or 'default'}{src_note}): SUCCESS, 0% loss{rtt_str}"
    loss = result.get("loss_pct", 100)
    err  = result.get("error", "")
    return f"✗ Ping {device} → {destination} (VRF: {vrf or 'default'}{src_note}): FAILED — {loss}% packet loss{(' — ' + err) if err else ''}"


@tool
async def test_tcp_port(
    device: str,
    destination: str,
    port: str,
    config: RunnableConfig,
    vrf: str = "",
) -> str:
    """
    Test TCP reachability to destination:port from a network device via live SSH.
    Use last_hop_device from trace_path as the device — it is closest to the destination.
    Call this when ping passes but application connectivity is still failing.

    Returns: reachable (True/False) and error details if unreachable.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Testing TCP {device} → {destination}:{port}...")

    try:
        result = await _nornir_post("/tcp-test",
                                    {"device": device, "destination": destination,
                                     "port": int(port), "vrf": vrf},
                                    timeout=30.0)
    except Exception as exc:
        result = {"reachable": False, "error": str(exc)}

    _store(session_id)["tcp_result"] = result

    if result.get("reachable"):
        return f"✓ TCP {destination}:{port} is reachable from {device} — service is accepting connections."
    err = result.get("error", "connection refused or timed out")
    return f"✗ TCP {destination}:{port} is NOT reachable from {device} — {err}"


@tool
async def check_routing(
    devices: list[str],
    destination: str,
    config: RunnableConfig,
    vrf: str = "",
) -> str:
    """
    Check the routing table on multiple devices for a destination IP via live SSH.
    Call this when ping fails to identify which hop loses the route.
    Pass all path devices from trace_path.

    Returns: per-device route info (next-hop, egress interface, VRF, protocol).
    """
    session_id = _sid(config)
    devs_str = ", ".join(devices)
    await _push_status(session_id, f"Checking routing on {devs_str}...")

    cb = CircuitBreaker.for_endpoint(NORNIR_AGENT_URL + "/routing-check")
    async def _do():
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{NORNIR_AGENT_URL}/routing-check",
                             json={"devices": devices, "destination": destination, "vrf": vrf})
            r.raise_for_status()
            return r.json()
    try:
        result = await retry_async(cb, _do,
            retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError))
    except Exception as exc:
        return _backend_unavailable("Nornir", "routing check", exc)

    lines = [f"Routing check for {destination}:"]
    for device, info in (result.get("hops") or {}).items():
        if not info.get("found"):
            lines.append(f"  {device}: ✗ no route — {info.get('error', 'no match')}")
        else:
            lines.append(
                f"  {device}: ✓ via {info.get('next_hop','directly connected')} "
                f"egress {info.get('interface','?')} ({info.get('protocol','?')}, VRF {info.get('vrf','default')})"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interface diagnostic tools
# ---------------------------------------------------------------------------

@tool
async def get_interface_counters(
    devices_and_interfaces: list[dict],
    config: RunnableConfig,
) -> str:
    """
    Poll interface error and discard counters 3× over ~9 seconds on path interfaces.
    Returns ONLY actively incrementing counters — clean interfaces are suppressed.
    Call in parallel with search_servicenow after trace_path.

    Args:
        devices_and_interfaces: List of {"device": str, "interfaces": [str]} dicts.
            Use path_hops_for_counters from the trace_path output.
            Example: [{"device": "arista1", "interfaces": ["Ethernet1", "Ethernet3"]}]
    """
    import aiohttp

    session_id = _sid(config)

    valid = []
    for entry in devices_and_interfaces:
        if not isinstance(entry, dict):
            continue
        device = str(entry.get("device", "")).strip()
        interfaces = entry.get("interfaces")
        if not device or not interfaces:
            continue
        valid.append({"device": device, "interfaces": interfaces})
    if not valid:
        return "No interface data available for counter polling."

    devs_str = ", ".join(e["device"] for e in valid)
    await _push_status(session_id, f"Polling interface counters: {devs_str}...")

    async def _fetch_one(session, entry: dict) -> tuple[str, dict]:
        device = entry["device"]
        interfaces = entry["interfaces"]
        try:
            async with session.post(
                f"{NORNIR_AGENT_URL}/interface-counters",
                json={"device": device, "interfaces": interfaces},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
        except Exception as exc:
            return f"{device}: unreachable ({exc})", {"device": device, "ssh_error": str(exc), "active": [], "clean": []}

        if "error" in data:
            return f"{device}: {data['error']}", {"device": device, "ssh_error": data["error"], "active": [], "clean": []}

        active   = data.get("active_errors", [])
        clean    = data.get("clean_interfaces", [])
        interval = data.get("poll_interval_s", 3)
        iters    = data.get("iterations", 3)
        window   = interval * (iters - 1)
        structured = {"device": device, "window_s": window, "active": active, "clean": clean}

        if not active:
            return f"{device}: all interfaces clean over {window}s", structured
        rows = []
        for c in active:
            intf = c.get("interface", "?")
            d    = c.get("delta_9s", {})
            parts = [f"{k}+{v}" for k, v in d.items() if v > 0]
            rows.append(f"  {intf}: ACTIVE — {', '.join(parts)} over {window}s")
        return f"{device}:\n" + "\n".join(rows), structured

    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[_fetch_one(session, e) for e in valid])

    store = _store(session_id)
    lines = []
    for text, structured in results:
        if text:
            lines.append(text)
        if structured and structured.get("device"):
            store["interface_counters"].append(structured)

    return "Interface counters:\n" + "\n".join(lines) if lines else "No counter data."


@tool
async def get_interface_detail(
    device: str,
    interface: str,
    config: RunnableConfig,
) -> str:
    """
    Fetch full operational status and error counters for a specific interface via live SSH.
    Use when path trace shows an interface DOWN or when you need oper_status details.

    Returns: line-protocol, oper_status, input/output errors, description.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Checking interface {device}/{interface}...")

    try:
        data = await _nornir_post("/interface-detail",
                                  {"device": device, "interface": interface},
                                  timeout=15.0)
        _store(session_id)["interface_details"][f"{device}:{interface}"] = data
        return json.dumps(data, indent=2)
    except Exception as exc:
        return _backend_unavailable("Nornir", "interface detail lookup", exc, subject=f"{device}/{interface}")


@tool
async def get_all_interfaces(device: str, config: RunnableConfig) -> str:
    """
    List all non-management interfaces, their up/down state, and primary IP on a device.
    Use for device health queries or when you need to map an OSPF/syslog interface IP to
    a concrete interface and determine whether that interface is down.

    Returns: per-interface oper_status, line-protocol, description, and primary IP.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Getting all interfaces on {device}...")

    try:
        data = await _nornir_post("/all-interfaces-status", {"device": device}, timeout=15.0)
    except Exception as exc:
        return _backend_unavailable("Nornir", "interface inventory lookup", exc, subject=device)

    _store(session_id)["all_interfaces"][device] = data
    interfaces = data.get("interfaces", [])
    if not interfaces:
        return f"No interface data returned for {device}."
    down = [i for i in interfaces if not i.get("up")]
    lines = [f"{device}: {len(interfaces)} interfaces, {len(down)} DOWN"]
    for i in interfaces:
        oper = i.get("oper_status", "")
        ip = i.get("primary_ip")
        plen = i.get("prefix_len")
        ip_text = f" ip {ip}/{plen}" if ip and plen is not None else ""
        desc_text = f" ({i['description']})" if i.get("description") else ""
        if i.get("up"):
            if ip:
                lines.append(f"  ✓ {i['interface']} — up{ip_text}{desc_text}")
            continue
        status = "ADMIN-DOWN" if oper in ("disabled", "adminDown") else f"link-down ({oper})"
        lines.append(f"  ✗ {i['interface']} — {status}{ip_text}{desc_text}")
    return "\n".join(lines)


@tool
async def get_device_syslog(device: str, config: RunnableConfig, interface: str = "") -> str:
    """
    Fetch recent syslog messages from a device via live SSH.
    Filters to link/down/flap events.  Pass interface to scope to a specific port.
    Use when you need to determine WHEN an interface went down or OSPF dropped.

    Returns: timestamped syslog lines.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Fetching syslog from {device}...")

    try:
        data = await _nornir_post("/show-logging", {"device": device, "lines": 100}, timeout=15.0)
    except Exception as exc:
        return _backend_unavailable("Nornir", "syslog lookup", exc, subject=device)

    logs = data.get("logs", [])
    kw = ["link", "down", "flap", "err-disable", "lineproto", "ospf", "adjacency"]
    if interface:
        short = interface.replace("GigabitEthernet", "Gi").replace("Ethernet", "Et")
        relevant = [l for l in logs
                    if interface.lower() in l.lower()
                    or short.lower() in l.lower()
                    or any(k in l.lower() for k in kw)][-20:]
    else:
        relevant = [l for l in logs if any(k in l.lower() for k in kw)][-30:]

    if not relevant:
        return f"{device}: no relevant syslog events found."

    store = _store(session_id)
    interface_inventory = (store.get("all_interfaces") or {}).get(device, {})
    interface_rows = interface_inventory.get("interfaces", []) if isinstance(interface_inventory, dict) else []
    ip_to_interface: dict[str, dict[str, Any]] = {}
    for row in interface_rows:
        ip = str(row.get("primary_ip") or "").strip()
        if ip:
            ip_to_interface[ip] = row

    correlations: list[dict[str, Any]] = []
    correlation_lines: list[str] = []
    for line in relevant:
        lower = line.lower()
        if "adjacency" not in lower and "ospf" not in lower:
            continue
        for ip in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", line):
            row = ip_to_interface.get(ip)
            if not row:
                continue
            oper = row.get("oper_status", "unknown")
            state = "up" if row.get("up") else f"down ({oper})"
            correlation = {
                "ip": ip,
                "interface": row.get("interface"),
                "up": row.get("up"),
                "oper_status": oper,
                "line": line,
            }
            if correlation not in correlations:
                correlations.append(correlation)
                correlation_lines.append(
                    f"  Correlated OSPF syslog IP {ip} -> {row.get('interface')} ({state})"
                )

    store["syslog"][device] = {
        "logs": logs,
        "relevant": relevant,
        "interface": interface,
        "correlations": correlations,
    }
    body = [f"{device} syslog:"]
    body.extend(f"  {l}" for l in relevant)
    if correlation_lines:
        body.append("  OSPF interface correlation:")
        body.extend(correlation_lines)
    return "\n".join(body)


@tool
async def inspect_ospf_peering(
    device_a: str,
    interface_a: str,
    device_b: str,
    interface_b: str,
    config: RunnableConfig,
    ip_a: str = "",
    ip_b: str = "",
) -> str:
    """
    Inspect a specific OSPF peering end-to-end on both devices via live SSH.
    Use this when routing history identifies a concrete peering pair such as
    `ai3 Ethernet2 <-> ai4 Ethernet2`.

    Returns:
      - interface state and primary IP on both sides
      - interface counters on both sides
      - interface detail on both sides
      - recent syslog with OSPF/IP correlation on both sides
      - bilateral ping results across the peering IPs when provided
      - an explicit diagnosis class and recommended next action
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Inspecting OSPF peering {device_a}/{interface_a} <-> {device_b}/{interface_b}...")

    async def _all_interfaces(device: str) -> dict[str, Any]:
        try:
            data = await _nornir_post("/all-interfaces-status", {"device": device}, timeout=15.0)
        except Exception as exc:
            data = {"device": device, "interfaces": [], "error": str(exc)}
        _store(session_id)["all_interfaces"][device] = data
        return data

    async def _interface_detail(device: str, interface: str) -> dict[str, Any]:
        try:
            data = await _nornir_post("/interface-detail", {"device": device, "interface": interface}, timeout=15.0)
        except Exception as exc:
            data = {"device": device, "interface": interface, "error": str(exc)}
        _store(session_id)["interface_details"][f"{device}:{interface}"] = data
        return data

    async def _interface_counters(device: str, interface: str) -> dict[str, Any]:
        try:
            data = await _nornir_post("/interface-counters", {"device": device, "interfaces": [interface]}, timeout=20.0)
        except Exception as exc:
            return {"device": device, "ssh_error": str(exc), "active_errors": [], "clean_interfaces": []}

        structured = {
            "device": device,
            "window_s": data.get("poll_interval_s", 3) * max(0, data.get("iterations", 3) - 1),
            "active": data.get("active_errors", []),
            "clean": data.get("clean_interfaces", []),
        }
        _store(session_id)["interface_counters"].append(structured)
        return data

    async def _syslog(device: str) -> dict[str, Any]:
        try:
            data = await _nornir_post("/show-logging", {"device": device, "lines": 100}, timeout=15.0)
        except Exception as exc:
            data = {"device": device, "logs": [], "relevant": [], "error": str(exc)}
        logs = data.get("logs", [])
        kw = ["link", "down", "flap", "err-disable", "lineproto", "ospf", "adjacency"]
        relevant = [l for l in logs if any(k in l.lower() for k in kw)][-30:]
        return {"logs": logs, "relevant": relevant, **({"error": data["error"]} if data.get("error") else {})}

    async def _ping(device: str, destination: str, source_interface: str) -> dict[str, Any]:
        try:
            result = await _nornir_post(
                "/ping",
                {"device": device, "destination": destination, "source_interface": source_interface, "vrf": ""},
                timeout=30.0,
            )
        except Exception as exc:
            result = {"success": False, "error": str(exc), "device": device, "destination": destination}
        _store(session_id)["ping_results"].append({
            "device": device,
            "destination": destination,
            "source_interface": source_interface,
            "vrf": "default",
            **result,
        })
        return result

    all_a, all_b, detail_a, detail_b, counters_a, counters_b, syslog_a, syslog_b = await asyncio.gather(
        _all_interfaces(device_a),
        _all_interfaces(device_b),
        _interface_detail(device_a, interface_a),
        _interface_detail(device_b, interface_b),
        _interface_counters(device_a, interface_a),
        _interface_counters(device_b, interface_b),
        _syslog(device_a),
        _syslog(device_b),
    )

    def _pick_interface(data: dict[str, Any], interface: str) -> dict[str, Any]:
        for row in data.get("interfaces", []) or []:
            if row.get("interface") == interface:
                return row
        return {}

    row_a = _pick_interface(all_a, interface_a)
    row_b = _pick_interface(all_b, interface_b)
    live_ip_a = str(row_a.get("primary_ip") or ip_a or "").strip()
    live_ip_b = str(row_b.get("primary_ip") or ip_b or "").strip()

    def _correlate_syslog(device: str, row: dict[str, Any], syslog_data: dict[str, Any]) -> list[str]:
        ip = str(row.get("primary_ip") or "").strip()
        if not ip:
            return []
        lines = []
        for line in syslog_data.get("relevant", []) or []:
            if ip in line and ("adjacency" in line.lower() or "ospf" in line.lower()):
                oper = row.get("oper_status", "unknown")
                state = "up" if row.get("up") else f"down ({oper})"
                lines.append(f"{device}: syslog local IP {ip} belongs to {row.get('interface')} ({state})")
        return lines

    correlations = _correlate_syslog(device_a, row_a, syslog_a) + _correlate_syslog(device_b, row_b, syslog_b)
    _store(session_id)["syslog"][device_a] = {**syslog_a, "interface": interface_a, "correlations": correlations}
    _store(session_id)["syslog"][device_b] = {**syslog_b, "interface": interface_b, "correlations": correlations}

    ping_a = ping_b = None
    if live_ip_b and row_a.get("up"):
        ping_a = await _ping(device_a, live_ip_b, interface_a)
    if live_ip_a and row_b.get("up"):
        ping_b = await _ping(device_b, live_ip_a, interface_b)

    def _state_line(device: str, interface: str, row: dict[str, Any], detail: dict[str, Any]) -> str:
        ip = row.get("primary_ip")
        plen = row.get("prefix_len")
        ip_text = f" {ip}/{plen}" if ip and plen is not None else ""
        admin = detail.get("oper_status") or row.get("oper_status") or "unknown"
        proto = detail.get("line_protocol") or row.get("line_protocol") or "unknown"
        state = "UP" if row.get("up") else f"DOWN ({admin})"
        return f"  {device} {interface}:{ip_text} state={state}, line_protocol={proto}"

    lines = [
        f"OSPF peering inspection for {device_a} {interface_a} <-> {device_b} {interface_b}:",
        _state_line(device_a, interface_a, row_a, detail_a),
        _state_line(device_b, interface_b, row_b, detail_b),
    ]
    for line in correlations:
        lines.append(f"  {line}")

    if ping_a is not None:
        lines.append(
            f"  Ping {device_a} {interface_a} -> {live_ip_b}: "
            f"{'SUCCESS' if ping_a.get('success') else 'FAILED'}"
        )
    if ping_b is not None:
        lines.append(
            f"  Ping {device_b} {interface_b} -> {live_ip_a}: "
            f"{'SUCCESS' if ping_b.get('success') else 'FAILED'}"
        )

    both_down = row_a.get("up") is False and row_b.get("up") is False
    one_down = row_a.get("up") is False or row_b.get("up") is False
    active_a = counters_a.get("active_errors", []) or []
    active_b = counters_b.get("active_errors", []) or []
    diagnosis_class = "unknown"
    recommended_action = ""

    if both_down:
        diagnosis_class = "peering_admin_down_both_sides"
        recommended_action = (
            f"Re-enable {interface_a} on {device_a} and {interface_b} on {device_b}, then verify the OSPF adjacency reforms "
            "and the route is readvertised upstream."
        )
        lines.append(
            f"  Evidence summary: both ends of the peering are down/admin-down ({device_a} {interface_a} and {device_b} {interface_b}). "
            "This is sufficient to explain the OSPF adjacency loss and route withdrawal."
        )
    elif one_down:
        diagnosis_class = "peering_admin_down_one_side"
        down_side = f"{device_a} {interface_a}" if row_a.get("up") is False else f"{device_b} {interface_b}"
        recommended_action = (
            f"Re-enable {down_side}, then verify the OSPF adjacency reforms and the route is readvertised upstream."
        )
        lines.append(
            f"  Evidence summary: {down_side} is down/admin-down. That is sufficient to explain the OSPF adjacency loss."
        )
    elif active_a or active_b:
        diagnosis_class = "peering_interface_errors"
        noisy_side = f"{device_a} {interface_a}" if active_a else f"{device_b} {interface_b}"
        recommended_action = (
            f"Treat {noisy_side} as a physical-layer suspect: inspect/reseat or replace the cable/optic/transceiver, "
            "check CRC/error counters and port health, then verify peer-IP reachability and confirm the OSPF adjacency reforms."
        )
        lines.append(
            f"  Evidence summary: {noisy_side} shows active interface errors while the peering is failing. "
            "This points to a physical/link-quality problem on the peering."
        )
    elif ping_a is not None and ping_b is not None and not ping_a.get("success") and not ping_b.get("success"):
        diagnosis_class = "peering_bidirectional_reachability_failure"
        recommended_action = (
            f"Treat the {device_a} {interface_a} <-> {device_b} {interface_b} peering as a physical/link-path suspect: "
            "inspect and reseat or replace the cable/optic/transceiver on both ends, verify interface counters and port health, "
            "then re-test direct peer-IP reachability before changing OSPF timers or policy."
        )
        lines.append(
            "  Evidence summary: both interfaces are up but bidirectional peer-IP reachability fails on the peering. "
            "This points to a peering/link problem rather than the destination LAN interface."
        )
    elif ping_a is not None and ping_a.get("success") is False:
        diagnosis_class = "peering_one_way_reachability_failure"
        recommended_action = (
            f"Start on {device_a} {interface_a} as the likely physical-layer suspect: inspect/reseat or replace the local cable/optic/transceiver, "
            f"check interface counters and port health, then re-test peer-IP reachability toward {live_ip_b} before changing OSPF settings."
        )
        lines.append(
            f"  Evidence summary: both peering interfaces are currently up, but {device_a} {interface_a} cannot reach peer IP {live_ip_b}. "
            "This localizes the failure to the peering itself and makes a generic route recommendation inappropriate."
        )
        lines.append(
            f"  OSPF impact: the OSPF adjacency on {device_a} {interface_a} <-> {device_b} {interface_b} is down/lost because peer-IP reachability failed on that peering."
        )
    elif ping_b is not None and ping_b.get("success") is False:
        diagnosis_class = "peering_one_way_reachability_failure"
        recommended_action = (
            f"Start on {device_b} {interface_b} as the likely physical-layer suspect: inspect/reseat or replace the local cable/optic/transceiver, "
            f"check interface counters and port health, then re-test peer-IP reachability toward {live_ip_a} before changing OSPF settings."
        )
        lines.append(
            f"  Evidence summary: both peering interfaces are currently up, but {device_b} {interface_b} cannot reach peer IP {live_ip_a}. "
            "This localizes the failure to the peering itself and makes a generic route recommendation inappropriate."
        )
        lines.append(
            f"  OSPF impact: the OSPF adjacency on {device_a} {interface_a} <-> {device_b} {interface_b} is down/lost because peer-IP reachability failed on that peering."
        )
    elif correlations:
        diagnosis_class = "peering_correlated_syslog_only"
        recommended_action = (
            f"Use the correlated OSPF-facing interfaces ({device_a} {interface_a} and {device_b} {interface_b}) as the investigation focus, "
            "then verify peer-IP reachability and current OSPF neighbor state on that exact peering."
        )
        lines.append(
            "  Evidence summary: syslog/IP correlation identifies the exact OSPF-facing interface(s); use that evidence directly in Root Cause."
        )
    else:
        diagnosis_class = "peering_unresolved_after_inspection"
        recommended_action = (
            f"Continue on the {device_a} {interface_a} <-> {device_b} {interface_b} peering with targeted OSPF neighbor/parameter checks; "
            "the current evidence identifies the failing adjacency but not yet the exact trigger."
        )

    lines.append(f"  Diagnosis class: {diagnosis_class}")
    lines.append(f"  Recommended next action: {recommended_action}")

    _store(session_id)["peering_inspections"].append({
        "device_a": device_a,
        "interface_a": interface_a,
        "device_b": device_b,
        "interface_b": interface_b,
        "ip_a": live_ip_a,
        "ip_b": live_ip_b,
        "row_a": row_a,
        "row_b": row_b,
        "detail_a": detail_a,
        "detail_b": detail_b,
        "correlations": correlations,
        "ping_a": ping_a,
        "ping_b": ping_b,
        "counters_a": counters_a,
        "counters_b": counters_b,
        "diagnosis_class": diagnosis_class,
        "recommended_action": recommended_action,
        "summary": lines[-1] if lines else "",
    })

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OSPF diagnostic tools
# ---------------------------------------------------------------------------

@tool
async def check_ospf_neighbors(devices: list[str], config: RunnableConfig) -> str:
    """
    Check current OSPF neighbor adjacency state on each device via live SSH.
    Call in parallel with check_ospf_interfaces and lookup_ospf_history.
    Pass all path devices plus any historically known devices from lookup_routing_history.

    Returns: per-device neighbor count, router-IDs, interfaces, states.
    """
    session_id = _sid(config)
    devs_str = ", ".join(devices)
    await _push_status(session_id, f"Checking OSPF neighbors on {devs_str}...")

    try:
        data = await _nornir_post("/ospf-neighbors", {"devices": devices}, timeout=30.0)
    except Exception as exc:
        return _backend_unavailable("Nornir", "OSPF neighbor lookup", exc)

    _store(session_id)["ospf_neighbors"] = data

    neighbors = data.get("ospf_neighbors", {})
    lines = ["OSPF neighbors:"]
    for device, info in neighbors.items():
        count = info.get("count", 0)
        if count == 0:
            lines.append(f"  {device}: 0 OSPF neighbors")
        else:
            nbr_list = ", ".join(
                f"{n['router_id']} via {n['interface']} ({n['state']})"
                for n in info.get("neighbors", [])
            )
            lines.append(f"  {device}: {count} neighbor(s) — {nbr_list}")
    return "\n".join(lines)


@tool
async def check_ospf_interfaces(devices: list[str], config: RunnableConfig) -> str:
    """
    Check which interfaces are currently reported by 'show ip ospf interface brief' on each device.
    A device with ospf_interface_count=0 has no interfaces currently reported by that command, but
    this is NOT by itself proof of misconfiguration: an interface-down condition can also result in 0.
    Correlate with get_all_interfaces, get_device_syslog, and lookup_ospf_history before concluding
    whether the issue is misconfiguration or a physical/link failure.
    Call in parallel with check_ospf_neighbors and lookup_ospf_history.

    Returns: per-device ospf_interface_count and the list of interfaces currently reported by OSPF.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Checking OSPF interface config on {', '.join(devices)}...")

    try:
        data = await _nornir_post("/ospf-interfaces", {"devices": devices}, timeout=30.0)
    except Exception as exc:
        return _backend_unavailable("Nornir", "OSPF interface lookup", exc)

    _store(session_id)["ospf_interfaces"] = data

    intfs = data.get("ospf_interfaces", {})
    lines = ["OSPF interface configuration:"]
    for device, info in intfs.items():
        count  = info.get("ospf_interface_count", 0)
        ifaces = info.get("interfaces", [])
        if count == 0:
            lines.append(
                f"  {device}: ospf_interface_count=0 — "
                f"no interfaces currently reported by 'show ip ospf interface brief'; "
                f"correlate with interface state, syslog, and history before calling this a config issue"
            )
        else:
            lines.append(f"  {device}: {count} OSPF interface(s) — {', '.join(ifaces)}")
    return "\n".join(lines)


@tool
async def lookup_ospf_history(devices: list[str], config: RunnableConfig) -> str:
    """
    Compare each device's current OSPF neighbor count against its last 10 historical snapshots.
    Use to confirm whether a device previously had OSPF neighbors before they were lost.
    Call in parallel with check_ospf_neighbors and check_ospf_interfaces.

    Returns: per-device historical neighbor trend and current count.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Looking up OSPF history for {', '.join(devices)}...")

    try:
        try:
            from atlas.db import fetch as _fetch, fetchrow as _fetchrow
        except ImportError:
            from db import fetch as _fetch, fetchrow as _fetchrow  # type: ignore
    except Exception as exc:
        return f"OSPF history DB unavailable: {exc}"

    results: dict[str, Any] = {}
    for device in devices:
        try:
            snapshots = await _fetch(
                """
                SELECT date_trunc('minute', collected_at) AS t, count(*) AS n
                FROM ospf_history WHERE device=$1
                GROUP BY t ORDER BY t DESC LIMIT 10
                """,
                device,
            )
            current = await _fetch(
                "SELECT router_id, interface, state FROM ospf_neighbors WHERE device=$1",
                device,
            )
            results[device] = {
                "current_neighbor_count": len(current),
                "history": [{"snapshot_time": r["t"].isoformat(), "neighbor_count": r["n"]} for r in snapshots],
            }
        except Exception as exc:
            results[device] = {"error": str(exc)}

    _store(session_id)["ospf_history"] = results

    lines = ["OSPF neighbor history:"]
    for device, info in results.items():
        if "error" in info:
            lines.append(f"  {device}: DB error — {info['error']}")
            continue
        hist  = info.get("history", [])
        curr  = info.get("current_neighbor_count", 0)
        trend = " → ".join(str(s["neighbor_count"]) for s in reversed(hist))
        lines.append(f"  {device}: history [{trend}] → now: {curr}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routing history tool
# ---------------------------------------------------------------------------

@tool
async def lookup_routing_history(destination_ip: str, config: RunnableConfig) -> str:
    """
    Query the routing history database for:
    1. All devices that historically had a data-plane route to destination_ip.
    2. The last known good route (egress interface, next-hop, protocol, prefix).

    Use after trace_path to find devices that SHOULD be in the path but aren't
    (e.g. because they're down) — include these in OSPF checks.
    Call in parallel with search_servicenow and get_interface_counters.

    Returns: historically known path devices + last known route details.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Looking up routing history for {destination_ip}...")

    try:
        try:
            from atlas.db import fetch as _fetch, fetchrow as _fetchrow
        except ImportError:
            from db import fetch as _fetch, fetchrow as _fetchrow  # type: ignore
    except Exception as exc:
        return f"Routing history DB unavailable: {exc}"

    try:
        # All devices that ever had a data-plane route to this destination
        hist_devs = await _fetch(
            """
            SELECT DISTINCT device FROM routing_history
            WHERE $1::inet << prefix
              AND egress_interface IS NOT NULL
              AND egress_interface NOT ILIKE 'management%'
            """,
            destination_ip,
        )
        historical_devices = [r["device"] for r in hist_devs]

        # Most recent known route (any device)
        last_route = await _fetchrow(
            """
            SELECT device, egress_interface, next_hop::text, protocol, prefix::text, collected_at
            FROM routing_history
            WHERE $1::inet << prefix
              AND egress_interface NOT ILIKE 'management%'
            ORDER BY collected_at DESC LIMIT 1
            """,
            destination_ip,
        )

        # Most recent upstream learned route with a next-hop. This is more useful
        # than the destination gateway's connected route when we need to identify
        # the actual OSPF peering pair carrying the advertisement.
        last_upstream_route = await _fetchrow(
            """
            SELECT device, egress_interface, next_hop::text, protocol, prefix::text, collected_at
            FROM routing_history
            WHERE $1::inet << prefix
              AND egress_interface NOT ILIKE 'management%'
              AND next_hop IS NOT NULL
            ORDER BY collected_at DESC LIMIT 1
            """,
            destination_ip,
        )
    except Exception as exc:
        return f"Routing history query error: {exc}"

    peer_hint: dict[str, Any] | None = None
    peering_source = last_upstream_route or last_route
    if peering_source and peering_source.get("next_hop"):
        try:
            peer_data = await _nornir_post("/find-device", {"ip": str(peering_source["next_hop"]).split("/")[0]}, timeout=15.0)
            if peer_data.get("found"):
                peer_hint = {
                    "from_device": peering_source.get("device"),
                    "from_interface": peering_source.get("egress_interface"),
                    "next_hop_ip": peering_source.get("next_hop"),
                    "to_device": peer_data.get("device"),
                    "to_interface": peer_data.get("interface"),
                }
        except Exception:
            peer_hint = None

    store = _store(session_id)
    store["historical_devices"] = historical_devices
    store["routing_history"] = {
        "historical_devices": historical_devices,
        "last_route": dict(last_route) if last_route else None,
        "last_upstream_route": dict(last_upstream_route) if last_upstream_route else None,
        "peer_hint": peer_hint,
    }

    lines = [f"Routing history for {destination_ip}:"]
    if historical_devices:
        lines.append(f"  Historically known path devices: {', '.join(historical_devices)}")
        lines.append(f"  (Include these in OSPF checks even if not in current path)")
    else:
        lines.append("  No routing history found in DB.")

    if last_upstream_route:
        import datetime
        try:
            delta = datetime.datetime.now(datetime.timezone.utc) - \
                    datetime.datetime.fromisoformat(str(last_upstream_route["collected_at"]))
            age = f"{int(delta.total_seconds() // 3600)}h ago"
        except Exception:
            age = "unknown age"
        lines.append(
            f"  Primary upstream clue ({age}): {last_upstream_route['device']} learned "
            f"{destination_ip} via {last_upstream_route['egress_interface']} next-hop "
            f"{last_upstream_route['next_hop']} ({last_upstream_route['protocol']})"
        )
        if peer_hint:
            lines.append(
                f"  Primary OSPF peering to troubleshoot: {peer_hint['from_device']} {peer_hint['from_interface']} "
                f"<-> {peer_hint['to_device']} {peer_hint['to_interface']} (via {peer_hint['next_hop_ip']})"
            )
            lines.append(
                "  Troubleshoot this bilateral peering first. Do not stop at the destination gateway alone."
            )

    if last_route:
        import datetime
        try:
            delta = datetime.datetime.now(datetime.timezone.utc) - \
                    datetime.datetime.fromisoformat(str(last_route["collected_at"]))
            age = f"{int(delta.total_seconds() // 3600)}h ago"
        except Exception:
            age = "unknown age"
        lines.append(
            f"  Last known route ({age}): {last_route['device']} egress "
            f"{last_route['egress_interface']} via {last_route['next_hop']} "
            f"({last_route['protocol']}, {last_route['prefix']})"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Holistic connectivity snapshot
# ---------------------------------------------------------------------------

def _is_device_name(value: str) -> bool:
    value = str(value or "").strip()
    return bool(value) and bool(_HOSTNAME_RE.match(value)) and not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value)


def _extract_devices_from_hops(hops: list[dict[str, Any]]) -> list[str]:
    devices: list[str] = []
    for hop in hops or []:
        if not isinstance(hop, dict):
            continue
        for key in ("from_device", "to_device"):
            value = str(hop.get(key) or "").strip()
            if _is_device_name(value) and value not in devices:
                devices.append(value)
    return devices


def _extract_relevant_interfaces(hops: list[dict[str, Any]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for hop in hops or []:
        if not isinstance(hop, dict):
            continue
        from_device = str(hop.get("from_device") or "").strip()
        to_device = str(hop.get("to_device") or "").strip()
        out_interface = str(hop.get("out_interface") or "").strip()
        in_interface = str(hop.get("in_interface") or "").strip()
        if _is_device_name(from_device) and out_interface:
            result.setdefault(from_device, set()).add(out_interface)
        if _is_device_name(to_device) and in_interface:
            result.setdefault(to_device, set()).add(in_interface)
    return result


def _pick_interface_row(data: dict[str, Any], interface: str) -> dict[str, Any]:
    for row in (data.get("interfaces") or []):
        if row.get("interface") == interface:
            return row
    return {}


def _infer_destination_side_device_from_store(store: dict[str, Any], route_to_dest: dict[str, Any] | None = None) -> str:
    if route_to_dest:
        hops = route_to_dest.get("hops") or {}
        for device, route in hops.items():
            if not isinstance(route, dict):
                continue
            if not route.get("found"):
                continue
            protocol = str(route.get("protocol") or "").lower()
            if protocol in {"connected", "local"} and _is_device_name(str(device)):
                return str(device)

    reverse_hops = store.get("reverse_path_hops") or []
    if reverse_hops and isinstance(reverse_hops, list):
        first = reverse_hops[0]
        if isinstance(first, dict):
            candidate = str(first.get("to_device") or "").strip()
            if _is_device_name(candidate):
                return candidate
    meta = store.get("path_meta") or {}
    candidate = str(meta.get("last_hop_device") or "").strip()
    return candidate if _is_device_name(candidate) else ""


@tool
async def collect_connectivity_snapshot(
    source_ip: str,
    dest_ip: str,
    config: RunnableConfig,
    port: str = "",
) -> str:
    """
    Collect a holistic connectivity snapshot for the current incident.

    This is a discovery-first evidence bundle for the agent:
    - current forward and reverse topology clues
    - historical devices / primary peering hint
    - per-device routing protocol discovery
    - spanning-tree mode when relevant
    - route status toward source/destination
    - relevant interface state/detail/counters
    - protocol-specific OSPF evidence only when OSPF is actually discovered
    - destination-side TCP check when a port is provided

    Use this before writing the final report when you need to reason about
    multiple simultaneous issues across the path.
    """
    session_id = _sid(config)
    store = _store(session_id)

    if not store.get("path_hops"):
        text, hops, flags = await _live_path_trace(source_ip, dest_ip, session_id=session_id)
        store["path_hops"] = hops
        store["path_flags"] = flags
        meta = _extract_path_metadata(hops)
        meta["src_vrf"] = await _infer_vrf(source_ip, meta.get("first_hop_device", ""))
        store["path_meta"] = meta
        store["path_text"] = text
    if not store.get("reverse_path_hops"):
        text, hops, _ = await _live_path_trace(dest_ip, source_ip, session_id=session_id)
        store["reverse_path_hops"] = hops
        store["reverse_path_text"] = text

    if not store.get("routing_history"):
        try:
            try:
                from atlas.db import fetch as _fetch, fetchrow as _fetchrow
            except ImportError:
                from db import fetch as _fetch, fetchrow as _fetchrow  # type: ignore

            hist_devs = await _fetch(
                """
                SELECT DISTINCT device FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface IS NOT NULL
                  AND egress_interface NOT ILIKE 'management%'
                """,
                dest_ip,
            )
            last_route = await _fetchrow(
                """
                SELECT device, egress_interface, next_hop::text, protocol, prefix::text, collected_at
                FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface NOT ILIKE 'management%'
                ORDER BY collected_at DESC LIMIT 1
                """,
                dest_ip,
            )
            last_upstream_route = await _fetchrow(
                """
                SELECT device, egress_interface, next_hop::text, protocol, prefix::text, collected_at
                FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface NOT ILIKE 'management%'
                  AND next_hop IS NOT NULL
                ORDER BY collected_at DESC LIMIT 1
                """,
                dest_ip,
            )
            peer_hint = None
            peering_source = last_upstream_route or last_route
            if peering_source and peering_source.get("next_hop"):
                try:
                    peer_data = await _cached_nornir_post(
                        session_id,
                        "/find-device",
                        {"ip": str(peering_source["next_hop"]).split("/")[0]},
                        timeout=15.0,
                    )
                    if peer_data.get("found"):
                        peer_hint = {
                            "from_device": peering_source.get("device"),
                            "from_interface": peering_source.get("egress_interface"),
                            "next_hop_ip": peering_source.get("next_hop"),
                            "to_device": peer_data.get("device"),
                            "to_interface": peer_data.get("interface"),
                        }
                except Exception:
                    peer_hint = None
            store["routing_history"] = _json_safe({
                "historical_devices": [r["device"] for r in hist_devs],
                "last_route": dict(last_route) if last_route else None,
                "last_upstream_route": dict(last_upstream_route) if last_upstream_route else None,
                "peer_hint": peer_hint,
            })
        except Exception as exc:
            store["routing_history"] = {"error": str(exc)}

    path_hops = store.get("path_hops") or []
    reverse_hops = store.get("reverse_path_hops") or []
    routing_history = store.get("routing_history") or {}
    peer_hint = routing_history.get("peer_hint") or {}

    devices: list[str] = []
    for device in (
        _extract_devices_from_hops(path_hops)
        + _extract_devices_from_hops(reverse_hops)
        + list(routing_history.get("historical_devices") or [])
        + [peer_hint.get("from_device"), peer_hint.get("to_device")]
    ):
        device = str(device or "").strip()
        if _is_device_name(device) and device not in devices:
            devices.append(device)

    if not devices:
        return "Connectivity snapshot unavailable: no in-scope devices were discovered from live path or history."

    relevant_interfaces = _extract_relevant_interfaces(path_hops)
    for device, interfaces in _extract_relevant_interfaces(reverse_hops).items():
        relevant_interfaces.setdefault(device, set()).update(interfaces)
    if peer_hint.get("from_device") and peer_hint.get("from_interface"):
        relevant_interfaces.setdefault(str(peer_hint["from_device"]), set()).add(str(peer_hint["from_interface"]))
    if peer_hint.get("to_device") and peer_hint.get("to_interface"):
        relevant_interfaces.setdefault(str(peer_hint["to_device"]), set()).add(str(peer_hint["to_interface"]))

    syslog_devices: set[str] = set()
    for key in ("from_device", "to_device"):
        dev = str(peer_hint.get(key) or "").strip()
        if _is_device_name(dev):
            syslog_devices.add(dev)
    if not syslog_devices:
        reverse_meta = store.get("reverse_path_meta") or {}
        for dev in (
            str((store.get("path_meta") or {}).get("first_hop_device") or "").strip(),
            str(reverse_meta.get("reverse_first_hop_device") or "").strip(),
        ):
            if _is_device_name(dev):
                syslog_devices.add(dev)

    async def _device_snapshot(device: str) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=25.0) as c:
                resp = await c.post(
                    f"{NORNIR_AGENT_URL}/device-snapshot",
                    json={
                        "device": device,
                        "source_ip": source_ip,
                        "dest_ip": dest_ip,
                        "relevant_interfaces": sorted(relevant_interfaces.get(device, set())),
                        "include_syslog": device in syslog_devices,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            data = {"device": device, "error": str(exc)}
        return data

    await _push_status(
        session_id,
        f"Collecting parallel device snapshots for {len(devices)} device{'s' if len(devices) != 1 else ''}..."
    )
    snapshot_results_list = await asyncio.gather(*(_device_snapshot(device) for device in devices))
    device_snapshots = dict(zip(devices, snapshot_results_list))

    protocol_results: dict[str, Any] = {}
    interface_results: dict[str, Any] = {}
    syslog_results: dict[str, Any] = {}
    route_to_dest: dict[str, Any] = {"hops": {}}
    route_to_src: dict[str, Any] = {"hops": {}}
    ospf_neighbors: dict[str, Any] = {}
    ospf_interfaces: dict[str, Any] = {}
    details: dict[tuple[str, str], dict[str, Any]] = {}
    counter_results_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    snapshot_errors: dict[str, str] = {}
    live_snapshot_devices: list[str] = []

    for device, snap in device_snapshots.items():
        if snap.get("error"):
            snapshot_errors[device] = str(snap.get("error") or "unknown snapshot error")
        else:
            live_snapshot_devices.append(device)

        proto = snap.get("protocol_discovery") or {
            "device": device,
            "routing_protocols": [],
            "configured_routing_protocols": [],
            "observed_route_types": [],
            "l2_control_plane": {
                "spanning_tree_mode": "unknown",
                "spanning_tree_enabled": None,
                "summary_lines": [],
            },
            "errors": {"device_snapshot": snap.get("error")} if snap.get("error") else {},
        }
        protocol_results[device] = proto
        store["protocol_discovery"][device] = proto

        iface_payload = snap.get("all_interfaces") or {"device": device, "interfaces": [], "error": snap.get("error")}
        interface_results[device] = iface_payload
        store["all_interfaces"][device] = iface_payload

        syslog_payload = snap.get("syslog") or {"device": device, "logs": [], "relevant": [], "error": snap.get("error")}
        syslog_results[device] = syslog_payload
        store["syslog"][device] = syslog_payload

        route_to_dest["hops"][device] = snap.get("route_to_destination") or {"found": False, "error": snap.get("error", "missing")}
        route_to_src["hops"][device] = snap.get("route_to_source") or {"found": False, "error": snap.get("error", "missing")}

        if "ospf" in (proto.get("routing_protocols") or []):
            ospf_neighbors[device] = snap.get("ospf_neighbors") or {"device": device, "count": 0, "neighbors": []}
            ospf_interfaces[device] = snap.get("ospf_interfaces") or {"device": device, "ospf_enabled_interfaces": [], "ospf_interface_count": 0}

        snap_details = snap.get("interface_details") or {}
        for interface, detail in snap_details.items():
            details[(device, interface)] = detail
            store["interface_details"][f"{device}:{interface}"] = detail
            active = []
            if any((detail.get(k) or 0) > 0 for k in ("input_errors", "output_errors", "input_discards", "output_discards")):
                active.append({
                    "interface": interface,
                    "input_errors": detail.get("input_errors", 0),
                    "output_errors": detail.get("output_errors", 0),
                    "input_discards": detail.get("input_discards", 0),
                    "output_discards": detail.get("output_discards", 0),
                })
            counter_payload = {"active_errors": active, "clean_interfaces": [] if active else [interface]}
            counter_results_by_key[(device, interface)] = counter_payload
            store["interface_counters"].append({
                "device": device,
                "window_s": 0,
                "active": active,
                "clean": counter_payload["clean_interfaces"],
            })

    destination_side_device = _infer_destination_side_device_from_store(store, route_to_dest)
    service_snapshot: dict[str, Any] | None = None
    if port and destination_side_device:
        await _push_status(
            session_id,
            f"Testing destination-side TCP from {destination_side_device} to {dest_ip}:{port}..."
        )
        try:
            service_snapshot = await _nornir_post(
                "/tcp-test",
                {"device": destination_side_device, "destination": dest_ip, "port": int(port), "vrf": ""},
                timeout=30.0,
            )
            service_snapshot["device"] = destination_side_device
        except Exception as exc:
            service_snapshot = {
                "device": destination_side_device,
                "reachable": False,
                "error": str(exc),
                "destination": dest_ip,
                "port": int(port),
            }

    await _push_status(session_id, "Assembling structured connectivity findings...")
    findings: list[str] = []
    link_lines: list[str] = []
    if peer_hint:
        a_dev = str(peer_hint.get("from_device") or "")
        a_int = str(peer_hint.get("from_interface") or "")
        b_dev = str(peer_hint.get("to_device") or "")
        b_int = str(peer_hint.get("to_interface") or "")
        if a_dev and a_int and b_dev and b_int:
            a_row = _pick_interface_row(interface_results.get(a_dev, {}), a_int)
            b_row = _pick_interface_row(interface_results.get(b_dev, {}), b_int)
            a_state = "up" if a_row.get("up") else f"down ({a_row.get('oper_status', 'unknown')})"
            b_state = "up" if b_row.get("up") else f"down ({b_row.get('oper_status', 'unknown')})"
            a_ip = a_row.get("primary_ip") or "unknown"
            b_ip = b_row.get("primary_ip") or str(peer_hint.get("next_hop_ip") or "unknown")
            link_lines.append(
                f"- {a_dev} {a_int} ({a_ip}) <-> {b_dev} {b_int} ({b_ip}) | states: {a_state} / {b_state}"
            )

            inspection = None
            for item in reversed(store.get("peering_inspections") or []):
                if not isinstance(item, dict):
                    continue
                direct = item.get("device_a") == a_dev and item.get("interface_a") == a_int and item.get("device_b") == b_dev and item.get("interface_b") == b_int
                reverse = item.get("device_a") == b_dev and item.get("interface_a") == b_int and item.get("device_b") == a_dev and item.get("interface_b") == a_int
                if direct or reverse:
                    inspection = item
                    break
            if inspection:
                ping_forward = inspection.get("ping_a_success")
                ping_reverse = inspection.get("ping_b_success")
                link_lines.append(
                    f"  peer-IP reachability: {a_dev}->{b_ip}={'ok' if ping_forward else 'failed'}; "
                    f"{b_dev}->{a_ip}={'ok' if ping_reverse else 'failed'}"
                )
            if a_row.get("up") is False or b_row.get("up") is False:
                findings.append(
                    f"[interface] Primary peering {a_dev} {a_int} <-> {b_dev} {b_int} has an interface down/admin-down."
                )

    for device in devices:
        route_info = (route_to_dest.get("hops") or {}).get(device, {})
        if route_info.get("error") and route_info.get("found") is not True:
            findings.append(f"[evidence] Live routing data for {device} was unavailable: {route_info.get('error')}.")
        elif route_info.get("found") is False:
            findings.append(f"[routing] {device} currently has no route to {dest_ip}.")
        for line in syslog_results.get(device, {}).get("relevant", []):
            low = line.lower()
            if "crc" in low or "fcs" in low or "discard" in low:
                findings.append(f"[interface] {device} recent syslog mentions physical/interface errors: {line}")
                break

    if service_snapshot:
        if service_snapshot.get("reachable") is True:
            findings.append(f"[service] TCP port {port} is reachable from destination-side device {service_snapshot['device']}.")
        else:
            err = str(service_snapshot.get("error") or service_snapshot.get("output") or "").lower()
            if "refused" in err:
                findings.append(f"[service] TCP port {port} is actively refused from destination-side device {service_snapshot['device']}.")
            else:
                findings.append(f"[service] TCP port {port} could not be validated from destination-side device {service_snapshot['device']}.")

    snapshot = {
        "source_ip": source_ip,
        "dest_ip": dest_ip,
        "port": port,
        "devices": devices,
        "routing_history": routing_history,
        "destination_side_device": destination_side_device,
        "findings": findings,
        "errors": snapshot_errors,
        "live_evidence_available": bool(path_hops or reverse_hops or live_snapshot_devices),
        "service": service_snapshot,
    }
    store["connectivity_snapshot"] = _json_safe(snapshot)

    lines = [
        "Connectivity incident snapshot:",
        f"  source_ip: {source_ip}",
        f"  dest_ip: {dest_ip}",
    ]
    if port:
        lines.append(f"  requested_port: {port}")
    lines.extend([
        f"  forward_path_devices: {', '.join(_extract_devices_from_hops(path_hops)) or 'none'}",
        f"  reverse_path_devices: {', '.join(_extract_devices_from_hops(reverse_hops)) or 'none'}",
        f"  historical_devices: {', '.join(routing_history.get('historical_devices') or []) or 'none'}",
    ])
    if peer_hint:
        lines.append(
            f"  primary_historical_peering: {peer_hint.get('from_device')} {peer_hint.get('from_interface')} "
            f"<-> {peer_hint.get('to_device')} {peer_hint.get('to_interface')} via {peer_hint.get('next_hop_ip')}"
        )

    lines.append("Device summary:")
    for device in devices:
        proto = protocol_results.get(device, {})
        route_dest = (route_to_dest.get("hops") or {}).get(device, {})
        route_src = (route_to_src.get("hops") or {}).get(device, {})
        rel_intfs = sorted(relevant_interfaces.get(device, set()))
        route_dest_summary = (
            f"{route_dest.get('protocol')} via {route_dest.get('interface')} next-hop {route_dest.get('next_hop')}"
            if route_dest.get("found") else
            (
                f"unavailable ({route_dest.get('error', 'unknown')})"
                if route_dest.get("error")
                else f"no route ({route_dest.get('error', 'unknown')})"
            )
        )
        route_src_summary = (
            f"{route_src.get('protocol')} via {route_src.get('interface')} next-hop {route_src.get('next_hop')}"
            if route_src.get("found") else
            (
                f"unavailable ({route_src.get('error', 'unknown')})"
                if route_src.get("error")
                else f"no route ({route_src.get('error', 'unknown')})"
            )
        )
        lines.append(
            f"- {device}: protocols={', '.join(proto.get('routing_protocols') or []) or 'none'}; "
            f"route_to_destination={route_dest_summary}; route_to_source={route_src_summary}"
        )
        if "ospf" in (proto.get("routing_protocols") or []):
            neigh = ospf_neighbors.get(device, {})
            ospf_intf = ospf_interfaces.get(device, {})
            lines.append(
                f"  ospf_neighbors={neigh.get('count', 0)}; "
                f"ospf_interfaces={ospf_intf.get('ospf_interface_count', 0)}"
            )
        if rel_intfs:
            rel_bits: list[str] = []
            for interface in rel_intfs:
                row = _pick_interface_row(interface_results.get(device, {}), interface)
                counters = counter_results_by_key.get((device, interface), {})
                state = "up" if row.get("up") else f"down ({row.get('oper_status', 'unknown')})"
                ip_text = f"{row.get('primary_ip')}/{row.get('prefix_len')}" if row.get("primary_ip") else "no_ip"
                active_errors = counters.get("active_errors") or []
                err_text = " active_errors" if active_errors else ""
                rel_bits.append(f"{interface}={ip_text},{state}{err_text}")
            lines.append(f"  relevant_interfaces: {'; '.join(rel_bits)}")

    if link_lines:
        lines.append("Link summary:")
        lines.extend(link_lines)

    if service_snapshot:
        if service_snapshot.get("reachable") is True:
            lines.append(f"Service summary: tcp_port_{port} reachable from {service_snapshot.get('device')}")
        else:
            lines.append(
                f"Service summary: tcp_port_{port} unreachable/refused from {service_snapshot.get('device')} "
                f"({service_snapshot.get('error') or service_snapshot.get('output') or 'no detail'})"
            )

    lines.append("Candidate issues:")
    if findings:
        for finding in findings[:10]:
            lines.append(f"  - {finding}")
    else:
        lines.append("  - No explicit candidate issue was derived; reason from the summarized device, link, and service evidence above.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ServiceNow tools
# ---------------------------------------------------------------------------

@tool
async def search_servicenow(
    device_names: list[str],
    config: RunnableConfig,
    source_ip: str = "",
    dest_ip: str = "",
    port: str = "",
    hours_back: int = 24,
) -> str:
    """
    Search ServiceNow for incidents AND change requests related to devices in the path.
    ALWAYS call this after trace_path. Pass every device hostname from the path.
    Change requests look back further than incidents because relevant changes often
    predate the resulting outage by more than a week.

    Args:
        device_names: All device hostnames from trace_path (e.g. ["arista1", "arista2"])
        source_ip:    Source IP of the traffic flow
        dest_ip:      Destination IP of the traffic flow
        port:         Destination port if known (improves relevance filtering)
        hours_back:   Window for incidents (default 24h)
    """
    session_id = _sid(config)
    store = _store(session_id)

    discovered_devices: list[str] = []
    for device in device_names or []:
        device = str(device or "").strip()
        if _is_device_name(device) and device not in discovered_devices:
            discovered_devices.append(device)

    for device in _extract_devices_from_hops(store.get("path_hops") or []):
        if device not in discovered_devices:
            discovered_devices.append(device)
    for device in _extract_devices_from_hops(store.get("reverse_path_hops") or []):
        if device not in discovered_devices:
            discovered_devices.append(device)

    routing_history = store.get("routing_history") or {}
    for device in (routing_history.get("historical_devices") or []):
        device = str(device or "").strip()
        if _is_device_name(device) and device not in discovered_devices:
            discovered_devices.append(device)

    peer_hint = routing_history.get("peer_hint") or {}
    for key in ("from_device", "to_device"):
        device = str(peer_hint.get(key) or "").strip()
        if _is_device_name(device) and device not in discovered_devices:
            discovered_devices.append(device)

    if not discovered_devices and dest_ip:
        try:
            try:
                from atlas.db import fetch as _fetch
            except ImportError:
                from db import fetch as _fetch  # type: ignore
            hist_rows = await _fetch(
                """
                SELECT DISTINCT device FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface IS NOT NULL
                  AND egress_interface NOT ILIKE 'management%'
                ORDER BY device
                """,
                dest_ip,
            )
            for row in hist_rows or []:
                device = str((row or {}).get("device") or "").strip()
                if _is_device_name(device) and device not in discovered_devices:
                    discovered_devices.append(device)
        except Exception:
            pass

    await _push_status(session_id, f"Checking ServiceNow for {', '.join(discovered_devices)}...")

    terms = [str(d).strip() for d in discovered_devices if d]
    for ip in (source_ip, dest_ip):
        if ip and ip.strip():
            terms.append(ip.strip())
    if port and port.strip():
        terms.append(port.strip())
    terms = list(dict.fromkeys(terms))

    if not terms:
        return _set_servicenow_summary(session_id, "ServiceNow skipped: no device names or IPs provided.")

    query     = " OR ".join(terms)
    chg_hours = max(hours_back * 30, 720)

    try:
        inc_result, chg_result = await asyncio.wait_for(
            asyncio.gather(
                call_mcp_tool("search_servicenow_incidents",
                              {"query": query, "limit": 10, "updated_within_hours": hours_back},
                              timeout=4.0),
                call_mcp_tool("list_servicenow_change_requests",
                              {"query": query, "limit": 20, "updated_within_hours": chg_hours},
                              timeout=4.0),
            ),
            timeout=5.0,
        )
    except asyncio.TimeoutError:
        return _set_servicenow_summary(session_id, "ServiceNow timed out; continuing without ticket context.")
    except Exception as exc:
        return _set_servicenow_summary(session_id, f"ServiceNow unavailable: {exc}")

    inc_error = inc_result.get("error") if isinstance(inc_result, dict) else None
    chg_error = chg_result.get("error") if isinstance(chg_result, dict) else None
    if inc_error or chg_error:
        parts = []
        if inc_error:
            parts.append(f"incident search failed: {inc_error}")
        if chg_error:
            parts.append(f"change search failed: {chg_error}")
        return _set_servicenow_summary(session_id, "ServiceNow unavailable: " + "; ".join(parts))

    def _cell(v, n=60):
        s = str(v or "—").strip().replace("|", "/").replace("\n", " ")
        return s[:n] if len(s) > n else s

    def _fmt_incidents(rows):
        if not rows:
            return "No incidents found."
        lines = []
        for r in rows:
            ci = r.get("cmdb_ci") or {}
            ci_name = (ci.get("display_value") or ci.get("value") or "—") if isinstance(ci, dict) else str(ci or "—")
            lines.append(
                f"- **{_cell(r.get('number'))}**\n"
                f"  CI: {_cell(ci_name)}\n"
                f"  Description: {_cell(r.get('short_description'), 120)}\n"
                f"  State: {_cell(r.get('state'))}\n"
                f"  Priority: {_cell(r.get('priority'))}"
            )
        return "\n".join(lines)

    def _fmt_changes(rows):
        if not rows:
            return "No change requests found."
        lines = []
        for r in rows:
            ci = r.get("cmdb_ci") or {}
            ci_name = (ci.get("display_value") or ci.get("value") or "—") if isinstance(ci, dict) else str(ci or "—")
            lines.append(
                f"- **{_cell(r.get('number'))}**\n"
                f"  CI: {_cell(ci_name)}\n"
                f"  Description: {_cell(r.get('short_description'), 120)}\n"
                f"  State: {_cell(r.get('state'))}\n"
                f"  Risk: {_cell(r.get('risk'))}\n"
                f"  Scheduled: {_cell((r.get('start_date') or '—')[:16])}"
            )
        return "\n".join(lines)

    inc_rows = (inc_result or {}).get("result", [])
    chg_rows = (chg_result or {}).get("result", [])

    logger.info(
        "search_servicenow: session=%s devices=%s terms=%s incidents=%s changes=%s",
        session_id,
        discovered_devices,
        terms,
        len(inc_rows),
        len(chg_rows),
    )

    summary = (
        f"Incidents found: {len(inc_rows)}\n"
        f"Change requests found: {len(chg_rows)}\n\n"
        "### Incidents\n"
        f"{_fmt_incidents(inc_rows)}\n\n"
        "### Change Requests\n"
        f"{_fmt_changes(chg_rows)}"
    )
    return _set_servicenow_summary(session_id, summary)


@tool
async def get_incident_details(incident_number: str, config: RunnableConfig) -> str:
    """
    Fetch full details for a specific ServiceNow incident by number (e.g. INC0010035).
    Use when the user references a specific incident and wants its description or to troubleshoot it.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Fetching {incident_number}...")

    try:
        data = await call_mcp_tool(
            "get_servicenow_incident",
            {"number": incident_number.upper().strip()},
            timeout=20.0,
        )
    except Exception as exc:
        return f"Incident lookup error: {exc}"

    if "error" in data:
        return f"Incident not found: {data['error']}"
    r = data.get("result", {})
    lines = [
        f"**{r.get('number')}** — {r.get('short_description', '')}",
        f"State: {r.get('state')} | Priority: {r.get('priority')} | Opened: {r.get('opened_at')}",
        f"Assigned to: {(r.get('assigned_to') or {}).get('display_value', 'Unassigned')}",
    ]
    if desc := r.get("description") or r.get("short_description"):
        lines.append(f"Description: {desc}")
    if notes := r.get("close_notes"):
        lines.append(f"Resolution: {notes}")
    return "\n".join(lines)


@tool
async def get_change_request_details(change_number: str, config: RunnableConfig) -> str:
    """
    Fetch full details for a specific ServiceNow change request by number (e.g. CHG0010001).
    Use when the user references a specific change request and wants its details or current state.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Fetching {change_number}...")

    try:
        data = await call_mcp_tool(
            "get_servicenow_change_request",
            {"number": change_number.upper().strip()},
            timeout=20.0,
        )
    except Exception as exc:
        return f"Change request lookup error: {exc}"

    if "error" in data:
        return f"Change request not found: {data['error']}"
    r = data.get("result", {})
    lines = [
        f"**{r.get('number')}** — {r.get('short_description', '')}",
        f"State: {r.get('state')} | Risk: {r.get('risk', 'Unknown')}",
        f"Assignment group: {(r.get('assignment_group') or {}).get('display_value', 'Unassigned')}",
        f"Configuration Item: {(r.get('cmdb_ci') or {}).get('display_value', r.get('cmdb_ci', 'Unknown'))}",
    ]
    if desc := r.get("description") or r.get("short_description"):
        lines.append(f"Description: {desc}")
    if just := r.get("justification"):
        lines.append(f"Justification: {just}")
    if plan := r.get("implementation_plan"):
        lines.append(f"Implementation plan: {plan}")
    if notes := r.get("close_notes"):
        lines.append(f"Close notes: {notes}")
    return "\n".join(lines)


@tool
async def create_servicenow_incident(
    short_description: str,
    config: RunnableConfig,
    description: str = "",
    urgency: str = "2",
    impact: str = "2",
    ci_name: str = "",
) -> str:
    """
    Create a ServiceNow incident.
    Use for explicit requests to create/open/raise an incident or ticket.
    Returns the created incident number and key details.
    """
    session_id = _sid(config)
    await _push_status(session_id, "Creating ServiceNow incident...")

    try:
        result = await call_mcp_tool(
            "create_servicenow_incident",
            {
                "short_description": short_description,
                "description": description,
                "urgency": urgency,
                "impact": impact,
                "category": "network",
                "ci_name": ci_name,
            },
            timeout=20.0,
        )
    except Exception as exc:
        return f"Incident creation failed: {exc}"

    if not isinstance(result, dict):
        return f"Incident creation failed: unexpected response {result!r}"
    if "error" in result:
        return f"Incident creation failed: {result['error']}"

    r = result.get("result", {}) if isinstance(result.get("result"), dict) else result
    number = r.get("number") or r.get("display_value") or "unknown"
    sys_id = r.get("sys_id") or "unknown"
    if number != "unknown":
        verify = await call_mcp_tool(
            "get_servicenow_incident",
            {"number": str(number).upper().strip()},
            timeout=20.0,
        )
        if isinstance(verify, dict) and "error" in verify:
            return (
                f"Incident creation could not be verified for {number}.\n"
                f"lookup_error: {verify['error']}"
            )
    return (
        f"Created ServiceNow incident {number}.\n"
        f"sys_id: {sys_id}\n"
        f"short_description: {short_description}"
    )


@tool
async def create_servicenow_change_request(
    short_description: str,
    config: RunnableConfig,
    description: str = "",
    risk: str = "3",
    assignment_group: str = "",
    justification: str = "",
    implementation_plan: str = "",
    ci_name: str = "",
) -> str:
    """
    Create a ServiceNow change request.
    Use for explicit requests to create/open/submit a change request.
    Returns the created change number and key details.
    """
    session_id = _sid(config)
    await _push_status(session_id, "Creating ServiceNow change request...")

    try:
        result = await call_mcp_tool(
            "create_servicenow_change_request",
            {
                "short_description": short_description,
                "description": description,
                "risk": risk,
                "assignment_group": assignment_group,
                "justification": justification,
                "implementation_plan": implementation_plan,
                "ci_name": ci_name,
            },
            timeout=20.0,
        )
    except Exception as exc:
        return f"Change request creation failed: {exc}"

    if not isinstance(result, dict):
        return f"Change request creation failed: unexpected response {result!r}"
    if "error" in result:
        return f"Change request creation failed: {result['error']}"

    r = result.get("result", {}) if isinstance(result.get("result"), dict) else result
    number = r.get("number") or r.get("display_value") or "unknown"
    sys_id = r.get("sys_id") or "unknown"
    if number != "unknown":
        verify = await call_mcp_tool(
            "get_servicenow_change_request",
            {"number": str(number).upper().strip()},
            timeout=20.0,
        )
        if isinstance(verify, dict) and "error" in verify:
            return (
                f"Change request creation could not be verified for {number}.\n"
                f"lookup_error: {verify['error']}"
            )
    return (
        f"Created ServiceNow change request {number}.\n"
        f"sys_id: {sys_id}\n"
        f"short_description: {short_description}"
    )


@tool
async def update_servicenow_change_request(
    number: str,
    config: RunnableConfig,
    state: str = "",
    work_notes: str = "",
    assigned_to: str = "",
    close_notes: str = "",
) -> str:
    """
    Update an existing ServiceNow change request.
    Use for explicit requests to close/update a change request such as CHG0030001.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Updating ServiceNow change request {number}...")

    args: dict[str, Any] = {"number": number}
    if state:
        args["state"] = state
    if work_notes:
        args["work_notes"] = work_notes
    if assigned_to:
        args["assigned_to"] = assigned_to
    if close_notes:
        args["close_notes"] = close_notes

    try:
        result = await call_mcp_tool("update_servicenow_change_request", args, timeout=20.0)
    except Exception as exc:
        return f"Change request update failed: {exc}"

    if not isinstance(result, dict):
        return f"Change request update failed: unexpected response {result!r}"
    if "error" in result:
        return f"Change request update failed: {result['error']}"

    r = result.get("result", {}) if isinstance(result.get("result"), dict) else result
    out_number = r.get("number") or number
    out_state = r.get("state") or state or "updated"
    return f"Updated ServiceNow change request {out_number}.\nstate: {out_state}"


# ---------------------------------------------------------------------------
# Firewall / Splunk tools
# ---------------------------------------------------------------------------

@tool
async def check_panorama_policy(
    source_ip: str,
    dest_ip: str,
    firewall_hostnames: list[str],
    config: RunnableConfig,
    port: str = "",
    protocol: str = "tcp",
) -> str:
    """
    Run Panorama 'test security-policy-match' against each Palo Alto firewall in the path.
    Only call this if trace_path identified Palo Alto firewalls.
    Pass the firewall hostnames from the path trace output.

    Returns: per-firewall matching rule, action (allow/deny), zones.
    """
    session_id = _sid(config)
    fws_str = ", ".join(firewall_hostnames)
    await _push_status(session_id, f"Checking Panorama policy on {fws_str}...")

    if not firewall_hostnames:
        return "No Palo Alto firewalls provided — Panorama check skipped."

    if _STUB:
        return "Security policy match (STUB): rule=Allow-Internal action=allow zones=trust→untrust"

    async def _test_one(fw: str) -> str:
        args: dict[str, Any] = {"firewall_hostname": fw, "source_ip": source_ip, "dest_ip": dest_ip}
        if port:
            args["dest_port"] = port.strip()
        if protocol:
            args["protocol"] = protocol.strip()
        result = await call_mcp_tool("test_panorama_security_policy_match", args, timeout=45.0)
        if not result:
            return f"{fw}: no response from Panorama."
        if "error" in result:
            return f"{fw}: {result['error']}"
        rule    = result.get("matching_rule", "no-match")
        action  = result.get("action", "unknown")
        dg      = result.get("device_group", "?")
        from_z  = result.get("from_zone", "?")
        to_z    = result.get("to_zone", "?")
        port_info = f" port {port}/{protocol}" if port else ""
        icon    = "✅" if action.lower() == "allow" else "🚫"
        return f"{icon} {fw} (dg: {dg}){port_info}: rule='{rule}' action={action} zones={from_z}→{to_z}"

    results = await asyncio.gather(*[_test_one(fw) for fw in firewall_hostnames])
    return "Panorama policy results:\n" + "\n".join(results)


@tool
async def check_splunk(task: str, config: RunnableConfig) -> str:
    """
    Query Splunk for recent firewall deny events and traffic patterns.
    Use to correlate deny events with the path findings when firewalls are present.
    Pass a natural-language description including the source IP and time window.
    Example: "Check deny events for 10.0.0.1 → 11.0.0.1 in the last 24 hours."
    """
    session_id = _sid(config)
    await _push_status(session_id, "Querying Splunk for traffic events...")

    if _STUB:
        return "Splunk (STUB): 14 deny events on port 22 from 10.0.0.1 in the last 24h."

    import uuid
    cb = CircuitBreaker.for_endpoint(SPLUNK_AGENT_URL)

    async def _do() -> dict:
        async with httpx.AsyncClient(timeout=30.0) as c:
            resp = await c.post(SPLUNK_AGENT_URL, json={
                "id": str(uuid.uuid4()),
                "message": {"role": "user", "parts": [{"type": "text", "text": task}]},
            })
            resp.raise_for_status()
            return resp.json()

    try:
        data = await retry_async(
            cb, _do,
            retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError),
        )
        artifacts = data.get("artifacts", [])
        if artifacts:
            text = next(
                (p.get("text") for p in artifacts[0].get("parts", []) if p.get("type") == "text"),
                None,
            )
            if text:
                return text
        return "Splunk: no data returned."
    except Exception as exc:
        return f"Splunk unavailable: {exc}"


# ---------------------------------------------------------------------------
# Long-term memory recall — semantic search over past sessions + incidents
# ---------------------------------------------------------------------------

@tool
async def recall_similar_cases(
    query: str,
    devices: list[str],
    config: RunnableConfig,
) -> str:
    """
    Search long-term memory for past troubleshooting sessions and closed incidents
    semantically similar to the current query.

    Use this only after the current investigation has produced live evidence
    that suggests recurrence, instability, or an unresolved pattern.

    Args:
        query:   The current issue description (e.g. "10.0.100.100 can't reach 10.0.200.200 port 443").
        devices: Device hostnames in the path — used to match device-tagged incidents.
    """
    session_id = _sid(config)
    store = _store(session_id)
    signals = memory_manager.get_recall_signals(store)
    if not signals:
        return (
            "Memory recall deferred: gather live evidence first. "
            "Use recall only after live results show recurrence, instability, or an unresolved pattern."
        )

    await _push_status(session_id, "Searching past cases...")

    try:
        try:
            from atlas.agent_memory import recall_memory, recall_incidents_by_devices, format_memory_context
        except ImportError:
            from agent_memory import recall_memory, recall_incidents_by_devices, format_memory_context  # type: ignore

        past_sessions = await recall_memory(query, agent_type="troubleshoot", top_k=3)
        past_incidents = await recall_incidents_by_devices(devices, query=query, top_k=5) if devices else []

        combined = past_sessions + [i for i in past_incidents if i not in past_sessions]
        if not combined:
            return f"No similar past cases found in memory for signals: {', '.join(signals)}."
        context = format_memory_context(combined)
        return f"Memory recall triggered by live signals: {', '.join(signals)}.\n\n{context}"
    except Exception as exc:
        return f"Memory recall unavailable: {exc}"


# ---------------------------------------------------------------------------
# Vendor KB lookup — LLM-generated vendor-specific guidance
# ---------------------------------------------------------------------------

_PLATFORM_TO_VENDOR = {
    "arista": "arista_eos",
    "eos":    "arista_eos",
    "cisco":  "cisco_ios",
    "ios":    "cisco_ios",
    "nxos":   "cisco_nxos",
    "nx-os":  "cisco_nxos",
    "junos":  "junos",
    "panos":  "panos",
}

_VENDOR_SYSTEM_PROMPTS = {
    "arista_eos": """\
You are an expert Arista EOS network engineer writing concise vendor knowledge base entries.
Given a description of a diagnosed network problem, produce 2-3 KB entries.

Each entry must follow this exact format:
**[Short descriptive title]**
[2-4 sentences: explain the EOS-specific cause and include exact CLI commands to verify and fix.]
Reference: [EOS documentation section or relevant `show` command]

Rules:
- Use exact Arista EOS CLI syntax (e.g. `router ospf 1`, `network 10.0.0.0/8 area 0`, `ip ospf area 0`)
- Do not invent bug IDs or URLs
- Be specific and actionable""",

    "cisco_ios": """\
You are an expert Cisco IOS network engineer writing concise vendor knowledge base entries.
Given a description of a diagnosed network problem, produce 2-3 KB entries.

Each entry must follow this exact format:
**[Short descriptive title]**
[2-4 sentences: explain the IOS-specific cause and include exact CLI commands to verify and fix.]
Reference: [IOS documentation section or relevant `show` command]

Rules:
- Use exact Cisco IOS CLI syntax
- Do not invent bug IDs or URLs
- Be specific and actionable""",
}


async def _detect_vendor(devices: list[str]) -> str:
    """Return the primary vendor key for the given device list using the DB."""
    if not devices:
        return "unknown"
    try:
        try:
            from atlas.db import fetch
        except ImportError:
            from db import fetch  # type: ignore
        rows = await fetch(
            "SELECT platform FROM devices WHERE hostname = ANY($1::text[])",
            devices,
        )
        counts: dict[str, int] = {}
        for row in rows:
            platform = (row["platform"] or "").lower()
            vendor = next((v for k, v in _PLATFORM_TO_VENDOR.items() if k in platform), None)
            if vendor:
                counts[vendor] = counts.get(vendor, 0) + 1
        return max(counts, key=counts.get) if counts else "unknown"
    except Exception as exc:
        logger.warning("_detect_vendor: %s", exc)
        return "unknown"


@tool
async def lookup_vendor_kb(
    symptoms: str,
    devices: list[str],
    context: str | None = None,
) -> str:
    """
    Look up vendor-specific knowledge base entries for the given symptoms and devices.

    Use this when you need vendor-specific CLI commands, configuration examples, or
    troubleshooting tips — for example, exact Arista EOS or Cisco IOS syntax to fix
    a diagnosed problem. Call it as soon as you have enough context to describe the
    symptoms; you do not need to wait until the end of the investigation.

    The vendor is auto-detected from the device list via the device database.

    Args:
        symptoms: Natural language description of the issue
                  (e.g. "OSPF process running but ospf_interface_count=0 on arista2").
        devices:  List of device hostnames in the path (used to detect vendor).
        context:  Optional additional context from previous tool results
                  (e.g. "TCP port 443 connection refused from last hop device").
    """
    vendor = await _detect_vendor(devices)
    system_prompt = _VENDOR_SYSTEM_PROMPTS.get(vendor)
    if not system_prompt:
        return f"No vendor KB handler for vendor={vendor!r} (devices: {devices})."

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
        max_tokens=800,
    )
    user_content = f"Symptoms:\n{symptoms}"
    if context:
        user_content += f"\n\nAdditional context:\n{context}"
    user_content += "\n\nWrite 2-3 knowledge base entries."

    resp = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    result = (resp.content or "").strip()
    logger.info("lookup_vendor_kb: vendor=%s devices=%s, %d chars returned", vendor, devices, len(result))
    return result or "No KB entries generated."


# ---------------------------------------------------------------------------
# Tool lists — each agent gets only the tools it needs
# ---------------------------------------------------------------------------

# Full diagnostic tool set for the troubleshooting agent
ALL_TOOLS = [
    trace_path,
    trace_reverse_path,
    ping_device,
    test_tcp_port,
    check_routing,
    get_interface_counters,
    get_interface_detail,
    get_all_interfaces,
    get_device_syslog,
    inspect_ospf_peering,
    collect_connectivity_snapshot,
    check_ospf_neighbors,
    check_ospf_interfaces,
    lookup_ospf_history,
    lookup_routing_history,
    search_servicenow,
    get_incident_details,
    check_panorama_policy,
    check_splunk,
    recall_similar_cases,
    lookup_vendor_kb,
]

# Restricted tool set for the network-ops agent — path + policy + tickets only.
# No diagnostic tools (OSPF, counters, routing checks, ping) — those belong to
# the troubleshooting agent.
NETWORK_OPS_TOOLS = [
    trace_path,
    check_panorama_policy,
    search_servicenow,
    get_incident_details,
    get_change_request_details,
    create_servicenow_incident,
    create_servicenow_change_request,
    update_servicenow_change_request,
]

# Connectivity investigations should reason from live tools plus historical
# routing evidence, but not from recalled past narratives that can overwrite
# current-state observations.
CONNECTIVITY_TOOLS = [
    trace_path,
    trace_reverse_path,
    ping_device,
    check_routing,
    collect_connectivity_snapshot,
    lookup_routing_history,
    search_servicenow,
    get_incident_details,
    check_panorama_policy,
    check_splunk,
    lookup_vendor_kb,
]
