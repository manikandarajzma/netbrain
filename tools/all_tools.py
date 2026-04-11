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
import json
import logging
import os
import re
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.tools.resilience import retry_async, CircuitBreaker, CircuitOpenError
    from atlas.mcp_client import call_mcp_tool
except ImportError:
    from tools.resilience import retry_async, CircuitBreaker, CircuitOpenError
    from mcp_client import call_mcp_tool

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
    })


def pop_session_data(session_id: str) -> dict[str, Any]:
    """Read and remove all side-effect data after the agent completes."""
    return _session_store.pop(session_id, {})


def _sid(config: RunnableConfig) -> str:
    return (config or {}).get("configurable", {}).get("session_id", "default")


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


async def _nornir_post(endpoint: str, payload: dict, timeout: float = 30.0) -> dict:
    """
    POST to the Nornir HTTP backend with circuit-breaker + retry.
    Every Nornir tool call should go through here — never call httpx directly.
    Raises on failure; callers catch and return an error string.
    """
    url = f"{NORNIR_AGENT_URL}/{endpoint.lstrip('/')}"
    cb  = CircuitBreaker.for_endpoint(url)

    async def _do() -> dict:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    return await retry_async(
        cb, _do,
        retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError),
    )


_PLATFORM_TO_TYPE = {
    "arista_eos":  "arista switch",
    "cisco_ios":   "cisco router",
    "cisco_nxos":  "cisco switch",
    "cisco_iosxr": "cisco router",
}

_HOSTNAME_RE = re.compile(r'^[A-Za-z0-9]([A-Za-z0-9\-_\.]*[A-Za-z0-9])?$')


async def _live_path_trace(src_ip: str, dst_ip: str) -> tuple[str, list[dict], dict]:
    """
    Hop-by-hop live path trace via SSH.  No database reads.

    Returns:
        text_summary   — human-readable for the LLM
        structured_hops — list of hop dicts for PathVisualization
        flags           — anomaly flags (mgmt_routing_detected, no_route_device, …)
    """
    flags: dict[str, Any] = {}

    async def _find_device(ip: str) -> dict:
        async with httpx.AsyncClient(timeout=25.0) as c:
            r = await c.post(f"{NORNIR_AGENT_URL}/find-device", json={"ip": ip})
            return r.json()

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
                async with httpx.AsyncClient(timeout=20.0) as c:
                    r = await c.post(f"{NORNIR_AGENT_URL}/route",
                                     json={"device": device, "destination": src})
                    route = r.json()
                if route.get("found") and route.get("protocol", "").lower() == "connected":
                    return device, route.get("egress_interface")
            except Exception:
                continue
        return "", None

    text_hops: list[str] = []
    structured_hops: list[dict] = []
    seen: set[str] = set()
    MAX_HOPS = 15

    current_device, gw_iface = await _find_first_hop(src_ip)
    if not current_device:
        return (
            f"Could not find a network device with a connected route to {src_ip} — check inventory.",
            [], flags,
        )

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
            async with httpx.AsyncClient(timeout=15.0) as c:
                r = await c.post(f"{NORNIR_AGENT_URL}/route",
                                 json={"device": current_device, "destination": dst_ip})
                route = r.json()
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

        # Check egress interface line-protocol
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                rs = await c.post(f"{NORNIR_AGENT_URL}/interface-status",
                                  json={"device": current_device, "interface": egress})
                intf_status = rs.json()
            if not intf_status.get("up", True):
                text_hops.append(
                    f"  ⚠️  Hop {len(text_hops)+1}: {current_device} | {egress} is DOWN"
                )
                structured_hops.append({
                    "from_device": current_device, "from_device_type": "switch",
                    "out_interface": egress, "out_zone": None, "device_group": None,
                    "to_device": f"⚠️ {egress} DOWN", "to_device_type": "host",
                    "in_interface": None, "in_zone": None,
                })
                break
        except Exception:
            pass

        text_hops.append(
            f"  Hop {len(text_hops)+1}: {current_device} | Egress: {egress} | "
            f"Protocol: {protocol} | Prefix: {prefix} | Next-hop: {next_hop or 'directly connected'}"
        )

        if not next_hop:
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

        next_dev = await _find_device(next_hop)
        if not next_dev.get("found"):
            text_hops.append(f"  Next-hop {next_hop} not found — path ends here")
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

    text, hops, flags = await _live_path_trace(source_ip, dest_ip)

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

    text, hops, _ = await _live_path_trace(dest_ip, source_ip)

    if hops:
        _store(session_id)["reverse_path_hops"] = hops

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
    Reverse ping:  device=last_hop_device,  destination=src_ip,  source_interface=last_hop_lan_interface,  vrf=dst_vrf

    Returns: success/failure, packet loss %, RTT.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Pinging {device} → {destination}...")

    cb = CircuitBreaker.for_endpoint(NORNIR_AGENT_URL + "/ping")
    async def _do():
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{NORNIR_AGENT_URL}/ping",
                             json={"device": device, "destination": destination,
                                   "source_interface": source_interface, "vrf": vrf})
            r.raise_for_status()
            return r.json()
    try:
        result = await retry_async(cb, _do,
            retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError))
    except Exception as exc:
        result = {"success": False, "error": str(exc)}

    _store(session_id)["ping_result"] = result
    _store(session_id)["ping_results"].append({
        "device": device,
        "destination": destination,
        "source_interface": source_interface,
        "vrf": vrf or "default",
        **result,
    })

    if result.get("success"):
        rtt = result.get("rtt_avg_ms")
        rtt_str = f", RTT avg {rtt}ms" if rtt else ""
        return f"✓ Ping {device} → {destination} (VRF: {vrf or 'default'}): SUCCESS, 0% loss{rtt_str}"
    loss = result.get("loss_pct", 100)
    err  = result.get("error", "")
    return f"✗ Ping {device} → {destination} (VRF: {vrf or 'default'}): FAILED — {loss}% packet loss{(' — ' + err) if err else ''}"


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
        return f"Routing check error: {exc}"

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
        return f"Interface detail error: {exc}"


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
        return f"All-interfaces error: {exc}"

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
        return f"Syslog error on {device}: {exc}"

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
      - interface detail on both sides
      - recent syslog with OSPF/IP correlation on both sides
      - bilateral ping results across the peering IPs when provided
      - a short evidence summary the LLM can reason from directly
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Inspecting OSPF peering {device_a}/{interface_a} <-> {device_b}/{interface_b}...")

    async def _all_interfaces(device: str) -> dict[str, Any]:
        data = await _nornir_post("/all-interfaces-status", {"device": device}, timeout=15.0)
        _store(session_id)["all_interfaces"][device] = data
        return data

    async def _interface_detail(device: str, interface: str) -> dict[str, Any]:
        data = await _nornir_post("/interface-detail", {"device": device, "interface": interface}, timeout=15.0)
        _store(session_id)["interface_details"][f"{device}:{interface}"] = data
        return data

    async def _syslog(device: str) -> dict[str, Any]:
        data = await _nornir_post("/show-logging", {"device": device, "lines": 100}, timeout=15.0)
        logs = data.get("logs", [])
        kw = ["link", "down", "flap", "err-disable", "lineproto", "ospf", "adjacency"]
        relevant = [l for l in logs if any(k in l.lower() for k in kw)][-30:]
        return {"logs": logs, "relevant": relevant}

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

    all_a, all_b, detail_a, detail_b, syslog_a, syslog_b = await asyncio.gather(
        _all_interfaces(device_a),
        _all_interfaces(device_b),
        _interface_detail(device_a, interface_a),
        _interface_detail(device_b, interface_b),
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
    if ip_b and row_a.get("up"):
        ping_a = await _ping(device_a, ip_b, interface_a)
    if ip_a and row_b.get("up"):
        ping_b = await _ping(device_b, ip_a, interface_b)

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
            f"  Ping {device_a} {interface_a} -> {ip_b}: "
            f"{'SUCCESS' if ping_a.get('success') else 'FAILED'}"
        )
    if ping_b is not None:
        lines.append(
            f"  Ping {device_b} {interface_b} -> {ip_a}: "
            f"{'SUCCESS' if ping_b.get('success') else 'FAILED'}"
        )

    both_down = row_a.get("up") is False and row_b.get("up") is False
    one_down = row_a.get("up") is False or row_b.get("up") is False
    if both_down:
        lines.append(
            f"  Evidence summary: both ends of the peering are down/admin-down ({device_a} {interface_a} and {device_b} {interface_b}). "
            "This is sufficient to explain the OSPF adjacency loss and route withdrawal."
        )
    elif one_down:
        down_side = f"{device_a} {interface_a}" if row_a.get("up") is False else f"{device_b} {interface_b}"
        lines.append(
            f"  Evidence summary: {down_side} is down/admin-down. That is sufficient to explain the OSPF adjacency loss."
        )
    elif ping_a is not None and ping_b is not None and not ping_a.get("success") and not ping_b.get("success"):
        lines.append(
            "  Evidence summary: both interfaces are up but bidirectional peer-IP reachability fails on the peering. "
            "This points to a peering/link problem rather than the destination LAN interface."
        )
    elif correlations:
        lines.append(
            "  Evidence summary: syslog/IP correlation identifies the exact OSPF-facing interface(s); use that evidence directly in Root Cause."
        )

    _store(session_id)["peering_inspections"].append({
        "device_a": device_a,
        "interface_a": interface_a,
        "device_b": device_b,
        "interface_b": interface_b,
        "ip_a": ip_a,
        "ip_b": ip_b,
        "row_a": row_a,
        "row_b": row_b,
        "detail_a": detail_a,
        "detail_b": detail_b,
        "correlations": correlations,
        "ping_a": ping_a,
        "ping_b": ping_b,
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
        return f"OSPF neighbors error: {exc}"

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
        return f"OSPF interfaces error: {exc}"

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
    Change requests look back further (7× hours_back) since changes predate incidents.

    Args:
        device_names: All device hostnames from trace_path (e.g. ["arista1", "arista2"])
        source_ip:    Source IP of the traffic flow
        dest_ip:      Destination IP of the traffic flow
        port:         Destination port if known (improves relevance filtering)
        hours_back:   Window for incidents (default 24h)
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Checking ServiceNow for {', '.join(device_names)}...")

    terms = [str(d).strip() for d in device_names if d]
    for ip in (source_ip, dest_ip):
        if ip and ip.strip():
            terms.append(ip.strip())
    if port and port.strip():
        terms.append(port.strip())
    terms = list(dict.fromkeys(terms))

    if not terms:
        return "ServiceNow skipped: no device names or IPs provided."

    query     = " OR ".join(terms)
    chg_hours = max(hours_back * 7, 168)

    try:
        inc_result, chg_result = await asyncio.gather(
            call_mcp_tool("search_servicenow_incidents",
                          {"query": query, "limit": 10, "updated_within_hours": hours_back},
                          timeout=30.0),
            call_mcp_tool("list_servicenow_change_requests",
                          {"query": query, "limit": 20, "updated_within_hours": chg_hours},
                          timeout=30.0),
        )
    except Exception as exc:
        return f"ServiceNow unavailable: {exc}"

    def _cell(v, n=60):
        s = str(v or "—").strip().replace("|", "/").replace("\n", " ")
        return s[:n] if len(s) > n else s

    def _fmt_incidents(rows):
        if not rows:
            return "No incidents found."
        lines = ["| Number | CI | Description | State | Priority |",
                 "|--------|----|-------------|-------|----------|"]
        for r in rows:
            ci = r.get("cmdb_ci") or {}
            ci_name = (ci.get("display_value") or ci.get("value") or "—") if isinstance(ci, dict) else str(ci or "—")
            lines.append(
                f"| {_cell(r.get('number'))} | {_cell(ci_name)} | "
                f"{_cell(r.get('short_description'))} | {_cell(r.get('state'))} | "
                f"{_cell(r.get('priority'))} |"
            )
        return "\n".join(lines)

    def _fmt_changes(rows):
        if not rows:
            return "No change requests found."
        lines = ["| Number | CI | Description | State | Risk | Scheduled |",
                 "|--------|----|-------------|-------|------|-----------|"]
        for r in rows:
            ci = r.get("cmdb_ci") or {}
            ci_name = (ci.get("display_value") or ci.get("value") or "—") if isinstance(ci, dict) else str(ci or "—")
            lines.append(
                f"| {_cell(r.get('number'))} | {_cell(ci_name)} | "
                f"{_cell(r.get('short_description'))} | {_cell(r.get('state'))} | "
                f"{_cell(r.get('risk'))} | {_cell((r.get('start_date') or '—')[:16])} |"
            )
        return "\n".join(lines)

    inc_rows = (inc_result or {}).get("result", [])
    chg_rows = (chg_result or {}).get("result", [])

    return (
        f"INCIDENTS:\n{_fmt_incidents(inc_rows)}\n\n"
        f"CHANGE REQUESTS:\n{_fmt_changes(chg_rows)}"
    )


@tool
async def get_incident_details(incident_number: str, config: RunnableConfig) -> str:
    """
    Fetch full details for a specific ServiceNow incident by number (e.g. INC0010035).
    Use when the user references a specific incident and wants its description or to troubleshoot it.
    """
    session_id = _sid(config)
    await _push_status(session_id, f"Fetching {incident_number}...")

    try:
        try:
            from atlas.tools.servicenow_tools import get_servicenow_incident as _t
        except ImportError:
            from tools.servicenow_tools import get_servicenow_incident as _t  # type: ignore
        _fn = getattr(_t, "fn", None) or _t
        data = await _fn(incident_number.upper().strip())
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

    Call this early in the investigation — similar past cases often reveal the
    root cause before running live diagnostic tools.

    Args:
        query:   The current issue description (e.g. "10.0.100.100 can't reach 10.0.200.200 port 443").
        devices: Device hostnames in the path — used to match device-tagged incidents.
    """
    session_id = _sid(config)
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
            return "No similar past cases found in memory."
        return format_memory_context(combined)
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
]
