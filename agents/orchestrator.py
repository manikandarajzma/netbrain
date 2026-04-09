"""
Atlas Orchestrator — multi-domain network operations: path tracing, troubleshooting, ServiceNow, Splunk, Panorama, and more.

Uses a LangGraph ReAct agent where the LLM reasons at each step before deciding
which specialist agent to call next based on what the path and prior findings reveal.
"""
import asyncio
import hashlib
import logging
import os
import pathlib
import uuid

import httpx
from langchain_core.tools import tool
from tools.resilience import retry_async, CircuitBreaker, CircuitOpenError
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger("atlas.agents.troubleshoot")

NETBRAIN_AGENT_URL    = "http://localhost:8004"
PANORAMA_AGENT_URL    = "http://localhost:8003"
SPLUNK_AGENT_URL      = "http://localhost:8002"
SERVICENOW_AGENT_URL  = "http://localhost:8005"
NORNIR_AGENT_URL      = "http://localhost:8006"
NETBOX_AGENT_URL      = "http://localhost:8007"
DB_AGENT_URL          = "http://localhost:8008"

# Set STUB_UNREACHABLE_AGENTS=true to return realistic fake data when NetBrain/Panorama
# are offline — useful for testing the full flow end-to-end with real ServiceNow data.
def _stub_agents() -> bool:
    return os.getenv("STUB_UNREACHABLE_AGENTS", "").lower() in ("1", "true", "yes")

_NETBRAIN_STUB = """
Network path trace from 10.0.0.1 to 11.0.0.1 (STUB — NetBrain offline):

Hop 1: EDGE-RTR-01  | Type: Cisco Router    | Egress: GigabitEthernet0/0 → GigabitEthernet0/1
Hop 2: CORE-SW-01   | Type: Cisco Switch    | Egress: GigabitEthernet1/0/24 → GigabitEthernet1/0/1
Hop 3: PA-FW-01     | Type: Palo Alto NGFW  | Egress: ethernet1/1 (trust) → ethernet1/2 (untrust)
Hop 4: DIST-RTR-02  | Type: Cisco Router    | Egress: GigabitEthernet0/2 → GigabitEthernet0/3
Hop 5: 11.0.0.1     | Destination reached

Palo Alto firewalls in path: PA-FW-01
All devices in path: EDGE-RTR-01, CORE-SW-01, PA-FW-01, DIST-RTR-02
""".strip()

_PANORAMA_STUB = """
Security policy match results (STUB — Panorama offline):
PA-FW-01 (device_group: DataCenter-DG): matching rule='Allow-Internal-to-DMZ', action=allow, zones=trust->untrust.
""".strip()

_SPLUNK_STUB = """
Splunk firewall log summary for 10.0.0.1 → 11.0.0.1 (last 24 hours, STUB — Splunk offline):

Deny events: 14 total
  - PA-FW-01: 14 deny events on port 22 (SSH), source 10.0.0.1, destination 11.0.0.1
  - Most recent: 2026-03-25 22:41:07
  - Rule matched: Block-SSH-Inbound

Traffic summary (allowed):
  - 1,842 flows from 10.0.0.1 to 11.0.0.1 (ports 443, 80, 8443)
  - Peak traffic: 2026-03-25 18:00–19:00 UTC

Unique destinations from 10.0.0.1: 7 hosts on 11.0.0.0/24
""".strip()

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "orchestrator.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Redis result cache
# ---------------------------------------------------------------------------

# TTLs per agent type (seconds)
_CACHE_TTL = {
    "netbrain":   900,   # 15 min — paths rarely change mid-investigation
    "panorama":   600,   # 10 min — policies change infrequently
    "splunk":     120,   #  2 min — logs are near-real-time
    "servicenow":  60,   #  1 min — keep short so new incidents are visible quickly
}


def _cache_key(agent: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]
    return f"atlas:ts_cache:{agent}:{digest}"


def _cache_get(agent: str, *parts: str) -> str | None:
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        return r.get(_cache_key(agent, *parts))
    except Exception:
        return None


def _cache_set(agent: str, value: str, *parts: str) -> None:
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.setex(_cache_key(agent, *parts), _CACHE_TTL[agent], value)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Agent caller helper
# ---------------------------------------------------------------------------

async def _call_agent(url: str, task: str, timeout: float = 10.0) -> str:
    cb = CircuitBreaker.for_endpoint(url)

    async def _do_call() -> str:
        payload = {
            "id": str(uuid.uuid4()),
            "message": {"role": "user", "parts": [{"type": "text", "text": task}]},
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            artifacts = data.get("artifacts", [])
            if artifacts:
                text = next(
                    (p.get("text") for p in artifacts[0].get("parts", []) if p.get("type") == "text"),
                    None,
                )
                if text:
                    return text
            return "Agent returned no data."

    try:
        return await retry_async(
            cb, _do_call,
            retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError, httpx.ConnectError),
        )
    except CircuitOpenError as e:
        logger.warning("Agent call to %s blocked by circuit breaker: %s", url, e)
        return f"Agent unavailable: {e}"
    except Exception as e:
        logger.warning("Agent call to %s failed: %s", url, e)
        return f"Agent unavailable: {e}"


# ---------------------------------------------------------------------------
# Tools — each specialist agent exposed as a tool to the ReAct orchestrator
# ---------------------------------------------------------------------------

_session_id: str | None = None
_llm = None  # set by orchestrate_troubleshoot before agent runs
_session_path_hops: list | None = None          # structured path hops from DB trace, injected into final response
_session_reverse_path_hops: list | None = None  # return path hops (dst → src)
_session_path_devices: list[str] = []           # device names in path — used to force SNOW call if LLM skips it
_session_interface_counters: list[dict] = []    # interface counter results, injected into final response
_session_path_text: str = ""                    # raw path trace text — carries ⚠️ anomaly lines for synthesis
_session_path_flags: dict = {}                  # anomaly flags set by _live_path_trace, merged into runbook ctx


@tool
async def call_netbrain_agent(task: str) -> str:
    """
    Trace the hop-by-hop network path between two IP addresses and check if traffic is allowed.
    Use this first for any path troubleshooting query.
    Pass a natural language task describing the source and destination IPs.
    Example: "Trace the path from 10.0.0.1 to 10.0.1.1 and check if traffic is allowed."
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Tracing network path with NetBrain...")
    except Exception:
        pass
    global _session_path_devices
    if _stub_agents():
        logger.info("NetBrain stub active — returning fake path")
        _session_path_devices = _extract_devices_from_netbrain(_NETBRAIN_STUB)
        return _NETBRAIN_STUB
    cached = _cache_get("netbrain", task)
    if cached:
        logger.info("NetBrain cache hit")
        if not _session_path_devices:
            _session_path_devices = _extract_devices_from_netbrain(cached)
        return cached
    result = await _call_agent(NETBRAIN_AGENT_URL, task)
    if result.startswith("Agent unavailable"):
        return result
    _session_path_devices = _extract_devices_from_netbrain(result)
    _cache_set("netbrain", result, task)
    return result


_PLATFORM_TO_TYPE = {
    "arista_eos":   "arista switch",
    "cisco_ios":    "cisco router",
    "cisco_nxos":   "cisco switch",
    "cisco_iosxr":  "cisco router",
}


async def _live_path_trace(src_ip: str, dst_ip: str) -> tuple[str, list]:
    """
    Fully live hop-by-hop path trace via SSH — no database reads.
    1. Find first-hop by querying all devices for a connected route to src_ip's subnet.
       This works even when the gateway is a VIP not assigned to any device interface.
    2. Per-hop: /route (live SSH) → next-hop IP, then /find-device → next device.
    Returns (text_summary, path_hops) for PathVisualization.
    """
    async def _find_device_live(ip: str) -> dict:
        """Resolve a next-hop IP to a device by SSHing to all devices in inventory."""
        async with httpx.AsyncClient(timeout=25.0) as client:
            r = await client.post(f"{NORNIR_AGENT_URL}/find-device", json={"ip": ip})
            return r.json()

    async def _find_first_hop(src: str) -> tuple[str, str | None]:
        """
        Find the first-hop router for a source IP by querying each device for
        a connected route to src's subnet. Returns (device_name, egress_interface).
        This avoids the VIP problem: the gateway IP may not be a real interface address.
        """
        async with httpx.AsyncClient(timeout=25.0) as client:
            registry_resp = await client.get(f"{NORNIR_AGENT_URL}/devices")
            devices = registry_resp.json().get("devices", []) if registry_resp.status_code == 200 else []

        if not devices:
            # fallback: try known devices from nornir inventory
            try:
                import yaml
                from pathlib import Path as _Path
                hosts_file = _Path(__file__).resolve().parent.parent / "nornir" / "inventory" / "hosts.yaml"
                with open(hosts_file) as f:
                    devices = list(yaml.safe_load(f).keys())
            except Exception:
                devices = []

        for device in devices:
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    r = await client.post(f"{NORNIR_AGENT_URL}/route",
                        json={"device": device, "destination": src})
                    route = r.json()
                if route.get("found") and route.get("protocol", "").lower() == "connected":
                    return device, route.get("egress_interface")
            except Exception:
                continue
        return "", None

    text_hops = []
    structured_hops = []
    seen_devices = set()
    MAX_HOPS = 15

    # Step 1: find the first-hop device — the one with a connected route to src subnet
    current_device, gw_interface = await _find_first_hop(src_ip)
    if not current_device:
        return f"Could not find a network device with a connected route to {src_ip} — check inventory.", []
    logger.info("_live_path_trace(%s→%s): first hop = %s via %s", src_ip, dst_ip, current_device, gw_interface)

    # Prepend source host → first router hop
    structured_hops.append({
        "from_device":      src_ip,
        "from_device_type": "host",
        "out_interface":    None,
        "out_zone":         None,
        "device_group":     None,
        "to_device":        current_device,
        "to_device_type":   "switch",
        "in_interface":     gw_interface,
        "in_zone":          None,
    })

    for _ in range(MAX_HOPS):
        if current_device in seen_devices:
            text_hops.append(f"  !! Routing loop detected at {current_device}")
            break
        seen_devices.add(current_device)

        # Live SSH route lookup
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(f"{NORNIR_AGENT_URL}/route",
                    json={"device": current_device, "destination": dst_ip})
                route = r.json()
        except Exception as exc:
            text_hops.append(f"  Hop {len(text_hops)+1}: {current_device} — SSH error: {exc}")
            break

        if not route.get("found"):
            text_hops.append(f"  ⚠️  Hop {len(text_hops)+1}: {current_device} — no route to {dst_ip}")
            _session_path_flags["no_route_device"] = current_device
            break

        egress   = route.get("egress_interface") or ""
        next_hop = route.get("next_hop")
        protocol = route.get("protocol") or ""
        prefix   = route.get("prefix") or ""

        # Check if routing via management (data-plane failure)
        if egress and egress.lower().startswith("management"):
            text_hops.append(
                f"  ⚠️  Hop {len(text_hops)+1}: **{current_device}** is routing {dst_ip} via "
                f"**{egress}** (default route 0.0.0.0/0) — data-plane interfaces are likely DOWN. "
                f"Traffic is NOT taking the expected path."
            )
            structured_hops.append({
                "from_device":      current_device,
                "from_device_type": "switch",
                "out_interface":    egress,
                "out_zone":         None,
                "device_group":     None,
                "to_device":        f"⚠️ Mgmt fallback ({egress})",
                "to_device_type":   "host",
                "in_interface":     None,
                "in_zone":          None,
            })
            _session_path_flags["mgmt_routing_detected"] = True
            _session_path_flags["mgmt_routing_device"] = current_device
            break

        # Check if the egress interface is actually up
        intf_up = True
        if egress and not egress.lower().startswith("management"):
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    rs = await client.post(f"{NORNIR_AGENT_URL}/interface-status",
                        json={"device": current_device, "interface": egress})
                    intf_status = rs.json()
                    if not intf_status.get("up", True):
                        intf_up = False
                        text_hops.append(
                            f"  ⚠️  Hop {len(text_hops)+1}: {current_device} | Egress: {egress} "
                            f"is DOWN (line-protocol: {intf_status.get('line_protocol', '?')}) — "
                            f"traffic is likely following default route via management, not the data plane"
                        )
                        structured_hops.append({
                            "from_device":      current_device,
                            "from_device_type": "switch",
                            "out_interface":    egress,
                            "out_zone":         None,
                            "device_group":     None,
                            "to_device":        f"⚠️ {egress} DOWN",
                            "to_device_type":   "host",
                            "in_interface":     None,
                            "in_zone":          None,
                        })
                        break
            except Exception:
                pass  # if status check fails, continue tracing

        if not intf_up:
            break

        text_hops.append(
            f"  Hop {len(text_hops)+1}: {current_device} | Egress: {egress} | Protocol: {protocol} "
            f"| Prefix: {prefix} | Next-hop: {next_hop or 'directly connected'}"
        )

        if not next_hop:
            # Directly connected — live ARP lookup for in_interface
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    r = await client.post(f"{NORNIR_AGENT_URL}/arp",
                        json={"device": current_device, "ip": dst_ip})
                    arp = r.json()
            except Exception:
                arp = {}
            in_iface = arp.get("interface") if arp.get("found") else None
            if arp.get("found"):
                text_hops.append(
                    f"  Destination {dst_ip} reachable via ARP on {current_device} "
                    f"port {arp['interface']} (MAC {arp.get('mac', '?')})"
                )
            else:
                text_hops.append(
                    f"  Destination {dst_ip} directly connected on {current_device} (no ARP entry)"
                )
            structured_hops.append({
                "from_device":      current_device,
                "from_device_type": "switch",
                "out_interface":    egress,
                "out_zone":         None,
                "device_group":     None,
                "to_device":        dst_ip,
                "to_device_type":   "host",
                "in_interface":     in_iface,
                "in_zone":          None,
            })
            break

        # Map next-hop IP → device via live SSH (no DB)
        next_dev = await _find_device_live(next_hop)
        if not next_dev.get("found"):
            text_hops.append(f"  Next-hop {next_hop} not found on any live device — path ends here")
            break

        next_device = next_dev["device"]
        in_iface = next_dev["interface"]
        structured_hops.append({
            "from_device":      current_device,
            "from_device_type": "switch",
            "out_interface":    egress,
            "out_zone":         None,
            "device_group":     None,
            "to_device":        next_device,
            "to_device_type":   "switch",
            "in_interface":     in_iface,
            "in_zone":          None,
        })
        current_device = next_device

    devices_in_path = list(seen_devices)
    text = (
        f"Path from {src_ip} to {dst_ip} (live):\n"
        + "\n".join(text_hops)
        + f"\n\nAll devices in path: {', '.join(devices_in_path)}"
        + "\nPalo Alto firewalls in path: none"
    )
    return text, structured_hops


@tool
async def call_nornir_path_agent(task: str) -> str:
    """
    Trace the hop-by-hop network path between two IP addresses using live SSH
    to each device — no database or cache. Always reflects current network state.
    Use this INSTEAD of call_netbrain_agent when the user explicitly says to avoid
    NetBrain (e.g. "don't use NetBrain", "use Nornir", "without NetBrain").
    Pass a natural language task with the source and destination IPs.
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Tracing network path via live SSH...")
    except Exception:
        pass
    global _session_path_devices

    src_ip, dst_ip = _extract_ips(task)
    if not src_ip or not dst_ip:
        return "Could not extract source and destination IPs from the task."

    try:
        result, path_hops = await _live_path_trace(src_ip, dst_ip)
    except Exception as e:
        logger.exception("Live path trace error")
        result, path_hops = f"Path trace error: {e}", []

    global _session_path_hops, _session_path_text
    if path_hops:
        _session_path_hops = path_hops
    _session_path_text = result
    _session_path_devices = _extract_devices_from_netbrain(result)

    return result


@tool
async def call_netbox_path_agent(task: str) -> str:
    """
    Trace the hop-by-hop network path using pre-collected routing/ARP/MAC data in PostgreSQL
    and NetBox IPAM — WITHOUT NetBrain or live SSH.
    Use this INSTEAD of call_netbrain_agent when the user asks to use the database
    or collected data (e.g. "use the database", "use collected data", "use NetBox").
    """
    return await call_nornir_path_agent.ainvoke({"task": task})


@tool
async def call_panorama_agent(
    source_ip: str,
    dest_ip: str,
    firewall_hostnames: list[str],
    port: str = "",
    protocol: str = "tcp",
) -> str:
    """
    Run Panorama's 'test security-policy-match' against each firewall in the path.
    This uses the firewall's own policy evaluation engine — the same test as Panorama's
    Test Configuration UI — and returns the exact matching rule for each firewall.

    Always call this AFTER call_netbrain_agent so you can pass the firewall hostnames
    found in the path. Extract every Palo Alto firewall hostname from the NetBrain path
    and pass them as firewall_hostnames.

    Args:
        source_ip:           Source IP address (e.g. "10.0.0.1")
        dest_ip:             Destination IP address (e.g. "11.0.0.1")
        firewall_hostnames:  List of Palo Alto firewall hostnames from the NetBrain path
                             (e.g. ["PA-FW-01", "PA-FW-02"])
        port:                Destination port if specified by user (e.g. "443"). Leave "" if unknown.
        protocol:            Protocol: "tcp", "udp", or "icmp". Defaults to "tcp".

    Example: source_ip="10.0.0.1", dest_ip="11.0.0.1", firewall_hostnames=["PA-FW-01"], port="443", protocol="tcp"
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Checking Panorama security policies...")
    except Exception:
        pass
    try:
        from atlas.mcp_client import call_mcp_tool
    except ImportError:
        from mcp_client import call_mcp_tool

    if not firewall_hostnames:
        return "No firewall hostnames provided — cannot run security policy match test. Pass the Palo Alto firewall names from the NetBrain path."

    if _stub_agents():
        logger.info("Panorama stub active — returning fake policy match for %s", firewall_hostnames)
        lines = [
            f"{fw} (device_group: DataCenter-DG): matching rule='Allow-Internal-to-DMZ', action=allow, zones=trust->untrust."
            for fw in firewall_hostnames
        ]
        return "Security policy match results:\n" + "\n".join(lines)

    import asyncio as _asyncio

    async def _test_one(fw: str) -> str:
        args: dict = {
            "firewall_hostname": fw,
            "source_ip": source_ip,
            "dest_ip": dest_ip,
        }
        if port:
            args["dest_port"] = port.strip()
        if protocol:
            args["protocol"] = protocol.strip()
        result = await call_mcp_tool("test_panorama_security_policy_match", args, timeout=45.0)
        if not result:
            return f"{fw}: no response from Panorama."
        if "error" in result:
            return f"{fw}: {result['error']}"
        rule = result.get("matching_rule") or "no-match"
        action = result.get("action", "unknown")
        dg = result.get("device_group", "unknown")
        from_z = result.get("from_zone") or "?"
        to_z = result.get("to_zone") or "?"
        port_info = f" port {port}/{protocol}" if port else ""
        return (
            f"{fw} (device_group: {dg}){port_info}: "
            f"matching rule='{rule}', action={action}, zones={from_z}->{to_z}."
        )

    cache_parts = [source_ip, dest_ip, port, protocol] + sorted(firewall_hostnames)
    cached = _cache_get("panorama", *cache_parts)
    if cached:
        logger.info("Panorama cache hit")
        return cached
    lines = await _asyncio.gather(*[_test_one(fw) for fw in firewall_hostnames])
    result = "Security policy match results:\n" + "\n".join(lines)
    _cache_set("panorama", result, *cache_parts)
    return result


@tool
async def call_splunk_agent(task: str) -> str:
    """
    Check Splunk firewall logs for recent deny events and traffic patterns for an IP address.
    Use when you need to correlate path findings with actual observed traffic — deny counts, blocked ports, destination spread.
    Pass a natural language task describing the IP and what to look for.
    Example: "Check Splunk for recent deny events and traffic summary for 10.0.0.1 in the last 24 hours."
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Querying Splunk for deny events...")
    except Exception:
        pass
    if _stub_agents():
        logger.info("Splunk stub active — returning mock deny events")
        return _SPLUNK_STUB
    cached = _cache_get("splunk", task)
    if cached:
        logger.info("Splunk cache hit")
        return cached
    result = await _call_agent(SPLUNK_AGENT_URL, task, timeout=10.0)
    if not result.startswith("Agent unavailable"):
        _cache_set("splunk", result, task)
    return result


@tool
async def call_servicenow_agent(
    device_names: str,
    source_ip: str = "",
    dest_ip: str = "",
    port: str = "",
    query_type: str = "incidents_and_changes",
    hours_back: int = 24,
) -> str:
    """
    Search ServiceNow for incidents AND change requests related to devices in the network path.
    ALWAYS call this after call_netbrain_agent. Pass every device hostname from the path.

    Args:
        device_names: All device hostnames from the NetBrain path (e.g. ["EDGE-RTR-01", "PA-FW-01"])
        source_ip:    Source IP address (e.g. "10.0.0.1")
        dest_ip:      Destination IP address (e.g. "11.0.0.1")
        port:         Destination port if the user specified one (e.g. "22" for SSH) — improves ticket relevance

    Example: device_names=["EDGE-RTR-01","CORE-SW-01","PA-FW-01","DIST-RTR-02"],
             source_ip="10.0.0.1", dest_ip="11.0.0.1", port="22"
    """
    import json as _json, ast as _ast
    if isinstance(device_names, str):
        try:
            device_names = _json.loads(device_names)
        except Exception:
            try:
                device_names = _ast.literal_eval(device_names)
            except Exception:
                device_names = [d.strip().strip("'\"") for d in device_names.strip("[]").split(",") if d.strip()]
    if not isinstance(device_names, list):
        device_names = [str(device_names)]

    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Checking ServiceNow for related incidents and changes...")
    except Exception:
        pass

    cache_parts = sorted(device_names) + [source_ip, dest_ip, port]
    cached = _cache_get("servicenow", *cache_parts)
    if cached:
        logger.info("ServiceNow cache hit")
        return cached

    try:
        try:
            from atlas.mcp_client import call_mcp_tool
        except ImportError:
            from mcp_client import call_mcp_tool

        import asyncio as _asyncio

        # Build OR query from device names + IPs (+ port / SSH hints for scoped searches)
        terms = [str(d).strip() for d in device_names if d and str(d).strip()]
        for ip in (source_ip, dest_ip):
            ip = (ip or "").strip()
            if ip:
                terms.append(ip)
        p = (port or "").strip()
        if p:
            terms.append(p)
            if p == "22":
                terms.append("ssh")
        terms = list(dict.fromkeys(terms))  # de-dupe, preserve order

        if not terms:
            return (
                "ServiceNow search skipped: no device names or IP addresses were provided. "
                "Pass device_names from the NetBrain path and source_ip/dest_ip from the problem."
            )

        query = " OR ".join(terms)

        # Run incident and/or change request searches based on query_type
        # Change requests use a wider window (at least 7 days) since changes often predate incidents.
        chg_hours = max(hours_back * 7, 168)
        if query_type == "changes_only":
            inc_result = {"result": []}
            chg_result = await call_mcp_tool(
                "list_servicenow_change_requests",
                {"query": query, "limit": 20, "updated_within_hours": hours_back},
                timeout=30.0,
            )
        elif query_type == "incidents_only":
            inc_result = await call_mcp_tool(
                "search_servicenow_incidents",
                {"query": query, "limit": 10, "updated_within_hours": hours_back},
                timeout=30.0,
            )
            chg_result = {"result": []}
        else:  # incidents_and_changes (default) or anything else
            inc_result, chg_result = await _asyncio.gather(
                call_mcp_tool("search_servicenow_incidents", {"query": query, "limit": 10, "updated_within_hours": hours_back}, timeout=30.0),
                call_mcp_tool("list_servicenow_change_requests", {"query": query, "limit": 20, "updated_within_hours": chg_hours}, timeout=30.0),
            )

        def _dedup(records: list[dict]) -> list[dict]:
            """Deduplicate records by number."""
            seen: set[str] = set()
            result: list[dict] = []
            for r in records:
                num = r.get("number", "")
                if num and num not in seen:
                    seen.add(num)
                    result.append(r)
            return result

        def _record_blob(r: dict) -> str:
            chunks: list[str] = []
            for k in ("short_description", "description", "work_notes", "close_notes", "cmdb_ci"):
                v = r.get(k)
                if isinstance(v, dict):
                    v = v.get("display_value", "") or v.get("value", "")
                chunks.append(str(v or ""))
            return " ".join(chunks).lower()

        def _heuristic_relevant(records: list[dict]) -> list[dict]:
            """Keep records where cmdb_ci matches a path device (primary), falling back to blob search."""
            if not records or not terms:
                return []
            needles = [t.lower() for t in terms if t]
            out: list[dict] = []
            for r in records:
                ci = r.get("cmdb_ci") or ""
                if isinstance(ci, dict):
                    ci = ci.get("display_value", "") or ci.get("value", "")
                ci_lower = str(ci).lower()
                # Primary: cmdb_ci is an exact/contains match on a path device name
                if ci_lower and any(n in ci_lower or ci_lower in n for n in needles):
                    out.append(r)
                    continue
                # Fallback: device name appears in short_description only (avoid noisy fields)
                desc = str(r.get("short_description") or "").lower()
                if any(n in desc for n in needles):
                    out.append(r)
            return out

        async def _llm_filter_relevant(records: list[dict], record_type: str) -> list[dict]:
            """Use LLM to filter records relevant to the problem being investigated."""
            import re as _re3
            if not records:
                return []
            prefix = "INC" if record_type == "incidents" else "CHG"
            summaries = "\n".join(
                f"{r.get('number','?')}: {r.get('short_description','')}"
                for r in records
            )
            port_bit = f", port {port}" if (port or "").strip() else ""
            devices_str = ", ".join(str(d) for d in device_names if d)
            # Build a labeled list for the LLM to judge one-by-one
            labeled = "\n".join(
                f"{r.get('number','?')}: {r.get('short_description','')}"
                for r in records
            )
            prompt = (
                f"Network path devices: {devices_str}\n"
                f"Traffic: {source_ip} -> {dest_ip}{port_bit}\n\n"
                f"For each record below, write RELEVANT or NOT_RELEVANT.\n"
                f"Mark RELEVANT only if it involves a path device "
                f"(any issue type — networking, hardware, CPU, memory, config, or service impact).\n\n"
                f"{labeled}\n\n"
                f"Answer in this exact format, one line per record:\n"
                + "\n".join(f"{r.get('number','?')}: RELEVANT or NOT_RELEVANT" for r in records)
            )
            try:
                from langchain_core.messages import HumanMessage
                response = await _llm.ainvoke([HumanMessage(content=prompt)])
                text = (response.content or "").strip()
                # Parse lines like "INC0010001: RELEVANT" (must end with RELEVANT, not "RELEVANT or NOT_RELEVANT")
                relevant_nums = set(
                    m.group(1).upper()
                    for m in _re3.finditer(r'\b(' + prefix + r'\d+)\b[^:\n]*:\s*RELEVANT\s*$', text, _re3.IGNORECASE | _re3.MULTILINE)
                )
                picked = [r for r in records if r.get("number", "").upper() in relevant_nums]
                return picked if picked else _heuristic_relevant(records)
            except Exception as e:
                logger.warning("LLM relevancy filter failed: %s — using heuristic fallback", e)
                return _heuristic_relevant(records)

        def _cell(v, maxlen=60):
            s = str(v or "-").strip().replace("|", "/").replace("\n", " ")
            return s[:maxlen] if len(s) > maxlen else s

        def _fmt_incidents(rows) -> str:
            if not rows:
                return "No incidents found."
            lines = [
                "| Number | Device/CI | Short Description | Status | Priority | Assigned To | Opened | Resolved | Resolution Notes |",
                "|--------|-----------|-------------------|--------|----------|-------------|--------|----------|-----------------|",
            ]
            for r in rows:
                num = _cell(r.get("number"))
                # Use cmdb_ci as Device/CI — fall back to extracting from short_description
                ci_raw = r.get("cmdb_ci") or ""
                if isinstance(ci_raw, dict):
                    ci_raw = ci_raw.get("display_value") or ci_raw.get("value") or ""
                if not ci_raw:
                    import re as _re2
                    desc_raw = r.get("short_description", "")
                    m = _re2.match(r'^([A-Za-z][A-Za-z0-9\-]+\d*)\s*[:\-]', desc_raw)
                    ci_raw = m.group(1) if m else "-"
                device = _cell(str(ci_raw) or "-")
                desc_raw = r.get("short_description", "")
                desc = _cell(desc_raw)
                state = _cell(r.get("state"))
                pri = _cell(r.get("priority"))
                who = _cell((r.get("assigned_to") or {}).get("display_value") if isinstance(r.get("assigned_to"), dict) else r.get("assigned_to"))
                opened = _cell((r.get("opened_at") or "-")[:16])
                resolved = _cell((r.get("resolved_at") or "-")[:16])
                notes = _cell(r.get("close_notes") or r.get("work_notes") or "-", 200)
                lines.append(f"| {num} | {device} | {desc} | {state} | {pri} | {who} | {opened} | {resolved} | {notes} |")
            return "\n".join(lines)

        def _fmt_changes(rows) -> str:
            if not rows:
                return "No change requests found."
            lines = [
                "| Number | Device/CI | Short Description | Status | Risk | Assigned To | Scheduled | Completed | Close Notes |",
                "|--------|-----------|-------------------|--------|------|-------------|-----------|-----------|-------------|",
            ]
            for r in rows:
                num = _cell(r.get("number"))
                desc_raw = r.get("short_description", "")
                import re as _re2
                # Try start of description first (e.g. "DIST-RTR-02: ..."), then "on DEVICE" pattern
                device_match = (
                    _re2.match(r'^([A-Z][A-Z0-9\-]+-\d+)\s*[:\-]', desc_raw) or
                    _re2.search(r'\bon\s+([A-Z][A-Z0-9\-]+-\d+)', desc_raw)
                )
                if device_match:
                    device = device_match.group(1)
                else:
                    ci = r.get("cmdb_ci", "")
                    device = (ci.get("display_value") or ci.get("value") or "-") if isinstance(ci, dict) else (ci or "-")
                desc = _cell(desc_raw)
                state = _cell(r.get("state"))
                risk = _cell(r.get("risk"))
                who = _cell((r.get("assigned_to") or {}).get("display_value") if isinstance(r.get("assigned_to"), dict) else r.get("assigned_to"))
                sched = _cell((r.get("start_date") or "-")[:16])
                done = _cell((r.get("end_date") or "-")[:16])
                notes = _cell(r.get("close_notes") or r.get("work_notes") or "-", 80)
                lines.append(f"| {num} | {device} | {desc} | {state} | {risk} | {who} | {sched} | {done} | {notes} |")
            return "\n".join(lines)

        inc_rows = _dedup(inc_result.get("result", []) if isinstance(inc_result, dict) else [])
        chg_rows = _dedup(chg_result.get("result", []) if isinstance(chg_result, dict) else [])

        # SNOW query is already scoped to device names — use heuristic filter only
        inc_filtered = _heuristic_relevant(inc_rows)
        chg_filtered = _heuristic_relevant(chg_rows)

        result = (
            f"INCIDENTS:\n{_fmt_incidents(inc_filtered)}\n\n"
            f"CHANGE REQUESTS:\n{_fmt_changes(chg_filtered)}"
        )
        _cache_set("servicenow", result, *cache_parts)
        return result

    except Exception as e:
        logger.warning("ServiceNow direct MCP call failed: %s", e)
        return f"ServiceNow unavailable: {e}"


@tool
async def get_incident_details(incident_number: str) -> str:
    """
    Look up a specific ServiceNow incident by number (e.g. INC0010035) and return its full details.
    Use this when the user references a specific incident number and wants to know its details or troubleshoot it.

    Args:
        incident_number: The incident number, e.g. "INC0010035"
    """
    try:
        from tools.servicenow_tools import get_servicenow_incident as _t
        _fn = getattr(_t, 'fn', None) or _t
        data = await _fn(incident_number.upper().strip())
        if "error" in data:
            return f"Incident not found: {data['error']}"
        r = data.get("result", {})
        lines = [
            f"**{r.get('number')}** — {r.get('short_description', '')}",
            f"State: {r.get('state', '?')} | Priority: {r.get('priority', '?')} | Impact: {r.get('impact', '?')}",
            f"Opened: {r.get('opened_at', '?')} | Assigned to: {r.get('assigned_to', {}).get('display_value', 'Unassigned')}",
            f"Assignment group: {r.get('assignment_group', {}).get('display_value', '?')}",
        ]
        desc = r.get('description') or r.get('short_description') or ''
        if desc:
            lines.append(f"Description: {desc}")
        close_notes = r.get('close_notes') or ''
        if close_notes:
            lines.append(f"Resolution: {close_notes}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error fetching incident: {e}"


@tool
async def call_interface_counters_agent(devices_and_interfaces: str) -> str:
    """
    Fetch interface error and discard counters for specific interfaces on path devices via SSH.
    Use after a path trace to check for CRC errors, input errors, output drops on each link.

    Args:
        devices_and_interfaces: JSON string of a list of {device: str, interfaces: [str]} dicts.
            Each entry specifies a device and which interfaces to check.
            Example: '[{"device": "arista1", "interfaces": ["Ethernet1", "Ethernet3"]}, {"device": "arista2", "interfaces": ["Ethernet2"]}]'

    If you don't have the interface names, use the path trace output — look for
    out_interface / in_interface fields on each hop.
    """
    import asyncio as _asyncio
    import aiohttp
    import json as _json

    # Parse JSON string input
    if isinstance(devices_and_interfaces, str):
        try:
            devices_and_interfaces = _json.loads(devices_and_interfaces)
        except Exception:
            return "Invalid devices_and_interfaces format — expected a JSON list of {device, interfaces} dicts."

    if not devices_and_interfaces:
        return "No devices/interfaces specified."

    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Checking interface error counters...")
    except Exception:
        pass

    async def _fetch_one(session, entry: dict) -> tuple[str, dict]:
        device = entry.get("device", "")
        interfaces = entry.get("interfaces", [])
        if not device or not interfaces:
            return "", {}
        try:
            async with session.post(
                f"{NORNIR_AGENT_URL}/interface-counters",
                json={"device": device, "interfaces": interfaces},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
        except Exception as exc:
            structured = {"device": device, "window_s": 6, "active": [], "clean": [], "ssh_error": str(exc)}
            return f"{device}: unreachable ({exc})", structured

        if "error" in data:
            structured = {"device": device, "window_s": 6, "active": [], "clean": [], "ssh_error": data["error"]}
            return f"{device}: {data['error']}", structured

        active   = data.get("active_errors", [])
        clean    = data.get("clean_interfaces", [])
        interval = data.get("poll_interval_s", 3)
        iters    = data.get("iterations", 3)
        window   = interval * (iters - 1)

        structured = {"device": device, "window_s": window, "active": active, "clean": clean}

        if not active:
            return f"{device}: all interfaces clean (no incrementing errors over {window}s)", structured

        rows = []
        for c in active:
            intf = c.get("interface", "?")
            if "error" in c:
                rows.append(f"  {intf}: SSH error — {c['error']}")
                continue
            d = c.get("delta_9s", {})
            parts = [f"{k}+{v}" for k, v in d.items() if v > 0]
            last = c.get("last_clear", "never")
            rows.append(f"  {intf}: ACTIVE — {', '.join(parts)} over {window}s (cleared: {last})")
        if clean:
            rows.append(f"  clean: {', '.join(clean)}")
        return f"{device} (polled {iters}×/{interval}s):\n" + "\n".join(rows), structured

    async with aiohttp.ClientSession() as session:
        raw_results = await _asyncio.gather(*[_fetch_one(session, e) for e in devices_and_interfaces])

    global _session_interface_counters
    lines = []
    structured_list = []
    for text, structured in raw_results:
        if text:
            lines.append(text)
        if structured and structured.get("device"):
            _session_interface_counters.append(structured)
            structured_list.append(structured)

    # Health baseline: store snapshot (fire-and-forget), surface trends, and device reputation
    try:
        from agent_memory import store_device_health_snapshot, get_health_trend, get_device_reputation
        import asyncio as _asyncio2
        _asyncio2.get_event_loop().run_in_executor(
            None, store_device_health_snapshot, structured_list
        )
        for snap in structured_list:
            dev = snap.get("device", "")
            for c in snap.get("active", []):
                intf = c.get("interface", "")
                if intf and "error" not in c:
                    trend = get_health_trend(dev, intf, c.get("delta_9s", {}))
                    if trend:
                        lines.append(f"  ⚡ TREND: {trend}")
            if dev:
                rep = get_device_reputation(dev, window_days=7)
                if rep["failure_count"] >= 3:
                    cats = ", ".join(f"{k}×{v}" for k, v in rep["categories"].items() if k != "was_in_path")
                    lines.append(f"  ⚠️ REPUTATION: {dev} was the diagnosed failure point {rep['failure_count']}x in the last 7 days ({cats})")
    except Exception as _he:
        logger.debug("Health baseline: %s", _he)

    return "Interface counters:\n" + "\n".join(lines) if lines else "No counter data returned."


TROUBLESHOOT_TOOLS = [
    call_netbrain_agent,
    call_nornir_path_agent,
    call_netbox_path_agent,
    call_panorama_agent,
    call_splunk_agent,
    call_servicenow_agent,
    call_interface_counters_agent,
]

# Map tool name strings back to callables for plan execution
_TOOL_MAP = {t.name: t for t in TROUBLESHOOT_TOOLS}

# Fingerprint of the current tool set — changes automatically when tools are added/removed
_TOOL_FINGERPRINT = hashlib.sha256(
    ",".join(sorted(t.name for t in TROUBLESHOOT_TOOLS)).encode()
).hexdigest()[:12]


def _extract_ips(text: str) -> tuple[str, str]:
    """Extract first two IPs from text (source, dest). Returns ('', '') if not found."""
    import re as _re
    ips = _re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
    src = ips[0] if len(ips) > 0 else ""
    dst = ips[1] if len(ips) > 1 else ""
    return src, dst


def _extract_port(text: str) -> str:
    """Extract port number from text like 'port 443' or 'tcp/443'."""
    import re as _re
    m = _re.search(r'\bport\s+(\d+)\b|/(\d+)\b|\b(\d+)/(tcp|udp)\b', text, _re.IGNORECASE)
    if m:
        return next(g for g in m.groups() if g and g.isdigit())
    return ""


def _extract_firewalls_from_netbrain(nb_output: str) -> list[str]:
    """Parse 'Palo Alto firewalls in path: FW1, FW2' from NetBrain output."""
    import re as _re
    m = _re.search(r'Palo Alto firewalls? in path:\s*(.+)', nb_output, _re.IGNORECASE)
    if m:
        return [fw.strip() for fw in m.group(1).split(',') if fw.strip() and fw.strip().lower() != 'none']
    # Fallback: find any line with "Palo Alto" and grab the device name
    devices = []
    for line in nb_output.splitlines():
        if 'palo alto' in line.lower():
            dm = _re.match(r'Hop\s+\d+:\s+(\S+)', line)
            if dm:
                devices.append(dm.group(1))
    return devices


def _extract_devices_from_netbrain(nb_output: str) -> list[str]:
    """Parse 'All devices in path: D1, D2, ...' from NetBrain output."""
    import re as _re
    m = _re.search(r'All devices in path:\s*(.+)', nb_output, _re.IGNORECASE)
    if m:
        return [d.strip() for d in m.group(1).split(',') if d.strip()]
    # Fallback: collect all Hop device names
    devices = []
    for line in nb_output.splitlines():
        dm = _re.match(r'Hop\s+\d+:\s+(\S+)', line)
        if dm and not _re.match(r'^\d', dm.group(1)):  # skip IPs
            devices.append(dm.group(1))
    return devices


def _extract_plan_from_messages(messages: list) -> list[str]:
    """Extract the ordered list of tool names called during a ReAct run."""
    from langchain_core.messages import AIMessage
    tool_sequence = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, 'tool_calls', None):
            for tc in m.tool_calls:
                name = tc.get('name') if isinstance(tc, dict) else getattr(tc, 'name', None)
                if name and name not in tool_sequence:
                    tool_sequence.append(name)
    return tool_sequence


async def _execute_plan(
    tool_sequence: list[str],
    prompt: str,
) -> list[str]:
    """Execute a cached tool plan without the ReAct LLM loop.
    Returns tool outputs in the same format as the ReAct agent produces."""
    import asyncio as _asyncio

    src_ip, dst_ip = _extract_ips(prompt)
    port = _extract_port(prompt)
    tool_outputs = []

    # Path agent always runs first (others depend on its output)
    nb_output = ""
    if "call_netbrain_agent" in tool_sequence:
        nb_output = await call_netbrain_agent.ainvoke({"task": prompt})
        tool_outputs.append(nb_output)
    elif "call_nornir_path_agent" in tool_sequence:
        nb_output = await call_nornir_path_agent.ainvoke({"task": prompt})
        tool_outputs.append(nb_output)
    elif "call_netbox_path_agent" in tool_sequence:
        nb_output = await call_netbox_path_agent.ainvoke({"task": prompt})
        tool_outputs.append(nb_output)

    # Parse device info from path output for downstream tools
    firewalls = _extract_firewalls_from_netbrain(nb_output) if nb_output else []
    devices = _extract_devices_from_netbrain(nb_output) if nb_output else []

    # Remaining tools run in parallel
    async def _run_tool(name: str):
        if name in ("call_netbrain_agent", "call_nornir_path_agent", "call_netbox_path_agent"):
            return None  # already ran
        if name == "call_panorama_agent":
            if not firewalls:
                return "No Palo Alto firewalls in path — Panorama check skipped."
            return await call_panorama_agent.ainvoke({
                "source_ip": src_ip,
                "dest_ip": dst_ip,
                "firewall_hostnames": firewalls,
                "port": port,
                "protocol": "tcp",
            })
        if name == "call_splunk_agent":
            return await call_splunk_agent.ainvoke({"task": prompt})
        if name == "call_servicenow_agent":
            import json as _j
            return await call_servicenow_agent.ainvoke({
                "device_names": _j.dumps(devices) if isinstance(devices, list) else devices,
                "source_ip": src_ip,
                "dest_ip": dst_ip,
                "port": port,
            })
        return None

    _PATH_AGENTS = {"call_netbrain_agent", "call_nornir_path_agent", "call_netbox_path_agent"}
    parallel_tools = [n for n in tool_sequence if n not in _PATH_AGENTS]
    results = await _asyncio.gather(*[_run_tool(n) for n in parallel_tools], return_exceptions=True)
    for r in results:
        if r and not isinstance(r, Exception):
            tool_outputs.append(str(r))

    return tool_outputs


# ---------------------------------------------------------------------------
# Runbook infrastructure
# ---------------------------------------------------------------------------

async def _ping_via_nornir(
    device: str, destination: str, source_interface: str = "", vrf: str = ""
) -> dict:
    """POST to nornir agent /ping and return the result dict."""
    cb = CircuitBreaker.for_endpoint(NORNIR_AGENT_URL + "/ping")
    async def _do():
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{NORNIR_AGENT_URL}/ping",
                json={
                    "device": device,
                    "destination": destination,
                    "source_interface": source_interface,
                    "vrf": vrf,
                },
            )
            resp.raise_for_status()
            return resp.json()
    try:
        return await retry_async(cb, _do, retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError))
    except CircuitOpenError as exc:
        return {"success": False, "error": str(exc), "device": device, "destination": destination}
    except Exception as exc:
        return {"success": False, "error": str(exc), "device": device, "destination": destination}


async def _routing_check_via_nornir(devices: list[str], destination: str, vrf: str = "") -> dict:
    """POST to nornir agent /routing-check and return the result dict."""
    cb = CircuitBreaker.for_endpoint(NORNIR_AGENT_URL + "/routing-check")
    async def _do():
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{NORNIR_AGENT_URL}/routing-check",
                json={"devices": devices, "destination": destination, "vrf": vrf},
            )
            resp.raise_for_status()
            return resp.json()
    try:
        return await retry_async(cb, _do, retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError))
    except Exception as exc:
        return {"error": str(exc), "destination": destination, "hops": {}}


async def _nornir_diagnose(task: str) -> str:
    """Send a free-form diagnostic task to the nornir agent's ReAct loop."""
    import uuid
    cb = CircuitBreaker.for_endpoint(NORNIR_AGENT_URL)
    async def _do():
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{NORNIR_AGENT_URL}/",
                json={
                    "id": str(uuid.uuid4()),
                    "message": {"parts": [{"type": "text", "text": task}]},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # A2A response: artifacts[0].parts[0].text
            if data.get("status", {}).get("state") == "failed":
                return ""
            try:
                return data["artifacts"][0]["parts"][0]["text"]
            except (KeyError, IndexError, TypeError):
                return ""
    try:
        return await retry_async(cb, _do, retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError))
    except Exception as exc:
        return f"Follow-up investigation unavailable: {exc}"


async def _infer_src_vrf(src_ip: str, first_hop_device: str) -> str:
    """Query the DB to find which VRF contains src_ip on first_hop_device.

    Strategy: longest-prefix-match in routing_table; fall back to interface_ips.
    Returns 'default' if nothing is found or DB is unavailable.
    """
    if not src_ip or not first_hop_device:
        return "default"
    try:
        try:
            from atlas.db import fetchrow
        except ImportError:
            from db import fetchrow

        # 1. Try routing table — longest-prefix match toward src_ip on the first hop
        row = await fetchrow(
            """
            SELECT vrf FROM routing_table
            WHERE device = $1 AND $2::inet << prefix
            ORDER BY masklen(prefix) DESC LIMIT 1
            """,
            first_hop_device, src_ip,
        )
        if row and row["vrf"]:
            return row["vrf"]

        # 2. Try interface_ips — maybe src_ip is directly connected to the device
        row = await fetchrow(
            "SELECT vrf FROM interface_ips WHERE device = $1 AND $2::inet << (ip::text || '/' || prefix_len)::cidr LIMIT 1",
            first_hop_device, src_ip,
        )
        if row and row["vrf"]:
            return row["vrf"]
    except Exception as exc:
        logger.warning("_infer_src_vrf: DB query failed: %s", exc)

    return "default"


async def _post_path_trace(result: str, ctx: dict) -> None:
    """Extract structured fields from path agent output into the runbook context."""
    import re as _re

    devices   = _extract_devices_from_netbrain(result)
    firewalls = _extract_firewalls_from_netbrain(result)

    ctx["path_devices"]       = devices
    ctx["has_firewalls"]      = bool(firewalls)
    ctx["firewall_hostnames"] = firewalls

    # Extract first hop device + interfaces from path text or structured hops.
    # first_hop_lan_interface  = interface on the first device facing the SOURCE (used for ping source)
    # first_hop_egress_interface = interface on the first device facing the DESTINATION
    if _session_path_hops:
        # First hop: from_device=src_ip, to_device=first_router, in_interface=LAN interface
        first_h = _session_path_hops[0]
        ctx["first_hop_device"]            = first_h.get("to_device", "") or (devices[0] if devices else "")
        ctx["first_hop_lan_interface"]     = first_h.get("in_interface", "") or ""
        # Egress interface: out_interface of the hop where from_device == first_hop_device
        ctx["first_hop_egress_interface"]  = ""
        first_dev = ctx["first_hop_device"]
        for h in _session_path_hops:
            if h.get("from_device") == first_dev and h.get("out_interface"):
                ctx["first_hop_egress_interface"] = h["out_interface"]
                break
        # Last-hop device: the network device directly connected to the destination
        # = from_device of the last hop where to_device is an IP (not a hostname)
        import re as _re2
        ctx["last_hop_device"] = ""
        ctx["last_hop_egress_interface"] = ""
        for h in reversed(_session_path_hops):
            to_d   = h.get("to_device", "")
            from_d = h.get("from_device", "")
            if _re2.match(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$", to_d) and from_d and not _re2.match(r"^\d", from_d):
                ctx["last_hop_device"] = from_d
                ctx["last_hop_egress_interface"] = h.get("out_interface", "")
                break
    else:
        # Stub/NetBrain format: "Hop 1: EDGE-RTR-01  | ... | Egress: Gi0/0 → Gi0/1"
        m = _re.search(
            r"Hop\s+1:\s+(\S+).*?Egress:\s+(\S+)",
            result,
            _re.IGNORECASE | _re.DOTALL,
        )
        if m:
            ctx["first_hop_device"]           = m.group(1).rstrip(",|:()")
            egress = m.group(2)
            ctx["first_hop_egress_interface"] = egress.split("→")[0].strip().rstrip(",|:()")
        elif devices:
            ctx["first_hop_device"]           = devices[0]
            ctx["first_hop_egress_interface"] = ""
        else:
            ctx["first_hop_device"]           = ""
            ctx["first_hop_egress_interface"] = ""
        # No structured hops — LAN interface unknown from text alone
        ctx["first_hop_lan_interface"] = ctx.get("first_hop_egress_interface", "")

    # Build devices+interfaces list for counter polling.
    # Use structured hops from DB trace if available; fall back to device list only.
    if _session_path_hops:
        dev_intfs: dict[str, set] = {}
        for hop in _session_path_hops:
            for dev_key, intf_key in [("from_device", "out_interface"), ("to_device", "in_interface")]:
                dev  = hop.get(dev_key, "")
                intf = hop.get(intf_key, "")
                if dev and not _re.match(r"^\d", dev):
                    dev_intfs.setdefault(dev, set())
                    if intf:
                        dev_intfs[dev].add(intf)
        ctx["path_hops_for_counters"] = [
            {"device": d, "interfaces": sorted(intfs)}
            for d, intfs in dev_intfs.items()
            if intfs
        ]
    else:
        ctx["path_hops_for_counters"] = [{"device": d, "interfaces": []} for d in devices]

    # VRF inference: determine which VRF the src_ip lives in on the first hop device.
    # This propagates into routing checks and pings so they query the correct VRF.
    src_vrf = await _infer_src_vrf(ctx.get("src_ip", ""), ctx.get("first_hop_device", ""))
    ctx["src_vrf"] = src_vrf
    logger.info(
        "_post_path_trace: src_vrf=%r for src_ip=%r on first_hop=%r",
        src_vrf, ctx.get("src_ip"), ctx.get("first_hop_device"),
    )

    # Merge path anomaly flags set by _live_path_trace
    ctx.update(_session_path_flags)
    # Unified investigation_device: whichever device triggered an anomaly
    ctx["investigation_device"] = (
        _session_path_flags.get("mgmt_routing_device")
        or _session_path_flags.get("no_route_device")
        or ""
    )
    ctx["path_anomaly_detected"] = bool(ctx["investigation_device"])
    logger.info(
        "_post_path_trace: path_anomaly_detected=%r investigation_device=%r",
        ctx["path_anomaly_detected"], ctx["investigation_device"],
    )


def _post_ping_test(result: dict, ctx: dict) -> None:
    """Populate ping_failed / ping_loss_pct into runbook context."""
    if isinstance(result, dict):
        success  = result.get("success", False)
        loss_pct = result.get("loss_pct", 100 if not success else 0)
    else:
        success, loss_pct = False, 100
    ctx["ping_failed"]   = not success
    ctx["ping_loss_pct"] = loss_pct


def _post_reverse_ping_test(result: dict, ctx: dict) -> None:
    """Populate reverse_ping_failed into runbook context."""
    if isinstance(result, dict):
        ctx["reverse_ping_failed"] = not result.get("success", False)
    else:
        ctx["reverse_ping_failed"] = True


def _post_tcp_test(result: dict, ctx: dict) -> None:
    """Populate tcp_reachable into runbook context."""
    if isinstance(result, dict):
        ctx["tcp_reachable"] = result.get("reachable", False)
    else:
        ctx["tcp_reachable"] = False


def _post_historical_route(result: dict, ctx: dict) -> None:
    """Store routing_history result and set historical_egress / historical_egress_found."""
    if isinstance(result, dict) and result.get("found"):
        ctx["historical_egress"]       = result.get("egress_interface", "")
        ctx["historical_egress_found"] = bool(ctx["historical_egress"])
        ctx["historical_route"]        = result
    else:
        ctx["historical_egress"]       = ""
        ctx["historical_egress_found"] = False
        ctx["historical_route"]        = {}


def _post_interface_detail(result: dict, ctx: dict) -> None:
    """Store interface detail result."""
    if isinstance(result, dict) and "error" not in result:
        ctx["investigation_intf_detail"] = result
    else:
        ctx["investigation_intf_detail"] = {}


def _post_syslog(result: dict, ctx: dict) -> None:
    """Store syslog result."""
    if isinstance(result, dict):
        ctx["investigation_syslog"] = result.get("logs", [])
    else:
        ctx["investigation_syslog"] = []


def _post_all_interfaces(result: dict, ctx: dict) -> None:
    """Store all-interfaces status; extract list of DOWN non-management interfaces."""
    if isinstance(result, dict) and "interfaces" in result:
        ctx["all_interfaces"] = result["interfaces"]
        ctx["down_interfaces"] = [
            i for i in result["interfaces"] if not i.get("up", True)
        ]
    else:
        ctx["all_interfaces"] = []
        ctx["down_interfaces"] = []


_RUNBOOK_MAP = {
    "blocked":      "connectivity.yaml",
    "slow":         "performance.yaml",
    "intermittent": "intermittent.yaml",
    "device":       "device_health.yaml",
    "path_changed": "path_change.yaml",
    "general":      "connectivity.yaml",
}
_RUNBOOKS_DIR = pathlib.Path(__file__).parent.parent / "runbooks"


def _select_runbook(issue_type: str) -> pathlib.Path:
    name = _RUNBOOK_MAP.get(issue_type, "connectivity.yaml")
    path = _RUNBOOKS_DIR / name
    if not path.exists():
        logger.warning("Runbook %s not found — falling back to connectivity.yaml", name)
        path = _RUNBOOKS_DIR / "connectivity.yaml"
    logger.info("Selected runbook: %s (issue_type=%s)", path.name, issue_type)
    return path


def _build_runbook_tools(prompt: str) -> dict:
    """Return the tool registry for the connectivity runbook."""

    async def _path_agent(prompt: str) -> str:
        return await call_nornir_path_agent.ainvoke({"task": prompt})

    async def _servicenow_agent(
        device_names, source_ip="", dest_ip="", port="",
        query_type="incidents_and_changes", hours_back=24,
    ) -> str:
        import json as _j
        _dn = device_names if isinstance(device_names, list) else []
        return await call_servicenow_agent.ainvoke({
            "device_names": _j.dumps(_dn),
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "port": port,
            "query_type": query_type,
            "hours_back": int(hours_back) if hours_back else 24,
        })

    async def _counters_agent(devices_and_interfaces) -> str:
        if not isinstance(devices_and_interfaces, list):
            return "No interface data."
        # Filter out entries without interface names — counter polling needs at least one intf
        valid = [e for e in devices_and_interfaces if e.get("interfaces")]
        if not valid:
            return "No interface data available for counter polling."
        import json as _j
        return await call_interface_counters_agent.ainvoke({"devices_and_interfaces": _j.dumps(valid)})

    async def _ping_agent(
        device: str, destination: str, source_interface: str = "", vrf: str = ""
    ) -> dict:
        return await _ping_via_nornir(device, destination, source_interface, vrf)

    async def _routing_agent(devices, destination: str, vrf: str = "") -> dict:
        devs = devices if isinstance(devices, list) else []
        return await _routing_check_via_nornir(devs, destination, vrf)

    async def _reverse_path_agent(prompt: str) -> str:
        """Trace the return path (dst → src), storing hops for visualization."""
        global _session_reverse_path_hops
        src_ip, dst_ip = _extract_ips(prompt)
        logger.info("_reverse_path_agent: src_ip=%r dst_ip=%r (will trace %r → %r)", src_ip, dst_ip, dst_ip, src_ip)
        if not src_ip or not dst_ip:
            return "Could not extract IPs for reverse path trace."
        try:
            text, hops = await _live_path_trace(dst_ip, src_ip)  # swapped
            logger.info("_reverse_path_agent: got %d hops, text=%r", len(hops), text[:120])
            if hops:
                _session_reverse_path_hops = hops
            return text
        except Exception as exc:
            logger.exception("_reverse_path_agent: error")
            return f"Reverse path trace error: {exc}"

    async def _tcp_port_agent(device: str, destination: str, port, vrf: str = "") -> dict:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{NORNIR_AGENT_URL}/tcp-test",
                    json={"device": device, "destination": destination,
                          "port": int(port), "vrf": vrf},
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            return {"reachable": False, "error": str(exc), "device": device,
                    "destination": destination, "port": port}

    async def _panorama_agent(
        source_ip: str, dest_ip: str, firewall_hostnames,
        port: str = "", protocol: str = "tcp",
    ) -> str:
        fws = firewall_hostnames if isinstance(firewall_hostnames, list) else []
        return await call_panorama_agent.ainvoke({
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "firewall_hostnames": fws,
            "port": port,
            "protocol": protocol,
        })

    async def _splunk_agent(task: str) -> str:
        return await call_splunk_agent.ainvoke({"task": task})

    async def _routing_history_agent(device: str, destination: str) -> dict:
        """Query routing_history DB for the last known data-plane route on device to destination."""
        try:
            from db import fetchrow as _frow
            hist = await _frow(
                """
                SELECT egress_interface, next_hop::text, protocol, prefix::text, collected_at
                FROM routing_history
                WHERE device = $1
                  AND $2::inet << prefix
                  AND egress_interface IS NOT NULL
                  AND egress_interface NOT ILIKE 'management%'
                ORDER BY masklen(prefix) DESC, collected_at DESC
                LIMIT 1
                """,
                device, destination,
            )
        except Exception as exc:
            logger.warning("routing_history_agent: DB error: %s", exc)
            return {"found": False, "device": device, "error": str(exc)}
        if not hist:
            return {"found": False, "device": device}
        return {
            "found":             True,
            "device":            device,
            "egress_interface":  hist["egress_interface"],
            "next_hop":          hist["next_hop"],
            "protocol":          hist["protocol"],
            "prefix":            hist["prefix"],
            "collected_at":      hist["collected_at"].isoformat(),
        }

    async def _interface_detail_agent(device: str, interface: str) -> dict:
        """Fetch full interface stats (error counters, line-protocol) from a live device via SSH."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(f"{NORNIR_AGENT_URL}/interface-detail",
                    json={"device": device, "interface": interface})
                return r.json()
        except Exception as exc:
            return {"error": str(exc), "device": device, "interface": interface}

    async def _syslog_agent(device: str, interface: str = "") -> dict:
        """Fetch recent syslog from a device, filtered to lines relevant to the given interface (or all link events if interface is empty)."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(f"{NORNIR_AGENT_URL}/show-logging",
                    json={"device": device, "lines": 100})
                data = r.json()
                logs = data.get("logs", [])
                if interface:
                    intf_short = interface.replace("Ethernet", "Et").replace("GigabitEthernet", "Gi")
                    relevant = [
                        l for l in logs
                        if interface.lower() in l.lower()
                        or intf_short.lower() in l.lower()
                        or any(kw in l.lower() for kw in ["link", "down", "flap", "err-disable"])
                    ][-20:]
                else:
                    # No specific interface — return all link/down events
                    relevant = [
                        l for l in logs
                        if any(kw in l.lower() for kw in ["link", "down", "flap", "err-disable", "lineproto"])
                    ][-30:]
                return {"device": device, "interface": interface, "logs": relevant}
        except Exception as exc:
            return {"error": str(exc), "device": device, "interface": interface, "logs": []}

    async def _all_interfaces_agent(device: str) -> dict:
        """Get status of all non-management interfaces on a device via live SSH."""
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(f"{NORNIR_AGENT_URL}/all-interfaces-status",
                    json={"device": device})
                return r.json()
        except Exception as exc:
            return {"error": str(exc), "device": device, "interfaces": []}

    async def _ospf_interfaces_agent(devices) -> dict:
        """Check which interfaces are OSPF-enabled on each device. Empty = no network commands configured."""
        devs = devices if isinstance(devices, list) else []
        if not devs:
            return {"ospf_interfaces": {}}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(f"{NORNIR_AGENT_URL}/ospf-interfaces",
                    json={"devices": devs})
                return r.json()
        except Exception as exc:
            return {"error": str(exc), "ospf_interfaces": {}}

    async def _ospf_agent(devices) -> dict:
        """Check OSPF neighbor state on each device in the path."""
        devs = devices if isinstance(devices, list) else []
        if not devs:
            return {"ospf_neighbors": {}}
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(f"{NORNIR_AGENT_URL}/ospf-neighbors",
                    json={"devices": devs})
                return r.json()
        except Exception as exc:
            return {"error": str(exc), "ospf_neighbors": {}}

    async def _ospf_history_agent(devices) -> dict:
        """Compare each device's current OSPF neighbor count against its own historical snapshots."""
        devs = devices if isinstance(devices, list) else []
        if not devs:
            return {"ospf_history": {}}
        try:
            from db import fetch as _fetch
            results = {}
            for device in devs:
                # Pull the last 10 collection snapshots for this device, grouped by time bucket
                snapshots = await _fetch(
                    """
                    SELECT
                        date_trunc('minute', collected_at) AS snapshot_time,
                        count(*) AS neighbor_count,
                        array_agg(state) AS states
                    FROM ospf_history
                    WHERE device = $1
                    GROUP BY date_trunc('minute', collected_at)
                    ORDER BY snapshot_time DESC
                    LIMIT 10
                    """,
                    device,
                )
                # Current state from latest collection
                current = await _fetch(
                    "SELECT router_id, interface, state FROM ospf_neighbors WHERE device = $1",
                    device,
                )
                results[device] = {
                    "current_neighbor_count": len(current),
                    "history": [
                        {
                            "snapshot_time": r["snapshot_time"].isoformat(),
                            "neighbor_count": r["neighbor_count"],
                            "states": r["states"],
                        }
                        for r in snapshots
                    ],
                }
            return {"ospf_history": results}
        except Exception as exc:
            logger.warning("ospf_history_agent: %s", exc)
            return {"error": str(exc), "ospf_history": {}}

    return {
        "path_agent":               _path_agent,
        "reverse_path_agent":       _reverse_path_agent,
        "servicenow_agent":         _servicenow_agent,
        "interface_counters_agent": _counters_agent,
        "ping_agent":               _ping_agent,
        "tcp_port_agent":           _tcp_port_agent,
        "routing_check_agent":      _routing_agent,
        "ospf_agent":               _ospf_agent,
        "ospf_interfaces_agent":    _ospf_interfaces_agent,
        "ospf_history_agent":       _ospf_history_agent,
        "panorama_agent":           _panorama_agent,
        "splunk_agent":             _splunk_agent,
        "routing_history_agent":    _routing_history_agent,
        "interface_detail_agent":   _interface_detail_agent,
        "syslog_agent":             _syslog_agent,
        "all_interfaces_agent":     _all_interfaces_agent,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _build_deterministic_root_cause(ctx: dict, src_ip: str, dst_ip: str) -> str:
    """
    Build a deterministic Root Cause + Recommendation when we have definitive findings.
    Returns empty string to fall through to LLM synthesis when findings are ambiguous.
    """
    # ── OSPF misconfiguration: device has OSPF process but zero enabled interfaces ──
    ospf_interfaces = ctx.get("ospf_interfaces", {}).get("ospf_interfaces", {})
    ospf_history    = ctx.get("ospf_history", {}).get("ospf_history", {})
    ospf_neighbors  = ctx.get("ospf_neighbors", {}).get("ospf_neighbors", {})
    if ospf_interfaces:
        for device, intf_data in ospf_interfaces.items():
            if intf_data.get("ospf_interface_count", -1) == 0:
                hist_snaps = ospf_history.get(device, {}).get("history", [])
                had_neighbors = any(s.get("neighbor_count", 0) > 0 for s in hist_snaps)
                if had_neighbors:
                    max_hist = max(s["neighbor_count"] for s in hist_snaps)
                    syslog_r = ctx.get("syslog", {})
                    syslog_block = ""
                    if syslog_r and syslog_r.get("logs"):
                        syslog_block = "\n\nSyslog events on {}:\n".format(device) + "\n".join(
                            f"  - {l}" for l in syslog_r["logs"][-10:]
                        )
                    return (
                        f"## Root Cause\n\n"
                        f"**OSPF misconfiguration on {device}** — the OSPF process is running "
                        f"(router-ID exists) but **no interfaces are participating in OSPF** "
                        f"(`ospf_interface_count: 0`). This means no `network` command or "
                        f"`ip ospf area` is configured on any interface. "
                        f"Historically {device} had **{max_hist} OSPF neighbor(s)**; loss of "
                        f"OSPF routes caused traffic to fall back to a static default via "
                        f"Management0 ({src_ip} → {dst_ip} blackholed)."
                        f"{syslog_block}\n\n"
                        f"## Recommendation\n\n"
                        f"- Re-add the OSPF `network` statement (or `ip ospf area <id>` on the "
                        f"relevant interfaces) on **{device}**\n"
                        f"- Verify OSPF adjacencies reconverge with `show ip ospf neighbor`\n"
                        f"- Confirm the route to {dst_ip} is learned via OSPF (not Management0)\n"
                        f"- Validate end-to-end connectivity from {src_ip} to {dst_ip}"
                    )

    device  = ctx.get("investigation_device", "")
    detail  = ctx.get("investigation_intf_detail", {})
    syslog  = ctx.get("investigation_syslog", [])
    hist    = ctx.get("historical_route", {})

    if not device or not detail or "error" in detail:
        return ""

    intf     = detail.get("interface", "")
    oper     = detail.get("oper_status", "")
    lp       = detail.get("line_protocol", "")
    admin_dn = oper in ("disabled", "adminDown")
    link_dn  = lp == "down" and not admin_dn

    if not admin_dn and not link_dn:
        return ""

    # Build syslog evidence block
    syslog_lines = []
    for l in syslog:
        syslog_lines.append(f"  - {l}")
    syslog_block = "\n".join(syslog_lines) if syslog_lines else "  - No syslog events captured."

    # Historical route context
    hist_line = ""
    if hist and hist.get("found"):
        hist_line = (
            f" The last known data-plane route to {dst_ip} used **{hist['egress_interface']}** "
            f"({hist['protocol']}, {hist['prefix']})."
        )

    if admin_dn:
        rc = (
            f"## Root Cause\n\n"
            f"**{intf}** on **{device}** is **administratively shut down** (`shutdown` applied).{hist_line} "
            f"With no data-plane egress, {device} is forwarding via its management interface, "
            f"blackholing traffic between {src_ip} and {dst_ip}.\n\n"
            f"Syslog events on {device}:\n{syslog_block}\n\n"
            f"## Recommendation\n\n"
            f"- Run `no shutdown` on `interface {intf}` on **{device}**\n"
            f"- Verify OSPF adjacency reconverges after the interface comes up\n"
            f"- Confirm end-to-end connectivity from {src_ip} to {dst_ip}"
        )
    else:
        rc = (
            f"## Root Cause\n\n"
            f"**{intf}** on **{device}** has lost its physical link (line-protocol: down).{hist_line} "
            f"This has taken down the data-plane path between {src_ip} and {dst_ip}.\n\n"
            f"Syslog events on {device}:\n{syslog_block}\n\n"
            f"## Recommendation\n\n"
            f"- Check the physical cable and SFP on `{intf}` on **{device}** and its peer port\n"
            f"- Review syslog above for the exact time and cause of the link-down event\n"
            f"- Confirm OSPF adjacency reconverges once the link is restored"
        )
    return rc


# ---------------------------------------------------------------------------

async def orchestrate_troubleshoot(
    prompt: str,
    username: str | None = None,
    session_id: str | None = None,
    issue_type: str = "general",
) -> dict:
    global _session_id, _llm, _session_path_hops, _session_path_devices, _session_interface_counters, _session_reverse_path_hops, _session_path_text, _session_path_flags
    _session_id = session_id
    _session_path_hops = None
    _session_reverse_path_hops = None
    _session_path_devices = []
    _session_interface_counters = []
    _session_path_text = ""
    _session_path_flags = {}
    """
    Run the troubleshoot ReAct agent.
    The LLM reasons at each step before deciding which specialist agent to call next.
    Returns a structured troubleshooting report.
    """
    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    logger.info("Troubleshoot ReAct agent: %s (user=%s)", prompt, username)

    _llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )

    # --- Tool plan cache: skip the ReAct LLM loop on cache hit ---
    from langchain_core.messages import ToolMessage, HumanMessage
    import re

    # Queries with explicit tool-routing instructions bypass the cache and ReAct loop —
    # we execute a deterministic plan directly to avoid the local model misbehaving.
    _NORNIR_PATTERNS = re.compile(
        r"without netbrain|no netbrain|use nornir|use the database|use netbox|"
        r"live data|collected data|don.t use netbrain",
        re.IGNORECASE,
    )
    _NETBRAIN_PATTERNS = re.compile(r"use netbrain|with netbrain", re.IGNORECASE)

    tool_outputs: list[str] = []
    _plan_used = False
    _rb_ctx: dict = {}  # structured runbook context — used for deterministic section building

    # --- Resolve INC number → IPs when prompt references an incident but has no IPs ---
    _INC_RE = re.compile(r'\bINC\d+\b', re.IGNORECASE)
    _inc_match = _INC_RE.search(prompt)
    _inc_summary: dict | None = None  # populated below if INC resolves
    if _inc_match and not _extract_ips(prompt)[0]:
        try:
            from tools.servicenow_tools import get_servicenow_incident as _get_inc_tool
            # FastMCP wraps the function in a FunctionTool — use .fn to get the raw coroutine
            _get_inc_fn = getattr(_get_inc_tool, 'fn', None) or _get_inc_tool
            _inc_data = await _get_inc_fn(_inc_match.group(0).upper())
            _inc_desc = ""
            if "result" in _inc_data:
                _r = _inc_data["result"]
                _inc_desc = _r.get("description") or _r.get("short_description") or ""
                _inc_summary = {
                    "number": _r.get("number", _inc_match.group(0).upper()),
                    "short_description": _r.get("short_description", ""),
                    "state": _r.get("state", ""),
                    "priority": _r.get("priority", ""),
                    "opened_at": _r.get("opened_at", ""),
                    "assigned_to": (_r.get("assigned_to") or {}).get("display_value") or "Unassigned",
                    "assignment_group": (_r.get("assignment_group") or {}).get("display_value") or "",
                }
            _inc_ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', _inc_desc)
            if len(_inc_ips) >= 2:
                _port_hint = re.search(r'\bport\s+(\d+)\b', _inc_desc, re.IGNORECASE)
                _port_str = f" port {_port_hint.group(1)}" if _port_hint else ""
                prompt = f"{prompt} (source: {_inc_ips[0]}, destination: {_inc_ips[1]}{_port_str})"
                logger.info("INC→IP resolved: %s → %s → %s", _inc_match.group(0), _inc_desc[:80], prompt[-60:])
        except Exception as _inc_exc:
            logger.warning("INC→IP resolution failed: %s", _inc_exc)

    # --- Runbook executor: use when both source and destination IPs are present ---
    _rb_src_ip, _rb_dst_ip = _extract_ips(prompt)
    if _rb_src_ip and _rb_dst_ip and not _NORNIR_PATTERNS.search(prompt) and not _NETBRAIN_PATTERNS.search(prompt):
        try:
            try:
                from atlas.runbook_executor import RunbookExecutor
            except ImportError:
                from runbook_executor import RunbookExecutor

            _rb_path = _select_runbook(issue_type)
            _rb_executor = RunbookExecutor(
                _rb_path,
                tools=_build_runbook_tools(prompt),
                post_processors={
                    "path_trace":        _post_path_trace,
                    "ping_test":         _post_ping_test,
                    "reverse_ping_test": _post_reverse_ping_test,
                    "tcp_test":          _post_tcp_test,
                    "historical_route":    _post_historical_route,
                    "interface_detail":    _post_interface_detail,
                    "device_syslog":       _post_syslog,
                    "all_interfaces":      _post_all_interfaces,
                    "device_syslog_all":   _post_syslog,
                },
                session_id=session_id or "default",
            )
            tool_outputs = await _rb_executor.run({
                "prompt":   prompt,
                "src_ip":   _rb_src_ip,
                "dst_ip":   _rb_dst_ip,
                "port":     _extract_port(prompt),
                "protocol": "tcp",
            })
            _rb_ctx = _rb_executor.ctx  # capture structured context for section builders
            _plan_used = True
            logger.info("Runbook executor completed — %d outputs collected", len(tool_outputs))
        except Exception as _rb_exc:
            logger.warning("Runbook executor failed (%s) — falling back to deterministic plan", _rb_exc)

    # Deterministic plan for explicit routing queries — skip cache and ReAct loop entirely
    if not _plan_used and _NORNIR_PATTERNS.search(prompt):
        _deterministic_plan = ["call_nornir_path_agent", "call_panorama_agent", "call_splunk_agent", "call_servicenow_agent"]
        logger.info("Deterministic plan (no-NetBrain): %s", _deterministic_plan)
        tool_outputs = await _execute_plan(_deterministic_plan, prompt)
        _plan_used = True
    elif not _plan_used and _NETBRAIN_PATTERNS.search(prompt):
        _deterministic_plan = ["call_netbrain_agent", "call_panorama_agent", "call_splunk_agent", "call_servicenow_agent"]
        logger.info("Deterministic plan (NetBrain): %s", _deterministic_plan)
        tool_outputs = await _execute_plan(_deterministic_plan, prompt)
        _plan_used = True

    if not _plan_used:
        try:
            from agent_memory import recall_tool_plan, store_tool_plan
            cached_plan = await recall_tool_plan(prompt, _TOOL_FINGERPRINT)
            if cached_plan:
                logger.info("Tool plan cache hit — executing %s directly", cached_plan)
                try:
                    import atlas.status_bus as status_bus
                    await status_bus.push(session_id or "default", "Using cached agent plan...")
                except Exception:
                    pass
                tool_outputs = await _execute_plan(cached_plan, prompt)
                _plan_used = True
        except Exception as _plan_exc:
            logger.warning("Tool plan cache check failed: %s", _plan_exc)

    if not _plan_used:
        try:
            from atlas.agents.agent_team import run_agent_team
        except ImportError:
            from agent_team import run_agent_team

        _team_tools: dict[str, list] = {
            "path_agent":     [call_nornir_path_agent, call_netbrain_agent],
            "evidence_agent": [call_servicenow_agent, get_incident_details],
            "device_agent":   [call_interface_counters_agent],
            "security_agent": [call_panorama_agent],
        }
        tool_outputs = await run_agent_team(
            query=prompt,
            llm=_llm,
            tools_by_agent=_team_tools,
            session_id=session_id or "default",
        )

    # Force ServiceNow / interface counters only when the runbook did NOT already run them.
    # When _plan_used via runbook executor, these were already called as runbook steps — skip.
    _snow_already_called = any(
        o.startswith("INCIDENTS:") or o.startswith("ServiceNow unavailable")
        for o in tool_outputs
    )
    if not _plan_used and _session_path_devices and not _snow_already_called:
        try:
            src_ip_forced, dst_ip_forced = _extract_ips(prompt)
            import json as _j
            snow_out = await call_servicenow_agent.ainvoke({
                "device_names": _j.dumps(list(_session_path_devices)),
                "source_ip": src_ip_forced or "",
                "dest_ip": dst_ip_forced or "",
                "port": _extract_port(prompt),
            })
            if snow_out:
                tool_outputs.append(snow_out)
        except Exception as _snow_exc:
            logger.warning("Forced SNOW call failed: %s", _snow_exc)

    if not _plan_used and not _session_interface_counters and (_session_path_hops or _session_path_devices):
        _dev_intfs: dict[str, set] = {}
        if _session_path_hops:
            _IP_RE = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}')
            for _hop in _session_path_hops:
                for _dev, _intf in (
                    (_hop.get("from_device", ""), _hop.get("out_interface")),
                    (_hop.get("to_device", ""),   _hop.get("in_interface")),
                ):
                    if _dev and not _IP_RE.match(_dev) and _intf:
                        _dev_intfs.setdefault(_dev, set()).add(_intf)
        if not _dev_intfs and _session_path_devices:
            _HOP_RE = re.compile(r'Hop\s+\d+:\s+(\S+).*?Egress:\s*(\S+)', re.IGNORECASE)
            for _out in tool_outputs:
                for _m in _HOP_RE.finditer(_out):
                    _dev_intfs.setdefault(_m.group(1), set()).add(_m.group(2))
        _forced_entries = [{"device": d, "interfaces": sorted(i)} for d, i in _dev_intfs.items() if i]
        if _forced_entries:
            try:
                import json as _j
                await call_interface_counters_agent.ainvoke({"devices_and_interfaces": _j.dumps(_forced_entries)})
            except Exception as _intf_exc:
                logger.warning("Forced interface counters call failed: %s", _intf_exc)

    past_memories: list[dict] = []
    past_incidents: list[dict] = []
    confirmed_context = ""

    if tool_outputs:
        import re as _re

        # Strip internal reasoning tags
        cleaned_outputs = []
        for out in tool_outputs:
            out = _re.sub(r"<plan>.*?</plan>", "", out, flags=_re.DOTALL)
            out = _re.sub(r"<reflection>.*?</reflection>", "", out, flags=_re.DOTALL)
            out = _re.sub(r"<plan>.*", "", out, flags=_re.DOTALL)
            out = _re.sub(r"<reflection>.*", "", out, flags=_re.DOTALL)
            out = out.strip()
            if out:
                cleaned_outputs.append(out)

        # Separate ServiceNow output — injected directly, not passed to synthesis LLM
        snow_raw = None
        non_snow_outputs = []
        for out in cleaned_outputs:
            if out.startswith("INCIDENTS:") or out.startswith("ServiceNow unavailable"):
                snow_raw = out
            elif "agent unavailable" in out.lower() and "splunk" in out.lower():
                non_snow_outputs.append("Splunk: No log data available — agent unreachable.")
            else:
                non_snow_outputs.append(out)

        # Prefer post-processor result; fall back to text extraction
        _path_has_firewalls = _rb_ctx.get("has_firewalls") or bool(
            _extract_firewalls_from_netbrain("\n".join(cleaned_outputs))
        )

        path_devices: list[str] = (
            list(_session_path_devices) if _session_path_devices
            else (_rb_ctx.get("path_devices") or [])
        )

        # ── Deterministic section builders ──────────────────────────────────

        def _build_path_section() -> str:
            if _session_path_hops:
                rows = [
                    "| Hop | From | To | Egress | Ingress |",
                    "|-----|------|----|--------|---------|",
                ]
                for i, h in enumerate(_session_path_hops, 1):
                    rows.append(
                        f"| {i} | {h.get('from_device','—')} | {h.get('to_device','—')} "
                        f"| {h.get('out_interface','—')} | {h.get('in_interface','—')} |"
                    )
                return "## Path Summary\n\n" + "\n".join(rows)
            raw = str(_rb_ctx.get("path_trace", ""))
            if not raw:
                return ""
            lines = ["| Hop | Device | Egress Interface |", "|-----|--------|-----------------|"]
            for line in raw.splitlines():
                m = _re.match(r'Hop\s+(\d+):\s+(\S+).*?Egress:\s*(\S+)', line, _re.IGNORECASE)
                if m:
                    lines.append(f"| {m.group(1)} | {m.group(2)} | {m.group(3)} |")
            if len(lines) > 2:
                return "## Path Summary\n\n" + "\n".join(lines)
            return f"## Path Summary\n\n{raw[:400]}"

        def _build_reverse_path_section() -> str:
            raw = str(_rb_ctx.get("reverse_path_trace", ""))
            if not raw or "error" in raw.lower():
                return ""
            lines = ["| Hop | Device | Egress Interface |", "|-----|--------|-----------------|"]
            for line in raw.splitlines():
                m = _re.match(r'Hop\s+(\d+):\s+(\S+).*?Egress:\s*(\S+)', line, _re.IGNORECASE)
                if m:
                    lines.append(f"| {m.group(1)} | {m.group(2)} | {m.group(3)} |")
            if len(lines) > 2:
                return "## Reverse Path\n\n" + "\n".join(lines)
            if raw.strip():
                return f"## Reverse Path\n\n{raw[:400]}"
            return ""

        def _fmt_ping(r) -> str:
            """Format a single ping result dict into one line."""
            if not isinstance(r, dict):
                return str(r)[:200]
            device = r.get("device", "unknown")
            dest   = r.get("destination", "")
            vrf    = r.get("vrf", "default")
            loss   = r.get("loss_pct", 0 if r.get("success") else 100)
            rtt    = r.get("rtt_avg_ms")
            if r.get("success"):
                rtt_str = f", RTT avg {rtt}ms" if rtt else ""
                return f"✓ **{device}** → **{dest}** (VRF: {vrf}): success, 0% loss{rtt_str}"
            else:
                return f"✗ **{device}** → **{dest}** (VRF: {vrf}): **{loss}% packet loss**"

        def _build_ping_section() -> str:
            r  = _rb_ctx.get("ping_test")
            rr = _rb_ctx.get("reverse_ping_test")
            if not r and not rr:
                return "## Ping Test\n\nPing test not performed — no first-hop device in inventory."
            lines = ["## Ping Test"]
            if r:
                lines.append(_fmt_ping(r))
            if rr:
                lines.append(_fmt_ping(rr))
            return "\n\n".join(lines)

        def _build_interface_section() -> str:
            if not _session_interface_counters:
                return "## Interface Errors\n\nInterface counter data not available."
            rows = []
            any_errors = False
            for entry in _session_interface_counters:
                device = entry.get("device", "?")
                active = entry.get("active", [])
                clean  = entry.get("clean", [])
                window = entry.get("window_s", "?")
                err    = entry.get("ssh_error", "")
                if err:
                    rows.append(f"**{device}**: unreachable — {err}")
                    continue
                if active:
                    any_errors = True
                    for c in active:
                        intf = c.get("interface", "?")
                        if "error" in c:
                            rows.append(f"**{device}** {intf}: SSH error — {c['error']}")
                        else:
                            d = c.get("delta_9s", {})
                            parts = [f"{k}+{v}" for k, v in d.items() if v > 0]
                            last  = c.get("last_clear", "never")
                            rows.append(
                                f"**{device}** {intf}: {', '.join(parts) or 'errors'} "
                                f"over {window}s (cleared: {last})"
                            )
                else:
                    clean_str = ", ".join(clean) if clean else "all polled interfaces"
                    rows.append(f"**{device}**: clean — no incrementing errors ({clean_str})")
            if not any_errors:
                return "## Interface Errors\n\nNo incrementing errors detected on path interfaces."
            return "## Interface Errors\n\n" + "\n".join(rows)

        def _build_tcp_section() -> str:
            r = _rb_ctx.get("tcp_test")
            if not r:
                return ""
            if isinstance(r, dict):
                device = r.get("device", "unknown")
                dest   = r.get("destination", _rb_ctx.get("dst_ip", ""))
                port   = r.get("port", _rb_ctx.get("port", ""))
                if "error" in r:
                    return f"## TCP Port Test\n\n{device} → {dest}:{port} — error: {r['error']}"
                if r.get("reachable"):
                    return (f"## TCP Port Test\n\n"
                            f"✓ TCP {dest}:{port} reachable from **{device}** — "
                            "service is accepting connections.")
                else:
                    return (f"## TCP Port Test\n\n"
                            f"✗ TCP {dest}:{port} unreachable from **{device}** — "
                            "service is not accepting connections (port closed or filtered).")
            return ""

        def _build_routing_section() -> str:
            r = _rb_ctx.get("routing_check")
            if not r:
                ping_r = _rb_ctx.get("ping_test")
                if isinstance(ping_r, dict) and ping_r.get("success"):
                    return "## Routing Analysis\n\nRouting check skipped — ping successful."
                return "## Routing Analysis\n\nRouting check not performed."
            if isinstance(r, dict):
                dest = r.get("destination", _rb_ctx.get("dst_ip", ""))
                hops = r.get("hops", {})
                if not hops:
                    return f"## Routing Analysis\n\nNo routing data returned for {dest}."
                lines = [
                    f"Destination: **{dest}**", "",
                    "| Device | VRF | Next Hop | Interface | Status |",
                    "|--------|-----|----------|-----------|--------|",
                ]
                for device, info in hops.items():
                    if not info.get("found") and "error" in info:
                        lines.append(f"| {device} | — | — | — | {info['error']} |")
                    else:
                        vrf_  = info.get("vrf", "default")
                        nh    = info.get("next_hop") or "—"
                        iface = info.get("interface") or "—"
                        proto = info.get("protocol", "")
                        status = f"✓ {proto}".strip() if info.get("found") else "✗ no route"
                        lines.append(f"| {device} | {vrf_} | {nh} | {iface} | {status} |")
                return "## Routing Analysis\n\n" + "\n".join(lines)
            return f"## Routing Analysis\n\n{str(r)[:400]}"

        def _build_firewall_section() -> str:
            if not _path_has_firewalls:
                return ""
            panorama_raw = next(
                (o for o in non_snow_outputs if "matching rule=" in o or "action=" in o), None
            )
            if not panorama_raw:
                return "## Firewall Policy Check\n\nPanorama policy data not available."
            lines = [
                "| Firewall | Device Group | Rule | Action | Zones |",
                "|----------|-------------|------|--------|-------|",
            ]
            for line in panorama_raw.splitlines():
                m = _re.search(
                    r'(\S+)\s*\(device_group:\s*([^)]+)\).*?rule=\'([^\']+)\'.*?action=(\w+).*?zones=([^\s.]+)',
                    line,
                )
                if m:
                    fw, dg, rule, action, zones = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
                    action_fmt = f"✅ {action}" if action.lower() == "allow" else f"🚫 {action}"
                    lines.append(f"| {fw} | {dg} | {rule} | {action_fmt} | {zones} |")
            if len(lines) > 2:
                return "## Firewall Policy Check\n\n" + "\n".join(lines)
            return f"## Firewall Policy Check\n\n{panorama_raw}"

        def _build_splunk_section() -> str:
            if not _path_has_firewalls:
                return ""
            splunk_raw = next((o for o in non_snow_outputs if "splunk" in o.lower()), None)
            if not splunk_raw:
                return ""
            if "no log data" in splunk_raw.lower() or "agent unreachable" in splunk_raw.lower():
                return "## Splunk Traffic Analysis\n\nNo log data available — Splunk agent unreachable. Log correlation skipped."
            return f"## Splunk Traffic Analysis\n\n{splunk_raw}"

        def _build_investigation_section() -> str:
            """Build investigation findings from routing_history + interface detail/all + syslog."""
            device        = _rb_ctx.get("investigation_device", "")
            hist          = _rb_ctx.get("historical_route", {})
            detail        = _rb_ctx.get("investigation_intf_detail", {})
            syslog        = _rb_ctx.get("investigation_syslog", [])
            all_intfs     = _rb_ctx.get("all_interfaces", [])
            down_intfs    = _rb_ctx.get("down_interfaces", [])
            if not device or (not hist and not detail and not syslog and not all_intfs):
                return ""

            lines = [f"## Data-Plane Investigation — {device}"]

            if hist and hist.get("found"):
                age = ""
                try:
                    import datetime
                    delta = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(hist["collected_at"])
                    age = f" (collected {int(delta.total_seconds() // 3600)}h ago)"
                except Exception:
                    pass
                lines.append(
                    f"Last known data-plane route{age}: "
                    f"egress **{hist['egress_interface']}** via {hist['next_hop'] or 'directly connected'} "
                    f"({hist['protocol']}, {hist['prefix']})"
                )
            elif hist is not None:
                lines.append("No routing history found in DB — device may not have been collected yet.")

            if detail and "error" not in detail:
                lp       = detail.get("line_protocol", "unknown")
                oper     = detail.get("oper_status", "unknown")
                in_err   = detail.get("input_errors", 0)
                out_err  = detail.get("output_errors", 0)
                in_disc  = detail.get("input_discards", 0)
                desc     = detail.get("description", "")
                admin_down = oper in ("disabled", "adminDown")
                status_str = (
                    "**ADMINISTRATIVELY SHUT DOWN (disabled)**"
                    if admin_down else
                    f"line-protocol: **{lp}**, oper_status: {oper}"
                )
                lines.append(
                    f"Interface **{detail.get('interface')}** — {status_str}"
                    + (f", description: {desc!r}" if desc else "")
                    + f" | input_errors={in_err} output_errors={out_err} input_discards={in_disc}"
                )

            if all_intfs:
                up_count   = sum(1 for i in all_intfs if i.get("up"))
                down_count = len(down_intfs)
                lines.append(f"Interface summary: {up_count} up, {down_count} DOWN")
                if down_intfs:
                    def _intf_status_str(i):
                        oper = i.get("oper_status", "")
                        if oper in ("disabled", "adminDown"):
                            return "**ADMINISTRATIVELY SHUT DOWN**"
                        return f"line-protocol: {i.get('line_protocol','?')}"
                    down_list = ", ".join(
                        f"**{i['interface']}** ({_intf_status_str(i)})"
                        + (f" — {i['description']!r}" if i.get("description") else "")
                        for i in down_intfs
                    )
                    lines.append(f"DOWN interfaces: {down_list}")

            if syslog:
                lines.append(
                    "Recent syslog (interface/link events):\n"
                    + "\n".join(f"  {l}" for l in syslog)
                )

            return "\n\n".join(lines)

        def _build_ospf_section() -> str:
            neighbors  = _rb_ctx.get("ospf_neighbors", {}).get("ospf_neighbors", {})
            interfaces = _rb_ctx.get("ospf_interfaces", {}).get("ospf_interfaces", {})
            history    = _rb_ctx.get("ospf_history", {}).get("ospf_history", {})
            syslog_r   = _rb_ctx.get("syslog", {})
            if not neighbors and not interfaces and not history and not syslog_r:
                return ""
            lines = ["## OSPF Analysis"]

            # Per-device neighbor + interface counts
            all_devices = set(list(neighbors.keys()) + list(interfaces.keys()) + list(history.keys()))
            for device in sorted(all_devices):
                nbr_data  = neighbors.get(device, {})
                intf_data = interfaces.get(device, {})
                hist_data = history.get(device, {})

                nbr_count       = nbr_data.get("count", 0)
                intf_count      = intf_data.get("ospf_interface_count", 0)
                current_in_hist = hist_data.get("current_neighbor_count", nbr_count)
                hist_snaps      = hist_data.get("history", [])

                # Build the status line
                if intf_count == 0 and any(s.get("neighbor_count", 0) > 0 for s in hist_snaps):
                    max_hist = max(s["neighbor_count"] for s in hist_snaps)
                    status = (
                        f"⚠️ **OSPF MISCONFIGURATION** — 0 OSPF-enabled interfaces "
                        f"(no `network` command or `ip ospf area` on any interface). "
                        f"Historically had {max_hist} neighbor(s)."
                    )
                elif nbr_count == 0 and intf_count > 0:
                    status = f"⚠️ 0 OSPF neighbors — {intf_count} interface(s) configured but no adjacency formed."
                else:
                    nbr_list = ", ".join(
                        f"{n['router_id']} via {n['interface']} ({n['state']})"
                        for n in nbr_data.get("neighbors", [])
                    )
                    status = f"✓ {nbr_count} neighbor(s): {nbr_list}"

                lines.append(f"**{device}**: {status}")

                # Historical trend
                if hist_snaps:
                    trend = " → ".join(
                        f"{s['neighbor_count']} neighbors @ {s['snapshot_time'][:16]}"
                        for s in reversed(hist_snaps)
                    )
                    lines.append(f"  History: {trend} → now: {nbr_count}")

            return "\n".join(lines)

        def _build_vendor_kb_section() -> str:
            kb       = _rb_ctx.get("vendor_kb") or {}
            results  = kb.get("kb_results") or []
            symptoms = kb.get("symptoms") or []
            vendor   = kb.get("vendor", "unknown")
            if not results and not symptoms:
                return ""
            lines = [f"## Vendor Knowledge Base ({vendor})"]
            if symptoms:
                sym_labels = ", ".join(f"`{s['type']}`" + (f" on {s['device']}" if s.get('device') else "") for s in symptoms)
                lines.append(f"**Detected symptoms:** {sym_labels}")
                lines.append("")
            if results:
                for r in results:
                    lines.append(f"**{r['title']}**")
                    lines.append(r["snippet"])
                    if r.get("url"):
                        lines.append(f"*{r['url']}*")
                    lines.append("")
            else:
                lines.append("*No relevant vendor articles found.*")
            return "\n".join(lines)

        # Build all sections
        report_sections: list[str] = []
        path_sec = _build_path_section()
        if path_sec:
            report_sections.append(path_sec)
        investigation_sec = _build_investigation_section()
        logger.info(
            "investigation_sec built: device=%r hist_found=%r detail_oper=%r syslog_lines=%d all_intfs=%d",
            _rb_ctx.get("investigation_device"),
            _rb_ctx.get("historical_route", {}).get("found"),
            _rb_ctx.get("investigation_intf_detail", {}).get("oper_status"),
            len(_rb_ctx.get("investigation_syslog") or []),
            len(_rb_ctx.get("all_interfaces") or []),
        )
        if investigation_sec:
            report_sections.append(investigation_sec)
        # Connectivity-specific sections only make sense when src+dst IPs are present
        if _rb_src_ip and _rb_dst_ip:
            if not _path_has_firewalls:
                for sec in (_build_ping_section(), _build_tcp_section(), _build_interface_section(), _build_routing_section(), _build_ospf_section(), _build_vendor_kb_section()):
                    if sec:
                        report_sections.append(sec)
            else:
                for sec in (_build_firewall_section(), _build_splunk_section(), _build_vendor_kb_section()):
                    if sec:
                        report_sections.append(sec)

        # Parse ServiceNow change block
        chg_block = ""
        if snow_raw and snow_raw.startswith("INCIDENTS:"):
            parts = _re.split(r'\n\nCHANGE REQUESTS:', snow_raw, maxsplit=1)
            chg_block = parts[1].strip() if len(parts) == 2 else ""

        chg_section = ""
        if _rb_src_ip and _rb_dst_ip:
            chg_section = (
                "\n\n## Recent Changes\n\n" + chg_block
                if chg_block and chg_block != "No change requests found."
                else "\n\n## Recent Changes\nNo related changes found for devices in the path."
            )
        elif chg_block and chg_block != "No change requests found.":
            chg_section = "\n\n## Recent Changes\n\n" + chg_block

        # ── Vendor KB lookup + Memory recall (parallel) ──────────────────────
        logger.info("Memory recall prep: path_devices=%s", path_devices)
        memory_context = ""
        vendor_kb: dict = {}
        try:
            import atlas.status_bus as status_bus
            await status_bus.push(_session_id or "default", "Searching vendor knowledge base and recalling past cases...")
        except Exception:
            pass

        # Vendor KB lookup
        try:
            try:
                from agents.vendor_lookup_agent import lookup as _vendor_lookup
            except ImportError:
                from vendor_lookup_agent import lookup as _vendor_lookup
            vendor_kb = await _vendor_lookup(_rb_ctx, path_devices)
            _rb_ctx["vendor_kb"] = vendor_kb
            logger.info("vendor_lookup: vendor=%r symptoms=%d kb_articles=%d",
                        vendor_kb.get("vendor"), len(vendor_kb.get("symptoms", [])), len(vendor_kb.get("kb_results", [])))
        except Exception as _vk_exc:
            logger.warning("vendor_lookup failed: %s", _vk_exc)

        try:
            from agent_memory import (
                recall_memory, recall_incidents_by_devices, format_memory_context,
                get_confirmed_resolutions, format_confirmed_resolutions,
            )
            past_memories, semantic_incidents, device_incidents = await asyncio.gather(
                recall_memory(prompt, agent_type="atlas", top_k=3),
                recall_memory(prompt, agent_type="incident", top_k=5, min_similarity=0.40),
                recall_incidents_by_devices(path_devices, query=prompt),
            )
            seen_keys: set[str] = set()
            past_incidents: list[dict] = []
            for inc in semantic_incidents + device_incidents:
                key = inc.get("result_summary", "")[:30]
                if key not in seen_keys:
                    seen_keys.add(key)
                    past_incidents.append(inc)
            all_memories = past_memories + past_incidents
            memory_context = format_memory_context(all_memories) if all_memories else ""

            # Confirmed resolutions from closed SNOW tickets for path devices
            confirmed_fixes = get_confirmed_resolutions(path_devices, max_per_device=2)
            confirmed_context = format_confirmed_resolutions(confirmed_fixes) if confirmed_fixes else ""

            logger.info("Memory recall: atlas=%d incidents=%d confirmed_fixes=%d", len(past_memories), len(past_incidents), len(confirmed_fixes))
        except Exception as _mem_exc:
            logger.warning("Memory recall failed: %s", _mem_exc)
            memory_context = ""
            confirmed_context = ""

        # ── LLM: Root Cause + Recommendation only ────────────────────────────
        sections_text = "\n\n".join(report_sections)

        # Skip Root Cause synthesis when there's no connectivity diagnostic data (no src/dst IPs).
        # For device health queries, just output the agent team results directly.
        if not _rb_src_ip or not _rb_dst_ip:
            parts = [p for p in [sections_text, chg_section.strip(), "\n\n".join(non_snow_outputs)] if p]
            final = "\n\n".join(parts).strip()
            if memory_context:
                final += "\n\n---\n\n" + memory_context
        else:
            # ── Fast-path: deterministic root cause when we have definitive findings ──
            _det_rc = _build_deterministic_root_cause(_rb_ctx, _rb_src_ip, _rb_dst_ip)
            if _det_rc:
                _kb_sec = _build_vendor_kb_section()
                final = sections_text + chg_section + "\n\n" + _det_rc
                if _kb_sec:
                    final += "\n\n" + _kb_sec
                logger.info("Using deterministic root cause (skipping LLM synthesis)")

            if not _det_rc:
                _synthesis_prefix = "\n\n".join(filter(None, [confirmed_context, memory_context]))
                synthesis_system = (
                    ((_synthesis_prefix + "\n\n") if _synthesis_prefix else "") +
                    "You are writing the final section of a network troubleshooting report.\n"
                    "The diagnostic sections above have already been built from structured data.\n"
                    "Your ONLY job is to write:\n\n"
                    "  ## Root Cause\n  (one or two sentences based strictly on the diagnostic data)\n\n"
                    "  ## Recommendation\n  (specific, actionable steps that match the root cause)\n\n"
                    "Rules:\n"
                    "- Do NOT repeat or summarise the diagnostic sections.\n"
                    "- Do NOT mention firewalls, Panorama, or Splunk if there are no firewalls in the path.\n"
                    "- Do NOT write 'Unable to determine root cause' — always draw a conclusion.\n"
                    "- If a 'Data-Plane Investigation' section is present: name the device and interface, "
                    "state whether it is admin-shutdown or link-down, and include the timestamp from syslog "
                    "showing when the interface went down. If OSPF dropped, mention it.\n"
                    "- If the OSPF Analysis section shows '⚠️ OSPF MISCONFIGURATION' (ospf_interface_count=0 "
                    "with historical neighbors): the root cause is a missing OSPF network command or "
                    "ip ospf area config — NOT a physical link issue. Say this explicitly.\n"
                    "- If routing via Management: data-plane interfaces are DOWN or OSPF is misconfigured.\n"
                    "- If ping failed: name the specific hop or device where routing breaks.\n"
                    "- If ping succeeded and interfaces are clean: say the path is healthy and the issue is likely "
                    "at the application or service layer.\n"
                    "- If a firewall denied traffic: name the rule and device.\n"
                    "- If a 'Vendor Knowledge Base' section is present: use the article titles and "
                    "snippets to add vendor-specific config commands, known bug references, or "
                    "documentation links to the Recommendation. Do not quote snippets verbatim — "
                    "synthesise them into actionable steps.\n\n"
                    "DIAGNOSTIC DATA:\n" + sections_text
                )
                synthesis_messages = [
                    SystemMessage(content=_load_skill()),
                    HumanMessage(content=prompt),
                    SystemMessage(content=synthesis_system),
                    HumanMessage(content="Write ONLY ## Root Cause and ## Recommendation based on the diagnostic data above."),
                ]
                synthesis_response = await _llm.ainvoke(synthesis_messages)
                rc_text = synthesis_response.content or ""

                rc_match = _re.search(r'## Root Cause', rc_text)
                if rc_match:
                    rc_text = rc_text[rc_match.start():]

                has_changes = chg_block and chg_block != "No change requests found."
                if has_changes and "## Recommendation" in rc_text:
                    rc_text = rc_text.rstrip() + (
                        "\n- Recent changes were made to path devices — correlate these with the onset of the issue."
                    )

                final = sections_text + chg_section + "\n\n" + rc_text

    else:
        # No tool results — fall back to whatever the agent produced
        final = next(
            (m.content for m in reversed(messages) if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)),
            "Investigation complete — no summary generated.",
        )

    # Strip <plan>...</plan> and <reflection>...</reflection> tags — internal reasoning, not for display
    final = re.sub(r"<plan>.*?</plan>", "", final, flags=re.DOTALL).strip()
    final = re.sub(r"<reflection>.*?</reflection>", "", final, flags=re.DOTALL).strip()

    # When DB path hops are available the visual diagram replaces the text Path Summary
    if _session_path_hops:
        final = re.sub(
            r'## Path Summary.*?(?=## Ping Test|## Interface Errors|## Routing Analysis'
            r'|## Firewall Policy Check|## Splunk|## Recent Incidents|## Recent Changes|## Root Cause|$)',
            '', final, flags=re.DOTALL
        ).strip()

    # Store this session in semantic memory — extract Root Cause + Recommendation as the summary
    try:
        from agent_memory import store_memory, record_device_incident
        # Store only the diagnostic content — strip injected ServiceNow/memory sections
        storable = final.split("\n\n---\n\n")[0].strip()
        summary_match = re.search(r'## Root Cause(.*?)(?=## Recommendation|$)', storable, re.DOTALL)
        summary = summary_match.group(1).strip()[:800] if summary_match else storable[:800]
        await store_memory(prompt, summary, agent_type="atlas")

        # Determine root cause category and failure device from runbook context
        if _rb_src_ip and _rb_dst_ip:
            _ping_ok = not _rb_ctx.get("ping_failed", True)
            _failure_dev = ""
            _rc_category = "healthy"
            if not _ping_ok:
                _rc_category = "routing"
                _failure_dev = _rb_ctx.get("first_hop_device", "")
            elif _rb_ctx.get("has_firewalls"):
                _rc_category = "firewall"
            elif _session_interface_counters and any(e.get("active") for e in (_session_interface_counters or [])):
                _rc_category = "interface_error"
            _session_devs = _rb_ctx.get("path_devices") or path_devices or []
            _sid = session_id or "default"
            for _dev in _session_devs:
                _cat = "failure_point" if _dev == _failure_dev else "was_in_path"
                record_device_incident(_dev, _cat, _sid)
    except Exception as _mem_exc:
        logger.debug("agent_memory: store skipped: %s", _mem_exc)

    all_recalled = past_memories + past_incidents  # defined in if tool_outputs block; fallback [] if else branch ran
    logger.info("all_recalled=%d serializing...", len(all_recalled))

    # Strip non-serializable fields from memories before returning to frontend.
    # Only surface incident-type memories to the user — atlas memories are internal
    # LLM context only (they contain raw prompts, not human-readable incident titles).
    serializable_memories = [
        {k: v for k, v in m.items() if k in ("query", "result_summary", "resolution", "timestamp", "similarity", "match_type")}
        for m in all_recalled
        if m.get("agent_type") == "incident"
    ]
    logger.info("serializable_memories=%d", len(serializable_memories))

    # For DB path traces, drop the report when it contains only empty/default sections
    # (no firewalls, no incidents, no changes, Splunk unreachable) — the visual is sufficient.
    _EMPTY_REPORT_PHRASES = [
        "no palo alto firewalls were found",
        "no related changes found",
        "splunk agent unreachable",
        "log correlation skipped",
    ]
    if _session_path_hops:
        report_lower = final.lower()
        _all_empty = all(phrase in report_lower for phrase in _EMPTY_REPORT_PHRASES)
        if _all_empty:
            final = ""

    content: dict = {}
    if final:
        content["direct_answer"] = final
    if _inc_summary:
        content["incident_summary"] = _inc_summary
    if _session_path_hops:
        content["path_hops"] = _session_path_hops
        content["source"] = src_ip if (src_ip := _extract_ips(prompt)[0]) else ""
        content["destination"] = dst_ip if (dst_ip := _extract_ips(prompt)[1]) else ""
    if _session_reverse_path_hops:
        content["reverse_path_hops"] = _session_reverse_path_hops
    if _session_interface_counters:
        content["interface_counters"] = _session_interface_counters

    logger.info("Final content keys: %s | path_hops=%s | reverse_path_hops=%s",
        list(content.keys()),
        len(content.get("path_hops") or []),
        len(content.get("reverse_path_hops") or []),
    )
    return {
        "role": "assistant",
        "content": content,
        "memories": serializable_memories,
    }
