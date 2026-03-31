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
    payload = {
        "id": str(uuid.uuid4()),
        "message": {"role": "user", "parts": [{"type": "text", "text": task}]},
    }
    try:
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
    except Exception as e:
        logger.warning("Agent call to %s failed: %s", url, e)
        return f"Agent unavailable: {e}"


# ---------------------------------------------------------------------------
# Tools — each specialist agent exposed as a tool to the ReAct orchestrator
# ---------------------------------------------------------------------------

_session_id: str | None = None
_llm = None  # set by orchestrate_troubleshoot before agent runs
_session_path_hops: list | None = None          # structured path hops from DB trace, injected into final response
_session_path_devices: list[str] = []           # device names in path — used to force SNOW call if LLM skips it
_session_interface_counters: list[dict] = []    # interface counter results, injected into final response


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


async def _db_path_trace(src_ip: str, dst_ip: str) -> tuple[str, list]:
    """
    Inline hop-by-hop path trace using the collected DB and NetBox.
    Returns (text_summary, path_hops) where path_hops is ready for PathVisualization.
    """
    try:
        from atlas.db import fetchrow, fetch
    except ImportError:
        from db import fetchrow, fetch

    try:
        from atlas.tools.netbox_tools import get_gateway_for_prefix
    except ImportError:
        from tools.netbox_tools import get_gateway_for_prefix

    text_hops = []
    structured_hops = []
    seen_devices = set()
    MAX_HOPS = 15

    # Cache device platforms to avoid repeated DB hits
    _platform_cache: dict[str, str] = {}

    async def _device_type(hostname: str) -> str:
        if hostname not in _platform_cache:
            row = await fetchrow("SELECT platform FROM devices WHERE hostname = $1", hostname)
            platform = row["platform"] if row else ""
            _platform_cache[hostname] = _PLATFORM_TO_TYPE.get(platform, "switch")
        return _platform_cache[hostname]

    # Step 1: find first-hop gateway — prefer NetBox, fall back to routing DB
    gw_info = get_gateway_for_prefix.fn(src_ip)
    if "error" not in gw_info:
        gw_ip = gw_info["gateway"]
        row = await fetchrow(
            "SELECT device, interface FROM interface_ips WHERE ip = $1::inet LIMIT 1", gw_ip
        )
        if not row:
            msg = f"Gateway {gw_ip} not found in interface_ips table — run collect_devices.py first."
            return msg, []
        current_device = row["device"]
        gw_interface = row["interface"]
    else:
        # NetBox unavailable — derive gateway from directly-connected route in DB
        logger.info("NetBox unavailable (%s), falling back to DB gateway lookup", gw_info["error"][:80])
        row = await fetchrow(
            """
            SELECT rt.device, rt.egress_interface, ii.ip::text AS gateway_ip
            FROM routing_table rt
            JOIN interface_ips ii
              ON ii.device = rt.device AND ii.interface = rt.egress_interface
            WHERE $1::inet << rt.prefix AND rt.next_hop IS NULL
            ORDER BY masklen(rt.prefix) DESC
            LIMIT 1
            """,
            src_ip,
        )
        if not row:
            msg = f"Cannot determine gateway for {src_ip}: NetBox offline and no directly-connected route in DB."
            return msg, []
        current_device = row["device"]
        gw_interface = row["egress_interface"]

    # Prepend source host → first router hop
    structured_hops.append({
        "from_device":      src_ip,
        "from_device_type": "host",
        "out_interface":    None,
        "out_zone":         None,
        "device_group":     None,
        "to_device":        current_device,
        "to_device_type":   await _device_type(current_device),
        "in_interface":     gw_interface,
        "in_zone":          None,
    })

    for _ in range(MAX_HOPS):
        if current_device in seen_devices:
            text_hops.append(f"  !! Routing loop detected at {current_device}")
            break
        seen_devices.add(current_device)

        route = await fetchrow(
            """
            SELECT prefix, next_hop::text, egress_interface, protocol
            FROM routing_table
            WHERE device = $1 AND $2::inet << prefix
            ORDER BY masklen(prefix) DESC LIMIT 1
            """,
            current_device, dst_ip,
        )
        if not route:
            text_hops.append(f"  Hop {len(text_hops)+1}: {current_device} — no route to {dst_ip}")
            break

        egress = route["egress_interface"] or ""
        next_hop = route["next_hop"]
        protocol = route["protocol"] or ""
        text_hops.append(
            f"  Hop {len(text_hops)+1}: {current_device} | Egress: {egress} | Protocol: {protocol} "
            f"| Next-hop: {next_hop or 'directly connected'}"
        )

        if not next_hop:
            # Directly connected — resolve via ARP for last-hop in_interface
            arp = await fetchrow(
                "SELECT mac, interface FROM arp_table WHERE device = $1 AND ip = $2::inet LIMIT 1",
                current_device, dst_ip,
            )
            in_iface = arp["interface"] if arp else None
            if arp:
                text_hops.append(
                    f"  Destination {dst_ip} reachable via ARP on {current_device} "
                    f"port {arp['interface']} (MAC {arp['mac']})"
                )
            else:
                text_hops.append(
                    f"  Destination {dst_ip} directly connected on {current_device} (no ARP entry yet)"
                )
            structured_hops.append({
                "from_device":      current_device,
                "from_device_type": await _device_type(current_device),
                "out_interface":    egress,
                "out_zone":         None,
                "device_group":     None,
                "to_device":        dst_ip,
                "to_device_type":   "host",
                "in_interface":     None,
                "in_zone":          None,
            })
            break

        # Follow next-hop to next device
        next_row = await fetchrow(
            "SELECT device, interface FROM interface_ips WHERE ip = $1::inet LIMIT 1", next_hop
        )
        if not next_row:
            text_hops.append(f"  Next-hop {next_hop} not found in inventory — path ends here")
            break

        next_device = next_row["device"]
        in_iface = next_row["interface"]
        structured_hops.append({
            "from_device":      current_device,
            "from_device_type": await _device_type(current_device),
            "out_interface":    egress,
            "out_zone":         None,
            "device_group":     None,
            "to_device":        next_device,
            "to_device_type":   await _device_type(next_device),
            "in_interface":     in_iface,
            "in_zone":          None,
        })
        current_device = next_device

    devices_in_path = list(seen_devices)
    text = (
        f"Path from {src_ip} to {dst_ip} (via collected DB):\n"
        + "\n".join(text_hops)
        + f"\n\nAll devices in path: {', '.join(devices_in_path)}"
        + "\nPalo Alto firewalls in path: none"
    )
    return text, structured_hops


@tool
async def call_nornir_path_agent(task: str) -> str:
    """
    Trace the hop-by-hop network path between two IP addresses using the collected
    routing/ARP/MAC database and NetBox IPAM — WITHOUT NetBrain.
    Use this INSTEAD of call_netbrain_agent when the user explicitly says to avoid
    NetBrain (e.g. "don't use NetBrain", "use Nornir", "use the database", "without NetBrain").
    Pass a natural language task with the source and destination IPs.
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Tracing network path via device database...")
    except Exception:
        pass
    global _session_path_devices
    cached = _cache_get("netbox_path", task)
    if cached:
        logger.info("DB path cache hit")
        if not _session_path_devices:
            _session_path_devices = _extract_devices_from_netbrain(cached)
        return cached

    src_ip, dst_ip = _extract_ips(task)
    if not src_ip or not dst_ip:
        return "Could not extract source and destination IPs from the task."

    try:
        result, path_hops = await _db_path_trace(src_ip, dst_ip)
    except Exception as e:
        logger.exception("DB path trace error")
        result, path_hops = f"Path trace error: {e}", []

    global _session_path_hops
    if path_hops:
        _session_path_hops = path_hops
    _session_path_devices = _extract_devices_from_netbrain(result)

    _cache_set("netbox_path", result, task)
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
    device_names: list[str],
    source_ip: str = "",
    dest_ip: str = "",
    port: str = "",
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

        # Run incident and change request searches in parallel (both must use the same scoped query)
        inc_result, chg_result = await _asyncio.gather(
            call_mcp_tool("search_servicenow_incidents", {"query": query, "limit": 10, "updated_within_hours": 24}, timeout=30.0),
            call_mcp_tool("list_servicenow_change_requests", {"query": query, "limit": 20, "updated_within_hours": 1}, timeout=30.0),
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
async def call_interface_counters_agent(devices_and_interfaces: list[dict]) -> str:
    """
    Fetch interface error and discard counters for specific interfaces on path devices via SSH.
    Use after a path trace to check for CRC errors, input errors, output drops on each link.

    Args:
        devices_and_interfaces: List of {device: str, interfaces: [str]} dicts.
            Each entry specifies a device and which interfaces to check.
            Example: [{"device": "arista1", "interfaces": ["Ethernet1", "Ethernet3"]},
                      {"device": "arista2", "interfaces": ["Ethernet2"]}]

    If you don't have the interface names, use the path trace output — look for
    out_interface / in_interface fields on each hop.
    """
    import asyncio as _asyncio
    import aiohttp

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
    for text, structured in raw_results:
        if text:
            lines.append(text)
        if structured and structured.get("device"):
            _session_interface_counters.append(structured)

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
            return await call_servicenow_agent.ainvoke({
                "device_names": devices,
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
# Main entry point
# ---------------------------------------------------------------------------

async def orchestrate_troubleshoot(
    prompt: str,
    username: str | None = None,
    session_id: str | None = None,
) -> dict:
    global _session_id, _llm, _session_path_hops, _session_path_devices, _session_interface_counters
    _session_id = session_id
    _session_path_hops = None  # reset for each new invocation
    _session_path_devices = []
    _session_interface_counters = []
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

    # Deterministic plan for explicit routing queries — skip cache and ReAct loop entirely
    if _NORNIR_PATTERNS.search(prompt):
        _deterministic_plan = ["call_nornir_path_agent", "call_panorama_agent", "call_splunk_agent", "call_servicenow_agent"]
        logger.info("Deterministic plan (no-NetBrain): %s", _deterministic_plan)
        tool_outputs = await _execute_plan(_deterministic_plan, prompt)
        _plan_used = True
    elif _NETBRAIN_PATTERNS.search(prompt):
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
        agent = create_react_agent(
            model=_llm,
            tools=TROUBLESHOOT_TOOLS,
            prompt=SystemMessage(content=_load_skill()),
        )

        result = await agent.ainvoke(
            {"messages": [("user", prompt)]},
            config={"recursion_limit": 25},
        )

        messages = result.get("messages", [])
        _PATH_AGENTS = {"call_netbrain_agent", "call_nornir_path_agent", "call_netbox_path_agent"}
        _path_agent_seen = False
        for m in messages:
            if isinstance(m, ToolMessage) and m.content:
                # Deduplicate: only keep the first path agent output
                tool_name = getattr(m, "name", "") or ""
                if tool_name in _PATH_AGENTS:
                    if _path_agent_seen:
                        continue
                    _path_agent_seen = True
                tool_outputs.append(str(m.content))

        # Store the tool sequence for future cache hits (skip for explicit tool-routing queries)
        try:
            plan = _extract_plan_from_messages(messages)
            if plan and not _skip_cache:
                await store_tool_plan(prompt, plan, _TOOL_FINGERPRINT)
        except Exception as _store_exc:
            logger.warning("Tool plan store failed: %s", _store_exc)

    # Force ServiceNow call if path devices are known but SNOW wasn't called
    _snow_already_called = any(
        o.startswith("INCIDENTS:") or o.startswith("ServiceNow unavailable")
        for o in tool_outputs
    )
    if _session_path_devices and not _snow_already_called:
        try:
            src_ip_forced, dst_ip_forced = _extract_ips(prompt)
            snow_out = await call_servicenow_agent.ainvoke({
                "device_names": _session_path_devices,
                "source_ip": src_ip_forced or "",
                "dest_ip": dst_ip_forced or "",
                "port": _extract_port(prompt),
            })
            if snow_out:
                tool_outputs.append(snow_out)
        except Exception as _snow_exc:
            logger.warning("Forced SNOW call failed: %s", _snow_exc)

    # Force interface counters call if path is known but LLM skipped it
    if not _session_interface_counters and (_session_path_hops or _session_path_devices):
        # Build device → interfaces mapping from structured path hops (DB trace)
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
        # Fall back to device names only if hops gave us nothing (NetBrain path)
        if not _dev_intfs and _session_path_devices:
            # Parse raw tool outputs for "Hop N: DEVICE ... Egress: INTF" lines
            _HOP_RE = re.compile(r'Hop\s+\d+:\s+(\S+).*?Egress:\s*(\S+)', re.IGNORECASE)
            for _out in tool_outputs:
                for _m in _HOP_RE.finditer(_out):
                    _dev_intfs.setdefault(_m.group(1), set()).add(_m.group(2))

        _forced_entries = [{"device": d, "interfaces": sorted(i)} for d, i in _dev_intfs.items() if i]
        logger.info("Forced interface counters: path_hops=%s path_devices=%s entries=%s",
                    bool(_session_path_hops), _session_path_devices, _forced_entries)
        if _forced_entries:
            try:
                await call_interface_counters_agent.ainvoke({"devices_and_interfaces": _forced_entries})
            except Exception as _intf_exc:
                logger.warning("Forced interface counters call failed: %s", _intf_exc)

    past_memories: list[dict] = []
    past_incidents: list[dict] = []

    if tool_outputs:
        # Strip internal reasoning tags from tool outputs before passing to synthesis
        import re as _re
        cleaned_outputs = []
        for out in tool_outputs:
            out = _re.sub(r"<plan>.*?</plan>", "", out, flags=_re.DOTALL)
            out = _re.sub(r"<reflection>.*?</reflection>", "", out, flags=_re.DOTALL)
            out = _re.sub(r"<plan>.*", "", out, flags=_re.DOTALL)
            out = _re.sub(r"<reflection>.*", "", out, flags=_re.DOTALL)
            out = out.strip()
            if out:
                cleaned_outputs.append(out)

        def _fmt_path_table(raw: str) -> str:
            """Convert raw NetBrain hop text into a markdown table."""
            lines = [
                "| Hop | Device | Type | Egress Interface |",
                "|-----|--------|------|-----------------|",
            ]
            for line in raw.splitlines():
                m = _re.match(r'Hop\s+(\d+):\s+(\S+)\s*\|?\s*Type:\s*([^|]+?)\s*\|?\s*Egress:\s*(.+)', line)
                if m:
                    hop, device, dtype, egress = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
                    lines.append(f"| {hop} | {device} | {dtype} | {egress} |")
                elif _re.match(r'Hop\s+\d+.*Destination reached', line):
                    dm = _re.match(r'Hop\s+(\d+):\s+(\S+)', line)
                    if dm:
                        lines.append(f"| {dm.group(1)} | {dm.group(2)} | — | Destination reached |")
            return "\n".join(lines) if len(lines) > 2 else raw

        def _fmt_firewall_table(raw: str) -> str:
            """Convert raw Panorama policy text into a markdown table."""
            lines = [
                "| Firewall | Device Group | Rule | Action | Zones |",
                "|----------|-------------|------|--------|-------|",
            ]
            for line in raw.splitlines():
                m = _re.search(
                    r'(\S+)\s*\(device_group:\s*([^)]+)\).*?rule=\'([^\']+)\'.*?action=(\w+).*?zones=([^\s.]+)',
                    line
                )
                if m:
                    fw, dg, rule, action, zones = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
                    action_fmt = f"✅ {action}" if action.lower() == "allow" else f"🚫 {action}"
                    lines.append(f"| {fw} | {dg} | {rule} | {action_fmt} | {zones} |")
            return "\n".join(lines) if len(lines) > 2 else raw

        # Separate ServiceNow output from the rest — we inject it directly
        # so the synthesis LLM never has to copy tables verbatim.
        # Also normalise Splunk unavailable messages so synthesis LLM sees clean text.
        snow_raw = None
        non_snow_outputs = []
        for out in cleaned_outputs:
            if out.startswith("INCIDENTS:") or out.startswith("ServiceNow unavailable"):
                snow_raw = out
            elif "agent unavailable" in out.lower() and "splunk" in out.lower():
                non_snow_outputs.append("Splunk: No log data available — agent unreachable.")
            else:
                non_snow_outputs.append(out)

        verbatim_block = "\n\n---\n\n".join(non_snow_outputs) if non_snow_outputs else "(no data)"

        # Determine whether any Palo Alto firewalls are in the path — used to tailor the synthesis prompt
        _path_has_firewalls = bool(_extract_firewalls_from_netbrain("\n".join(cleaned_outputs)))

        # Extract device names for device-based memory recall.
        # Prefer the already-parsed list set by the path agent tool; fall back to text extraction.
        path_devices: list[str] = list(_session_path_devices) if _session_path_devices else []
        if not path_devices:
            for out in cleaned_outputs:
                found = _extract_devices_from_netbrain(out)
                if found:
                    path_devices = found
                    break

        logger.info("Memory recall prep: path_devices=%s session_path_devices=%s", path_devices, _session_path_devices)
        # Recall semantically similar past cases (agent sessions + closed ServiceNow incidents)
        memory_context = ""
        try:
            import atlas.status_bus as status_bus
            await status_bus.push(_session_id or "default", "Recalling similar past cases from memory...")
        except Exception:
            pass
        try:
            from agent_memory import recall_memory, recall_incidents_by_devices, format_memory_context
            past_memories, semantic_incidents, device_incidents = await asyncio.gather(
                recall_memory(prompt, agent_type="atlas", top_k=3),
                recall_memory(prompt, agent_type="incident", top_k=5, min_similarity=0.40),
                recall_incidents_by_devices(path_devices, query=prompt),
            )
            # Merge semantic + device incidents, deduplicate by incident number in result_summary
            seen_keys: set[str] = set()
            past_incidents: list[dict] = []
            for inc in semantic_incidents + device_incidents:
                key = inc.get("result_summary", "")[:30]
                if key not in seen_keys:
                    seen_keys.add(key)
                    past_incidents.append(inc)
            all_memories = past_memories + past_incidents
            memory_context = format_memory_context(all_memories) if all_memories else ""
            logger.info("Memory recall: path_devices=%s atlas=%d incidents=%d (semantic=%d device=%d)",
                        path_devices, len(past_memories), len(past_incidents),
                        len(semantic_incidents), len(device_incidents))
        except Exception as _mem_exc:
            logger.warning("Memory recall failed: %s", _mem_exc)
            memory_context = ""

        if _path_has_firewalls:
            _conclusion_rules = (
                "CRITICAL CONCLUSION RULES:\n"
                "- If Panorama action is 'allow' for the firewalls in the path: the firewall is NOT blocking the traffic. "
                "Root Cause MUST say the firewall permits this traffic and the issue lies elsewhere "
                "(application layer, routing, or endpoint). Do NOT write 'Unable to determine root cause'.\n"
                "- If Panorama action is 'deny'/'drop' for any firewall: that firewall IS the likely cause — name the rule.\n"
                "- If Splunk data is unavailable but Panorama shows 'allow': still conclude the firewall "
                "is not the cause — Splunk absence does not change the Panorama result.\n"
                "- Only write 'Unable to determine root cause' if every agent returned no data at all.\n\n"
                "REPORT SECTIONS (the Recent Changes section will be added separately):\n"
                "1. ## Path Summary — list hops from the path trace\n"
                "2. ## Firewall Policy Check — Panorama rule name, action, zones per firewall\n"
                "3. ## Splunk Traffic Analysis — deny counts and traffic patterns. "
                "If the Splunk data says 'No log data available — agent unreachable', write exactly: "
                "'No log data available — Splunk agent unreachable. Log correlation skipped.'\n"
                "4. ## Root Cause and ## Recommendation — synthesise all findings\n\n"
            )
        else:
            _conclusion_rules = (
                "CRITICAL CONCLUSION RULES:\n"
                "- There are NO Palo Alto firewalls in this path. Do NOT mention firewalls, Panorama, or Splunk.\n"
                "- Read the path data carefully and report what it actually shows — do NOT assume a problem exists.\n"
                "- If the path trace shows successful ARP resolution or 'Destination reachable': "
                "Root Cause MUST say the destination is reachable and the path is healthy. "
                "Recommendation should focus on verifying application/service layer, not routing.\n"
                "- If the path trace shows a routing gap, missing route, or loop: Root Cause should name that specific issue.\n"
                "- Do NOT write 'Unable to determine root cause' — the path data is always sufficient to draw a conclusion.\n"
                "- Do NOT suggest checking OSPF or interfaces unless the path data shows an actual error on those.\n\n"
                "REPORT SECTIONS (the Recent Changes section will be added separately):\n"
                "1. ## Path Summary — list hops from the path trace\n"
                "2. ## Root Cause — one or two sentences based strictly on what the path data shows\n"
                "3. ## Recommendation — specific actionable steps matching the root cause\n\n"
            )

        synthesis_system = (
            (memory_context + "\n\n") if memory_context else ""
        ) + (
            "You are writing the final troubleshooting report. "
            "Below is the VERBATIM DATA returned by each specialist agent. "
            "You MUST use ONLY the information in this data block — do NOT invent, guess, or paraphrase "
            "any device names, policy names, IP addresses, hop counts, deny counts, or zone names. "
            "If a value is not present in the data block, say it was not available.\n\n"
            + _conclusion_rules
            + "VERBATIM AGENT DATA:\n"
            f"{verbatim_block}"
        )
        synthesis_messages = [
            SystemMessage(content=_load_skill()),
            HumanMessage(content=prompt),
            SystemMessage(content=synthesis_system),
            HumanMessage(content="Now write the final report using ONLY the verbatim data above. Do not invent any values."),
        ]
        synthesis_response = await _llm.ainvoke(synthesis_messages)
        final = synthesis_response.content or "Investigation complete — no summary generated."

        # Inject ServiceNow sections directly — parse from raw tool output
        chg_block = ""
        if snow_raw and snow_raw.startswith("INCIDENTS:"):
            parts = _re.split(r'\n\nCHANGE REQUESTS:', snow_raw, maxsplit=1)
            chg_block = parts[1].strip() if len(parts) == 2 else ""

        chg_section = (
            "\n\n## Recent Changes\n\n" + chg_block
            if chg_block and chg_block != "No change requests found."
            else "\n\n## Recent Changes\nNo related changes found for devices in the path."
        )

        # --- Inject Path Summary from raw NetBrain output ---
        netbrain_raw = next((o for o in cleaned_outputs if "Hop " in o or "path trace" in o.lower()), None)
        if netbrain_raw:
            path_table = _fmt_path_table(netbrain_raw)
            final = _re.sub(
                r'## Path Summary.*?(?=## Firewall Policy Check|## Splunk|## Root Cause|$)',
                f'## Path Summary\n\n{path_table}\n\n', final, flags=_re.DOTALL
            )

        # --- Inject Firewall Policy Check from raw Panorama output ---
        panorama_raw = next((o for o in cleaned_outputs if "matching rule=" in o or "action=" in o), None)
        if panorama_raw and _path_has_firewalls:
            fw_table = _fmt_firewall_table(panorama_raw)
            final = _re.sub(
                r'## Firewall Policy Check.*?(?=## Splunk|## Recent Incidents|## Root Cause|$)',
                f'## Firewall Policy Check\n\n{fw_table}\n\n', final, flags=_re.DOTALL
            )
        elif not _path_has_firewalls:
            # No firewalls in path — strip the section entirely
            final = _re.sub(
                r'## Firewall Policy Check.*?(?=## Splunk|## Recent Incidents|## Root Cause|$)',
                '', final, flags=_re.DOTALL
            ).strip()

        # --- Inject Splunk section (only when firewalls are in the path) ---
        splunk_raw = next((o for o in non_snow_outputs if "splunk" in o.lower()), None)
        if splunk_raw and _path_has_firewalls:
            if "no log data" in splunk_raw.lower() or "agent unreachable" in splunk_raw.lower():
                splunk_section = "## Splunk Traffic Analysis\n\nNo log data available — Splunk agent unreachable. Log correlation skipped.\n\n"
            else:
                splunk_section = f"## Splunk Traffic Analysis\n\n{splunk_raw}\n\n"
            final = _re.sub(
                r'## Splunk Traffic Analysis.*?(?=## Recent Incidents|## Root Cause|## Recommendation|$)',
                splunk_section, final, flags=_re.DOTALL
            )
        elif not _path_has_firewalls:
            # No firewalls — strip Splunk section (data would be misleading)
            final = _re.sub(
                r'## Splunk Traffic Analysis.*?(?=## Recent Incidents|## Root Cause|## Recommendation|$)',
                '', final, flags=_re.DOTALL
            ).strip()

        # Strip any synthesis-generated Recent Incidents/Changes sections, inject correct ones
        final = _re.sub(
            r'## Recent Incidents.*?(?=## Recent Changes|## Root Cause|## Recommendation|$)',
            '', final, flags=_re.DOTALL
        ).strip()
        final = _re.sub(
            r'## Recent Changes.*?(?=## Root Cause|## Recommendation|$)',
            '', final, flags=_re.DOTALL
        ).strip()

        root_cause_match = _re.search(r'## Root Cause', final)
        if root_cause_match:
            insert_pos = root_cause_match.start()
            final = final[:insert_pos].rstrip() + chg_section + "\n\n" + final[insert_pos:]
        else:
            final = final.rstrip() + chg_section

        # --- Append change context to Recommendation ---
        has_changes = chg_block and chg_block != "No change requests found."
        if has_changes:
            addendum = "- Recent changes were made to path devices — correlate these with the onset of the issue."
            final = _re.sub(
                r'(## Recommendation.*?)$',
                lambda m: m.group(1).rstrip() + "\n" + addendum,
                final, flags=_re.DOTALL
            )
    else:
        # No tool results — fall back to whatever the agent produced
        final = next(
            (m.content for m in reversed(messages) if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)),
            "Investigation complete — no summary generated.",
        )

    # Strip <plan>...</plan> and <reflection>...</reflection> tags — internal reasoning, not for display
    final = re.sub(r"<plan>.*?</plan>", "", final, flags=re.DOTALL).strip()
    final = re.sub(r"<reflection>.*?</reflection>", "", final, flags=re.DOTALL).strip()

    # Ensure known section headers are on their own paragraph line.
    # The LLM sometimes writes "## Section content" inline; split to "## Section\n\ncontent".
    final = re.sub(
        r'(## (?:Path Summary|Firewall Policy Check|Splunk Traffic Analysis|Root Cause|Recommendation|Recent Incidents|Recent Changes))[ \t]+(?=\S)',
        lambda m: f'{m.group(1)}\n\n',
        final,
    )

    # When DB path hops are available the visual diagram replaces the text Path Summary
    if _session_path_hops:
        final = re.sub(
            r'## Path Summary.*?(?=## Firewall Policy Check|## Splunk|## Recent Incidents|## Root Cause|$)',
            '', final, flags=re.DOTALL
        ).strip()

    # Store this session in semantic memory — extract Root Cause + Recommendation as the summary
    try:
        from agent_memory import store_memory
        # Store only the diagnostic content — strip injected ServiceNow/memory sections
        storable = final.split("\n\n---\n\n")[0].strip()
        summary_match = re.search(r'## Root Cause(.*?)(?=## Recommendation|$)', storable, re.DOTALL)
        summary = summary_match.group(1).strip()[:800] if summary_match else storable[:800]
        await store_memory(prompt, summary, agent_type="atlas")
    except Exception as _mem_exc:
        logger.debug("agent_memory: store skipped: %s", _mem_exc)

    all_recalled = past_memories + past_incidents  # defined in if tool_outputs block; fallback [] if else branch ran
    logger.info("all_recalled=%d serializing...", len(all_recalled))

    # Strip non-serializable fields from memories before returning to frontend
    serializable_memories = [
        {k: v for k, v in m.items() if k in ("query", "result_summary", "resolution", "timestamp", "similarity", "match_type")}
        for m in all_recalled
        if not (m.get("agent_type") == "atlas" and float(m.get("similarity", 0) or 0) >= 0.99)
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
    if _session_path_hops:
        content["path_hops"] = _session_path_hops
        content["source"] = src_ip if (src_ip := _extract_ips(prompt)[0]) else ""
        content["destination"] = dst_ip if (dst_ip := _extract_ips(prompt)[1]) else ""
    if _session_interface_counters:
        content["interface_counters"] = _session_interface_counters

    return {
        "role": "assistant",
        "content": content,
        "memories": serializable_memories,
    }
