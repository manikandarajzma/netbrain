"""
Troubleshoot Orchestrator — diagnoses connectivity problems between two endpoints.

Uses a LangGraph ReAct agent where the LLM reasons at each step before deciding
which specialist agent to call next based on what the path and prior findings reveal.
"""
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

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "troubleshoot_orchestrator.md"


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
    "servicenow": 300,   #  5 min — tickets update occasionally
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

async def _call_agent(url: str, task: str, timeout: float = 60.0) -> str:
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
    if _stub_agents():
        logger.info("NetBrain stub active — returning fake path")
        return _NETBRAIN_STUB
    cached = _cache_get("netbrain", task)
    if cached:
        logger.info("NetBrain cache hit")
        return cached
    result = await _call_agent(NETBRAIN_AGENT_URL, task)
    if result.startswith("Agent unavailable"):
        return result
    _cache_set("netbrain", result, task)
    return result


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
            call_mcp_tool("search_servicenow_incidents", {"query": query, "limit": 10}, timeout=30.0),
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
            """Keep rows whose text mentions at least one search term (API may still return noise)."""
            if not records or not terms:
                return []
            needles = [t.lower() for t in terms if t]
            out: list[dict] = []
            for r in records:
                blob = _record_blob(r)
                if any(n in blob for n in needles):
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
                f"Mark RELEVANT only if it involves a path device AND describes a networking issue "
                f"(BGP, routing, firewall, interface, connectivity, outage, or config change).\n\n"
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
                # Extract device name from short_description (first word before colon or dash)
                desc_raw = r.get("short_description", "")
                import re as _re2
                device_match = _re2.match(r'^([A-Z][A-Z0-9\-]+(?:-\d+)?)\s*[:\-]', desc_raw)
                device = device_match.group(1) if device_match else "-"
                desc = _cell(desc_raw)
                state = _cell(r.get("state"))
                pri = _cell(r.get("priority"))
                who = _cell((r.get("assigned_to") or {}).get("display_value") if isinstance(r.get("assigned_to"), dict) else r.get("assigned_to"))
                opened = _cell((r.get("opened_at") or "-")[:16])
                resolved = _cell((r.get("resolved_at") or "-")[:16])
                notes = _cell(r.get("close_notes") or r.get("work_notes") or "-", 80)
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
                device = device_match.group(1) if device_match else "-"
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

        import asyncio as _asyncio
        inc_filtered, chg_filtered = await _asyncio.gather(
            _llm_filter_relevant(inc_rows, "incidents"),
            _llm_filter_relevant(chg_rows, "change requests"),
        )

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
async def call_cisco_agent(task: str) -> str:
    """
    Check interface errors, drops, and hardware faults on a Cisco device.
    Use when a Cisco switch or router in the path has suspected interface issues.
    Pass a natural language task describing the device and interfaces to check.
    Example: "Check interface errors on SW-EDGE-02 Gi0/1 and Gi0/2."
    NOTE: This agent is not yet available. Return a note that Cisco agent is pending implementation.
    """
    logger.info("Cisco agent called (not yet implemented): %s", task)
    return "Cisco agent is not yet available. Interface-level diagnostics for Cisco devices are pending implementation."


TROUBLESHOOT_TOOLS = [
    call_netbrain_agent,
    call_panorama_agent,
    call_splunk_agent,
    call_servicenow_agent,
    call_cisco_agent,
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def orchestrate_troubleshoot(
    prompt: str,
    username: str | None = None,
    session_id: str | None = None,
) -> dict:
    global _session_id, _llm
    _session_id = session_id
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

    agent = create_react_agent(
        model=_llm,
        tools=TROUBLESHOOT_TOOLS,
        prompt=SystemMessage(content=_load_skill()),
    )

    result = await agent.ainvoke(
        {"messages": [("user", prompt)]},
        config={"recursion_limit": 25},
    )

    # Collect verbatim tool results from ToolMessages in the message history.
    # These are used to ground the final synthesis so the LLM cannot hallucinate
    # policy names, hop counts, or other specific values.
    from langchain_core.messages import ToolMessage, HumanMessage
    import re

    messages = result.get("messages", [])
    tool_outputs: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage) and m.content:
            tool_outputs.append(str(m.content))

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
        synthesis_system = (
            "You are writing the final troubleshooting report. "
            "Below is the VERBATIM DATA returned by each specialist agent. "
            "You MUST use ONLY the information in this data block — do NOT invent, guess, or paraphrase "
            "any device names, policy names, IP addresses, hop counts, deny counts, or zone names. "
            "If a value is not present in the data block, say it was not available.\n\n"
            "CRITICAL CONCLUSION RULES:\n"
            "- If Panorama action is 'allow' for the firewalls in the path: the firewall is NOT blocking the traffic. "
            "Root Cause MUST say the firewall permits this traffic and the issue lies elsewhere "
            "(application layer, routing, or endpoint). Do NOT write 'Unable to determine root cause'.\n"
            "- If Panorama action is 'deny'/'drop' for any firewall: that firewall IS the likely cause — name the rule.\n"
            "- If Splunk data is unavailable but Panorama shows 'allow': still conclude the firewall "
            "is not the cause — Splunk absence does not change the Panorama result.\n"
            "- Only write 'Unable to determine root cause' if every agent returned no data at all.\n\n"
            "REPORT SECTIONS (write all four — the Recent Incidents and Recent Changes sections will be added separately):\n"
            "1. **Path Summary** — list hops from NetBrain\n"
            "2. **Firewall Policy Check** — Panorama rule name, action, zones per firewall\n"
            "3. **Splunk Traffic Analysis** — deny counts and traffic patterns. "
            "If the Splunk data says 'No log data available — agent unreachable', write exactly: "
            "'No log data available — Splunk agent unreachable. Log correlation skipped.'\n"
            "4. **Root Cause** and **Recommendation** — synthesise all findings\n\n"
            "VERBATIM AGENT DATA:\n"
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
        inc_block = ""
        chg_block = ""
        if snow_raw and snow_raw.startswith("INCIDENTS:"):
            inc_block = ""
            chg_block = ""
            parts = _re.split(r'\n\nCHANGE REQUESTS:', snow_raw, maxsplit=1)
            if len(parts) == 2:
                inc_block = parts[0].replace("INCIDENTS:\n", "", 1).strip()
                chg_block = parts[1].strip()
            else:
                inc_block = snow_raw.replace("INCIDENTS:\n", "", 1).strip()

            inc_section = (
                "\n\n**Recent Incidents**\n\n" + inc_block
                if inc_block and inc_block != "No incidents found."
                else "\n\n**Recent Incidents**\nNo related incidents found for devices in the path."
            )
            chg_section = (
                "\n\n**Recent Changes**\n\n" + chg_block
                if chg_block and chg_block != "No change requests found."
                else "\n\n**Recent Changes**\nNo related changes found for devices in the path."
            )
        else:
            inc_section = "\n\n**Recent Incidents**\nNo related incidents found for devices in the path."
            chg_section = "\n\n**Recent Changes**\nNo related changes found for devices in the path."

        # --- Inject Path Summary from raw NetBrain output ---
        netbrain_raw = next((o for o in cleaned_outputs if "Hop " in o or "path trace" in o.lower()), None)
        if netbrain_raw:
            path_table = _fmt_path_table(netbrain_raw)
            final = _re.sub(
                r'\*\*Path Summary\*\*.*?(?=\*\*Firewall Policy Check\*\*|\*\*Splunk|\*\*Root Cause\*\*|$)',
                f'**Path Summary**\n\n{path_table}\n\n', final, flags=_re.DOTALL
            )

        # --- Inject Firewall Policy Check from raw Panorama output ---
        panorama_raw = next((o for o in cleaned_outputs if "matching rule=" in o or "action=" in o), None)
        if panorama_raw:
            fw_table = _fmt_firewall_table(panorama_raw)
            final = _re.sub(
                r'\*\*Firewall Policy Check\*\*.*?(?=\*\*Splunk|\*\*Recent Incidents\*\*|\*\*Root Cause\*\*|$)',
                f'**Firewall Policy Check**\n\n{fw_table}\n\n', final, flags=_re.DOTALL
            )

        # --- Inject Splunk section ---
        splunk_raw = next((o for o in non_snow_outputs if "splunk" in o.lower()), None)
        if splunk_raw:
            if "no log data" in splunk_raw.lower() or "agent unreachable" in splunk_raw.lower():
                splunk_section = "**Splunk Traffic Analysis**\n\nNo log data available — Splunk agent unreachable. Log correlation skipped.\n\n"
            else:
                splunk_section = f"**Splunk Traffic Analysis**\n\n{splunk_raw}\n\n"
            final = _re.sub(
                r'\*\*Splunk Traffic Analysis\*\*.*?(?=\*\*Recent Incidents\*\*|\*\*Root Cause\*\*|\*\*Recommendation\*\*|$)',
                splunk_section, final, flags=_re.DOTALL
            )

        # Strip any synthesis-generated Recent Incidents/Changes sections (they may be wrong)
        # then insert the correct ones from the tool output
        final = _re.sub(
            r'\*\*Recent Incidents\*\*.*?(?=\*\*Recent Changes\*\*|\*\*Root Cause\*\*|\*\*Recommendation\*\*|$)',
            '', final, flags=_re.DOTALL
        ).strip()
        final = _re.sub(
            r'\*\*Recent Changes\*\*.*?(?=\*\*Root Cause\*\*|\*\*Recommendation\*\*|$)',
            '', final, flags=_re.DOTALL
        ).strip()

        root_cause_match = _re.search(r'\*\*Root Cause\*\*', final)
        if root_cause_match:
            insert_pos = root_cause_match.start()
            final = final[:insert_pos].rstrip() + inc_section + chg_section + "\n\n" + final[insert_pos:]
        else:
            final = final.rstrip() + inc_section + chg_section

        # --- Append incident/change context to Recommendation ---
        has_incidents = inc_block and inc_block != "No incidents found."
        has_changes = chg_block and chg_block != "No change requests found."
        if has_incidents or has_changes:
            addendum_lines = []
            if has_incidents:
                addendum_lines.append("- Open incidents exist on path devices — review these before making changes.")
            if has_changes:
                addendum_lines.append("- Recent changes were made to path devices — correlate these with the onset of the issue.")
            addendum = "\n".join(addendum_lines)
            final = _re.sub(
                r'(\*\*Recommendation\*\*.*?)$',
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

    return {
        "role": "assistant",
        "content": {"direct_answer": final},
    }
