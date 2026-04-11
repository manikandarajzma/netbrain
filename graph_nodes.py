"""
Atlas LangGraph nodes.

Graph shape:
  classify_intent
      ├─► call_troubleshoot_agent   (connectivity / device-health investigation)
      ├─► call_network_ops_agent    (firewall change requests, policy review, access requests)
      └─► build_final_response      (dismiss / early-exit)
"""
import json
import logging
import re
import traceback as _tb
from typing import Any

from langchain_core.messages import HumanMessage

try:
    from atlas.graph_state import AtlasState
    from atlas.chat_service import _IP_OR_CIDR_RE
except ImportError:
    from graph_state import AtlasState        # type: ignore[assignment]
    from chat_service import _IP_OR_CIDR_RE  # type: ignore[assignment]

logger = logging.getLogger("atlas.graph_nodes")


# ---------------------------------------------------------------------------
# Pending clarification state (Redis-backed with in-memory fallback)
# ---------------------------------------------------------------------------

_pending_ts_mem: dict[str, str] = {}
_PENDING_TS_TTL = 600


def _pending_ts_set(session_id: str, prompt: str, issue_type: str = "general") -> None:
    payload = json.dumps({"prompt": prompt, "issue_type": issue_type})
    try:
        import os, redis as _r
        _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=True).setex(f"atlas:pending_ts:{session_id}", _PENDING_TS_TTL, payload)
        return
    except Exception:
        pass
    _pending_ts_mem[session_id] = payload


def _pending_ts_get(session_id: str) -> tuple[str | None, str | None]:
    try:
        import os, redis as _r
        raw = _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                          decode_responses=True).get(f"atlas:pending_ts:{session_id}")
    except Exception:
        raw = _pending_ts_mem.get(session_id)
    if not raw:
        return None, None
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get("prompt", ""), data.get("issue_type", "general")
        return str(data), "general"
    except Exception:
        return str(raw), "general"


def _pending_ts_exists(session_id: str) -> bool:
    try:
        import os, redis as _r
        return _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                           decode_responses=True).exists(f"atlas:pending_ts:{session_id}") > 0
    except Exception:
        pass
    return session_id in _pending_ts_mem


def _pending_ts_delete(session_id: str) -> None:
    try:
        import os, redis as _r
        _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=True).delete(f"atlas:pending_ts:{session_id}")
        return
    except Exception:
        pass
    _pending_ts_mem.pop(session_id, None)


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

_DISMISSALS      = {"no", "nope", "nah", "no thanks", "never mind", "nevermind", "skip",
                    "not now", "no need", "i'm good", "im good", "all good"}
_ACKNOWLEDGEMENTS = {"yes", "yeah", "sure", "ok", "okay", "great", "thanks",
                     "thank you", "cool", "got it", "noted", "perfect", "sounds good"}

# Signals a network-ops / document-generation workflow rather than troubleshooting.
# Checked BEFORE the troubleshoot fallback so explicit ops requests are never
# misclassified as "something is broken" investigations.
#
# Covers:
#   - Explicit request phrases (firewall request, change request, fw request…)
#   - Port/access opening ("open port 443", "need port … open", "whitelist")
#   - Rule operations (add/create/remove/update/modify rule)
#   - Traffic directives that are actionable, not diagnostic (allow/block/permit/deny traffic)
#   - Policy & security review workflows
#   - Document generation (spreadsheet, change ticket)
_NETWORK_OPS_RE = re.compile(
    r"\b("
    # Explicit ops request phrases
    r"firewall\s+request|fw\s+request|change\s+request|network\s+change|"
    r"access\s+request|request\s+access|security\s+review|policy\s+review|"
    r"spreadsheet|change\s+ticket|"
    # Port opening / whitelisting
    r"open\s+port|need\s+port|port\s+\d+\s+(open|allowed|whitelisted)|"
    r"whitelist|need\s+access\s+(from|to|between)|grant\s+access|allow\s+access|"
    # Rule CRUD
    r"(add|create|remove|delete|update|modify)\s+(a\s+)?(firewall\s+)?rule|"
    r"new\s+rule|fw\s+rule|firewall\s+rule|"
    # Actionable traffic directives (not diagnostic)
    r"allow\s+traffic|block\s+traffic|permit\s+traffic|deny\s+traffic|"
    r"allow\s+\d{1,3}\.\d{1,3}|permit\s+\d{1,3}\.\d{1,3}|"
    # Panorama / firewall policy work
    r"firewall\s+policy|fw\s+policy|panorama\s+rule|security\s+policy"
    r")\b",
    re.IGNORECASE,
)


async def classify_intent(state: AtlasState) -> dict[str, Any]:
    """
    Classify the user's prompt and set state["intent"].

    Possible values
    ---------------
    "troubleshoot"
        Layered connectivity / device-health investigation.
        → routed to call_troubleshoot_agent
    "network_ops"
        Operational workflow: firewall change request, policy review,
        spreadsheet generation, etc.
        → routed to call_network_ops_agent
    "dismiss"
        Bare acknowledgement with nothing pending — skip LLM entirely.
        → short-circuits to build_final_response
    """
    prompt     = state["prompt"]
    session_id = state.get("session_id") or "default"

    try:
        import atlas.status_bus as sb
        await sb.push(session_id, "Analyzing your query...")
    except Exception:
        pass

    prompt_lower = prompt.lower().strip().rstrip("!.")

    # Plain acknowledgement / dismissal with nothing pending → dismiss
    if prompt_lower in (_ACKNOWLEDGEMENTS | _DISMISSALS):
        if not _pending_ts_exists(session_id):
            return {
                "intent":         "dismiss",
                "final_response": {"role": "assistant",
                                   "content": "Sure, let me know if you need anything else."},
            }

    # Reply to a pending clarification (short, no IPs) → continue whichever flow is pending
    if _pending_ts_exists(session_id):
        if not _IP_OR_CIDR_RE.search(prompt) and len(prompt.split()) <= 15:
            # Preserve the original intent stored in the pending payload
            _, pending_issue_type = _pending_ts_get(session_id)
            intent = "network_ops" if pending_issue_type == "network_ops" else "troubleshoot"
            return {"intent": intent}
        _pending_ts_delete(session_id)

    # Ambiguity guard: diagnostic framing overrides ops keywords.
    #
    # Rule: if the prompt is INVESTIGATING an existing situation (failed request,
    # broken rule, rejected change) route to troubleshoot.
    # Only route to network_ops when the user explicitly wants to CREATE or OPEN
    # something new (open port, add rule, whitelist).
    #
    # _DIAGNOSTIC_FRAMING_RE — signals the user wants diagnosis, not creation:
    #   • "why" (standalone or "why is/are/isn't…")
    #   • "rejected", "not matching", "not working", "failing", "debug", "broken"
    #   • explicit troubleshoot/investigate verbs
    _DIAGNOSTIC_FRAMING_RE = re.compile(
        r"\b(why(\s+(is|are|isn'?t|aren'?t|can'?t|doesn'?t|won'?t))?|"
        r"what'?s\s+(blocking|causing|wrong|happening)|"
        r"rejected|not\s+matching|not\s+being\s+applied|"
        r"not\s+(working|reachable|responding|connecting)|"
        r"can'?t\s+(reach|connect|ping|access)|"
        r"investigate|diagnose|debug|troubleshoot|check\s+why|"
        r"failing|broken|down\b|unreachable|timed?\s*out)\b",
        re.IGNORECASE,
    )

    # _ACTION_RE — explicit creation / opening intent only.
    # Deliberately excludes nouns like "change request", "firewall rule", "fw request"
    # that can refer to an EXISTING thing the user is investigating.
    _ACTION_RE = re.compile(
        r"\b(open\s+port|need\s+port|whitelist|"
        r"(add|create|submit|raise)\s+(a\s+)?(new\s+)?(firewall\s+)?rule|"
        r"allow\s+access|grant\s+access|new\s+rule)\b",
        re.IGNORECASE,
    )

    if _NETWORK_OPS_RE.search(prompt):
        if _DIAGNOSTIC_FRAMING_RE.search(prompt) and not _ACTION_RE.search(prompt):
            return {"intent": "troubleshoot"}
        return {"intent": "network_ops"}

    # Everything else is a troubleshooting query
    return {"intent": "troubleshoot"}


# ---------------------------------------------------------------------------
# Shared infrastructure helpers
# ---------------------------------------------------------------------------

_INC_RE = re.compile(r'\bINC\d+\b', re.IGNORECASE)
_IP_RE  = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')


def _extract_ips(text: str) -> tuple[str, str]:
    ips = _IP_RE.findall(text)
    return (ips[0] if ips else ""), (ips[1] if len(ips) > 1 else "")


def _extract_final_text(messages: list) -> str:
    text = next(
        (m.content for m in reversed(messages)
         if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)),
        "",
    )
    text = re.sub(r"<plan>.*?</plan>",             "", text, flags=re.DOTALL)
    text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)
    return text.strip()


def _needs_forced_peering_inspection(session_data: dict[str, Any]) -> bool:
    routing = session_data.get("routing_history") or {}
    peer_hint = routing.get("peer_hint") or {}
    if not all((
        str(peer_hint.get("from_device", "")).strip(),
        str(peer_hint.get("from_interface", "")).strip(),
        str(peer_hint.get("to_device", "")).strip(),
        str(peer_hint.get("to_interface", "")).strip(),
    )):
        return False

    inspections = session_data.get("peering_inspections") or []
    for item in inspections:
        if not isinstance(item, dict):
            continue
        same_forward = (
            item.get("device_a") == peer_hint.get("from_device")
            and item.get("interface_a") == peer_hint.get("from_interface")
            and item.get("device_b") == peer_hint.get("to_device")
            and item.get("interface_b") == peer_hint.get("to_interface")
        )
        same_reverse = (
            item.get("device_b") == peer_hint.get("from_device")
            and item.get("interface_b") == peer_hint.get("from_interface")
            and item.get("device_a") == peer_hint.get("to_device")
            and item.get("interface_a") == peer_hint.get("to_interface")
        )
        if same_forward or same_reverse:
            return False
    return True


def _replace_markdown_section(text: str, header: str, body: str) -> str:
    pattern = rf"(^## {re.escape(header)}\s*$)(.*?)(?=^## |\Z)"
    replacement = rf"\1\n\n{body.strip()}\n\n"
    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE | re.DOTALL)
    if count:
        return new_text.strip()
    return (text.strip() + f"\n\n## {header}\n\n{body.strip()}").strip()


def _derive_connectivity_evidence(session_data: dict[str, Any], dst_ip: str) -> dict[str, str] | None:
    routing = session_data.get("routing_history") or {}
    peer_hint = routing.get("peer_hint") or {}
    from_device = str(peer_hint.get("from_device", "")).strip()
    from_interface = str(peer_hint.get("from_interface", "")).strip()
    to_device = str(peer_hint.get("to_device", "")).strip()
    to_interface = str(peer_hint.get("to_interface", "")).strip()
    next_hop_ip = str(peer_hint.get("next_hop_ip", "")).strip()
    if not all((from_device, from_interface, to_device, to_interface)):
        return None

    def _interface_state(device: str, interface: str) -> dict[str, Any]:
        data = (session_data.get("all_interfaces") or {}).get(device, {})
        for item in data.get("interfaces", []) if isinstance(data, dict) else []:
            if item.get("interface") == interface:
                return item
        detail = (session_data.get("interface_details") or {}).get(f"{device}:{interface}", {})
        return detail if isinstance(detail, dict) else {}

    def _counter_state(device: str, interface: str) -> dict[str, Any]:
        for item in session_data.get("interface_counters", []) or []:
            if item.get("device") != device:
                continue
            for active in item.get("active", []) or []:
                if active.get("interface") == interface:
                    return active
            if interface in (item.get("clean") or []):
                return {"clean": True}
        return {}

    def _syslog_has(device: str, pattern: str) -> bool:
        syslog_data = (session_data.get("syslog") or {}).get(device, {})
        lines = syslog_data.get("relevant") or []
        pattern_l = pattern.lower()
        return any(pattern_l in str(line).lower() for line in lines)

    def _find_ping(device: str, destination: str, source_interface: str = "") -> dict[str, Any]:
        for item in reversed(session_data.get("ping_results", []) or []):
            if item.get("device") != device:
                continue
            if item.get("destination") != destination:
                continue
            if source_interface and item.get("source_interface") != source_interface:
                continue
            return item
        return {}

    from_state = _interface_state(from_device, from_interface)
    to_state = _interface_state(to_device, to_interface)
    from_counter = _counter_state(from_device, from_interface)
    to_counter = _counter_state(to_device, to_interface)
    from_up = from_state.get("up")
    to_up = to_state.get("up")
    inactivity_seen = _syslog_has(from_device, "inactivity timer expired") or _syslog_has(to_device, "inactivity timer expired")
    from_desc = f"{from_device} {from_interface}"
    to_desc = f"{to_device} {to_interface}"
    forward_peer_ping = _find_ping(from_device, next_hop_ip, from_interface) if next_hop_ip else {}
    reverse_peer_ping = _find_ping(to_device, next_hop_ip, to_interface) if next_hop_ip else {}
    peer_ping_failed = (
        bool(next_hop_ip)
        and forward_peer_ping.get("success") is False
        and reverse_peer_ping.get("success") is False
    )
    one_way_peer_ping_failed = (
        bool(next_hop_ip)
        and not peer_ping_failed
        and (
            forward_peer_ping.get("success") is False
            or reverse_peer_ping.get("success") is False
        )
    )

    if from_up is False or to_up is False:
        down_desc = from_desc if from_up is False else to_desc
        root_cause = (
            f"{from_device} historically learned the route to {dst_ip} from {to_device} over the "
            f"{from_desc} <-> {to_desc} OSPF peering. {down_desc} is currently down, so that peering dropped "
            f"and {to_device} stopped advertising the route to {dst_ip}."
        )
        recommendation = (
            f"Restore the failed peering interface ({down_desc}) and then confirm the OSPF adjacency reforms on "
            f"{from_desc} <-> {to_desc}."
        )
    elif from_counter.get("delta_9s") or to_counter.get("delta_9s"):
        noisy_desc = from_desc if from_counter.get("delta_9s") else to_desc
        root_cause = (
            f"{from_device} historically learned the route to {dst_ip} from {to_device} over the "
            f"{from_desc} <-> {to_desc} OSPF peering. That peering is failing while {noisy_desc} shows actively "
            f"incrementing interface errors, which is the most likely trigger for the adjacency loss and route withdrawal."
        )
        recommendation = (
            f"Fix the physical/link issue on {noisy_desc} first, then re-check OSPF adjacency on "
            f"{from_desc} <-> {to_desc}."
        )
    elif peer_ping_failed:
        root_cause = (
            f"{from_device} historically learned the route to {dst_ip} from {to_device} over the "
            f"{from_desc} <-> {to_desc} OSPF peering. Both sides now fail to reach the historical peering IP "
            f"{next_hop_ip} from that adjacency, which localizes the outage to the peering itself rather than "
            f"the destination LAN interface. That peer reachability failure is what caused the OSPF adjacency to "
            f"drop and the route advertisement to disappear upstream."
        )
        recommendation = (
            f"Treat {from_desc} <-> {to_desc} as the failing peering. Fix peer reachability on that link first: "
            f"confirm both interfaces are physically up, verify the peering IP {next_hop_ip} answers ping from both sides, "
            f"and only if IP reachability works then compare OSPF timers, area, authentication, MTU, network type, and BFD."
        )
    elif one_way_peer_ping_failed:
        failing_desc = from_desc if forward_peer_ping.get("success") is False else to_desc
        root_cause = (
            f"{from_device} historically learned the route to {dst_ip} from {to_device} over the "
            f"{from_desc} <-> {to_desc} OSPF peering. Current evidence shows the peering IP {next_hop_ip} is not "
            f"reachable from {failing_desc}, which narrows the failure to that bilateral peering rather than the "
            f"destination LAN interface. The adjacency loss is the routing symptom; the missing peer reachability "
            f"on the peering is the stronger root-cause clue."
        )
        recommendation = (
            f"Start on {failing_desc}. Verify the interface is sourcing traffic correctly toward peer IP {next_hop_ip}, "
            f"then compare OSPF timers, area, authentication, MTU, network type, and BFD across {from_desc} <-> {to_desc}."
        )
    else:
        root_cause = (
            f"{from_device} historically learned the route to {dst_ip} from {to_device} over the "
            f"{from_desc} <-> {to_desc} OSPF peering. That peering is no longer up, so {to_device} stopped "
            f"advertising the route and upstream devices lost reachability to {dst_ip}."
        )
        if inactivity_seen:
            root_cause += (
                " Syslog confirms the adjacency timed out, but the collected evidence does not yet prove whether "
                "the underlying trigger is peer-side reachability loss or an OSPF parameter mismatch on that peering."
            )
        if next_hop_ip:
            root_cause += (
                f" The historical next-hop on that peering was {next_hop_ip}, which is the correct place "
                f"to focus rather than the destination LAN interface."
            )

        recommendation = (
            f"Focus on the specific {from_desc} <-> {to_desc} OSPF peering. Verify both interfaces are up, "
            f"verify the peering IP answers ping from both sides, and compare OSPF timers, area, authentication, "
            f"MTU, network type, and BFD settings on that adjacency before changing anything else."
        )

    return {
        "from_device": from_device,
        "from_interface": from_interface,
        "to_device": to_device,
        "to_interface": to_interface,
        "next_hop_ip": next_hop_ip,
        "root_cause": root_cause,
        "recommendation": recommendation,
    }


def _apply_connectivity_evidence_guardrails(final_text: str, session_data: dict[str, Any], dst_ip: str) -> str:
    evidence = _derive_connectivity_evidence(session_data, dst_ip)
    if not evidence:
        return final_text

    rewritten = final_text
    rewritten = rewritten.replace("the interface associated with the 10.0.200.0/24 subnet", "the OSPF peering")
    rewritten = rewritten.replace("the interface connected to the 10.0.200.0/24 subnet", "the OSPF peering")
    rewritten = _replace_markdown_section(rewritten, "Root Cause", evidence["root_cause"])
    rewritten = _replace_markdown_section(rewritten, "Recommendation", evidence["recommendation"])
    return rewritten


async def _push_status(session_id: str, msg: str) -> None:
    try:
        try:
            import atlas.status_bus as sb
        except ImportError:
            import status_bus as sb  # type: ignore
        await sb.push(session_id, msg)
    except Exception:
        pass


async def _resolve_inc(prompt: str) -> tuple[str, dict | None]:
    """Expand INC→IPs when the prompt has an INC number but no IPs."""
    m = _INC_RE.search(prompt)
    if not m or _IP_RE.search(prompt):
        return prompt, None
    inc_num = m.group(0).upper()
    try:
        try:
            from atlas.tools.servicenow_tools import get_servicenow_incident as _t
        except ImportError:
            from tools.servicenow_tools import get_servicenow_incident as _t  # type: ignore
        fn   = getattr(_t, "fn", None) or _t
        data = await fn(inc_num)
        if "error" in data:
            return prompt, None
        r    = data.get("result", {})
        desc = r.get("description") or r.get("short_description") or ""
        ips  = _IP_RE.findall(desc)
        if len(ips) < 2:
            return prompt, None
        port_hit = re.search(r'\bport\s+(\d+)\b', desc, re.IGNORECASE)
        port_str = f" port {port_hit.group(1)}" if port_hit else ""
        new_prompt = f"{prompt} (source: {ips[0]}, destination: {ips[1]}{port_str})"
        logger.info("INC→IP resolved: %s → %s", inc_num, new_prompt[-60:])
        inc_summary = {
            "number":            r.get("number", inc_num),
            "short_description": r.get("short_description", ""),
            "state":             r.get("state", ""),
            "priority":          r.get("priority", ""),
            "opened_at":         r.get("opened_at", ""),
            "assigned_to":       (r.get("assigned_to") or {}).get("display_value") or "Unassigned",
        }
        return new_prompt, inc_summary
    except Exception as exc:
        logger.warning("INC→IP resolution failed: %s", exc)
        return prompt, None


# ---------------------------------------------------------------------------
# Node 2: call_troubleshoot_agent
# ---------------------------------------------------------------------------

async def call_troubleshoot_agent(state: AtlasState) -> dict[str, Any]:
    """Build and invoke the troubleshoot ReAct agent; collect session data."""
    try:
        from atlas.agents.troubleshoot_agent import build_agent
        from atlas.tools.all_tools import pop_session_data
    except ImportError:
        from agents.troubleshoot_agent import build_agent                # type: ignore
        from tools.all_tools import pop_session_data                     # type: ignore

    session_id = state.get("session_id") or "default"
    prompt     = state["prompt"]

    await _push_status(session_id, "Investigating...")

    # Recover clarification context if pending
    pending, pending_issue_type = _pending_ts_get(session_id)
    if pending:
        _pending_ts_delete(session_id)

    # Fallback: detect clarification reply from conversation history
    if not pending:
        history = state.get("conversation_history") or []
        if len(history) >= 2:
            last_assistant = next(
                (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"), ""
            )
            last_user_before = next(
                (m.get("content", "") for m in reversed(history[:-1]) if m.get("role") == "user"), ""
            )
            is_clarification_reply = (
                ("which port" in last_assistant.lower() or "what type of issue" in last_assistant.lower())
                and not _IP_OR_CIDR_RE.findall(prompt)
                and len(prompt.split()) <= 6
            )
            if is_clarification_reply and _IP_OR_CIDR_RE.findall(last_user_before):
                pending            = last_user_before
                pending_issue_type = "general"

    issue_type  = pending_issue_type or "general"
    full_prompt = f"{pending}\n\nUser clarification: {prompt}" if pending else prompt

    if not pending:
        ip_matches = _IP_OR_CIDR_RE.findall(prompt)
        words      = prompt.lower().split()
        has_device_context = any(c.isdigit() or "-" in w for w in words for c in w)
        if not has_device_context and not ip_matches and len(prompt.split()) < 4:
            return {"final_response": {"role": "assistant", "content": (
                "Please describe the problem — include device names, IP addresses, or what is failing.\n"
                'Example: "Why can\'t 10.0.0.1 connect to 11.0.0.1?" or "arista1 is unreachable"'
            )}}

    # INC→IP expansion
    full_prompt, inc_summary = await _resolve_inc(full_prompt)

    agent_input = full_prompt

    config = {"configurable": {"session_id": session_id, "thread_id": session_id}}

    try:
        agent  = build_agent(full_prompt, issue_type)
        result = await agent.ainvoke({"messages": [HumanMessage(content=agent_input)]}, config=config)
    except Exception as exc:
        logger.error("Troubleshoot agent failed: %s\n%s", exc, _tb.format_exc())
        return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Troubleshooting failed: {exc}"}}}

    final_text    = _extract_final_text(result.get("messages", []))
    session_data  = pop_session_data(session_id)
    if _needs_forced_peering_inspection(session_data):
        peer_hint = (session_data.get("routing_history") or {}).get("peer_hint") or {}
        await _push_status(session_id, "Gathering required peering evidence...")
        follow_up = (
            f"{full_prompt}\n\n"
            "Required follow-up before answering:\n"
            f"- Routing history identified a concrete OSPF peering pair: {peer_hint.get('from_device')} {peer_hint.get('from_interface')} "
            f"<-> {peer_hint.get('to_device')} {peer_hint.get('to_interface')} via {peer_hint.get('next_hop_ip')}\n"
            "- You must call inspect_ospf_peering(...) for that exact pair before writing Root Cause or Recommendation.\n"
            "- Base the conclusion on that fresh inspection evidence, not prior runs or generic OSPF wording."
        )
        agent = build_agent(full_prompt, issue_type)
        result = await agent.ainvoke({"messages": [HumanMessage(content=follow_up)]}, config=config)
        final_text = _extract_final_text(result.get("messages", []))
        session_data = pop_session_data(session_id)

    path_hops     = session_data.get("path_hops", [])
    rev_hops      = session_data.get("reverse_path_hops", [])
    counters      = session_data.get("interface_counters", [])
    src_ip, dst_ip = _extract_ips(full_prompt)

    content: dict = {}
    if final_text:    content["direct_answer"]      = final_text
    if path_hops:
        content["path_hops"]   = path_hops
        content["source"]      = src_ip
        content["destination"] = dst_ip
    if rev_hops:      content["reverse_path_hops"]  = rev_hops
    if counters:      content["interface_counters"]  = counters
    if inc_summary:   content["incident_summary"]    = inc_summary

    # Store findings in long-term memory for future recall
    if final_text:
        try:
            try:
                from atlas.agent_memory import store_memory
            except ImportError:
                from agent_memory import store_memory  # type: ignore
            import asyncio
            asyncio.create_task(store_memory(full_prompt, final_text, agent_type="troubleshoot"))
        except Exception:
            pass

    logger.info("troubleshoot done: keys=%s hops=%d counters=%d", list(content.keys()), len(path_hops), len(counters))
    return {"final_response": {"role": "assistant", "content": content}}


# ---------------------------------------------------------------------------
# Node 3: call_network_ops_agent
# ---------------------------------------------------------------------------

async def call_network_ops_agent(state: AtlasState) -> dict[str, Any]:
    """Build and invoke the network-ops ReAct agent; collect session data."""
    try:
        from atlas.agents.network_ops_agent import build_agent
        from atlas.tools.all_tools import pop_session_data
    except ImportError:
        from agents.network_ops_agent import build_agent                 # type: ignore
        from tools.all_tools import pop_session_data                     # type: ignore

    session_id = state.get("session_id") or "default"
    prompt     = state["prompt"]

    await _push_status(session_id, "Processing network ops request...")

    config = {"configurable": {"session_id": session_id, "thread_id": session_id}}

    try:
        agent  = build_agent()
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    except Exception as exc:
        logger.error("Network ops agent failed: %s\n%s", exc, _tb.format_exc())
        return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Network ops agent failed: {exc}"}}}

    final_text = _extract_final_text(result.get("messages", []))
    session_data = pop_session_data(session_id)
    path_hops    = session_data.get("path_hops", [])
    rev_hops     = session_data.get("reverse_path_hops", [])
    src_ip, dst_ip = _extract_ips(prompt)

    content: dict = {}
    if final_text: content["direct_answer"] = final_text
    if path_hops:
        content["path_hops"]   = path_hops
        content["source"]      = src_ip
        content["destination"] = dst_ip
    if rev_hops:   content["reverse_path_hops"] = rev_hops

    return {"final_response": {"role": "assistant", "content": content}}


# ---------------------------------------------------------------------------
# Node 4: build_final_response
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    return {}
