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

    # Auto-inject long-term memory context before the agent starts reasoning
    memory_context = ""
    try:
        try:
            from atlas.agent_memory import recall_memory, recall_incidents_by_devices, format_memory_context
        except ImportError:
            from agent_memory import recall_memory, recall_incidents_by_devices, format_memory_context  # type: ignore
        past = await recall_memory(full_prompt, agent_type="troubleshoot", top_k=3, min_similarity=0.65)
        if past:
            memory_context = format_memory_context(past)
            logger.info("injecting %d past cases into context", len(past))
    except Exception as exc:
        logger.debug("memory recall skipped: %s", exc)

    agent_input = full_prompt
    if memory_context:
        agent_input = f"{memory_context}\n\n---\n\n{full_prompt}"

    config = {"configurable": {"session_id": session_id, "thread_id": session_id}}

    try:
        agent  = build_agent(full_prompt, issue_type)
        result = await agent.ainvoke({"messages": [HumanMessage(content=agent_input)]}, config=config)
    except Exception as exc:
        logger.error("Troubleshoot agent failed: %s\n%s", exc, _tb.format_exc())
        return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Troubleshooting failed: {exc}"}}}

    final_text    = _extract_final_text(result.get("messages", []))
    session_data  = pop_session_data(session_id)
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
