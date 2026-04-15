"""
Atlas LangGraph nodes.

Graph shape:
  classify_intent
      ├─► call_troubleshoot_agent   (connectivity / device-health investigation)
      ├─► call_network_ops_agent    (firewall change requests, policy review, access requests)
      └─► build_final_response      (dismiss / early-exit)
"""
import logging
import re
import traceback as _tb
from typing import Any

from langchain_core.messages import HumanMessage

try:
    from atlas.graph_state import AtlasState
    from atlas.chat_service import _IP_OR_CIDR_RE
    from atlas.services.memory_manager import memory_manager
    from atlas.services.request_preprocessor import (
        extract_final_text,
        extract_ips,
        extract_port,
        looks_like_clarification_request,
        resolve_incident_prompt,
    )
    from atlas.services.response_presenter import (
        response_presenter,
    )
    from atlas.services.runtime_helpers import (
        merge_session_data,
        missing_path_visuals,
        needs_connectivity_snapshot,
        push_status,
    )
except ImportError:
    from graph_state import AtlasState        # type: ignore[assignment]
    from chat_service import _IP_OR_CIDR_RE  # type: ignore[assignment]
    from services.memory_manager import memory_manager  # type: ignore
    from services.request_preprocessor import (  # type: ignore
        extract_final_text,
        extract_ips,
        extract_port,
        looks_like_clarification_request,
        resolve_incident_prompt,
    )
    from services.response_presenter import (  # type: ignore
        response_presenter,
    )
    from services.runtime_helpers import (  # type: ignore
        merge_session_data,
        missing_path_visuals,
        needs_connectivity_snapshot,
        push_status,
    )

logger = logging.getLogger("atlas.graph_nodes")


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
#   - Explicit ops request phrases (firewall request, change request, fw request…)
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
    r"create\s+an?\s+incident|open\s+an?\s+incident|raise\s+an?\s+incident|"
    r"create\s+a\s+ticket|open\s+a\s+ticket|raise\s+a\s+ticket|"
    r"details?\s+about\s+(inc|chg)\d+|show\s+(inc|chg)\d+|get\s+(inc|chg)\d+|"
    r"status\s+of\s+(inc|chg)\d+|close\s+inc\d+|update\s+inc\d+|"
    r"inc\d+|chg\d+|"
    r"close\s+chg\d+|update\s+chg\d+|close\s+change\s+request|update\s+change\s+request|"
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

_TROUBLESHOOT_RE = re.compile(
    r"\b(troubleshoot|investigate|diagnose|debug|connectivity|packet\s+loss|latency|slow|unreachable|can't\s+reach|cannot\s+reach|ping|ospf|bgp|eigrp|isis|route|routing|tcp\s+port|port\s+\d+)\b",
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

    await push_status(session_id, "Analyzing your query...")

    prompt_lower = prompt.lower().strip().rstrip("!.")

    # Plain acknowledgement / dismissal with nothing pending → dismiss
    if prompt_lower in (_ACKNOWLEDGEMENTS | _DISMISSALS):
        if not memory_manager.has_pending_context(session_id):
            return {
                "intent":         "dismiss",
                "final_response": {"role": "assistant",
                                   "content": "Sure, let me know if you need anything else."},
            }

    # Reply to a pending clarification → continue whichever flow is pending.
    # Network-ops follow-ups are often long structured field lists, so do not
    # require them to be short.
    if memory_manager.has_pending_context(session_id):
        pending_prompt, pending_issue_type = memory_manager.get_pending_context(session_id)
        if pending_issue_type == "network_ops":
            return {"intent": "network_ops"}
        if not _IP_OR_CIDR_RE.search(prompt) and len(prompt.split()) <= 15:
            intent = "network_ops" if pending_issue_type == "network_ops" else "troubleshoot"
            return {"intent": intent}
        memory_manager.clear_pending_context(session_id)

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
        r"allow\s+access|grant\s+access|new\s+rule|"
        r"(create|open|raise)\s+(an?\s+)?(incident|ticket))\b",
        re.IGNORECASE,
    )

    if _NETWORK_OPS_RE.search(prompt):
        if _DIAGNOSTIC_FRAMING_RE.search(prompt) and not _ACTION_RE.search(prompt):
            return {"intent": "troubleshoot"}
        return {"intent": "network_ops"}

    if _TROUBLESHOOT_RE.search(prompt):
        return {"intent": "troubleshoot"}

    return {
        "intent": "dismiss",
        "final_response": {
            "role": "assistant",
            "content": "Atlas is not equipped to help with it.",
        },
    }
# ---------------------------------------------------------------------------
# Node 2: call_troubleshoot_agent
# ---------------------------------------------------------------------------

async def call_troubleshoot_agent(state: AtlasState) -> dict[str, Any]:
    """Build and invoke the troubleshoot ReAct agent; collect session data."""
    try:
        from atlas.agents.troubleshoot_agent import build_agent
        from atlas.tools.all_tools import pop_session_data, clear_session_cache
    except ImportError:
        from agents.troubleshoot_agent import build_agent                # type: ignore
        from tools.all_tools import pop_session_data, clear_session_cache  # type: ignore

    session_id = state.get("session_id") or "default"
    prompt     = state["prompt"]

    # Start each troubleshoot run from fresh live tool state.
    clear_session_cache(session_id)
    pop_session_data(session_id)

    await push_status(session_id, "Investigating...")

    # Recover clarification context if pending
    pending, pending_issue_type = memory_manager.get_pending_context(session_id)
    if pending:
        memory_manager.clear_pending_context(session_id)

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
    full_prompt, inc_summary = await resolve_incident_prompt(full_prompt)
    if inc_summary:
        src_ip_hint, dst_ip_hint = extract_ips(full_prompt)
        if src_ip_hint and dst_ip_hint:
            issue_type = "connectivity"

    agent_input = full_prompt

    config = {"configurable": {"session_id": session_id, "thread_id": session_id}}

    try:
        agent  = build_agent(full_prompt, issue_type)
        result = await agent.ainvoke({"messages": [HumanMessage(content=agent_input)]}, config=config)
    except Exception as exc:
        logger.error("Troubleshoot agent failed: %s\n%s", exc, _tb.format_exc())
        clear_session_cache(session_id)
        return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Troubleshooting failed: {exc}"}}}

    final_text    = extract_final_text(result.get("messages", []))
    session_data  = pop_session_data(session_id)

    src_ip, dst_ip = extract_ips(full_prompt)
    port = extract_port(full_prompt)
    if needs_connectivity_snapshot(session_data, src_ip, dst_ip):
        await push_status(session_id, "Gathering holistic connectivity evidence...")
        follow_up = (
            f"{full_prompt}\n\n"
            "Required follow-up before answering:\n"
            f"- Call collect_connectivity_snapshot(source_ip={src_ip}, dest_ip={dst_ip}, port={port or ''}) before writing the report.\n"
            "- Use that snapshot as the primary evidence bundle.\n"
            "- If it surfaces multiple independent issues, keep the strongest end-to-end blocker as Root Cause and preserve the others under Additional Findings or Connectivity Test.\n"
            "- Do not stop at one issue if the snapshot shows more than one blocker."
        )
        agent = build_agent(full_prompt, issue_type)
        result = await agent.ainvoke({"messages": [HumanMessage(content=follow_up)]}, config=config)
        final_text = extract_final_text(result.get("messages", []))
        session_data = merge_session_data(session_data, pop_session_data(session_id))

    if missing_path_visuals(session_data, src_ip, dst_ip):
        await push_status(session_id, "Gathering required path visualizations...")
        try:
            try:
                from atlas.tools.all_tools import trace_path, trace_reverse_path
            except ImportError:
                from tools.all_tools import trace_path, trace_reverse_path  # type: ignore
            await trace_path.ainvoke({"source_ip": src_ip, "dest_ip": dst_ip}, config=config)
            await trace_reverse_path.ainvoke({"source_ip": src_ip, "dest_ip": dst_ip}, config=config)
            session_data = merge_session_data(session_data, pop_session_data(session_id))
        except Exception as exc:
            logger.warning("mandatory path visualization collection failed: %s", exc)

    content = response_presenter.build_troubleshoot_content(final_text, session_data, full_prompt, inc_summary)

    # Store findings in long-term memory for future recall
    if final_text:
        await memory_manager.store_agent_memory_entry(full_prompt, final_text, agent_type="troubleshoot")

    logger.info(
        "troubleshoot done: keys=%s hops=%d counters=%d",
        list(content.keys()),
        len(content.get("path_hops", []) or []),
        len(content.get("interface_counters", []) or []),
    )
    clear_session_cache(session_id)
    return {"final_response": {"role": "assistant", "content": content}}


# ---------------------------------------------------------------------------
# Node 3: call_network_ops_agent
# ---------------------------------------------------------------------------

async def call_network_ops_agent(state: AtlasState) -> dict[str, Any]:
    """Build and invoke the network-ops ReAct agent; collect session data."""
    try:
        from atlas.agents.network_ops_agent import build_agent
        from atlas.tools.all_tools import pop_session_data, clear_session_cache
    except ImportError:
        from agents.network_ops_agent import build_agent                 # type: ignore
        from tools.all_tools import pop_session_data, clear_session_cache  # type: ignore

    session_id = state.get("session_id") or "default"
    prompt     = state["prompt"]

    await push_status(session_id, "Processing network ops request...")

    pending, pending_issue_type = memory_manager.get_pending_context(session_id)
    if pending_issue_type == "network_ops" and pending:
        memory_manager.clear_pending_context(session_id)
        prompt = f"{pending}\n\nUser clarification: {prompt}"

    clear_session_cache(session_id)
    pop_session_data(session_id)

    config = {"configurable": {"session_id": session_id, "thread_id": session_id}}

    try:
        agent  = build_agent()
        result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    except Exception as exc:
        logger.error("Network ops agent failed: %s\n%s", exc, _tb.format_exc())
        clear_session_cache(session_id)
        return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Network ops agent failed: {exc}"}}}

    final_text = extract_final_text(result.get("messages", []))
    if looks_like_clarification_request(final_text):
        memory_manager.set_pending_context(session_id, prompt, "network_ops")
    session_data = pop_session_data(session_id)
    content = response_presenter.build_network_ops_content(final_text, session_data, prompt)

    clear_session_cache(session_id)
    return {"final_response": {"role": "assistant", "content": content}}


# ---------------------------------------------------------------------------
# Node 4: build_final_response
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    return {}
