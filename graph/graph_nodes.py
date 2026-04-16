"""
Atlas LangGraph nodes.

Graph shape:
  classify_intent
      ├─► call_troubleshoot_agent   (connectivity / device-health investigation)
      ├─► call_network_ops_agent    (incident, change, and operational request workflows)
      └─► build_final_response      (dismiss / early-exit)
"""
import logging
import re
from typing import Any

try:
    from atlas.graph.graph_state import AtlasState
    from atlas.application.chat_service import _IP_OR_CIDR_RE
    from atlas.services.memory_manager import memory_manager
    from atlas.services.metrics import metrics_recorder
    from atlas.services.network_ops_workflow_service import network_ops_workflow_service
    from atlas.services.observability import log_event
    from atlas.services.request_preprocessor import (
        extract_ips,
    )
    from atlas.services.status_service import status_service
    from atlas.services.troubleshoot_workflow_service import troubleshoot_workflow_service
except ImportError:
    from graph.graph_state import AtlasState        # type: ignore[assignment]
    from application.chat_service import _IP_OR_CIDR_RE  # type: ignore[assignment]
    from services.memory_manager import memory_manager  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.network_ops_workflow_service import network_ops_workflow_service  # type: ignore
    from services.observability import log_event  # type: ignore
    from services.request_preprocessor import (  # type: ignore
        extract_ips,
    )
    from services.status_service import status_service  # type: ignore
    from services.troubleshoot_workflow_service import troubleshoot_workflow_service  # type: ignore

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
    # Firewall / security policy work
    r"firewall\s+policy|fw\s+policy|security\s+policy"
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
    request_id = state.get("request_id")

    await status_service.push(session_id, "Analyzing your query...")

    prompt_lower = prompt.lower().strip().rstrip("!.")

    # Plain acknowledgement / dismissal with nothing pending → dismiss
    if prompt_lower in (_ACKNOWLEDGEMENTS | _DISMISSALS):
        if not memory_manager.has_pending_context(session_id):
            log_event(
                logger,
                "intent_classified",
                request_id=request_id,
                session_id=session_id,
                intent="dismiss",
                reason="acknowledgement_without_pending_context",
            )
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
            log_event(
                logger,
                "intent_classified",
                request_id=request_id,
                session_id=session_id,
                intent="network_ops",
                reason="pending_network_ops_context",
            )
            return {"intent": "network_ops"}
        if not _IP_OR_CIDR_RE.search(prompt) and len(prompt.split()) <= 15:
            intent = "network_ops" if pending_issue_type == "network_ops" else "troubleshoot"
            log_event(
                logger,
                "intent_classified",
                request_id=request_id,
                session_id=session_id,
                intent=intent,
                reason="pending_short_follow_up",
            )
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
            metrics_recorder.increment("atlas.intent.classified", intent="troubleshoot")
            log_event(
                logger,
                "intent_classified",
                request_id=request_id,
                session_id=session_id,
                intent="troubleshoot",
                reason="diagnostic_framing_overrode_network_ops",
            )
            return {"intent": "troubleshoot"}
        metrics_recorder.increment("atlas.intent.classified", intent="network_ops")
        log_event(
            logger,
            "intent_classified",
            request_id=request_id,
            session_id=session_id,
            intent="network_ops",
            reason="network_ops_regex_match",
        )
        return {"intent": "network_ops"}

    if _TROUBLESHOOT_RE.search(prompt):
        metrics_recorder.increment("atlas.intent.classified", intent="troubleshoot")
        log_event(
            logger,
            "intent_classified",
            request_id=request_id,
            session_id=session_id,
            intent="troubleshoot",
            reason="troubleshoot_regex_match",
        )
        return {"intent": "troubleshoot"}

    metrics_recorder.increment("atlas.intent.classified", intent="dismiss")
    log_event(
        logger,
        "intent_classified",
        request_id=request_id,
        session_id=session_id,
        intent="dismiss",
        reason="unsupported_prompt",
    )
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
    """Delegate troubleshoot orchestration to the owned workflow service."""
    return await troubleshoot_workflow_service.run(state)


# ---------------------------------------------------------------------------
# Node 3: call_network_ops_agent
# ---------------------------------------------------------------------------

async def call_network_ops_agent(state: AtlasState) -> dict[str, Any]:
    """Delegate network-ops orchestration to the owned workflow service."""
    return await network_ops_workflow_service.run(state)


# ---------------------------------------------------------------------------
# Node 4: build_final_response
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    return {}
