"""
Atlas LangGraph nodes.

Graph shape:
  classify_intent → call_troubleshoot_agent → build_final_response
"""
import json
import logging
import re
from typing import Any

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
# Matched BEFORE the troubleshooting keywords so explicit ops requests don't get
# misclassified as "why is X broken" investigations.
_NETWORK_OPS_RE = re.compile(
    r"\b(firewall\s+request|change\s+request|spreadsheet|policy\s+review|"
    r"open\s+port|allow\s+traffic|create\s+(a\s+)?rule|new\s+rule|"
    r"request\s+access|access\s+request|fw\s+request|security\s+review)\b",
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

    # Network-ops workflow detection (checked before generic troubleshooting keywords)
    if _NETWORK_OPS_RE.search(prompt):
        return {"intent": "network_ops"}

    # Everything else is a troubleshooting query
    return {"intent": "troubleshoot"}


# ---------------------------------------------------------------------------
# Node 2: call_troubleshoot_agent
# ---------------------------------------------------------------------------

async def call_troubleshoot_agent(state: AtlasState) -> dict[str, Any]:
    """Invoke the troubleshoot ReAct agent and return its structured result."""
    try:
        from atlas.agents.troubleshoot_agent import orchestrate_troubleshoot
    except ImportError:
        from agents.troubleshoot_agent import orchestrate_troubleshoot  # type: ignore

    session_id = state.get("session_id") or "default"
    prompt     = state["prompt"]

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
                pending           = last_user_before
                pending_issue_type = "general"

    issue_type = pending_issue_type or "general"

    if pending:
        full_prompt = f"{pending}\n\nUser clarification: {prompt}"
    else:
        ip_matches = _IP_OR_CIDR_RE.findall(prompt)

        # Require at minimum some device/IP context for very short prompts
        words = prompt.lower().split()
        has_device_context = any(c.isdigit() or "-" in w for w in words for c in w)
        if not has_device_context and len(ip_matches) == 0 and len(prompt.split()) < 4:
            return {"final_response": {"role": "assistant", "content": (
                "Please describe the problem — include device names, IP addresses, or what is failing.\n"
                'Example: "Why can\'t 10.0.0.1 connect to 11.0.0.1?" or "arista1 is unreachable"'
            )}}

        full_prompt = prompt

    try:
        result = await orchestrate_troubleshoot(
            full_prompt,
            username=state.get("username"),
            session_id=session_id,
            issue_type=issue_type,
        )
    except Exception as exc:
        import traceback as _tb
        logger.error("Troubleshoot agent failed: %s\n%s", exc, _tb.format_exc())
        result = {"role": "assistant", "content": f"Troubleshooting failed: {exc}"}

    return {"final_response": result}


# ---------------------------------------------------------------------------
# Node 3: call_network_ops_agent
# ---------------------------------------------------------------------------

async def call_network_ops_agent(state: AtlasState) -> dict[str, Any]:
    """
    Route network-ops workflows (firewall change requests, spreadsheet generation,
    policy reviews, etc.) to the dedicated network_ops_agent.

    This agent shares ALL_TOOLS with troubleshoot_agent but runs a different
    system prompt focused on structured output and document generation rather
    than step-by-step diagnostic investigation.
    """
    try:
        from atlas.agents.network_ops_agent import handle as network_ops_handle
    except ImportError:
        from agents.network_ops_agent import handle as network_ops_handle  # type: ignore

    session_id = state.get("session_id") or "default"
    prompt     = state["prompt"]

    try:
        result = await network_ops_handle(
            prompt,
            username=state.get("username"),
            session_id=session_id,
        )
    except Exception as exc:
        import traceback as _tb
        logger.error("Network ops agent failed: %s\n%s", exc, _tb.format_exc())
        result = {"role": "assistant", "content": f"Network ops agent failed: {exc}"}

    return {"final_response": result}


# ---------------------------------------------------------------------------
# Node 4: build_final_response
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    return {}
