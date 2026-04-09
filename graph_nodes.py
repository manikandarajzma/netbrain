"""
Atlas LangGraph nodes — troubleshooting flow only.
"""
import json
import logging
import re
from typing import Any

try:
    from atlas.graph_state import AtlasState
    from atlas.chat_service import _IP_OR_CIDR_RE
except ImportError:
    from graph_state import AtlasState  # type: ignore[assignment]
    from chat_service import _IP_OR_CIDR_RE  # type: ignore[assignment]

logger = logging.getLogger("atlas.graph_nodes")


# ---------------------------------------------------------------------------
# Pending clarification state (Redis-backed, 10-min TTL)
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
# Context checker — decides if a troubleshoot query has enough info
# ---------------------------------------------------------------------------

_ISSUE_WORDS = re.compile(
    r"\b(troubleshoot|help|why|cannot|can.?t|not|unable|fail|issue|problem|error|"
    r"block|drop|deny|denied|slow|latency|down|unreachable|intermittent|flap|"
    r"investigate|debug|diagnose|check|what.?s wrong|broken|outage|incident)\b",
    re.IGNORECASE,
)
_PORT_WORDS = re.compile(
    r"\b(port\s*\d+|tcp|udp|icmp|https?|ssh|dns|smtp|snmp|bgp|ospf|any)\b",
    re.IGNORECASE,
)
_ISSUE_TYPE_MAP = [
    (re.compile(r"\b(block|drop|deny|denied|reject|filter)\b", re.IGNORECASE), "blocked"),
    (re.compile(r"\b(slow|latency|lag|delay|degraded|performance)\b", re.IGNORECASE), "slow"),
    (re.compile(r"\b(intermittent|flap|unstable|sporadic|random)\b", re.IGNORECASE), "intermittent"),
    (re.compile(r"\b(device|router|switch|firewall|fw|pa-|arista|cisco)\b", re.IGNORECASE), "device"),
    (re.compile(r"\b(path.?change|route.?change|reroute|asymmetric)\b", re.IGNORECASE), "path_changed"),
]


def _regex_check_ts_context(prompt: str) -> tuple[bool, bool, str]:
    """Fast regex-based context check. Returns (has_issue_type, has_port, issue_type)."""
    has_issue = bool(_ISSUE_WORDS.search(prompt))
    has_port = bool(_PORT_WORDS.search(prompt))
    issue_type = "general"
    for pattern, itype in _ISSUE_TYPE_MAP:
        if pattern.search(prompt):
            issue_type = itype
            break
    return has_issue, has_port, issue_type


async def _llm_check_ts_context(prompt: str) -> tuple[bool, bool, str]:
    """Return (has_issue_type, has_port, issue_type).

    Uses regex first — only calls the LLM when the regex result is ambiguous
    (no clear issue words found in a short prompt with no IPs).
    """
    has_issue, has_port, issue_type = _regex_check_ts_context(prompt)

    # Regex was confident — skip the LLM call entirely
    if has_issue or has_port or len(prompt.split()) > 6:
        return has_issue, has_port, issue_type

    # Only fall back to LLM for very short ambiguous prompts (< 6 words, no
    # recognisable issue or port keywords) where we genuinely can't tell.
    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    _TS_CONTEXT_PROMPT = (
        'Analyse this network query. Reply with ONLY JSON, no explanation:\n'
        '{"has_issue_type": <true|false>, "has_port": <true|false>, "issue_type": "<blocked|slow|intermittent|device|path_changed|general>"}\n'
        '- has_issue_type: true if ANY problem is described\n'
        '- has_port: true if a port, protocol, or service is mentioned'
    )

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL,
                     temperature=0.0, api_key="docker", max_tokens=50)
    try:
        resp = await llm.ainvoke([SystemMessage(content=_TS_CONTEXT_PROMPT),
                                  HumanMessage(content=prompt)])
        data = json.loads(resp.content.strip())
        itype = data.get("issue_type", "general")
        valid = {"blocked", "slow", "intermittent", "device", "path_changed", "general"}
        return bool(data.get("has_issue_type")), bool(data.get("has_port")), (itype if itype in valid else "general")
    except Exception as exc:
        logger.warning("_llm_check_ts_context failed: %s", exc)
        return True, True, "general"


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

_DISMISSALS = {"no", "nope", "nah", "no thanks", "never mind", "nevermind", "skip",
               "not now", "no need", "i'm good", "im good", "all good"}
_ACKNOWLEDGEMENTS = {"yes", "yeah", "sure", "ok", "okay", "great", "thanks",
                     "thank you", "cool", "got it", "noted", "perfect", "sounds good"}


async def classify_intent(state: AtlasState) -> dict[str, Any]:
    prompt = state["prompt"]
    session_id = state.get("session_id") or "default"

    try:
        import atlas.status_bus as sb
        await sb.push(session_id, "Analyzing your query...")
    except Exception:
        pass

    prompt_lower = prompt.lower().strip().rstrip("!.")

    # Acknowledgement with nothing pending → dismiss
    if prompt_lower in _ACKNOWLEDGEMENTS | _DISMISSALS:
        if not _pending_ts_exists(session_id):
            return {"intent": "dismiss",
                    "final_response": {"role": "assistant",
                                       "content": "Sure, let me know if you need anything else."}}

    # Pending clarification reply (short, no IPs) → continue troubleshoot
    if _pending_ts_exists(session_id):
        if not _IP_OR_CIDR_RE.search(prompt) and len(prompt.split()) <= 15:
            return {"intent": "troubleshoot"}
        else:
            _pending_ts_delete(session_id)

    return {"intent": "troubleshoot"}


# ---------------------------------------------------------------------------
# Node 2: troubleshoot_orchestrator
# ---------------------------------------------------------------------------

async def troubleshoot_orchestrator(state: AtlasState) -> dict[str, Any]:
    try:
        from atlas.agents.orchestrator import orchestrate_troubleshoot
    except ImportError:
        from agents.orchestrator import orchestrate_troubleshoot

    session_id = state.get("session_id") or "default"
    prompt = state["prompt"]

    try:
        import atlas.status_bus as sb
        await sb.push(session_id, "Investigating...")
    except Exception:
        pass

    # Recover pending clarification context
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
                pending = last_user_before
                pending_issue_type = "general"

    issue_type = pending_issue_type or "general"

    if pending:
        full_prompt = f"{pending}\n\nUser clarification: {prompt}"
    else:
        ip_matches = _IP_OR_CIDR_RE.findall(prompt)
        is_connectivity = len(ip_matches) >= 2

        # Minimal context check — only ask if very vague
        words = prompt.lower().split()
        has_device_context = any(c.isdigit() or "-" in w for w in words for c in w)
        if not has_device_context and len(ip_matches) == 0 and len(prompt.split()) < 4:
            return {"final_response": {"role": "assistant", "content": (
                "Please describe the problem — include device names, IP addresses, or what is failing.\n"
                "Example: \"Why can't 10.0.0.1 connect to 11.0.0.1?\" or \"arista1 is unreachable\""
            )}}

        _PATH_TRACE_RE = re.compile(
            r"\b(trace\s+path|show\s+(me\s+)?the\s+route|find\s+(the\s+)?path|traceroute)\b",
            re.IGNORECASE,
        )

        if _PATH_TRACE_RE.search(prompt):
            issue_type = "general"
        else:
            issue_clear, port_clear, issue_type = await _llm_check_ts_context(prompt)

            if not issue_clear or (is_connectivity and not port_clear):
                parts = []
                if not issue_clear:
                    parts.append(
                        "**What type of issue are you seeing?**\n"
                        "- Blocked / denied — traffic is being dropped\n"
                        "- Slow / high latency — performance degraded\n"
                        "- Intermittent — drops in and out unpredictably\n"
                        "- Device issue — specific device misbehaving\n"
                        "- Path changed — routing different from expected"
                    )
                if is_connectivity and not port_clear:
                    parts.append(
                        "**Which port or protocol is affected?**\n"
                        "e.g. TCP 443, UDP 53 — or 'any' if protocol-agnostic"
                    )
                _pending_ts_set(session_id, prompt, issue_type)
                return {"final_response": {"role": "assistant", "content": "\n\n".join(parts)}}

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
        full_tb = _tb.format_exc()
        logger.error("Troubleshoot orchestrator failed: %s\nFULL TRACEBACK:\n%s", exc, full_tb)
        result = {"role": "assistant", "content": f"Troubleshooting failed: {exc}"}

    return {"final_response": result}


# ---------------------------------------------------------------------------
# Node 3: build_final_response
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    return {}
