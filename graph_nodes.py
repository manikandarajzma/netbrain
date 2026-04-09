"""
Atlas LangGraph nodes.
All logic is extracted from chat_service.py — no new behaviour introduced.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any

from atlas.graph_state import AtlasState
from atlas.chat_service import (
    _is_doc_query,
    _build_llm_messages,
    _to_openai_tool,
    _fetch_mcp_tools,
    _normalize_result,
    _check_tool_access,
    _TOOL_TIMEOUTS,
    TOOL_DISPLAY_NAMES,
    get_tool_display_name,
    MAX_AGENT_ITERATIONS,
    _IP_OR_CIDR_RE,
)

logger = logging.getLogger("atlas.graph_nodes")


# ---------------------------------------------------------------------------
# ServiceNow response cache (keyed on prompt, skips port-8005 round-trip)
# ---------------------------------------------------------------------------

_SNOW_RESPONSE_TTL = 120  # seconds

_SNOW_WRITE_RE = re.compile(
    r'\b(create|update|close|resolve|assign|add note|work note|open|submit|delete)\b',
    re.IGNORECASE,
)


def _snow_resp_key(prompt: str) -> str:
    digest = hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:20]
    return f"atlas:snow:resp:{digest}"


def _snow_resp_get(prompt: str) -> str | None:
    if _SNOW_WRITE_RE.search(prompt):
        return None
    try:
        import redis as _r
        return _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                           decode_responses=True).get(_snow_resp_key(prompt))
    except Exception:
        return None


def _snow_resp_set(prompt: str, text: str) -> None:
    if _SNOW_WRITE_RE.search(prompt):
        return
    try:
        import redis as _r
        _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"),
                    decode_responses=True).setex(_snow_resp_key(prompt), _SNOW_RESPONSE_TTL, text)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

_CONFIRMATIONS = {"yes", "yeah", "sure", "yep", "ok", "okay", "please", "go ahead", "do it", "show me", "yep please"}
_DISMISSALS = {"no", "nope", "nah", "no thanks", "don't", "dont", "never mind", "nevermind", "skip", "not now", "no need", "i'm good", "im good", "all good", "that's fine", "thats fine"}


_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a network security tool. Classify the message into exactly one category.

CATEGORIES:

network — Looking up firewall objects: what address group an IP belongs to, group members, policies referencing a group, or details about a specific policy. The user wants to know WHAT something IS, not fix a problem.
  "what group is 10.0.0.5 in?"  →  network
  "what address group is 11.0.0.1 part of?"  →  network
  "show me the members of leander_web"  →  network
  "what policies reference this group? also give me the members"  →  network
  "give me the group members and the policies that reference it"  →  network
  "what is the action for policy Allow HTTPS from DMZ to Internal"  →  network
  "show me the policy details for X"  →  network
  "what does policy X do?"  →  network

troubleshoot — Any network or infrastructure PROBLEM the user wants diagnosed or investigated. Covers connectivity issues, device outages, performance degradation, interface errors, policy violations, security events, and change-related problems. The user wants to know WHY something is broken or behaving unexpectedly.
  "why can't 10.0.0.1 connect to 11.0.0.1?"  →  troubleshoot
  "traffic from 10.0.0.1 to 11.0.0.1 is being blocked"  →  troubleshoot
  "is traffic from A to B allowed on port 443?"  →  troubleshoot
  "PA-FW-01 is dropping packets"  →  troubleshoot
  "why is CORE-SW-01 having high CPU?"  →  troubleshoot
  "users in building A can't reach the internet"  →  troubleshoot
  "troubleshoot connectivity between 10.0.0.1 and 11.0.0.1"  →  troubleshoot
  "why is there high latency between 10.0.0.1 and 11.0.0.1?"  →  troubleshoot
  "what's going on with arista1?"  →  troubleshoot
  "is arista1 healthy?"  →  troubleshoot
  "what's happening with CORE-SW-01?"  →  troubleshoot
  "check interface errors on arista1"  →  troubleshoot
  "help me troubleshoot INC0010035"  →  troubleshoot
  "troubleshoot INC0010001"  →  troubleshoot
  "investigate INC0010005"  →  troubleshoot

risk — Security posture or risk assessment for a single IP or host.
  "what is the risk of 10.0.0.5?"  →  risk
  "is this IP risky?"  →  risk

netbrain — Tracing or visualising the network path between two IPs, using NetBrain specifically. Only use this when the user does NOT mention Nornir, the database, or "without NetBrain".
  "trace path from A to B"  →  netbrain
  "show me the route from X to Y"  →  netbrain
  "trace path from A to B without NetBrain"  →  troubleshoot
  "trace path using Nornir"  →  troubleshoot
  "trace path using the database"  →  troubleshoot

servicenow — Anything about ITSM tickets, incidents, change requests, problems, CMDB CIs, or ServiceNow users. Use this when the user references a ticket number (INC..., CHG..., PRB...), asks about tickets or incidents, or wants to create/update/search records in ServiceNow.
  "show me open incidents"  →  servicenow
  "any historical ticket related to PA-FW-01"  →  servicenow
  "any issues related to PA-FW-01"  →  servicenow
  "give me details about INC0010005"  →  servicenow
  "give me details about NC0010005"  →  servicenow
  "details for INC0010001"  →  servicenow
  "create an incident for network outage"  →  servicenow
  "what change requests are scheduled this week?"  →  servicenow
  "find the CI for 10.0.0.1"  →  servicenow
  "is there an open ticket for this device?"  →  servicenow
  "look up user john.smith"  →  servicenow
  "search knowledge base for VPN setup"  →  servicenow
  "any known issues for this firewall?"  →  servicenow

doc — A how-to or conceptual question about firewall/network/security tools or processes.
  "how do I request firewall access?"  →  doc
  "what is a DMZ?"  →  doc

dismiss — Off-topic, greeting, or unrelated to firewalls and network security.
  "what's the weather?"  →  dismiss
  "hello"  →  dismiss
  "what?"  →  dismiss

Reply with ONLY the category name. No explanation. No punctuation.\
"""


async def _llm_classify_intent(prompt: str, session_id: str | None = None) -> str:
    """Call the local LLM to classify the intent of a user prompt. Result is cached in Redis."""
    import time as _time
    cache_key = f"atlas:intent:{hashlib.sha256(prompt.strip().lower().encode()).hexdigest()[:20]}"
    try:
        import redis as _r
        _rc = _r.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        cached = _rc.get(cache_key)
        if cached:
            logger.info("classify_intent: cache hit %r -> %r", prompt[:60], cached)
            return cached
    except Exception:
        pass

    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
        max_tokens=20,
    )
    try:
        _t0 = _time.perf_counter()
        response = await llm.ainvoke([
            SystemMessage(content=_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        _elapsed = _time.perf_counter() - _t0
        logger.info("_llm_classify_intent: LLM took %.2fs for %r", _elapsed, prompt[:60])
        raw = (response.content or "").strip().lower().split()[0].rstrip(".,;:")
        valid = {"troubleshoot", "risk", "netbrain", "doc", "network", "dismiss", "servicenow"}
        result = raw if raw in valid else "doc"
        if raw not in valid:
            logger.warning("_llm_classify_intent: unexpected label %r for %r — falling back to doc", raw, prompt[:80])
        try:
            _rc.setex(cache_key, 3600, result)  # cache intent for 1 hour
        except Exception:
            pass
        return result
    except Exception as exc:
        logger.warning("_llm_classify_intent failed: %s — falling back to doc", exc)
        return "doc"


def _last_assistant_content(history: list) -> str:
    """Return the last assistant message content as a string."""
    for msg in reversed(history):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, dict):
            content = content.get("direct_answer", "")
        return str(content)
    return ""


_CREATE_FORM_PHRASES = (
    "need a few details to create",
    "almost there",
    "still need the following",
    "just need a few more",
)


_CHG_USER_RE = re.compile(
    r'\b(create|open|raise|submit|log)\b.{0,40}\b(change request|change req|change|chg|cr)\b',
    re.IGNORECASE,
)
_INC_USER_RE = re.compile(
    r'\b(create|open|raise|submit|log)\b.{0,40}\b(incident|inc)\b',
    re.IGNORECASE,
)


def _resolve_active_flow(active_flow: str | None, history: list) -> str | None:
    """Return the current create-flow type.

    History (from DB) is the primary source of truth.  The LangGraph checkpoint
    value is NOT trusted because it may have been written by a previous buggy run.
    """
    # 1. Is a form still active? The last assistant message must contain a form phrase.
    last = _last_assistant_content(history)
    if not any(p in last.lower() for p in _CREATE_FORM_PHRASES):
        return None

    # 2. Scan history for the clearest signal of CHG vs INC.
    #    Check user messages (explicit create intent) AND assistant form messages.
    flow_type: str | None = None
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if isinstance(content, dict):
            content = content.get("direct_answer", "")
        low = str(content).lower()

        if role == "user":
            if _CHG_USER_RE.search(low):
                flow_type = "create_change_request"
            elif _INC_USER_RE.search(low):
                flow_type = "create_incident"
        elif role == "assistant" and any(p in low for p in _CREATE_FORM_PHRASES):
            if "change request" in low or "change req" in low or "change_request" in low:
                flow_type = "create_change_request"
            elif "incident" in low and flow_type is None:
                flow_type = "create_incident"

    if flow_type is not None:
        return flow_type

    # 3. Nothing found in history — fall back to checkpoint (last resort)
    if active_flow in ("create_change_request", "create_incident"):
        return active_flow

    return None


async def classify_intent(state: AtlasState) -> dict[str, Any]:
    import time as _time
    _sid = state.get("session_id") or "default"
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_sid, "Classifying your query...")
    except Exception:
        pass
    prompt = state["prompt"]
    prefilled = state.get("prefilled_tool_name")
    history = state.get("conversation_history") or []

    if prefilled and state.get("prefilled_tool_params") is not None:
        messages = _build_llm_messages(prompt, history)
        return {"intent": "prefilled", "messages": messages, "iteration": 0}

    # active_flow is persisted via Redis checkpointer; _resolve_active_flow falls back to
    # history scan when the checkpoint has no state (e.g. after a server restart).
    active_flow = _resolve_active_flow(state.get("active_flow"), history)
    _SNOW_RECORD_RE = re.compile(r'\b(INC|CHG|PRB)\d+\b')
    _last = _last_assistant_content(history)

    # Determine if this is a clear servicenow context without needing the LLM
    _PATH_TRACE_RE = re.compile(r'\b(\d{1,3}\.){3}\d{1,3}\b.*\b(\d{1,3}\.){3}\d{1,3}\b|trace\s+path|path\s+trace', re.IGNORECASE)
    _DEVICE_HEALTH_RE = re.compile(r"what'?s?\s+(going\s+on|happening)|is\s+\S+\s+healthy|check\s+(interface|error|counter)|what'?s?\s+the\s+status\s+of", re.IGNORECASE)
    _TROUBLESHOOT_RE = re.compile(r'\btroubleshoot\b', re.IGNORECASE)
    _forced_servicenow = (
        active_flow in ("create_change_request", "create_incident")
        or (_SNOW_RECORD_RE.search(_last) and len(prompt.split()) <= 5
            and not _PATH_TRACE_RE.search(prompt)
            and not _DEVICE_HEALTH_RE.search(prompt)
            and not _TROUBLESHOOT_RE.search(prompt))
    )

    # discover_only: return display label — never invoke real agents.
    # We resolve the effective intent here too (from context or LLM) so the label is correct.
    if state.get("discover_only"):
        if _forced_servicenow:
            effective_intent = "servicenow"
        elif _TROUBLESHOOT_RE.search(prompt):
            effective_intent = "troubleshoot"
        elif _pending_ts_exists(state.get("session_id") or "default"):
            # Clarification answer in progress — label as troubleshoot, not whatever the LLM thinks
            effective_intent = "troubleshoot"
        else:
            effective_intent = await _llm_classify_intent(prompt)
        display_map = {
            "troubleshoot": "Orchestrator",
            "netbrain": "NetBrain",
            "network": "Panorama",
            "servicenow": "ServiceNow",
            "risk": "Risk Assessment",
            "doc": "Documentation",
        }
        return {
            "intent": "dismiss",
            "final_response": {
                "tool_display_name": display_map.get(effective_intent, "ServiceNow"),
                "intent": effective_intent,
            },
        }

    if _forced_servicenow:
        logger.info(
            "classify_intent: forced servicenow (active_flow=%r, last_has_record=%s)",
            active_flow, bool(_SNOW_RECORD_RE.search(_last)),
        )
        return {"intent": "servicenow", "messages": [], "iteration": 0}

    # Hard override: "troubleshoot" keyword always routes to orchestrator regardless of LLM
    if _TROUBLESHOOT_RE.search(prompt):
        logger.info("classify_intent: hard override → troubleshoot (keyword match)")
        return {"intent": "troubleshoot", "messages": [], "iteration": 0}

    conversation_history = state.get("conversation_history") or []
    prompt_lower_strip = prompt.lower().strip().rstrip("!.")

    # last_follow_up_action is extracted once from history before graph entry (chat_service.py)
    follow_up_action = state.get("last_follow_up_action")
    follow_up_text = None
    if follow_up_action:
        # Find the matching follow_up text for display context
        for msg in reversed(conversation_history):
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str) and content.startswith("{"):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, ValueError):
                    pass
            if isinstance(content, dict):
                if content.get("follow_up_action") == follow_up_action:
                    follow_up_text = content.get("follow_up")
                    break
                for r in (content.get("multi_results") or []):
                    if isinstance(r, dict) and r.get("follow_up_action") == follow_up_action:
                        follow_up_text = r.get("follow_up")
                        break
            if follow_up_text is not None:
                break

    if follow_up_action and follow_up_action.get("tool"):
        user_lower = prompt.lower().strip().rstrip("!.")
        logger.info("classify_intent: follow_up_action present tool=%s, user_lower=%r", follow_up_action.get("tool"), user_lower)
        # Pure dismissal → acknowledge and stop
        if user_lower in _DISMISSALS:
            return {"intent": "dismiss", "final_response": {"role": "assistant", "content": "Sure, let me know if you need anything else."}, "messages": [], "iteration": 0}
        # Pure confirmation → execute follow-up directly
        if user_lower in _CONFIRMATIONS:
            messages = _build_llm_messages(prompt, conversation_history)
            logger.info("classify_intent: routing confirmation to prefilled tool=%s", follow_up_action["tool"])
            return {
                "intent": "prefilled",
                "prefilled_tool_name": follow_up_action["tool"],
                "prefilled_tool_params": follow_up_action.get("params", {}),
                "messages": messages,
                "iteration": 0,
            }
        # User asks for something adjacent to the offer, e.g. "no. find the policies though"
        # Only applies to short follow-up replies — not new queries with IPs or many words.
        user_wants_policies = any(w in user_lower for w in ("policies", "policy", "rules", "security rules", "firewall rules"))
        user_wants_members = any(w in user_lower for w in ("members", "contents", "what's in", "whats in"))
        is_short_followup = len(user_lower.split()) <= 10 and not _IP_OR_CIDR_RE.search(user_lower)

        if is_short_followup and (user_wants_policies or user_wants_members):
            fa_tool = follow_up_action.get("tool", "")
            fa_params = dict(follow_up_action.get("params", {}))

            # If the follow_up_action targets query_panorama_address_group_members we can
            # inject _policies_only when the user is asking for policies instead of members.
            if fa_tool == "query_panorama_address_group_members":
                # Always set explicitly — never inherit stale _policies_only from previous context
                fa_params["_policies_only"] = user_wants_policies
                messages = _build_llm_messages(prompt, conversation_history)
                return {
                    "intent": "prefilled",
                    "prefilled_tool_name": fa_tool,
                    "prefilled_tool_params": fa_params,
                    "messages": messages,
                    "iteration": 0,
                }

            # Generic: follow_up offered something and user is referencing it by concept
            if follow_up_text:
                fu_lower = follow_up_text.lower()
                offered_policies = "policies" in fu_lower or "policy" in fu_lower
                offered_members = "members" in fu_lower
                if (user_wants_policies and offered_policies) or (user_wants_members and offered_members):
                    messages = _build_llm_messages(prompt, conversation_history)
                    return {
                        "intent": "prefilled",
                        "prefilled_tool_name": fa_tool,
                        "prefilled_tool_params": fa_params,
                        "messages": messages,
                        "iteration": 0,
                    }
        # Anything else → fall through to normal routing with full history

    # Short acknowledgement with no follow_up pending → dismiss rather than re-run a tool
    _ACKNOWLEDGEMENTS = _CONFIRMATIONS | _DISMISSALS | {"ok", "okay", "great", "thanks", "thank you", "cool", "got it", "noted", "perfect", "sounds good"}
    if prompt.lower().strip().rstrip("!.") in _ACKNOWLEDGEMENTS and not state.get("discover_only"):
        return {"intent": "dismiss", "final_response": {"role": "assistant", "content": "Sure, let me know if you need anything else."}, "messages": [], "iteration": 0}

    # Pending troubleshoot clarification — route back if this looks like an answer (short, no IPs).
    # If the new message has IPs it's a fresh query — discard the stale pending so clarification
    # is re-evaluated cleanly instead of consuming the old pending as if it were an answer.
    if not state.get("discover_only"):
        session_id = state.get("session_id") or "default"
        has_pending_ts = _pending_ts_exists(session_id)
        if has_pending_ts:
            if not _IP_OR_CIDR_RE.search(prompt) and len(prompt.split()) <= 15:
                # Short, no-IP reply → treat as clarification answer
                return {"intent": "troubleshoot", "messages": [], "iteration": 0}
            else:
                # New query with IPs → discard stale pending, evaluate fresh
                _pending_ts_delete(session_id)
                logger.info("classify_intent: discarding stale pending_ts for session %s (new IP query)", session_id)

    # Deterministic override: "without NetBrain" / Nornir keywords must always route to
    # troubleshoot — skip the LLM and the Redis cache to avoid stale netbrain classifications.
    _NO_NETBRAIN_RE = re.compile(
        r"without netbrain|no netbrain|use nornir|use the database|use netbox|don.t use netbrain",
        re.IGNORECASE,
    )
    if _NO_NETBRAIN_RE.search(prompt):
        logger.info("classify_intent: deterministic override → troubleshoot (no-NetBrain pattern)")
        return {"intent": "troubleshoot", "messages": [], "iteration": 0}

    # LLM-based intent classification (result cached in Redis to avoid repeat LLM calls)
    _t_classify = _time.perf_counter()
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_sid, "Selecting agent...")
    except Exception:
        pass
    intent = await _llm_classify_intent(prompt, session_id=_sid)
    _classify_ms = (_time.perf_counter() - _t_classify) * 1000
    logger.info("classify_intent: LLM classified %r as %r in %.0fms", prompt[:80], intent, _classify_ms)
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_sid, f"Agent selected: {intent} ({_classify_ms:.0f}ms)")
    except Exception:
        pass

    if intent == "dismiss":
        return {"intent": "dismiss", "final_response": {"role": "assistant", "content": "I am not equipped to answer this question. If you feel its a mistake, reach out to the atlas team."}, "messages": [], "iteration": 0}

    if intent in ("troubleshoot", "netbrain", "risk", "servicenow"):
        return {"intent": intent, "messages": [], "iteration": 0}

    messages = _build_llm_messages(prompt, state.get("conversation_history") or [])
    return {"intent": intent, "messages": messages, "iteration": 0}


# ---------------------------------------------------------------------------
# Node: risk_orchestrator — fans out to Panorama + Splunk A2A agents
# ---------------------------------------------------------------------------

async def risk_orchestrator(state: AtlasState) -> dict[str, Any]:
    """Fan out to Panorama and Splunk A2A agents, synthesize with Ollama."""
    from agents.orchestrator import orchestrate_ip_risk
    prompt = state["prompt"]
    username = state.get("username")
    session_id = state.get("session_id")
    result = await orchestrate_ip_risk(prompt, username=username, session_id=session_id)
    return {"final_response": result}


# ---------------------------------------------------------------------------
# Node: netbrain_agent — delegates path queries to the NetBrain A2A agent
# ---------------------------------------------------------------------------

async def netbrain_agent(state: AtlasState) -> dict[str, Any]:
    """Send the path query to the NetBrain A2A agent and return its response."""
    import uuid
    import httpx

    prompt = state["prompt"]
    netbrain_url = "http://localhost:8004"

    task = {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": prompt}],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(netbrain_url, json=task)
            response.raise_for_status()
            data = response.json()
            artifacts = data.get("artifacts", [])
            text = None
            if artifacts:
                text = next(
                    (p.get("text") for p in artifacts[0].get("parts", []) if p.get("type") == "text"),
                    None,
                )
            if text:
                return {"final_response": {"role": "assistant", "content": {"direct_answer": text}}}
            return {"final_response": {"role": "assistant", "content": "NetBrain agent returned no data."}}
    except Exception as e:
        logger.warning("NetBrain agent call failed: %s", e)
        return {"final_response": {"role": "assistant", "content": f"NetBrain agent unavailable: {e}"}}


# ---------------------------------------------------------------------------
# Node: servicenow_agent — ServiceNow ITSM queries
# ---------------------------------------------------------------------------

_SNOW_RECORD_HEADING_RE = re.compile(r'^##\s+(INC|CHG|PRB)\d+', re.MULTILINE)
_SNOW_CREATED_RE = re.compile(r'\b(INC|CHG|PRB)\d{7,10}\b', re.IGNORECASE)


def _active_flow_from_response(text: str, current_flow: str | None) -> str | None:
    """Determine the new active_flow value from a ServiceNow agent response."""
    low = text.lower()
    # Create form was requested — set/keep the flow type
    if "need a few details to create the change request" in low or "need a few details to create the change" in low:
        return "create_change_request"
    if "need a few details to create the incident" in low:
        return "create_incident"
    # Still gathering details — preserve current flow
    if any(p in low for p in ("almost there", "still need the following", "just need a few more")):
        return current_flow
    # Record heading present → create/lookup completed; clear the flow
    if _SNOW_RECORD_HEADING_RE.search(text):
        return None
    # Record number returned without a create-form phrase → completed or plain lookup
    if _SNOW_CREATED_RE.search(text):
        return None
    # Default: clear flow (handles errors, plain lists, etc.)
    return None


async def servicenow_agent(state: AtlasState) -> dict[str, Any]:
    """Send the ServiceNow query to the ServiceNow A2A agent and return its response."""
    import uuid
    import httpx

    prompt = state["prompt"]
    servicenow_url = "http://localhost:8005"
    history = state.get("conversation_history") or []
    current_flow = _resolve_active_flow(state.get("active_flow"), history)

    # Short follow-up with no record number? Inject the last known record number so the
    # LLM knows which record to act on (e.g. "close it" after CHG0030022 was created).
    _SNOW_RECORD_RE2 = re.compile(r'\b(INC|CHG|PRB)\d+\b')
    _ACTION_RE = re.compile(r'\b(close|update|resolve|assign|add note|reopen|cancel|approve)\b', re.IGNORECASE)
    if not _SNOW_RECORD_RE2.search(prompt) and len(prompt.split()) <= 8 and _ACTION_RE.search(prompt):
        _last_content = _last_assistant_content(history)
        m = _SNOW_RECORD_RE2.search(_last_content)
        if m:
            prompt = f"{prompt} {m.group(0)}"
            logger.info("servicenow_agent: injected record number %s into prompt", m.group(0))

    # If active_flow is set, the user is replying to a create-details form.
    # Collect all user replies since the form was shown and send a combined prompt.
    if current_flow in ("create_change_request", "create_incident"):
        form_text = _last_assistant_content(history)
        # Gather user turns after the last assistant form prompt
        collecting = False
        user_replies = []
        _FORM_START = ("need a few details to create",)
        _FORM_CONT  = ("almost there", "still need", "please provide")
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, dict):
                content = content.get("direct_answer", "")
            low = str(content).lower()
            if role == "assistant":
                if any(p in low for p in _FORM_START):
                    collecting = True
                    user_replies = []  # reset only on the initial form prompt
                elif any(p in low for p in _FORM_CONT):
                    collecting = True   # keep collecting but do NOT reset user_replies
                continue
            if collecting and role == "user":
                user_replies.append(str(content))
        user_replies.append(prompt)
        combined = "\n".join(user_replies)
        prompt = f"[CREATE FORM] {current_flow}\n{form_text}\n\n[USER PROVIDED]\n{combined}"

    # Check response cache before doing anything — skips LLM + API call entirely
    cached = _snow_resp_get(prompt)
    if cached:
        logger.info("ServiceNow node: response cache hit for prompt=%r", prompt[:60])
        return {"active_flow": _active_flow_from_response(cached, current_flow),
                "final_response": {"role": "assistant", "content": {"direct_answer": cached}}}

    try:
        import atlas.status_bus as status_bus
        await status_bus.push(state.get("session_id") or "default", "Checking ServiceNow...")
    except Exception:
        pass

    task = {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": prompt}],
        },
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(servicenow_url, json=task)
            response.raise_for_status()
            data = response.json()
            artifacts = data.get("artifacts", [])
            text = next(
                (p.get("text") for p in artifacts[0].get("parts", []) if p.get("type") == "text"),
                None,
            ) if artifacts else None
            if text:
                _snow_resp_set(prompt, text)
                new_flow = _active_flow_from_response(text, current_flow)
                logger.info("ServiceNow node: active_flow %r -> %r", current_flow, new_flow)
                return {
                    "active_flow": new_flow,
                    "final_response": {"role": "assistant", "content": {"direct_answer": text}},
                }
            return {
                "active_flow": None,
                "final_response": {"role": "assistant", "content": "ServiceNow agent returned no data."},
            }
    except Exception as e:
        logger.warning("ServiceNow agent call failed: %s", e)
        return {
            "active_flow": None,
            "final_response": {"role": "assistant", "content": f"ServiceNow agent unavailable: {e}"},
        }


# ---------------------------------------------------------------------------
# Node: troubleshoot_orchestrator — multi-agent troubleshooting coordinator
# ---------------------------------------------------------------------------

_TS_CONTEXT_PROMPT = """\
You are analysing a network troubleshooting query. Determine what context is present.

Reply with ONLY a JSON object on one line, no explanation:
{"has_issue_type": <true|false>, "has_port": <true|false>, "issue_type": "<type>"}

- has_issue_type: true if the query describes ANY problem or investigation. Only false if the query is so vague it has no context (e.g. just two IPs with no verb).
- has_port: true if the query mentions a port, protocol, or service (TCP, UDP, port 443, HTTPS, SSH, DNS, "any", etc.). For device-based queries (no src/dst IPs), has_port is always true.
- issue_type: one of exactly: "blocked", "slow", "intermittent", "device", "path_changed", "general"
  - "blocked"      — traffic denied, dropped, filtered, can't connect, firewall blocking
  - "slow"         — high latency, slow, performance degraded, packet loss, high RTT
  - "intermittent" — flapping, drops in and out, sometimes works, unstable
  - "device"       — specific device problem: high CPU, memory, interface errors, device down
  - "path_changed" — routing changed, unexpected path, traffic going different route
  - "general"      — general connectivity troubleshoot or unknown

Examples:
"why can I not connect from 10.0.0.1 to 11.0.0.1" → {"has_issue_type": true, "has_port": false, "issue_type": "blocked"}
"traffic is blocked from 10.0.0.1 to 11.0.0.1 on TCP 443" → {"has_issue_type": true, "has_port": true, "issue_type": "blocked"}
"troubleshoot connectivity between 10.0.0.1 and 11.0.0.1" → {"has_issue_type": true, "has_port": false, "issue_type": "general"}
"latency is high from 10.0.0.1 to 11.0.0.1" → {"has_issue_type": true, "has_port": false, "issue_type": "slow"}
"connection drops intermittently" → {"has_issue_type": true, "has_port": false, "issue_type": "intermittent"}
"PA-FW-01 is dropping packets" → {"has_issue_type": true, "has_port": true, "issue_type": "device"}
"why is CORE-SW-01 having high CPU?" → {"has_issue_type": true, "has_port": true, "issue_type": "device"}
"traffic seems to be taking a different path" → {"has_issue_type": true, "has_port": false, "issue_type": "path_changed"}
"10.0.0.1 11.0.0.1" → {"has_issue_type": false, "has_port": false, "issue_type": "general"}\
"""

_VALID_ISSUE_TYPES = {"blocked", "slow", "intermittent", "device", "path_changed", "general"}


async def _llm_check_ts_context(prompt: str) -> tuple[bool, bool, str]:
    """Return (has_issue_type, has_port, issue_type) for a troubleshoot prompt."""
    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
        max_tokens=50,
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=_TS_CONTEXT_PROMPT),
            HumanMessage(content=prompt),
        ])
        import json as _json
        data = _json.loads(response.content.strip())
        issue_type = data.get("issue_type", "general")
        if issue_type not in _VALID_ISSUE_TYPES:
            issue_type = "general"
        return bool(data.get("has_issue_type")), bool(data.get("has_port")), issue_type
    except Exception as exc:
        logger.warning("_llm_check_ts_context failed: %s — assuming context missing", exc)
        return False, False, "general"


# Pending clarification prompts keyed by session_id.
# Stored in Redis (with 10-minute TTL) so they survive server reloads.
# Falls back to an in-memory dict if Redis is unavailable.
_pending_ts_mem: dict[str, str] = {}
_PENDING_TS_TTL = 600  # 10 minutes


def _pending_ts_set(session_id: str, prompt: str, issue_type: str = "general") -> None:
    import json as _json
    payload = _json.dumps({"prompt": prompt, "issue_type": issue_type})
    try:
        import os, redis as _redis
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.setex(f"atlas:pending_ts:{session_id}", _PENDING_TS_TTL, payload)
        return
    except Exception:
        pass
    _pending_ts_mem[session_id] = payload


def _pending_ts_get(session_id: str) -> tuple[str, str] | tuple[None, None]:
    """Return (prompt, issue_type) or (None, None) if not found."""
    import json as _json
    try:
        import os, redis as _redis
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        raw = r.get(f"atlas:pending_ts:{session_id}")
    except Exception:
        raw = _pending_ts_mem.get(session_id)
    if not raw:
        return None, None
    try:
        data = _json.loads(raw)
        if isinstance(data, dict):
            return data.get("prompt", ""), data.get("issue_type", "general")
        # Legacy: plain string stored before this change
        return str(data), "general"
    except Exception:
        return str(raw), "general"


def _pending_ts_exists(session_id: str) -> bool:
    try:
        import os, redis as _redis
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        return r.exists(f"atlas:pending_ts:{session_id}") > 0
    except Exception:
        pass
    return session_id in _pending_ts_mem


def _pending_ts_delete(session_id: str) -> None:
    try:
        import os, redis as _redis
        r = _redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.delete(f"atlas:pending_ts:{session_id}")
        return
    except Exception:
        pass
    _pending_ts_mem.pop(session_id, None)


async def troubleshoot_orchestrator(state: AtlasState) -> dict[str, Any]:
    """
    Multi-agent troubleshooting coordinator.

    If the prompt is missing port or issue-type context, returns a clarifying
    question and saves the original prompt so the next reply can continue.
    Uses application-layer state (not LangGraph interrupt) to stay simple.
    """
    try:
        from atlas.agents.orchestrator import orchestrate_troubleshoot
    except ImportError:
        from agents.orchestrator import orchestrate_troubleshoot

    session_id = state.get("session_id") or "default"
    prompt = state["prompt"]

    try:
        import atlas.status_bus as status_bus
        await status_bus.push(session_id, "Analyzing your query...")
    except Exception:
        pass

    # If a previous turn asked for clarification, combine with the original prompt.
    # Primary: Redis-backed store (survives reloads). Fallback: conversation history scan.
    pending, pending_issue_type = _pending_ts_get(session_id)
    if pending:
        _pending_ts_delete(session_id)
    if not pending:
        # Fallback: check if last assistant message was a clarification question and
        # the user's reply looks like a port/issue-type answer (short, no new IPs).
        history = state.get("conversation_history") or []
        if len(history) >= 2:
            last_assistant = next(
                (m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"),
                ""
            )
            last_user_before = next(
                (m.get("content", "") for m in reversed(history[:-1]) if m.get("role") == "user"),
                ""
            )
            is_clarification_reply = (
                ("which port" in last_assistant.lower() or "what type of issue" in last_assistant.lower())
                and len(_IP_OR_CIDR_RE.findall(prompt)) == 0
                and len(prompt.split()) <= 6
            )
            if is_clarification_reply and _IP_OR_CIDR_RE.findall(last_user_before):
                pending = last_user_before
                pending_issue_type = "general"
                logger.info("Recovered pending troubleshoot prompt from conversation history")

    issue_type = "general"

    if pending:
        full_prompt = f"{pending}\n\nUser clarification: {prompt}"
        issue_type = pending_issue_type or "general"
    else:
        # For connectivity queries (2 IPs), check if we have enough context.
        # For device-based queries (device name, single IP), proceed directly — no IP count gate.
        ip_matches_in_prompt = _IP_OR_CIDR_RE.findall(prompt)
        is_connectivity_query = len(ip_matches_in_prompt) >= 2
        if not is_connectivity_query and len(ip_matches_in_prompt) == 0:
            # No IPs and no obvious device name context — ask for more info
            words = prompt.lower().split()
            has_device_context = any(c.isdigit() or "-" in w for w in words for c in w)
            if not has_device_context and len(prompt.split()) < 4:
                return {"final_response": {"role": "assistant", "content": (
                    "Please describe the problem — include device names, IP addresses, or a description of what is failing.\n"
                    "Example: \"Why can't 10.0.0.1 connect to 11.0.0.1?\" or \"PA-FW-01 is dropping traffic\""
                )}}
        # Path trace requests are self-contained — skip clarification regardless of issue/port context.
        _PATH_TRACE_RE = re.compile(
            r"\b(trace\s+path|show\s+(me\s+)?the\s+route|find\s+(the\s+)?path|show\s+hops|traceroute)\b",
            re.IGNORECASE,
        )
        is_path_trace = bool(_PATH_TRACE_RE.search(prompt))

        if is_path_trace:
            issue_clear, port_clear, issue_type = True, True, "general"
        else:
            issue_clear, port_clear, issue_type = await _llm_check_ts_context(prompt)
        logger.info(
            "Troubleshoot context check: has_issue_type=%s has_port=%s issue_type=%s is_path_trace=%s",
            issue_clear, port_clear, issue_type, is_path_trace,
        )

        if not issue_clear or (is_connectivity_query and not port_clear):
            parts = []
            if not issue_clear:
                parts.append(
                    "**What type of issue are you seeing?**\n"
                    "- Blocked / denied — traffic is being dropped or a policy is denying it\n"
                    "- Slow / high latency — connectivity works but performance is degraded\n"
                    "- Intermittent — connectivity drops in and out unpredictably\n"
                    "- Device issue — a specific device is misbehaving (errors, CPU, outage)\n"
                    "- Path changed — routing appears different from what is expected"
                )
            if is_connectivity_query and not port_clear:
                parts.append(
                    "**Which port or protocol is affected?**\n"
                    "e.g. TCP 443, UDP 53, TCP 22 — or 'any' if protocol-agnostic"
                )
            question = "\n\n".join(parts)
            _pending_ts_set(session_id, prompt, issue_type)
            logger.info("Troubleshoot clarification requested for session %s (issue_type=%s)", session_id, issue_type)
            return {"final_response": {"role": "assistant", "content": question}}

        full_prompt = prompt

    try:
        result = await orchestrate_troubleshoot(
            full_prompt,
            username=state.get("username"),
            session_id=session_id,
            issue_type=issue_type,
        )
    except Exception as exc:
        logger.exception("Troubleshoot orchestrator failed: %s", exc)
        result = {
            "role": "assistant",
            "content": f"Troubleshooting failed: {exc}",
        }
    return {"final_response": result}


# ---------------------------------------------------------------------------
# Node 2: check_rbac
# ---------------------------------------------------------------------------

async def check_rbac(state: AtlasState) -> dict[str, Any]:
    tool_name = state.get("selected_tool_name") or state.get("prefilled_tool_name")
    username = state.get("username")
    session_id = state.get("session_id")
    err = _check_tool_access(username, tool_name, session_id) if tool_name else None
    return {"rbac_error": err}


# ---------------------------------------------------------------------------
# Node 3: fetch_mcp_tools
# ---------------------------------------------------------------------------

async def fetch_mcp_tools(state: AtlasState) -> dict[str, Any]:
    # Just a pass-through — tools are fetched inside tool_selector
    # This node exists as a named step for graph clarity
    return {}


# ---------------------------------------------------------------------------
# Node 4a: planner_node  (LLM generates a plan before any tool is called)
# ---------------------------------------------------------------------------

async def planner_node(state: AtlasState) -> dict[str, Any]:
    """
    Plan-and-Execute step 0: call the LLM without tools to generate a <plan>.
    The plan is injected as a SystemMessage so tool_selector can follow it step by step.
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage as _SM

    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    llm = ChatOpenAI(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, api_key="docker")
    messages = list(state["messages"])

    planning_instruction = _SM(content=(
        "Before calling any tool, state your plan: list every piece of information you need "
        "to fully answer the question, and which tool you will call for each step. "
        "Use exact tool names. Only plan — do not call any tool yet."
    ))

    try:
        plan_response = await asyncio.wait_for(
            llm.ainvoke(messages + [planning_instruction]),
            timeout=30.0,
        )
        if plan_response.content:
            plan_msg = _SM(content=(
                f"PLAN (execute every step — do not stop early):\n{plan_response.content}"
            ))
            messages = messages + [plan_msg]
            logger.info("planner_node: %s", plan_response.content[:300])
    except Exception as exc:
        logger.debug("planner_node: skipped (%s)", exc)

    return {"messages": messages}


# ---------------------------------------------------------------------------
# Node 4b: tool_selector  (LLM picks a tool)
# ---------------------------------------------------------------------------

async def tool_selector(state: AtlasState) -> dict[str, Any]:
    from langchain_openai import ChatOpenAI

    mcp_tools = await _fetch_mcp_tools()
    if not mcp_tools:
        return {"final_response": {"role": "assistant", "content": "Could not connect to MCP server."}}

    openai_tools = [_to_openai_tool(t) for t in mcp_tools]

    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    llm = ChatOpenAI(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, api_key="docker")

    iteration = state.get("iteration", 0)
    tool_choice = "required" if iteration == 0 else "auto"
    llm_with_tools = llm.bind_tools(openai_tools, tool_choice=tool_choice)
    messages = state["messages"]

    try:
        ai_msg = await asyncio.wait_for(llm_with_tools.ainvoke(messages), timeout=90.0)
    except asyncio.TimeoutError:
        return {"final_response": {"role": "assistant", "content": "Tool selection timed out. Please try again."}}

    # LLM chose to stop — use its text as the final answer
    if not ai_msg.tool_calls:
        accumulated = state.get("accumulated_results") or []
        if len(accumulated) > 1:
            return {"final_response": {"role": "assistant", "content": {"multi_results": accumulated}}}
        if len(accumulated) == 1:
            return {"final_response": {"role": "assistant", "content": accumulated[0]}}
        return {"final_response": {"role": "assistant", "content": ai_msg.content or "I could not determine how to answer that."}}

    tool_call = ai_msg.tool_calls[0]
    sel_tool_name = tool_call["name"]
    tool_args = {k: v for k, v in dict(tool_call["args"]).items() if v not in ({}, "")}
    tool_call_id = tool_call.get("id") or f"call_{sel_tool_name}_{iteration}"

    # discover_only: return tool metadata without executing
    if state.get("discover_only"):
        return {
            "final_response": {
                "tool_name": sel_tool_name,
                "parameters": tool_args,
                "tool_display_name": get_tool_display_name(sel_tool_name),
                "intent": tool_args.get("intent"),
                "format": "table",
            }
        }

    updated_messages = list(messages) + [ai_msg]
    return {
        "selected_tool_name": sel_tool_name,
        "selected_tool_args": tool_args,
        "tool_call_id": tool_call_id,
        "messages": updated_messages,
    }


# ---------------------------------------------------------------------------
# Node 5: doc_tool_caller
# ---------------------------------------------------------------------------

async def doc_tool_caller(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client import call_mcp_tool
    prompt = state["prompt"]
    result = await call_mcp_tool("search_documentation", {"query": prompt}, timeout=30.0)
    if result and not (isinstance(result, dict) and "error" in result):
        return {"final_response": {"role": "assistant", "content": result}}
    # Fall through to network path by setting intent
    return {"intent": "network", "tool_error": "doc search returned nothing"}


# ---------------------------------------------------------------------------
# Node 6: prefilled_tool_executor
# ---------------------------------------------------------------------------

async def prefilled_tool_executor(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client import call_mcp_tool
    from atlas.mcp_client_tool_selection import synthesize_final_answer

    tool_name = state["prefilled_tool_name"]
    tool_params = dict(state["prefilled_tool_params"] or {})
    prompt = state["prompt"]

    # discover_only: return tool metadata without executing the real tool.
    # Without this guard, a parallel /api/discover call for a follow-up confirmation
    # ("yes") would race with the main /api/chat call and hit the MCP tool twice.
    if state.get("discover_only"):
        return {
            "final_response": {
                "tool_name": tool_name,
                "tool_display_name": get_tool_display_name(tool_name),
                "parameters": {k: v for k, v in tool_params.items() if not k.startswith("_")},
            }
        }

    # Internal flag: caller wants policies from this group, not members
    policies_only = tool_params.pop("_policies_only", False)

    result = await call_mcp_tool(tool_name, tool_params, timeout=_TOOL_TIMEOUTS.get(tool_name, 65.0))

    if isinstance(result, dict) and result.get("requires_site"):
        return {"final_response": {"role": "assistant", "content": result}}
    if result is None or (isinstance(result, dict) and not result):
        msg = await synthesize_final_answer(prompt, tool_name, "No result returned.")
        return {"final_response": {"role": "assistant", "content": msg}}
    if isinstance(result, dict) and "error" in result:
        msg = await synthesize_final_answer(prompt, tool_name, result)
        return {"final_response": {"role": "assistant", "content": msg}}

    # policies_only: show policies from the group result instead of members
    if policies_only and isinstance(result, dict):
        group_name = result.get("address_group_name", tool_params.get("address_group_name", "this group"))
        policies = result.get("policies", [])
        if policies:
            count = len(policies)
            return {"final_response": {"role": "assistant", "content": {
                "direct_answer": f"Found {count} polic{'ies' if count != 1 else 'y'} referencing '{group_name}'",
                "policies": policies,
            }}}
        return {"final_response": {"role": "assistant", "content": {
            "direct_answer": f"No policies found referencing '{group_name}'.",
        }}}

    return {"final_response": {"role": "assistant", "content": _normalize_result(tool_name, result, prompt)}}


# ---------------------------------------------------------------------------
# Node 7: tool_executor
# ---------------------------------------------------------------------------

async def tool_executor(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client import call_mcp_tool
    from langchain_core.messages import ToolMessage

    tool_name = state["selected_tool_name"]
    tool_args = state.get("selected_tool_args") or {}
    tool_call_id = state.get("tool_call_id") or f"call_{tool_name}"
    iteration = state.get("iteration", 0)
    messages = state["messages"]

    result = await call_mcp_tool(tool_name, tool_args, timeout=_TOOL_TIMEOUTS.get(tool_name, 65.0))

    if isinstance(result, dict) and result.get("requires_site"):
        return {"final_response": {"role": "assistant", "content": result}}

    if result is None or (isinstance(result, dict) and not result):
        error = "No result returned. Check that the MCP server is running."
        updated_messages = messages + [ToolMessage(content=error, tool_call_id=tool_call_id)]
        return {"tool_error": error, "messages": updated_messages, "iteration": iteration + 1}

    if isinstance(result, dict) and "error" in result:
        updated_messages = messages + [ToolMessage(content=json.dumps(result), tool_call_id=tool_call_id)]
        return {"tool_error": result, "messages": updated_messages, "iteration": iteration + 1}

    # Success: normalize result and accumulate
    normalized = _normalize_result(tool_name, result, state.get("prompt", ""))
    # Strip follow_up offer from intermediate results that will be chained
    if tool_name == "query_panorama_ip_object_group" and isinstance(normalized, dict):
        normalized = {k: v for k, v in normalized.items() if k not in ("follow_up", "follow_up_action")}
    accumulated = list(state.get("accumulated_results") or [])
    accumulated.append(normalized)

    result_summary = json.dumps(result) if isinstance(result, dict) else str(result)
    # Truncate to avoid exceeding the model's context window on large Panorama results
    if len(result_summary) > 6000:
        result_summary = result_summary[:6000] + "... [truncated for context length]"
    updated_messages = messages + [ToolMessage(content=result_summary, tool_call_id=tool_call_id)]

    # Deterministic chaining: IP lookup always chains to members — no keyword matching needed.
    # If you know what group an IP is in, members + policies are always the natural next step.
    if tool_name == "query_panorama_ip_object_group" and isinstance(result, dict):
        address_groups = result.get("address_groups", [])
        if address_groups:
            first_group = address_groups[0]
            group_name = first_group.get("name")
            device_group = first_group.get("device_group")
            if group_name:
                members_params = {"address_group_name": group_name}
                if device_group:
                    members_params["device_group"] = device_group
                members_result = await call_mcp_tool(
                    "query_panorama_address_group_members",
                    members_params,
                    timeout=_TOOL_TIMEOUTS.get("query_panorama_address_group_members", 65.0),
                )
                if members_result and isinstance(members_result, dict) and "error" not in members_result:
                    members_normalized = _normalize_result("query_panorama_address_group_members", members_result, state.get("prompt", ""))
                    # Strip follow_up offer — we're chaining automatically, no need to prompt
                    if isinstance(members_normalized, dict):
                        members_normalized = {k: v for k, v in members_normalized.items() if k not in ("follow_up", "follow_up_action")}
                    accumulated.append(members_normalized)

                    # Also pull Splunk deny events for the IP
                    ip_address = tool_args.get("ip_address") or tool_args.get("ip")
                    if ip_address:
                        splunk_result = await call_mcp_tool(
                            "get_splunk_recent_denies",
                            {"ip_address": ip_address},
                            timeout=_TOOL_TIMEOUTS.get("get_splunk_recent_denies", 65.0),
                        )
                        if splunk_result and isinstance(splunk_result, dict) and "error" not in splunk_result:
                            splunk_normalized = _normalize_result("get_splunk_recent_denies", splunk_result, state.get("prompt", ""))
                            accumulated.append(splunk_normalized)

                    final = {"multi_results": accumulated} if len(accumulated) > 1 else members_normalized
                    return {"tool_raw_result": members_result, "accumulated_results": accumulated, "tool_error": None,
                            "final_response": {"role": "assistant", "content": final}, "messages": updated_messages, "iteration": iteration + 1}

    # If user asked for members + policies and this result already has both inline
    # (no follow_up_action needed), finalize immediately — don't let the LLM loop.
    prompt_lower_te = (state.get("prompt") or "").lower()
    if (
        tool_name == "query_panorama_address_group_members"
        and any(w in prompt_lower_te for w in ("policies", "policy"))
        and isinstance(normalized, dict)
        and not normalized.get("follow_up_action")
        and "members" in normalized
    ):
        final = {"multi_results": accumulated} if len(accumulated) > 1 else normalized
        return {"tool_raw_result": result, "accumulated_results": accumulated, "tool_error": None,
                "final_response": {"role": "assistant", "content": final}, "messages": updated_messages, "iteration": iteration + 1}

    return {"tool_raw_result": result, "accumulated_results": accumulated, "tool_error": None, "messages": updated_messages, "iteration": iteration + 1}


# ---------------------------------------------------------------------------
# Node 8: normalize_result
# ---------------------------------------------------------------------------

async def normalize_result(state: AtlasState) -> dict[str, Any]:
    tool_name = state["selected_tool_name"]
    result = state["tool_raw_result"]
    prompt = state.get("prompt", "")
    normalized = _normalize_result(tool_name, result, prompt)
    return {"final_response": {"role": "assistant", "content": normalized}}


# ---------------------------------------------------------------------------
# Node 9: synthesize_error
# ---------------------------------------------------------------------------

async def synthesize_error(state: AtlasState) -> dict[str, Any]:
    from atlas.mcp_client_tool_selection import synthesize_final_answer
    tool_name = state.get("selected_tool_name") or state.get("prefilled_tool_name") or "tool"
    error = state.get("tool_error") or "Unknown error"
    prompt = state.get("prompt", "")
    msg = await synthesize_final_answer(prompt, tool_name, error)
    return {"final_response": {"role": "assistant", "content": msg}}


# ---------------------------------------------------------------------------
# Node 10: enrich_with_insights
# ---------------------------------------------------------------------------

def _extract_member_ips(result: dict) -> list[str]:
    """Extract host IPs from address group member objects."""
    ips = []
    for m in result.get("members", []):
        if not isinstance(m, dict):
            continue
        if m.get("type") not in ("ip-netmask", "ip-range"):
            continue
        val = m.get("value", "")
        ip = val.split("/")[0] if "/" in val else val.split("-")[0]
        if ip and ip not in ips:
            ips.append(ip)
    return ips


async def enrich_with_insights(state: AtlasState) -> dict[str, Any]:
    """Add proactive insights to the final response before it is returned."""
    from atlas.mcp_client import call_mcp_tool

    final = state.get("final_response")
    if not final:
        return {}
    content = final.get("content") if isinstance(final, dict) else None
    if not content or isinstance(content, str):
        return {}

    # Collect all result dicts to analyse
    if isinstance(content, dict) and content.get("multi_results"):
        results = [r for r in content["multi_results"] if isinstance(r, dict)]
    elif isinstance(content, dict):
        results = [content]
    else:
        return {}

    insights: list[str] = []

    for result in results:
        # --- Hint-only: no extra tool calls ---
        # Address group with members but no policies → may be unused
        if "members" in result and "policies" in result and not result.get("policies"):
            group = result.get("address_group_name", "This group")
            insights.append(f"'{group}' is not referenced by any security policies — it may be unused.")

        # --- Inline Splunk check for member IPs ---
        ips = _extract_member_ips(result)
        if not ips:
            continue
        # RBAC: only check Splunk if the user has access
        rbac_err = _check_tool_access(state.get("username"), "get_splunk_recent_denies", state.get("session_id"))
        if rbac_err:
            continue
        tasks = [
            call_mcp_tool("get_splunk_recent_denies", {"ip_address": ip}, timeout=25.0)
            for ip in ips[:3]
        ]
        splunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        for ip, sr in zip(ips, splunk_results):
            if isinstance(sr, dict) and sr.get("count", 0) > 0:
                count = sr["count"]
                insights.append(f"{ip} has {count} recent deny event{'s' if count != 1 else ''} in Splunk.")

    if not insights:
        return {}

    # Attach insights to content
    if isinstance(content, dict) and content.get("multi_results"):
        updated = list(content["multi_results"])
        last = dict(updated[-1]) if isinstance(updated[-1], dict) else updated[-1]
        if isinstance(last, dict):
            last["insights"] = insights
            updated[-1] = last
        new_content = dict(content)
        new_content["multi_results"] = updated
    else:
        new_content = dict(content)
        new_content["insights"] = insights

    return {"final_response": {"role": "assistant", "content": new_content}}
