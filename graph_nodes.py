"""
Atlas LangGraph nodes.
All logic is extracted from chat_service.py — no new behaviour introduced.
"""
import asyncio
import json
import logging
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
# Node 1: classify_intent
# ---------------------------------------------------------------------------

_CONFIRMATIONS = {"yes", "yeah", "sure", "yep", "ok", "okay", "please", "go ahead", "do it", "show me", "yep please"}
_DISMISSALS = {"no", "nope", "nah", "no thanks", "don't", "dont", "never mind", "nevermind", "skip", "not now", "no need", "i'm good", "im good", "all good", "that's fine", "thats fine"}


_INTENT_SYSTEM_PROMPT = """\
You are an intent classifier for a network security tool. Classify the message into exactly one category.

CATEGORIES:

network — Looking up firewall objects: what address group an IP belongs to, group members, or which policies reference a group. The user wants to know WHAT something IS, not fix a problem.
  "what group is 10.0.0.5 in?"  →  network
  "what address group is 11.0.0.1 part of?"  →  network
  "show me the members of leander_web"  →  network
  "what policies reference this group? also give me the members"  →  network
  "give me the group members and the policies that reference it"  →  network

troubleshoot — A connectivity PROBLEM between two specific endpoints. The user cannot reach something and wants to know why.
  "why can't 10.0.0.1 connect to 11.0.0.1?"  →  troubleshoot
  "traffic from 10.0.0.1 to 11.0.0.1 is being blocked"  →  troubleshoot
  "is traffic from A to B allowed on port 443?"  →  troubleshoot

risk — Security posture or risk assessment for a single IP or host.
  "what is the risk of 10.0.0.5?"  →  risk
  "is this IP risky?"  →  risk

netbrain — Tracing or visualising the network path between two IPs.
  "trace path from A to B"  →  netbrain
  "show me the route from X to Y"  →  netbrain

doc — A how-to or conceptual question about firewall/network/security tools or processes.
  "how do I request firewall access?"  →  doc
  "what is a DMZ?"  →  doc

dismiss — Off-topic, greeting, or unrelated to firewalls and network security.
  "what's the weather?"  →  dismiss
  "hello"  →  dismiss
  "what?"  →  dismiss

Reply with ONLY the category name. No explanation. No punctuation.\
"""


async def _llm_classify_intent(prompt: str) -> str:
    """Call the local LLM to classify the intent of a user prompt."""
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
        max_tokens=10,
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        raw = (response.content or "").strip().lower().split()[0].rstrip(".,;:")
        valid = {"troubleshoot", "risk", "netbrain", "doc", "network", "dismiss"}
        if raw in valid:
            return raw
        logger.warning("_llm_classify_intent: unexpected label %r for %r — falling back to doc", raw, prompt[:80])
        return "doc"
    except Exception as exc:
        logger.warning("_llm_classify_intent failed: %s — falling back to doc", exc)
        return "doc"


async def classify_intent(state: AtlasState) -> dict[str, Any]:
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(state.get("session_id") or "default", "Classifying your query...")
    except Exception:
        pass
    prompt = state["prompt"]
    prefilled = state.get("prefilled_tool_name")

    if prefilled and state.get("prefilled_tool_params") is not None:
        messages = _build_llm_messages(prompt, state.get("conversation_history") or [])
        return {"intent": "prefilled", "messages": messages, "iteration": 0}

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
        has_pending_ts = session_id in _pending_ts_prompts
        if has_pending_ts:
            if not _IP_OR_CIDR_RE.search(prompt) and len(prompt.split()) <= 15:
                # Short, no-IP reply → treat as clarification answer
                return {"intent": "troubleshoot", "messages": [], "iteration": 0}
            else:
                # New query with IPs → discard stale pending, evaluate fresh
                _pending_ts_prompts.pop(session_id, None)
                logger.info("classify_intent: discarding stale pending_ts for session %s (new IP query)", session_id)

    # LLM-based intent classification
    intent = await _llm_classify_intent(prompt)
    logger.info("classify_intent: LLM classified %r as %r", prompt[:80], intent)

    if intent == "dismiss":
        return {"intent": "dismiss", "final_response": {"role": "assistant", "content": "I am not equipped to answer this question. If you feel its a mistake, reach out to the atlas team."}, "messages": [], "iteration": 0}

    if intent in ("troubleshoot", "netbrain", "risk"):
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
# Node: troubleshoot_orchestrator — multi-agent troubleshooting coordinator
# ---------------------------------------------------------------------------

_TS_CONTEXT_PROMPT = """\
You are analysing a network troubleshooting query. Determine what context is missing.

Reply with ONLY a JSON object on one line, no explanation:
{"has_issue_type": <true|false>, "has_port": <true|false>}

- has_issue_type: true if the query implies a connectivity failure or performance problem. "Cannot connect", "not connecting", "unable to connect", "can't reach", "not working" all count as has_issue_type=true (they imply blocked/unreachable). Only false if the query is completely neutral with no problem framing.
- has_port: true if the query mentions a port, protocol, or service (TCP, UDP, port 443, HTTPS, SSH, DNS, "any", etc.)

Examples:
"why can I not connect from 10.0.0.1 to 11.0.0.1" → {"has_issue_type": true, "has_port": false}
"why can't 10.0.0.1 reach 11.0.0.1" → {"has_issue_type": true, "has_port": false}
"traffic is blocked from 10.0.0.1 to 11.0.0.1 on TCP 443" → {"has_issue_type": true, "has_port": true}
"why is 10.0.0.1 slow reaching 11.0.0.1" → {"has_issue_type": true, "has_port": false}
"can 10.0.0.1 reach 11.0.0.1 over HTTPS" → {"has_issue_type": true, "has_port": true}
"check connectivity from 10.0.0.1 to 11.0.0.1" → {"has_issue_type": false, "has_port": false}\
"""


async def _llm_check_ts_context(prompt: str) -> tuple[bool, bool]:
    """Return (has_issue_type, has_port) for a troubleshoot prompt."""
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
        max_tokens=30,
    )
    try:
        response = await llm.ainvoke([
            SystemMessage(content=_TS_CONTEXT_PROMPT),
            HumanMessage(content=prompt),
        ])
        import json as _json
        data = _json.loads(response.content.strip())
        return bool(data.get("has_issue_type")), bool(data.get("has_port"))
    except Exception as exc:
        logger.warning("_llm_check_ts_context failed: %s — assuming context missing", exc)
        return False, False


# Pending clarification prompts keyed by session_id.
# When a troubleshoot query lacks port/issue context we return a clarifying
# question and store the original prompt here.  The next message from the same
# session is combined with it before running the orchestrator.
_pending_ts_prompts: dict[str, str] = {}


async def troubleshoot_orchestrator(state: AtlasState) -> dict[str, Any]:
    """
    Multi-agent troubleshooting coordinator.

    If the prompt is missing port or issue-type context, returns a clarifying
    question and saves the original prompt so the next reply can continue.
    Uses application-layer state (not LangGraph interrupt) to stay simple.
    """
    try:
        from atlas.agents.troubleshoot_orchestrator import orchestrate_troubleshoot
    except ImportError:
        from agents.troubleshoot_orchestrator import orchestrate_troubleshoot

    session_id = state.get("session_id") or "default"
    prompt = state["prompt"]

    try:
        import atlas.status_bus as status_bus
        await status_bus.push(session_id, "Analyzing your query...")
    except Exception:
        pass

    # If a previous turn asked for clarification, combine with the original prompt.
    # Do this before the IP check — the clarification answer won't contain IPs.
    pending = _pending_ts_prompts.pop(session_id, None)
    if pending:
        full_prompt = f"{pending}\n\nUser clarification: {prompt}"
    else:
        # Require at least two IPs to proceed — without them we can't troubleshoot anything.
        ip_matches_in_prompt = _IP_OR_CIDR_RE.findall(prompt)
        if len(ip_matches_in_prompt) < 2:
            return {"final_response": {"role": "assistant", "content": (
                "Please provide both a **source IP** and a **destination IP** to troubleshoot.\n"
                "Example: \"Why can't 10.0.0.1 connect to 11.0.0.1?\""
            )}}
        issue_clear, port_clear = await _llm_check_ts_context(prompt)
        logger.info("Troubleshoot context check: has_issue_type=%s has_port=%s", issue_clear, port_clear)

        if not issue_clear or not port_clear:
            parts = []
            if not issue_clear:
                parts.append(
                    "**What type of issue are you seeing?**\n"
                    "- Blocked / denied — traffic is being dropped or a policy is denying it\n"
                    "- Slow / high latency — connectivity works but performance is degraded\n"
                    "- Intermittent — connectivity drops in and out unpredictably\n"
                    "- Path changed — routing appears different from what is expected"
                )
            if not port_clear:
                parts.append(
                    "**Which port or protocol is affected?**\n"
                    "e.g. TCP 443, UDP 53, TCP 22 — or 'any' if protocol-agnostic"
                )
            question = "\n\n".join(parts)
            _pending_ts_prompts[session_id] = prompt
            logger.info("Troubleshoot clarification requested for session %s", session_id)
            return {"final_response": {"role": "assistant", "content": question}}

        full_prompt = prompt

    try:
        result = await orchestrate_troubleshoot(
            full_prompt,
            username=state.get("username"),
            session_id=session_id,
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
# Node 4: tool_selector  (LLM picks a tool)
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
    accumulated = list(state.get("accumulated_results") or [])
    accumulated.append(normalized)

    result_summary = json.dumps(result) if isinstance(result, dict) else str(result)
    # Truncate to avoid exceeding the model's context window on large Panorama results
    if len(result_summary) > 6000:
        result_summary = result_summary[:6000] + "... [truncated for context length]"
    updated_messages = messages + [ToolMessage(content=result_summary, tool_call_id=tool_call_id)]

    # Deterministic chaining: if IP lookup completed and user asked for members/policies,
    # call query_panorama_address_group_members directly — do not rely on LLM to chain.
    if tool_name == "query_panorama_ip_object_group" and isinstance(result, dict):
        prompt_lower_chain = (state.get("prompt") or "").lower()
        wants_members = any(w in prompt_lower_chain for w in ("members", "contents", "what's in", "whats in"))
        wants_policies = any(w in prompt_lower_chain for w in ("policies", "policy", "rules", "security rules", "firewall rules"))
        if wants_members or wants_policies:
            address_groups = result.get("address_groups", [])
            if address_groups:
                first_group = address_groups[0]
                group_name = first_group.get("name")
                device_group = first_group.get("device_group")
                if group_name:
                    members_params = {"address_group_name": group_name}
                    if device_group:
                        members_params["device_group"] = device_group
                    if wants_policies and not wants_members:
                        members_params["_policies_only"] = True
                    members_result = await call_mcp_tool(
                        "query_panorama_address_group_members",
                        members_params,
                        timeout=_TOOL_TIMEOUTS.get("query_panorama_address_group_members", 65.0),
                    )
                    if members_result and isinstance(members_result, dict) and "error" not in members_result:
                        members_normalized = _normalize_result("query_panorama_address_group_members", members_result, state.get("prompt", ""))
                        accumulated.append(members_normalized)
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

    # Stop auto-chaining when the result has a follow_up_action, unless the original
    # prompt explicitly asked for what this follow_up is offering.
    # e.g. "find group AND show members" → allow chaining to members step
    # but stop at members→policies unless prompt also asked for "policies".
    if isinstance(normalized, dict) and normalized.get("follow_up_action"):
        prompt_lower = (state.get("prompt") or "").lower()
        fu_lower = (normalized.get("follow_up") or "").lower()
        offered_members = "members" in fu_lower
        offered_policies = "policies" in fu_lower or "policy" in fu_lower
        user_asked_for_offer = (
            (offered_members and "members" in prompt_lower) or
            (offered_policies and any(w in prompt_lower for w in ("policies", "policy")))
        )
        if not user_asked_for_offer:
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
