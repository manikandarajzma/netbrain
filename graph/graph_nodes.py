"""
Atlas LangGraph nodes.

Graph shape:
  classify_intent
      ├─► dispatch_agent            (generic workflow dispatch for routed agents)
      └─► build_final_response      (dismiss / early-exit)
"""
import logging
from typing import Any

try:
    from atlas.agents.agent_registry import agent_registry
    from atlas.graph.graph_state import AtlasState
    from atlas.application.chat_service import _IP_OR_CIDR_RE
    from atlas.services.intent_routing_service import intent_routing_service
    from atlas.services.memory_manager import memory_manager
    from atlas.services.metrics import metrics_recorder
    from atlas.services.network_ops_workflow_service import network_ops_workflow_service
    from atlas.services.observability import log_event
    from atlas.services.pending_approval import pending_approval_store
    from atlas.services.status_service import status_service
    from atlas.services.troubleshoot_workflow_service import troubleshoot_workflow_service
    from atlas.services.workflow_registry import workflow_registry
except ImportError:
    from agents.agent_registry import agent_registry  # type: ignore
    from graph.graph_state import AtlasState        # type: ignore[assignment]
    from application.chat_service import _IP_OR_CIDR_RE  # type: ignore[assignment]
    from services.intent_routing_service import intent_routing_service  # type: ignore
    from services.memory_manager import memory_manager  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.network_ops_workflow_service import network_ops_workflow_service  # type: ignore
    from services.observability import log_event  # type: ignore
    from services.pending_approval import pending_approval_store  # type: ignore
    from services.status_service import status_service  # type: ignore
    from services.troubleshoot_workflow_service import troubleshoot_workflow_service  # type: ignore
    from services.workflow_registry import workflow_registry  # type: ignore

logger = logging.getLogger("atlas.graph_nodes")


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------

_DISMISSALS      = {"no", "nope", "nah", "no thanks", "never mind", "nevermind", "skip",
                    "not now", "no need", "i'm good", "im good", "all good"}
_ACKNOWLEDGEMENTS = {"yes", "yeah", "sure", "ok", "okay", "great", "thanks",
                     "thank you", "cool", "got it", "noted", "perfect", "sounds good"}


async def classify_intent(state: AtlasState) -> dict[str, Any]:
    """
    Classify the user's prompt and set state["intent"].

    Possible values
    ---------------
    "troubleshoot"
        Layered connectivity / device-health investigation.
        → routed through generic dispatch
    "network_ops"
        Operational workflow: firewall change request, policy review,
        spreadsheet generation, etc.
        → routed through generic dispatch
    "dismiss"
        Bare acknowledgement with nothing pending — skip LLM entirely.
        → short-circuits to build_final_response
    """
    prompt     = state["prompt"]
    session_id = state.get("session_id") or "default"
    request_id = state.get("request_id")

    await status_service.push(session_id, "Analyzing your query...")

    prompt_lower = prompt.lower().strip().rstrip("!.")

    if pending_approval_store.has(session_id):
        log_event(
            logger,
            "intent_classified",
            request_id=request_id,
            session_id=session_id,
            intent="network_ops",
            reason="pending_network_ops_approval",
        )
        return {"intent": "network_ops"}

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
            intent = pending_issue_type if pending_issue_type in agent_registry.valid_route_keys() else "troubleshoot"
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

    llm_decision = await intent_routing_service.route_prompt(prompt)
    if llm_decision:
        llm_intent = str(llm_decision.get("intent") or "").strip()
        metrics_recorder.increment("atlas.intent.classified", intent=llm_intent)
        log_event(
            logger,
            "intent_classified",
            request_id=request_id,
            session_id=session_id,
            intent=llm_intent,
            reason="llm_router",
            router_confidence=llm_decision.get("confidence"),
            router_note=llm_decision.get("reason"),
        )
        if llm_intent == "dismiss":
            return {
                "intent": "dismiss",
                "final_response": {
                    "role": "assistant",
                    "content": "Atlas is not equipped to help with it.",
                },
            }
        return {"intent": llm_intent}

    metrics_recorder.increment("atlas.intent.classified", intent="dismiss")
    log_event(
        logger,
        "intent_classified",
        request_id=request_id,
        session_id=session_id,
        intent="dismiss",
        reason="llm_router_unavailable_or_invalid",
    )
    return {
        "intent": "dismiss",
        "final_response": {
            "role": "assistant",
            "content": (
                "Atlas could not classify the request. Please state whether you want troubleshooting "
                "or an operational action such as incident or change management."
            ),
        },
    }
# ---------------------------------------------------------------------------
# Node 2: dispatch_agent
# ---------------------------------------------------------------------------

async def dispatch_agent(state: AtlasState) -> dict[str, Any]:
    """Dispatch to the workflow runner declared by the routed agent spec."""
    intent = str(state.get("intent") or "").strip()
    spec = agent_registry.get(intent)
    return await workflow_registry.run(spec, state)


# ---------------------------------------------------------------------------
# Compatibility wrappers
# ---------------------------------------------------------------------------

async def call_troubleshoot_agent(state: AtlasState) -> dict[str, Any]:
    """Compatibility wrapper for troubleshoot workflow dispatch."""
    return await troubleshoot_workflow_service.run(state)


async def call_network_ops_agent(state: AtlasState) -> dict[str, Any]:
    """Compatibility wrapper for network-ops workflow dispatch."""
    return await network_ops_workflow_service.run(state)


# ---------------------------------------------------------------------------
# Node 3: build_final_response
# ---------------------------------------------------------------------------

async def build_final_response(state: AtlasState) -> dict[str, Any]:
    if state.get("rbac_error"):
        return {"final_response": {"role": "assistant", "content": state["rbac_error"]}}
    return {}
