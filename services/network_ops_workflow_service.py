"""Owned network-ops workflow orchestration outside the graph node layer."""
from __future__ import annotations

import logging
import traceback as _tb
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage

try:
    from atlas.agents.network_ops_agent import build_agent
    from atlas.services.memory_manager import memory_manager
    from atlas.services.metrics import metrics_recorder
    from atlas.services.nornir_client import nornir_client
    from atlas.services.observability import elapsed_ms, log_event
    from atlas.services.request_preprocessor import extract_final_text, looks_like_clarification_request
    from atlas.services.response_presenter import response_presenter
    from atlas.services.session_store import session_store
    from atlas.services.status_service import status_service
except ImportError:
    from agents.network_ops_agent import build_agent  # type: ignore
    from services.memory_manager import memory_manager  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.nornir_client import nornir_client  # type: ignore
    from services.observability import elapsed_ms, log_event  # type: ignore
    from services.request_preprocessor import extract_final_text, looks_like_clarification_request  # type: ignore
    from services.response_presenter import response_presenter  # type: ignore
    from services.session_store import session_store  # type: ignore
    from services.status_service import status_service  # type: ignore


logger = logging.getLogger("atlas.network_ops_workflow")


class NetworkOpsWorkflowService:
    """Owns network-ops agent orchestration and follow-up handling."""

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        session_id = state.get("session_id") or "default"
        request_id = state.get("request_id")
        prompt = state["prompt"]
        started_at = perf_counter()

        await status_service.push(session_id, "Processing network ops request...")

        pending, pending_issue_type = memory_manager.get_pending_context(session_id)
        if pending_issue_type == "network_ops" and pending:
            memory_manager.clear_pending_context(session_id)
            prompt = f"{pending}\n\nUser clarification: {prompt}"

        nornir_client.clear_session_cache(session_id)
        session_store.pop(session_id)
        config = {"configurable": {"session_id": session_id, "thread_id": session_id}}

        try:
            agent = build_agent()
            result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        except Exception as exc:
            metrics_recorder.increment("atlas.agent.failed", agent_type="network_ops")
            log_event(
                logger,
                "network_ops_agent_failed",
                level="error",
                request_id=request_id,
                session_id=session_id,
                error=str(exc),
            )
            logger.error("Network ops agent failed: %s\n%s", exc, _tb.format_exc())
            nornir_client.clear_session_cache(session_id)
            return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Network ops agent failed: {exc}"}}}

        final_text = extract_final_text(result.get("messages", []))
        if looks_like_clarification_request(final_text):
            memory_manager.set_pending_context(session_id, prompt, "network_ops")
        session_data = session_store.pop(session_id)
        content = response_presenter.build_network_ops_content(final_text, session_data, prompt)

        duration_ms = elapsed_ms(started_at)
        metrics_recorder.increment("atlas.agent.completed", agent_type="network_ops")
        metrics_recorder.observe_ms("atlas.agent.duration_ms", duration_ms, agent_type="network_ops")
        log_event(
            logger,
            "network_ops_agent_completed",
            request_id=request_id,
            session_id=session_id,
            elapsed_ms=duration_ms,
            content_keys=list(content.keys()),
            has_path_hops=bool(content.get("path_hops")),
            has_reverse_path_hops=bool(content.get("reverse_path_hops")),
        )
        nornir_client.clear_session_cache(session_id)
        return {"final_response": {"role": "assistant", "content": content}}


network_ops_workflow_service = NetworkOpsWorkflowService()
