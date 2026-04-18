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
    from atlas.services.network_ops_scenario_service import network_ops_scenario_service
    from atlas.services.nornir_client import nornir_client
    from atlas.services.observability import elapsed_ms, log_event
    from atlas.services.pending_approval import pending_approval_store
    from atlas.services.request_preprocessor import extract_final_text, looks_like_clarification_request
    from atlas.services.response_presenter import response_presenter
    from atlas.services.session_store import session_store
    from atlas.services.status_service import status_service
    from atlas.tools.servicenow_agent_tools import execute_pending_write_action
except ImportError:
    from agents.network_ops_agent import build_agent  # type: ignore
    from services.memory_manager import memory_manager  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.network_ops_scenario_service import network_ops_scenario_service  # type: ignore
    from services.nornir_client import nornir_client  # type: ignore
    from services.observability import elapsed_ms, log_event  # type: ignore
    from services.pending_approval import pending_approval_store  # type: ignore
    from services.request_preprocessor import extract_final_text, looks_like_clarification_request  # type: ignore
    from services.response_presenter import response_presenter  # type: ignore
    from services.session_store import session_store  # type: ignore
    from services.status_service import status_service  # type: ignore
    from tools.servicenow_agent_tools import execute_pending_write_action  # type: ignore


logger = logging.getLogger("atlas.network_ops_workflow")


class NetworkOpsWorkflowService:
    """Owns network-ops agent orchestration and follow-up handling."""

    @staticmethod
    def _format_pending_fields(fields: dict[str, Any]) -> str:
        lines: list[str] = []
        for key, value in fields.items():
            if value in ("", None):
                continue
            label = str(key).replace("_", " ").title()
            lines.append(f"- {label}: {value}")
        return "\n".join(lines) if lines else "- No fields captured."

    def _build_pending_approval_edit_prompt(self, pending: dict[str, Any], user_edit: str) -> str:
        original_prompt = str(pending.get("original_prompt") or "").strip()
        action_label = str(pending.get("action_label") or "ServiceNow write action").strip()
        fields = pending.get("fields") or pending.get("arguments") or {}
        parts = []
        if original_prompt:
            parts.append(original_prompt)
        parts.append(
            "There is a pending ServiceNow write action that has not been executed yet.\n"
            f"Pending action: {action_label}\n"
            "Current fields:\n"
            f"{self._format_pending_fields(fields)}\n\n"
            f"User requested changes to the pending action: {user_edit}\n\n"
            "Update the pending action accordingly. Do not treat it as already executed. "
            "When ready, call the appropriate ServiceNow write tool again so Atlas can stage a fresh approval request."
        )
        return "\n\n".join(part for part in parts if part).strip()

    @staticmethod
    def _execution_succeeded(result_text: str, tool_name: str) -> bool:
        normalized = (result_text or "").strip()
        success_prefixes = {
            "create_servicenow_incident": "Created ServiceNow incident",
            "create_servicenow_change_request": "Created ServiceNow change request",
            "update_servicenow_change_request": "Updated ServiceNow change request",
        }
        prefix = success_prefixes.get(tool_name, "")
        return bool(prefix and normalized.startswith(prefix))

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        session_id = state.get("session_id") or "default"
        request_id = state.get("request_id")
        prompt = state["prompt"]
        user_prompt = prompt
        ui_action = state.get("ui_action") if isinstance(state.get("ui_action"), dict) else {}
        ui_action_type = str((ui_action or {}).get("type") or "").strip().lower()
        ui_action_approval_id = str((ui_action or {}).get("approval_id") or "").strip()
        started_at = perf_counter()

        await status_service.push(session_id, "Processing network ops request...")

        nornir_client.clear_session_cache(session_id)
        session_store.pop(session_id)

        pending_approval = pending_approval_store.get(session_id)
        if pending_approval:
            approval_prompt = str(pending_approval.get("original_prompt") or prompt)
            pending_approval_id = str(pending_approval.get("approval_id") or "").strip()
            if ui_action_type in {"confirm", "cancel"} and ui_action_approval_id and pending_approval_id and ui_action_approval_id != pending_approval_id:
                final_text = "That approval request is no longer current. Use the latest approval card."
                content = response_presenter.build_network_ops_content(
                    final_text,
                    {},
                    approval_prompt,
                    pending_approval=pending_approval,
                )
                nornir_client.clear_session_cache(session_id)
                return {"final_response": {"role": "assistant", "content": content}}
            if ui_action_type == "confirm":
                final_text = await execute_pending_write_action(pending_approval, session_id)
                if self._execution_succeeded(final_text, str(pending_approval.get("tool_name") or "")):
                    pending_approval_store.clear(session_id)
                    pending_approval = None
                content = response_presenter.build_network_ops_content(
                    final_text,
                    {},
                    approval_prompt,
                    pending_approval=pending_approval,
                )
                nornir_client.clear_session_cache(session_id)
                return {"final_response": {"role": "assistant", "content": content}}
            if ui_action_type == "cancel":
                pending_approval_store.clear(session_id)
                final_text = (
                    f"Cancelled the pending {str(pending_approval.get('action_label') or 'ServiceNow write action').lower()}. "
                    "Nothing was executed."
                )
                content = response_presenter.build_network_ops_content(final_text, {}, approval_prompt)
                nornir_client.clear_session_cache(session_id)
                return {"final_response": {"role": "assistant", "content": content}}
            pending_approval_store.clear(session_id)
            prompt = self._build_pending_approval_edit_prompt(pending_approval, prompt)

        pending, pending_issue_type = memory_manager.get_pending_context(session_id)
        if pending_issue_type == "network_ops" and pending:
            memory_manager.clear_pending_context(session_id)
            prompt = f"{pending}\n\nUser clarification: {prompt}"

        config = {
            "configurable": {
                "session_id": session_id,
                "thread_id": session_id,
                "request_prompt": user_prompt,
            }
        }
        scenario = await network_ops_scenario_service.select_scenario(prompt)

        try:
            agent = build_agent(prompt, scenario)
            result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]}, config=config)
        except Exception as exc:
            metrics_recorder.increment("atlas.agent.failed", agent_type="network_ops", scenario=scenario)
            log_event(
                logger,
                "network_ops_agent_failed",
                level="error",
                request_id=request_id,
                session_id=session_id,
                scenario=scenario,
                error=str(exc),
            )
            logger.error("Network ops agent failed: %s\n%s", exc, _tb.format_exc())
            nornir_client.clear_session_cache(session_id)
            return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Network ops agent failed: {exc}"}}}

        final_text = extract_final_text(result.get("messages", []))
        if looks_like_clarification_request(final_text):
            memory_manager.set_pending_context(session_id, prompt, "network_ops")
        session_data = session_store.pop(session_id)
        pending_approval = pending_approval_store.get(session_id)
        content = response_presenter.build_network_ops_content(
            final_text,
            session_data,
            user_prompt,
            pending_approval=pending_approval,
        )

        duration_ms = elapsed_ms(started_at)
        metrics_recorder.increment("atlas.agent.completed", agent_type="network_ops", scenario=scenario)
        metrics_recorder.observe_ms("atlas.agent.duration_ms", duration_ms, agent_type="network_ops", scenario=scenario)
        log_event(
            logger,
            "network_ops_agent_completed",
            request_id=request_id,
            session_id=session_id,
            scenario=scenario,
            elapsed_ms=duration_ms,
            content_keys=list(content.keys()),
            has_path_hops=bool(content.get("path_hops")),
            has_reverse_path_hops=bool(content.get("reverse_path_hops")),
        )
        nornir_client.clear_session_cache(session_id)
        return {"final_response": {"role": "assistant", "content": content}}


network_ops_workflow_service = NetworkOpsWorkflowService()
