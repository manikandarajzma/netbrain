"""Owned troubleshoot workflow orchestration outside the graph node layer."""
from __future__ import annotations

import logging
import re
import traceback as _tb
from time import perf_counter
from typing import Any

from langchain_core.messages import HumanMessage

try:
    from atlas.services.memory_manager import memory_manager
    from atlas.services.metrics import metrics_recorder
    from atlas.services.nornir_client import nornir_client
    from atlas.services.observability import elapsed_ms, log_event
    from atlas.services.request_preprocessor import (
        extract_final_text,
        extract_ips,
        extract_port,
        resolve_incident_prompt,
    )
    from atlas.services.response_presenter import response_presenter
    from atlas.services.session_store import session_store
    from atlas.services.status_service import status_service
    from atlas.services.workflow_state_service import workflow_state_service
    from atlas.tools.path_agent_tools import trace_path, trace_reverse_path
    from atlas.agents.troubleshoot_agent import build_agent
except ImportError:
    from services.memory_manager import memory_manager  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.nornir_client import nornir_client  # type: ignore
    from services.observability import elapsed_ms, log_event  # type: ignore
    from services.request_preprocessor import extract_final_text, extract_ips, extract_port, resolve_incident_prompt  # type: ignore
    from services.response_presenter import response_presenter  # type: ignore
    from services.session_store import session_store  # type: ignore
    from services.status_service import status_service  # type: ignore
    from services.workflow_state_service import workflow_state_service  # type: ignore
    from tools.path_agent_tools import trace_path, trace_reverse_path  # type: ignore
    from agents.troubleshoot_agent import build_agent  # type: ignore


logger = logging.getLogger("atlas.troubleshoot_workflow")
_IP_OR_CIDR_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")


class TroubleshootWorkflowService:
    """Owns troubleshoot-agent orchestration and evidence enforcement."""

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        session_id = state.get("session_id") or "default"
        request_id = state.get("request_id")
        prompt = state["prompt"]
        started_at = perf_counter()

        nornir_client.clear_session_cache(session_id)
        session_store.pop(session_id)
        await status_service.push(session_id, "Investigating...")

        pending, pending_issue_type = memory_manager.get_pending_context(session_id)
        if pending:
            memory_manager.clear_pending_context(session_id)

        if not pending:
            history = state.get("conversation_history") or []
            if len(history) >= 2:
                last_assistant = next((m.get("content", "") for m in reversed(history) if m.get("role") == "assistant"), "")
                last_user_before = next((m.get("content", "") for m in reversed(history[:-1]) if m.get("role") == "user"), "")
                is_clarification_reply = (
                    ("which port" in last_assistant.lower() or "what type of issue" in last_assistant.lower())
                    and not _IP_OR_CIDR_RE.findall(prompt)
                    and len(prompt.split()) <= 6
                )
                if is_clarification_reply and _IP_OR_CIDR_RE.findall(last_user_before):
                    pending = last_user_before
                    pending_issue_type = "general"

        issue_type = pending_issue_type or "general"
        full_prompt = f"{pending}\n\nUser clarification: {prompt}" if pending else prompt

        if not pending:
            ip_matches = _IP_OR_CIDR_RE.findall(prompt)
            words = prompt.lower().split()
            has_device_context = any(c.isdigit() or "-" in w for w in words for c in w)
            if not has_device_context and not ip_matches and len(prompt.split()) < 4:
                return {
                    "final_response": {
                        "role": "assistant",
                        "content": (
                            "Please describe the problem — include device names, IP addresses, or what is failing.\n"
                            'Example: "Why can\'t 10.0.0.1 connect to 11.0.0.1?" or "arista1 is unreachable"'
                        ),
                    }
                }

        full_prompt, inc_summary = await resolve_incident_prompt(full_prompt)
        if inc_summary:
            src_ip_hint, dst_ip_hint = extract_ips(full_prompt)
            if src_ip_hint and dst_ip_hint:
                issue_type = "connectivity"

        config = {"configurable": {"session_id": session_id, "thread_id": session_id}}
        agent_input = full_prompt

        try:
            agent = build_agent(full_prompt, issue_type)
            result = await agent.ainvoke({"messages": [HumanMessage(content=agent_input)]}, config=config)
        except Exception as exc:
            metrics_recorder.increment("atlas.agent.failed", agent_type="troubleshoot", issue_type=issue_type)
            log_event(
                logger,
                "troubleshoot_agent_failed",
                level="error",
                request_id=request_id,
                session_id=session_id,
                issue_type=issue_type,
                error=str(exc),
            )
            logger.error("Troubleshoot agent failed: %s\n%s", exc, _tb.format_exc())
            nornir_client.clear_session_cache(session_id)
            return {"final_response": {"role": "assistant", "content": {"direct_answer": f"Troubleshooting failed: {exc}"}}}

        final_text = extract_final_text(result.get("messages", []))
        session_data = session_store.pop(session_id)

        src_ip, dst_ip = extract_ips(full_prompt)
        port = extract_port(full_prompt)
        if workflow_state_service.needs_connectivity_snapshot(session_data, src_ip, dst_ip):
            await status_service.push(session_id, "Gathering holistic connectivity evidence...")
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
            session_data = workflow_state_service.merge_session_data(session_data, session_store.pop(session_id))

        if workflow_state_service.missing_path_visuals(session_data, src_ip, dst_ip):
            await status_service.push(session_id, "Gathering required path visualizations...")
            try:
                await trace_path.ainvoke({"source_ip": src_ip, "dest_ip": dst_ip}, config=config)
                await trace_reverse_path.ainvoke({"source_ip": src_ip, "dest_ip": dst_ip}, config=config)
                session_data = workflow_state_service.merge_session_data(session_data, session_store.pop(session_id))
            except Exception as exc:
                logger.warning("mandatory path visualization collection failed: %s", exc)

        if memory_manager.should_trigger_recall_follow_up(session_data):
            recall_follow_up = memory_manager.build_recall_follow_up(full_prompt, session_data)
            if recall_follow_up:
                await status_service.push(session_id, "Checking similar past cases...")
                agent = build_agent(full_prompt, issue_type)
                result = await agent.ainvoke({"messages": [HumanMessage(content=recall_follow_up)]}, config=config)
                final_text = extract_final_text(result.get("messages", [])) or final_text
                session_data = workflow_state_service.merge_session_data(session_data, session_store.pop(session_id))

        content = response_presenter.build_troubleshoot_content(final_text, session_data, full_prompt, inc_summary)
        if final_text:
            await memory_manager.store_agent_memory_entry(full_prompt, final_text, agent_type="troubleshoot")

        logger.info(
            "troubleshoot done: keys=%s hops=%d counters=%d",
            list(content.keys()),
            len(content.get("path_hops", []) or []),
            len(content.get("interface_counters", []) or []),
        )
        duration_ms = elapsed_ms(started_at)
        metrics_recorder.increment("atlas.agent.completed", agent_type="troubleshoot", issue_type=issue_type)
        metrics_recorder.observe_ms("atlas.agent.duration_ms", duration_ms, agent_type="troubleshoot", issue_type=issue_type)
        log_event(
            logger,
            "troubleshoot_agent_completed",
            request_id=request_id,
            session_id=session_id,
            issue_type=issue_type,
            elapsed_ms=duration_ms,
            content_keys=list(content.keys()),
            path_hops=len(content.get("path_hops", []) or []),
            reverse_path_hops=len(content.get("reverse_path_hops", []) or []),
            interface_counter_rows=len(content.get("interface_counters", []) or []),
        )
        nornir_client.clear_session_cache(session_id)
        return {"final_response": {"role": "assistant", "content": content}}


troubleshoot_workflow_service = TroubleshootWorkflowService()
