"""Top-level Atlas application owner."""
from __future__ import annotations

import logging
from time import perf_counter

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.services.diagnostics_service import diagnostics_service
    from atlas.services.device_diagnostics_service import device_diagnostics_service
    from atlas.services.graph_runtime import atlas_runtime
    from atlas.services.memory_manager import memory_manager
    from atlas.services.metrics import metrics_recorder
    from atlas.services.network_ops_workflow_service import network_ops_workflow_service
    from atlas.services.observability import elapsed_ms, log_event, new_request_id
    from atlas.services.routing_diagnostics_service import routing_diagnostics_service
    from atlas.services.response_presenter import response_presenter
    from atlas.services.session_store import session_store
    from atlas.services.status_service import status_service
    from atlas.services.troubleshoot_workflow_service import troubleshoot_workflow_service
    from atlas.services.workflow_state_service import workflow_state_service
    from atlas.tools.tool_registry import tool_registry
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from services.diagnostics_service import diagnostics_service  # type: ignore
    from services.device_diagnostics_service import device_diagnostics_service  # type: ignore
    from services.graph_runtime import atlas_runtime  # type: ignore
    from services.memory_manager import memory_manager  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.network_ops_workflow_service import network_ops_workflow_service  # type: ignore
    from services.observability import elapsed_ms, log_event, new_request_id  # type: ignore
    from services.routing_diagnostics_service import routing_diagnostics_service  # type: ignore
    from services.response_presenter import response_presenter  # type: ignore
    from services.session_store import session_store  # type: ignore
    from services.status_service import status_service  # type: ignore
    from services.troubleshoot_workflow_service import troubleshoot_workflow_service  # type: ignore
    from services.workflow_state_service import workflow_state_service  # type: ignore
    from tools.tool_registry import tool_registry  # type: ignore

logger = logging.getLogger("atlas.application")


class AtlasApplication:
    """Owns the runtime, agents, tools, memory, and presenters."""

    def __init__(self) -> None:
        self.runtime = atlas_runtime
        self.agent_factory = agent_factory
        self.device_diagnostics_service = device_diagnostics_service
        self.memory_manager = memory_manager
        self.network_ops_workflow_service = network_ops_workflow_service
        self.response_presenter = response_presenter
        self.routing_diagnostics_service = routing_diagnostics_service
        self.session_store = session_store
        self.status_service = status_service
        self.troubleshoot_workflow_service = troubleshoot_workflow_service
        self.tool_registry = tool_registry
        self.workflow_state_service = workflow_state_service
        self.diagnostics_service = diagnostics_service

    async def process_query(
        self,
        prompt: str,
        conversation_history: list[dict[str, str]],
        *,
        username: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        request_id = new_request_id()
        started_at = perf_counter()
        metrics_recorder.increment("atlas.query.started")
        log_event(
            logger,
            "query_started",
            request_id=request_id,
            session_id=session_id,
            username=username,
            prompt_chars=len(prompt or ""),
            history_messages=len(conversation_history or []),
        )
        result_state = await self.runtime.invoke_atlas_graph(
            prompt,
            conversation_history,
            username=username,
            session_id=session_id,
            request_id=request_id,
        )
        response = self.runtime.extract_final_response(result_state)
        content = response.get("content")
        content_type = "dict" if isinstance(content, dict) else "text"
        duration_ms = elapsed_ms(started_at)
        metrics_recorder.increment("atlas.query.completed", content_type=content_type)
        metrics_recorder.observe_ms("atlas.query.duration_ms", duration_ms, content_type=content_type)
        log_event(
            logger,
            "query_completed",
            request_id=request_id,
            session_id=session_id,
            elapsed_ms=duration_ms,
            response_keys=list(response.keys()),
            content_type=content_type,
        )
        return response

    async def get_diagnostics_snapshot(self) -> dict:
        return await self.diagnostics_service.build_snapshot()


atlas_application = AtlasApplication()
