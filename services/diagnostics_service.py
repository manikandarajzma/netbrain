"""Owned diagnostics snapshot service for internal Atlas inspection."""
from __future__ import annotations

from typing import Any

try:
    from atlas.agents.agent_registry import agent_registry
    from atlas.services.checkpointer_runtime import checkpointer_runtime
    from atlas.services.health_service import health_service
    from atlas.services.metrics import metrics_recorder
    from atlas.services.session_store import session_store
    from atlas.services.workflow_registry import workflow_registry
    from atlas.tools.tool_registry import tool_registry
except ImportError:
    from agents.agent_registry import agent_registry  # type: ignore
    from services.checkpointer_runtime import checkpointer_runtime  # type: ignore
    from services.health_service import health_service  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.session_store import session_store  # type: ignore
    from services.workflow_registry import workflow_registry  # type: ignore
    from tools.tool_registry import tool_registry  # type: ignore


class DiagnosticsService:
    """Owns inspectable runtime diagnostics for internal/admin use."""

    async def build_snapshot(self) -> dict[str, Any]:
        profiles = tool_registry.describe_profiles()
        registered_tools = tool_registry.describe_registered_tools()
        agents = agent_registry.describe()
        health = await health_service.build_snapshot()
        return {
            "owners": {
                "application": "AtlasApplication",
                "runtime": "AtlasRuntime",
                "agent_factory": "AgentFactory",
                "agent_registry": "AgentRegistry",
                "workflow_registry": "WorkflowRegistry",
                "device_diagnostics_service": "DeviceDiagnosticsService",
                "troubleshoot_workflow_service": "TroubleshootWorkflowService",
                "network_ops_workflow_service": "NetworkOpsWorkflowService",
                "response_presenter": "ResponsePresenter",
                "memory_manager": "MemoryManager",
                "routing_diagnostics_service": "RoutingDiagnosticsService",
                "status_service": "StatusService",
                "session_store": "SessionStore",
                "tool_registry": "ToolRegistry",
                "workflow_state_service": "WorkflowStateService",
                "diagnostics_service": "DiagnosticsService",
            },
            "runtime": {
                "checkpointer": checkpointer_runtime.get_status(),
                "active_tool_sessions": session_store.active_session_count(),
            },
            "health": health,
            "agents": {
                "count": len(agents),
                "registered": agents,
            },
            "workflows": workflow_registry.describe(),
            "tools": {
                "profile_count": len(profiles),
                "registered_tool_count": len(registered_tools),
                "profiles": profiles,
                "registered": registered_tools,
            },
            "metrics": metrics_recorder.snapshot(),
        }


diagnostics_service = DiagnosticsService()
