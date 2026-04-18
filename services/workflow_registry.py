"""Registry for agent workflow owners used by generic graph dispatch."""
from __future__ import annotations

from typing import Any

try:
    from atlas.agents.agent_registry import AgentSpec
    from atlas.services.network_ops_workflow_service import network_ops_workflow_service
    from atlas.services.troubleshoot_workflow_service import troubleshoot_workflow_service
except ImportError:
    from agents.agent_registry import AgentSpec  # type: ignore
    from services.network_ops_workflow_service import network_ops_workflow_service  # type: ignore
    from services.troubleshoot_workflow_service import troubleshoot_workflow_service  # type: ignore


WorkflowOwner = Any


class WorkflowRegistry:
    """Owns workflow-type-to-owner resolution for Atlas agents."""

    def __init__(self) -> None:
        self._registry: dict[str, WorkflowOwner] = {
            "troubleshoot": troubleshoot_workflow_service,
            "network_ops": network_ops_workflow_service,
        }

    def get(self, workflow_type: str) -> WorkflowOwner:
        try:
            return self._registry[workflow_type]
        except KeyError as exc:
            raise KeyError(f"Unknown workflow type: {workflow_type}") from exc

    async def run(self, spec: AgentSpec, state: dict[str, Any]) -> dict[str, Any]:
        owner = self.get(spec.workflow_type)
        return await owner.run(state)

    def describe(self) -> dict[str, str]:
        return {
            workflow_type: owner.__class__.__name__
            for workflow_type, owner in sorted(self._registry.items())
        }


workflow_registry = WorkflowRegistry()
