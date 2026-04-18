"""Compatibility wrapper around the generic spec-driven scenario selector."""
from __future__ import annotations

try:
    from atlas.agents.agent_registry import agent_registry
    from atlas.services.agent_scenario_service import agent_scenario_service
except ImportError:
    from agents.agent_registry import agent_registry  # type: ignore
    from services.agent_scenario_service import agent_scenario_service  # type: ignore


SPEC = agent_registry.get("network_ops")


class NetworkOpsScenarioService:
    """Compatibility wrapper for the network-ops scenario selector."""

    def _parse_decision(self, text: str):
        return agent_scenario_service._parse_decision(SPEC, text)

    async def select_scenario(self, prompt: str) -> str:
        return await agent_scenario_service.select_scenario(SPEC, prompt)


network_ops_scenario_service = NetworkOpsScenarioService()
