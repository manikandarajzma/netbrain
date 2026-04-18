"""
Atlas network operations agent.

Handles structured output workflows (firewall change requests, policy review,
access request documentation) using a restricted tool set.

Exports build_agent() which returns a pure specialized ReAct agent.
All infrastructure (status bus, session data, response formatting) lives
outside the agent layer.
"""
from __future__ import annotations

import logging

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.agents.agent_registry import agent_registry, build_agent_from_spec
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from agents.agent_registry import agent_registry, build_agent_from_spec  # type: ignore

logger = logging.getLogger("atlas.network_ops_agent")

SPEC_NAME = "network_ops"
SPEC = agent_registry.get(SPEC_NAME)
PROFILE_NAME = SPEC.default_tool_profile
NETWORK_OPS_TOOLS = SPEC.resolve_tools("", "general")
NETWORK_OPS_TOOLS_NO_PATH = SPEC.resolve_tools("ci name: arista-ai1", "record_lookup")


def _get_scenario_path(scenario: str) -> str | None:
    path = SPEC.resolve_scenario_path(scenario)
    return str(path) if path else None


def load_system_prompt(scenario: str = "general") -> str:
    scenario_path = _get_scenario_path(scenario)
    if scenario_path:
        logger.info("Loaded network-ops scenario: %s", scenario_path)
    return SPEC.load_system_prompt(scenario)


def _has_explicit_ci(prompt: str) -> bool:
    return SPEC.resolve_tool_profile(prompt, "incident_record") == "network_ops.no_path"


def _select_tools(prompt: str, scenario: str) -> tuple:
    return SPEC.resolve_tools(prompt, scenario)


def build_agent(prompt: str = "", scenario: str = "general", *, llm=None):
    """Return a pure specialized network-ops agent ready for ainvoke."""
    return build_agent_from_spec(
        SPEC,
        prompt=prompt,
        scenario=scenario,
        llm=llm,
        factory=agent_factory,
    )
