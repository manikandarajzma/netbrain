"""
Atlas troubleshooting agent.

Exports build_agent(prompt, scenario) which returns a pure specialized
ReAct agent. Infrastructure (status bus, session data, response formatting)
lives outside the agent layer.
"""
from __future__ import annotations

import logging

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.agents.agent_registry import agent_registry, build_agent_from_spec
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from agents.agent_registry import agent_registry, build_agent_from_spec  # type: ignore

logger = logging.getLogger("atlas.troubleshoot_agent")

SPEC_NAME = "troubleshoot"
SPEC = agent_registry.get(SPEC_NAME)

GENERAL_PROFILE = SPEC.default_tool_profile
CONNECTIVITY_PROFILE = SPEC.tool_profile_rules[0].profile_name

ALL_TOOLS = SPEC.resolve_tools("", "general")
CONNECTIVITY_TOOLS = SPEC.resolve_tools("", "connectivity")


def _get_scenario_path(scenario: str) -> str | None:
    path = SPEC.resolve_scenario_path(scenario)
    return str(path) if path else None


def load_system_prompt(scenario: str = "general") -> str:
    scenario_path = _get_scenario_path(scenario)
    if scenario_path:
        logger.info("Loaded scenario: %s", scenario_path)
    return SPEC.load_system_prompt(scenario)


def build_agent(prompt: str = "", scenario: str = "general", *, llm=None):
    """Return a pure specialized troubleshoot agent ready for ainvoke."""
    return build_agent_from_spec(
        SPEC,
        prompt=prompt,
        scenario=scenario,
        llm=llm,
        factory=agent_factory,
    )
