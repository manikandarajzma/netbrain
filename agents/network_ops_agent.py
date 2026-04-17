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
import pathlib

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.tools.tool_registry import tool_registry
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from tools.tool_registry import tool_registry          # type: ignore

logger = logging.getLogger("atlas.network_ops_agent")

PROFILE_NAME = "network_ops"
NETWORK_OPS_TOOLS = tool_registry.get_profile_tools(PROFILE_NAME)

_SKILLS_DIR = pathlib.Path(__file__).parent.parent / "skills"
_CORE_PROMPT = _SKILLS_DIR / "network_ops.md"
_SCENARIOS_DIR = _SKILLS_DIR / "network_ops_scenarios"

_SCENARIO_FILES = {
    "incident_record": "incident_record.md",
    "record_lookup": "record_lookup.md",
    "change_record": "change_record.md",
    "change_update": "change_update.md",
    "access_change": "access_change.md",
}


def _get_scenario_path(scenario: str) -> str | None:
    fname = _SCENARIO_FILES.get(str(scenario or "").strip())
    if not fname:
        return None
    path = _SCENARIOS_DIR / fname
    return str(path) if path.exists() else None


def load_system_prompt(scenario: str = "general") -> str:
    core = _CORE_PROMPT.read_text(encoding="utf-8").strip() if _CORE_PROMPT.exists() else ""
    scenario_path = _get_scenario_path(scenario)
    if scenario_path:
        scenario_text = pathlib.Path(scenario_path).read_text(encoding="utf-8").strip()
        logger.info("Loaded network-ops scenario: %s", scenario_path)
        return core + "\n\n---\n\n" + scenario_text
    return core


def build_agent(prompt: str = "", scenario: str = "general", *, llm=None):
    """Return a pure specialized network-ops agent ready for ainvoke."""
    llm = llm or agent_factory.build_network_ops_llm()
    return agent_factory.create_specialized_agent(
        llm,
        NETWORK_OPS_TOOLS,
        load_system_prompt(scenario),
        "network_ops",
    )
