"""
Atlas troubleshooting agent.

Exports build_agent(prompt, scenario) which returns a pure specialized
ReAct agent. Infrastructure (status bus, session data, response formatting)
lives outside the agent layer.
"""
from __future__ import annotations

import logging
import pathlib

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.tools.tool_registry import tool_registry
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from tools.tool_registry import tool_registry  # type: ignore

logger = logging.getLogger("atlas.troubleshoot_agent")

GENERAL_PROFILE = "troubleshoot.general"
CONNECTIVITY_PROFILE = "troubleshoot.connectivity"

ALL_TOOLS = tool_registry.get_profile_tools(GENERAL_PROFILE)
CONNECTIVITY_TOOLS = tool_registry.get_profile_tools(CONNECTIVITY_PROFILE)

_SKILLS_DIR    = pathlib.Path(__file__).parent.parent / "skills"
_CORE_PROMPT   = _SKILLS_DIR / "troubleshooter.md"
_SCENARIOS_DIR = _SKILLS_DIR / "troubleshooting_scenarios"

_SCENARIO_FILES = {
    "connectivity": "connectivity.md",
    "performance": "performance.md",
    "intermittent": "intermittent.md",
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
        logger.info("Loaded scenario: %s", scenario_path)
        return core + "\n\n---\n\n" + scenario_text
    return core


def build_agent(prompt: str = "", scenario: str = "general", *, llm=None):
    """Return a pure specialized troubleshoot agent ready for ainvoke."""
    llm = llm or agent_factory.build_troubleshoot_llm()
    system_prompt = load_system_prompt(scenario)
    scenario_path = _get_scenario_path(scenario) or ""
    tools = CONNECTIVITY_TOOLS if scenario_path.endswith("connectivity.md") else ALL_TOOLS
    return agent_factory.create_specialized_agent(llm, tools, system_prompt, "troubleshoot")
