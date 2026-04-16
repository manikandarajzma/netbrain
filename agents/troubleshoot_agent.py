"""
Atlas troubleshooting agent.

Exports build_agent(prompt, issue_type) which returns a pure specialized
ReAct agent. Infrastructure (status bus, session data, response formatting)
lives outside the agent layer.
"""
from __future__ import annotations

import logging
import pathlib
import re

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
    "blocked":      "connectivity.md",
    "connectivity": "connectivity.md",
    "slow":         "performance.md",
    "performance":  "performance.md",
    "intermittent": "intermittent.md",
    "flapping":     "intermittent.md",
}

_SCENARIO_KEYWORDS = [
    (re.compile(r"\b(block|deny|denied|reject|can.?t connect|port|tcp|udp|refused|unreachable)\b", re.IGNORECASE), "connectivity.md"),
    (re.compile(r"\b(slow|latency|lag|delay|degraded|throughput|performance|high rtt)\b",          re.IGNORECASE), "performance.md"),
    (re.compile(r"\b(intermittent|flap|unstable|sporadic|random|drops in and out)\b",              re.IGNORECASE), "intermittent.md"),
]

def _pick_scenario(prompt: str, issue_type: str) -> str | None:
    fname = _SCENARIO_FILES.get(issue_type)
    if not fname:
        for pattern, candidate in _SCENARIO_KEYWORDS:
            if pattern.search(prompt):
                fname = candidate
                break
    if not fname:
        return None
    path = _SCENARIOS_DIR / fname
    return str(path) if path.exists() else None


def load_system_prompt(prompt: str = "", issue_type: str = "general") -> str:
    core = _CORE_PROMPT.read_text(encoding="utf-8").strip() if _CORE_PROMPT.exists() else ""
    scenario_path = _pick_scenario(prompt, issue_type)
    if scenario_path:
        scenario_text = pathlib.Path(scenario_path).read_text(encoding="utf-8").strip()
        logger.info("Loaded scenario: %s", scenario_path)
        return core + "\n\n---\n\n" + scenario_text
    return core


def build_agent(prompt: str = "", issue_type: str = "general", *, llm=None):
    """Return a pure specialized troubleshoot agent ready for ainvoke."""
    llm = llm or agent_factory.build_default_llm()
    system_prompt = load_system_prompt(prompt, issue_type)
    scenario_path = _pick_scenario(prompt, issue_type) or ""
    tools = CONNECTIVITY_TOOLS if scenario_path.endswith("connectivity.md") else ALL_TOOLS
    return agent_factory.create_specialized_agent(llm, tools, system_prompt, "troubleshoot")
