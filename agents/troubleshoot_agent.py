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
    from atlas.agents.agent_factory import build_default_llm, create_specialized_agent
    from atlas.tools.all_tools import ALL_TOOLS
    from atlas.tools.all_tools import CONNECTIVITY_TOOLS
except ImportError:
    from agents.agent_factory import build_default_llm, create_specialized_agent  # type: ignore
    from tools.all_tools import ALL_TOOLS          # type: ignore
    from tools.all_tools import CONNECTIVITY_TOOLS  # type: ignore

logger = logging.getLogger("atlas.troubleshoot_agent")

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
    llm = llm or build_default_llm()
    system_prompt = load_system_prompt(prompt, issue_type)
    scenario_path = _pick_scenario(prompt, issue_type) or ""
    tools = CONNECTIVITY_TOOLS if scenario_path.endswith("connectivity.md") else ALL_TOOLS
    return create_specialized_agent(llm, tools, system_prompt, "troubleshoot")
