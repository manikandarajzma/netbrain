"""
Atlas troubleshooting agent.

Exports build_agent(prompt, issue_type) which returns a ready-to-invoke
create_react_agent. All infrastructure (status bus, session data, response
formatting) lives in graph_nodes.py.
"""
from __future__ import annotations

import logging
import pathlib
import re

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    from atlas.tools.all_tools import ALL_TOOLS
    from atlas.tools.all_tools import CONNECTIVITY_TOOLS
    from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
except ImportError:
    from tools.all_tools import ALL_TOOLS          # type: ignore
    from tools.all_tools import CONNECTIVITY_TOOLS  # type: ignore
    from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL  # type: ignore

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


def build_agent(prompt: str = "", issue_type: str = "general"):
    """Return a create_react_agent ready for ainvoke."""
    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )
    system_prompt = load_system_prompt(prompt, issue_type)
    scenario_path = _pick_scenario(prompt, issue_type) or ""
    tools = CONNECTIVITY_TOOLS if scenario_path.endswith("connectivity.md") else ALL_TOOLS
    return create_react_agent(llm, tools, prompt=SystemMessage(content=system_prompt))
