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
import re

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

_ACCESS_CHANGE_RE = re.compile(
    r"\b(open\s+port|need\s+port|whitelist|allow\s+access|grant\s+access|"
    r"(add|create|remove|delete|update|modify)\s+(a\s+)?(firewall\s+)?rule|"
    r"new\s+rule|firewall\s+policy|security\s+policy|fw\s+policy|allow\s+traffic|"
    r"block\s+traffic|permit\s+traffic|deny\s+traffic)\b",
    re.IGNORECASE,
)
_CHANGE_RECORD_RE = re.compile(
    r"\b(create|open|submit|raise)\s+(a\s+)?change\s+request\b|\bchange\s+request\b",
    re.IGNORECASE,
)


def _pick_scenario(prompt: str) -> str | None:
    text = prompt or ""
    if _ACCESS_CHANGE_RE.search(text):
        path = _SCENARIOS_DIR / "access_change.md"
        return str(path) if path.exists() else None
    if _CHANGE_RECORD_RE.search(text):
        path = _SCENARIOS_DIR / "change_record.md"
        return str(path) if path.exists() else None
    return None


def load_system_prompt(prompt: str = "") -> str:
    core = _CORE_PROMPT.read_text(encoding="utf-8").strip() if _CORE_PROMPT.exists() else ""
    scenario_path = _pick_scenario(prompt)
    if scenario_path:
        scenario_text = pathlib.Path(scenario_path).read_text(encoding="utf-8").strip()
        logger.info("Loaded network-ops scenario: %s", scenario_path)
        return core + "\n\n---\n\n" + scenario_text
    return core


def build_agent(prompt: str = "", *, llm=None):
    """Return a pure specialized network-ops agent ready for ainvoke."""
    llm = llm or agent_factory.build_default_llm()
    return agent_factory.create_specialized_agent(llm, NETWORK_OPS_TOOLS, load_system_prompt(prompt), "network_ops")
