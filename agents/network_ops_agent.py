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

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "network_ops.md"


def load_system_prompt() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


def build_agent(*, llm=None):
    """Return a pure specialized network-ops agent ready for ainvoke."""
    llm = llm or agent_factory.build_default_llm()
    return agent_factory.create_specialized_agent(llm, NETWORK_OPS_TOOLS, load_system_prompt(), "network_ops")
