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
    from atlas.agents.agent_factory import build_default_llm, create_specialized_agent
    from atlas.tools.all_tools import NETWORK_OPS_TOOLS
except ImportError:
    from agents.agent_factory import build_default_llm, create_specialized_agent  # type: ignore
    from tools.all_tools import NETWORK_OPS_TOOLS          # type: ignore

logger = logging.getLogger("atlas.network_ops_agent")

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "network_ops.md"


def load_system_prompt() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


def build_agent(
    *,
    llm=None,
    checkpointer=None,
    stream_mode: str | list[str] | None = None,
    **agent_kwargs,
):
    """Return a pure specialized network-ops agent ready for ainvoke."""
    llm = llm or build_default_llm()
    return create_specialized_agent(
        llm,
        NETWORK_OPS_TOOLS,
        load_system_prompt(),
        "network_ops",
        checkpointer=checkpointer,
        stream_mode=stream_mode,
        **agent_kwargs,
    )
