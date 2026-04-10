"""
Atlas network operations agent.

Handles structured output workflows (firewall change requests, policy review,
access request documentation) using a restricted tool set.

Exports build_agent() which returns a ready-to-invoke create_react_agent.
All infrastructure (status bus, session data, response formatting) lives in
graph_nodes.py.
"""
from __future__ import annotations

import logging
import pathlib

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    from atlas.tools.all_tools import NETWORK_OPS_TOOLS
    from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
except ImportError:
    from tools.all_tools import NETWORK_OPS_TOOLS          # type: ignore
    from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL  # type: ignore

logger = logging.getLogger("atlas.network_ops_agent")

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "network_ops.md"


def load_system_prompt() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


def build_agent():
    """Return a create_react_agent ready for ainvoke."""
    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )
    return create_react_agent(llm, NETWORK_OPS_TOOLS, prompt=SystemMessage(content=load_system_prompt()))
