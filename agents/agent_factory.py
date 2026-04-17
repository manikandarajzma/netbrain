"""Shared pure agent factory for Atlas specialized ReAct agents."""
from __future__ import annotations

import os

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    from atlas.tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL
except ImportError:
    from tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL  # type: ignore


class AgentFactory:
    """Owns agent-model defaults and specialized agent creation."""

    def __init__(self) -> None:
        self.router_model = os.getenv("ATLAS_ROUTER_MODEL", "gemma4:latest")
        self.selector_model = os.getenv("ATLAS_SELECTOR_MODEL", "gemma4:latest")
        self.network_ops_model = os.getenv("ATLAS_NETWORK_OPS_MODEL", "gemma4:latest")
        self.troubleshoot_model = os.getenv("ATLAS_TROUBLESHOOT_MODEL", "gemma4:latest")

    @staticmethod
    def _build_llm(model: str) -> ChatOpenAI:
        return ChatOpenAI(
            model=model,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            api_key="docker",
        )

    def build_default_llm(self) -> ChatOpenAI:
        """Return the default chat model used by Atlas agents."""
        return self._build_llm(OLLAMA_MODEL)

    def build_router_llm(self) -> ChatOpenAI:
        """Return the model used for coarse request routing."""
        return self._build_llm(self.router_model)

    def build_selector_llm(self) -> ChatOpenAI:
        """Return the model used for scenario selection."""
        return self._build_llm(self.selector_model)

    def build_network_ops_llm(self) -> ChatOpenAI:
        """Return the model used for network-operations agents."""
        return self._build_llm(self.network_ops_model)

    def build_troubleshoot_llm(self) -> ChatOpenAI:
        """Return the model used for troubleshoot agents."""
        return self._build_llm(self.troubleshoot_model)

    def configured_models(self) -> dict[str, str]:
        """Return the active model assignment for Atlas runtime roles."""
        return {
            "default": OLLAMA_MODEL,
            "router": self.router_model,
            "selector": self.selector_model,
            "network_ops": self.network_ops_model,
            "troubleshoot": self.troubleshoot_model,
        }

    def create_specialized_agent(self, llm, tools, system_prompt: str, agent_name: str):
        """
        Create a minimal specialized ReAct agent.

        Notes:
        - The installed LangGraph version in this environment still uses `prompt=`
          rather than `state_modifier=`.
        - Runtime/session/presentation behavior stays outside the agent layer.
        """
        return create_react_agent(
            llm,
            tools,
            prompt=SystemMessage(content=system_prompt),
            name=agent_name,
        )


agent_factory = AgentFactory()
