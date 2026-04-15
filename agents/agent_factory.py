"""Shared pure agent factory for Atlas specialized ReAct agents."""
from __future__ import annotations

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    from atlas.tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL
except ImportError:
    from tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL  # type: ignore


class AgentFactory:
    """Owns agent-model defaults and specialized agent creation."""

    def build_default_llm(self) -> ChatOpenAI:
        """Return the default chat model used by Atlas agents."""
        return ChatOpenAI(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.0,
            api_key="docker",
        )

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
