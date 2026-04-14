"""
Shared pure agent factory for Atlas specialized ReAct agents.

This module keeps agent construction minimal and consistent:
- one place to create the default LLM
- one place to wire create_react_agent

Infrastructure concerns such as status updates, session state, response shaping,
and validation stay outside the agent layer.
"""
from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    from atlas.tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL
except ImportError:
    from tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL  # type: ignore


def build_default_llm() -> ChatOpenAI:
    """Return the default chat model used by Atlas agents."""
    return ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )


def create_specialized_agent(
    llm,
    tools,
    system_prompt: str,
    agent_name: str,
    *,
    checkpointer=None,
    stream_mode: str | list[str] | None = None,
    response_format=None,
    pre_model_hook=None,
    post_model_hook=None,
    state_schema=None,
    context_schema=None,
    interrupt_before=None,
    interrupt_after=None,
    debug: bool = False,
    version: str = "v2",
    **extra_kwargs: Any,
):
    """
    Create a minimal specialized ReAct agent.

    Notes:
    - The installed LangGraph version in this environment still uses `prompt=`
      rather than `state_modifier=`.
    - Any runtime/session/presentation behavior should be handled outside the
      agent, not here.
    """
    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=system_prompt),
        checkpointer=checkpointer,
        response_format=response_format,
        pre_model_hook=pre_model_hook,
        post_model_hook=post_model_hook,
        state_schema=state_schema,
        context_schema=context_schema,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        version=version,
        name=agent_name,
        **extra_kwargs,
    )
    if stream_mode is not None:
        setattr(agent, "atlas_stream_mode", stream_mode)
    return agent
