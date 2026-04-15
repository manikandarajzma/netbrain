"""Top-level Atlas application owner."""
from __future__ import annotations

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.services.graph_runtime import atlas_runtime
    from atlas.services.memory_manager import memory_manager
    from atlas.services.response_presenter import response_presenter
    from atlas.tools.tool_registry import tool_registry
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from services.graph_runtime import atlas_runtime  # type: ignore
    from services.memory_manager import memory_manager  # type: ignore
    from services.response_presenter import response_presenter  # type: ignore
    from tools.tool_registry import tool_registry  # type: ignore


class AtlasApplication:
    """Owns the runtime, agents, tools, memory, and presenters."""

    def __init__(self) -> None:
        self.runtime = atlas_runtime
        self.agent_factory = agent_factory
        self.memory_manager = memory_manager
        self.response_presenter = response_presenter
        self.tool_registry = tool_registry

    async def process_query(
        self,
        prompt: str,
        conversation_history: list[dict[str, str]],
        *,
        username: str | None = None,
        session_id: str | None = None,
    ) -> dict:
        result_state = await self.runtime.invoke_atlas_graph(
            prompt,
            conversation_history,
            username=username,
            session_id=session_id,
        )
        return self.runtime.extract_final_response(result_state)


atlas_application = AtlasApplication()
