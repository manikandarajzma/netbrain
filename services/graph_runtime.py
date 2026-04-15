"""Graph execution helpers for Atlas chat entrypoints."""
from __future__ import annotations

from typing import Any

try:
    from atlas.services.checkpointer_runtime import ensure_checkpointer
except ImportError:
    from services.checkpointer_runtime import ensure_checkpointer  # type: ignore


class AtlasRuntime:
    """Owns graph execution state, config, invocation, and final payload extraction."""

    def build_initial_state(
        self,
        prompt: str,
        conversation_history: list[dict[str, str]],
        username: str | None,
        session_id: str | None,
    ) -> dict[str, Any]:
        return {
            "prompt": prompt,
            "conversation_history": conversation_history or [],
            "username": username,
            "session_id": session_id,
            "intent": None,
            "rbac_error": None,
            "final_response": None,
        }

    def build_graph_config(self, session_id: str | None) -> dict[str, Any]:
        config: dict[str, Any] = {"recursion_limit": 50}
        if session_id:
            config["configurable"] = {"thread_id": session_id}
        return config

    async def invoke_atlas_graph(
        self,
        prompt: str,
        conversation_history: list[dict[str, str]],
        *,
        username: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        await ensure_checkpointer()
        from atlas.graph_builder import atlas_graph

        initial_state = self.build_initial_state(prompt, conversation_history, username, session_id)
        config = self.build_graph_config(session_id)
        return await atlas_graph.ainvoke(initial_state, config=config)

    def extract_final_response(self, result_state: dict[str, Any]) -> dict[str, Any]:
        return result_state.get("final_response") or {
            "role": "assistant",
            "content": "Something went wrong — please try again.",
        }


atlas_runtime = AtlasRuntime()
