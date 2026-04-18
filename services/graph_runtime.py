"""Graph execution helpers for Atlas chat entrypoints."""
from __future__ import annotations

import logging
from time import perf_counter
from typing import Any

try:
    from atlas.services.checkpointer_runtime import checkpointer_runtime
    from atlas.services.metrics import metrics_recorder
    from atlas.services.observability import elapsed_ms, log_event, new_request_id
except ImportError:
    from services.checkpointer_runtime import checkpointer_runtime  # type: ignore
    from services.metrics import metrics_recorder  # type: ignore
    from services.observability import elapsed_ms, log_event, new_request_id  # type: ignore


logger = logging.getLogger("atlas.graph_runtime")


class AtlasRuntime:
    """Owns graph execution state, config, invocation, and final payload extraction."""

    def build_initial_state(
        self,
        prompt: str,
        conversation_history: list[dict[str, str]],
        username: str | None,
        session_id: str | None,
        request_id: str | None,
        ui_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = {
            "prompt": prompt,
            "conversation_history": conversation_history or [],
            "username": username,
            "session_id": session_id,
            "request_id": request_id or new_request_id(),
            "intent": None,
            "rbac_error": None,
            "final_response": None,
        }
        if ui_action is not None:
            state["ui_action"] = ui_action
        return state

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
        request_id: str | None = None,
        ui_action: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started_at = perf_counter()
        await checkpointer_runtime.ensure_ready()
        try:
            from atlas.graph.graph_builder import graph_builder
        except ImportError:
            from graph.graph_builder import graph_builder  # type: ignore

        initial_state = self.build_initial_state(prompt, conversation_history, username, session_id, request_id, ui_action)
        request_id = initial_state.get("request_id")
        config = self.build_graph_config(session_id)
        metrics_recorder.increment("atlas.graph.invoke.started")
        log_event(
            logger,
            "graph_invoke_started",
            request_id=request_id,
            session_id=session_id,
            prompt_chars=len(prompt or ""),
            history_messages=len(conversation_history or []),
            recursion_limit=config.get("recursion_limit"),
        )
        result = await graph_builder.get_graph().ainvoke(initial_state, config=config)
        duration_ms = elapsed_ms(started_at)
        metrics_recorder.increment(
            "atlas.graph.invoke.completed",
            has_final_response=bool(isinstance(result, dict) and result.get("final_response")),
        )
        metrics_recorder.observe_ms("atlas.graph.invoke.duration_ms", duration_ms)
        log_event(
            logger,
            "graph_invoke_completed",
            request_id=request_id,
            session_id=session_id,
            elapsed_ms=duration_ms,
            state_keys=list(result.keys()) if isinstance(result, dict) else [],
            has_final_response=bool(isinstance(result, dict) and result.get("final_response")),
        )
        return result

    def extract_final_response(self, result_state: dict[str, Any]) -> dict[str, Any]:
        return result_state.get("final_response") or {
            "role": "assistant",
            "content": "Something went wrong — please try again.",
        }


atlas_runtime = AtlasRuntime()
