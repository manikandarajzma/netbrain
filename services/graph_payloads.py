"""Helpers for building LangGraph input state and extracting output payloads."""
from __future__ import annotations

from typing import Any


def build_initial_state(
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


def build_graph_config(session_id: str | None) -> dict[str, Any]:
    config: dict[str, Any] = {"recursion_limit": 50}
    if session_id:
        config["configurable"] = {"thread_id": session_id}
    return config


def extract_final_response(result_state: dict[str, Any]) -> dict[str, Any]:
    return result_state.get("final_response") or {
        "role": "assistant",
        "content": "Something went wrong — please try again.",
    }
