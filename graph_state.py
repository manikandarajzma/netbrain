"""
Atlas LangGraph state definition.
"""
from typing import Any, Literal
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class AtlasState(TypedDict, total=False):
    # Input (set once at entry)
    prompt: str
    conversation_history: list[dict[str, str]]
    username: str | None
    session_id: str | None
    discover_only: bool
    prefilled_tool_name: str | None
    prefilled_tool_params: dict[str, Any] | None
    max_iterations: int

    # Routing signals
    intent: Literal["doc", "network", "prefilled", "dismiss", "risk", "netbrain", "troubleshoot", "servicenow"] | None
    rbac_error: str | None

    # LLM tool selection
    messages: list[BaseMessage]
    selected_tool_name: str | None
    selected_tool_args: dict[str, Any] | None
    tool_call_id: str | None
    iteration: int

    # Tool execution
    tool_raw_result: dict[str, Any] | str | None
    accumulated_results: list
    requires_site: bool
    tool_error: str | dict | None

    # Most recent follow_up_action from any tool result (set at graph entry, not scanned in nodes)
    last_follow_up_action: dict[str, Any] | None

    # Final answer
    final_response: dict[str, Any] | None

    # Conversation flow context — persisted across turns via LangGraph checkpointer.
    # "create_change_request" | "create_incident" | None
    active_flow: str | None
