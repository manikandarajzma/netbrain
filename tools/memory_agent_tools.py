"""Agent-facing memory tools for historical case recall."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.services.memory_manager import memory_manager
    from atlas.services.session_store import session_store
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from services.memory_manager import memory_manager  # type: ignore
    from services.session_store import session_store  # type: ignore
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


@tool
async def recall_similar_cases(
    query: str,
    devices: list[str],
    config: RunnableConfig,
) -> str:
    """
    Search long-term memory for past troubleshooting sessions and closed incidents
    semantically similar to the current query.

    Use this only after the current investigation has produced live evidence
    that suggests recurrence, instability, or an unresolved pattern.

    Args:
        query:   The current issue description (e.g. "10.0.100.100 can't reach 10.0.200.200 port 443").
        devices: Device hostnames in the path — used to match device-tagged incidents.
    """
    session_id = sid_from_config(config)
    store = session_store.get(session_id)
    signals = memory_manager.get_recall_signals(store)
    if not signals:
        return (
            "Memory recall deferred: gather live evidence first. "
            "Use recall only after live results show recurrence, instability, or an unresolved pattern."
        )

    await push_status(session_id, "Searching past cases...")

    try:
        try:
            from atlas.agent_memory import recall_memory, recall_incidents_by_devices, format_memory_context
        except ImportError:
            from agent_memory import recall_memory, recall_incidents_by_devices, format_memory_context  # type: ignore

        past_sessions = await recall_memory(query, agent_type="troubleshoot", top_k=3)
        past_incidents = await recall_incidents_by_devices(devices, query=query, top_k=5) if devices else []

        combined = past_sessions + [i for i in past_incidents if i not in past_sessions]
        if not combined:
            return f"No similar past cases found in memory for signals: {', '.join(signals)}."
        context = format_memory_context(combined)
        return f"Memory recall triggered by live signals: {', '.join(signals)}.\n\n{context}"
    except Exception as exc:
        return f"Memory recall unavailable: {exc}"


MEMORY_TOOL_CAPABILITIES = (
    (recall_similar_cases, ("memory.recall",)),
)
