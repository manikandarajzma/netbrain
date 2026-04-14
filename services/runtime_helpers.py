"""Runtime helpers for status bus, session merge, and side effects."""
from __future__ import annotations

from typing import Any


def merge_session_data(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge per-pass tool side effects so follow-up passes don't erase earlier evidence."""
    merged = dict(base or {})
    for key, value in (new or {}).items():
        if key in {"path_hops", "reverse_path_hops", "interface_counters", "ping_results", "peering_inspections"}:
            existing = merged.get(key)
            if not existing:
                merged[key] = value
                continue
            if isinstance(existing, list) and isinstance(value, list):
                merged[key] = existing + [item for item in value if item not in existing]
            continue
        if key in {"all_interfaces", "interface_details", "syslog", "protocol_discovery", "routing_history", "connectivity_snapshot"}:
            existing = merged.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                merged[key] = {**existing, **value}
            elif value and not existing:
                merged[key] = value
            continue
        if key not in merged or not merged.get(key):
            merged[key] = value
    return merged


def missing_path_visuals(session_data: dict[str, Any], src_ip: str, dst_ip: str) -> bool:
    return bool(
        src_ip and dst_ip and (
            not session_data.get("path_hops") or not session_data.get("reverse_path_hops")
        )
    )


def needs_connectivity_snapshot(session_data: dict[str, Any], src_ip: str, dst_ip: str) -> bool:
    if not src_ip or not dst_ip:
        return False
    return not bool(session_data.get("connectivity_snapshot"))


async def push_status(session_id: str, message: str) -> None:
    try:
        try:
            import atlas.status_bus as status_bus
        except ImportError:
            import status_bus as status_bus  # type: ignore
        await status_bus.push(session_id, message)
    except Exception:
        pass


async def store_agent_memory_entry(prompt: str, final_text: str, agent_type: str = "troubleshoot") -> None:
    if not final_text:
        return
    try:
        try:
            from atlas.agent_memory import store_memory
        except ImportError:
            from agent_memory import store_memory  # type: ignore
        import asyncio

        asyncio.create_task(store_memory(prompt, final_text, agent_type=agent_type))
    except Exception:
        pass

