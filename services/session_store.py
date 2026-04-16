"""Owned per-session side-effect store for tool output."""
from __future__ import annotations

from typing import Any


class SessionStore:
    """Owns the transient per-session data written by workflow tools."""

    @staticmethod
    def _new_session_data() -> dict[str, Any]:
        return {
            "path_hops": [],
            "reverse_path_hops": [],
            "interface_counters": [],
            "routing_history": {},
            "ping_results": [],
            "peering_inspections": [],
            "all_interfaces": {},
            "interface_details": {},
            "syslog": {},
            "protocol_discovery": {},
            "connectivity_snapshot": {},
            "ip_owners": {},
            "servicenow_summary": "",
            "memory_recall_used": False,
            "memory_recall_signals": [],
        }

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}

    def get(self, session_id: str) -> dict[str, Any]:
        return self._sessions.setdefault(session_id, self._new_session_data())

    def set_servicenow_summary(self, session_id: str, summary: str) -> str:
        self.get(session_id)["servicenow_summary"] = summary
        return summary

    def pop(self, session_id: str) -> dict[str, Any]:
        return self._sessions.pop(session_id, {})

    def active_session_count(self) -> int:
        return len(self._sessions)


session_store = SessionStore()
