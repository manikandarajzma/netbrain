"""Owned helpers for merging and validating transient workflow state."""
from __future__ import annotations

from typing import Any


class WorkflowStateService:
    """Owns session-side-effect merge behavior and workflow guard checks."""

    def merge_session_data(self, base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
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
            if key in {
                "all_interfaces",
                "interface_details",
                "syslog",
                "protocol_discovery",
                "routing_history",
                "connectivity_snapshot",
            }:
                existing = merged.get(key)
                if isinstance(existing, dict) and isinstance(value, dict):
                    merged[key] = {**existing, **value}
                elif value and not existing:
                    merged[key] = value
                continue
            if key not in merged or not merged.get(key):
                merged[key] = value
        return merged

    def missing_path_visuals(self, session_data: dict[str, Any], src_ip: str, dst_ip: str) -> bool:
        return bool(src_ip and dst_ip and (not session_data.get("path_hops") or not session_data.get("reverse_path_hops")))

    def needs_connectivity_snapshot(self, session_data: dict[str, Any], src_ip: str, dst_ip: str) -> bool:
        if not src_ip or not dst_ip:
            return False
        return not bool(session_data.get("connectivity_snapshot"))


workflow_state_service = WorkflowStateService()
