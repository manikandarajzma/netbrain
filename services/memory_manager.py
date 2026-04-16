"""Owned short-term and long-term memory helpers for Atlas."""
from __future__ import annotations

import asyncio
import re
from typing import Any

try:
    from atlas.services.pending_context import pending_context_store
except ImportError:
    from services.pending_context import pending_context_store  # type: ignore


class MemoryManager:
    """Owns pending clarification state and long-term memory storage hooks."""

    _HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?$")

    def set_pending_context(self, session_id: str, prompt: str, issue_type: str = "general") -> None:
        pending_context_store.set(session_id, prompt, issue_type)

    def get_pending_context(self, session_id: str) -> tuple[str | None, str | None]:
        return pending_context_store.get(session_id)

    def has_pending_context(self, session_id: str) -> bool:
        return pending_context_store.has(session_id)

    def clear_pending_context(self, session_id: str) -> None:
        pending_context_store.clear(session_id)

    def get_recall_signals(self, store: dict[str, Any]) -> list[str]:
        signals: list[str] = []

        path_flags = store.get("path_flags") or {}
        if any(
            path_flags.get(key)
            for key in (
                "no_route_device",
                "next_hop_resolution_failed",
                "mgmt_routing_detected",
                "missing_next_hop_device",
            )
        ):
            signals.append("path anomaly")

        for counter in store.get("interface_counters") or []:
            if isinstance(counter, dict):
                if counter.get("ssh_error"):
                    signals.append("interface counter collection failure")
                    break
                if counter.get("active"):
                    signals.append("active interface errors")
                    break

        interface_details = store.get("interface_details") or {}
        if isinstance(interface_details, dict):
            for detail in interface_details.values():
                if not isinstance(detail, dict):
                    continue
                if detail.get("error"):
                    signals.append("interface detail lookup failure")
                    break
                if str(detail.get("oper_status") or "").lower() not in {"", "up"}:
                    signals.append("interface state anomaly")
                    break
                if str(detail.get("line_protocol") or "").lower() not in {"", "up"}:
                    signals.append("line protocol anomaly")
                    break
                if any((detail.get(k) or 0) > 0 for k in ("input_errors", "output_errors", "input_discards", "output_discards")):
                    signals.append("interface error counters")
                    break

        syslog = store.get("syslog") or {}
        if isinstance(syslog, dict):
            for payload in syslog.values():
                if not isinstance(payload, dict):
                    continue
                if payload.get("error"):
                    signals.append("syslog collection failure")
                    break
                if payload.get("relevant"):
                    signals.append("recent device events")
                    break

        ospf_history = store.get("ospf_history") or {}
        if isinstance(ospf_history, dict):
            for info in ospf_history.values():
                if not isinstance(info, dict):
                    continue
                if info.get("error"):
                    signals.append("OSPF history lookup failure")
                    break
                counts = [snap.get("neighbor_count") for snap in info.get("history", []) if isinstance(snap, dict)]
                if counts and len(set(counts)) > 1:
                    signals.append("OSPF instability")
                    break

        for inspection in store.get("peering_inspections") or []:
            if not isinstance(inspection, dict):
                continue
            diagnosis = str(inspection.get("diagnosis_class") or "").strip().lower()
            if diagnosis and diagnosis not in {"healthy", "clean"}:
                signals.append("peer diagnosis anomaly")
                break
            if inspection.get("ping_a_success") is False or inspection.get("ping_b_success") is False:
                signals.append("peer reachability failure")
                break

        for ping in store.get("ping_results") or []:
            if isinstance(ping, dict) and ping.get("success") is False:
                signals.append("failed reachability test")
                break

        connectivity_snapshot = store.get("connectivity_snapshot") or {}
        if isinstance(connectivity_snapshot, dict):
            findings = connectivity_snapshot.get("findings") or []
            if findings:
                signals.append("unresolved connectivity findings")
            errors = connectivity_snapshot.get("errors") or {}
            if errors:
                signals.append("partial evidence gaps")
            service = connectivity_snapshot.get("service") or {}
            if isinstance(service, dict) and service and service.get("reachable") is False:
                signals.append("service reachability failure")

        deduped: list[str] = []
        for signal in signals:
            if signal not in deduped:
                deduped.append(signal)
        return deduped

    def should_recall(self, store: dict[str, Any]) -> bool:
        return bool(self.get_recall_signals(store))

    @classmethod
    def _is_device_name(cls, value: str) -> bool:
        value = str(value or "").strip()
        return bool(value) and bool(cls._HOSTNAME_RE.match(value)) and not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value)

    @classmethod
    def _extract_devices_from_hops(cls, hops: list[dict[str, Any]] | None) -> list[str]:
        devices: list[str] = []
        for hop in hops or []:
            if not isinstance(hop, dict):
                continue
            for key in ("from_device", "to_device"):
                value = str(hop.get(key) or "").strip()
                if cls._is_device_name(value) and value not in devices:
                    devices.append(value)
        return devices

    def get_recall_devices(self, store: dict[str, Any]) -> list[str]:
        devices: list[str] = []

        for device in (
            self._extract_devices_from_hops(store.get("path_hops") or [])
            + self._extract_devices_from_hops(store.get("reverse_path_hops") or [])
            + list((store.get("routing_history") or {}).get("historical_devices") or [])
        ):
            device = str(device or "").strip()
            if self._is_device_name(device) and device not in devices:
                devices.append(device)

        peer_hint = (store.get("routing_history") or {}).get("peer_hint") or {}
        for key in ("from_device", "to_device"):
            device = str(peer_hint.get(key) or "").strip()
            if self._is_device_name(device) and device not in devices:
                devices.append(device)

        return devices

    def should_trigger_recall_follow_up(self, store: dict[str, Any]) -> bool:
        if store.get("memory_recall_used"):
            return False
        return self.should_recall(store)

    def build_recall_follow_up(self, prompt: str, store: dict[str, Any]) -> str:
        signals = self.get_recall_signals(store)
        devices = self.get_recall_devices(store)
        if not signals:
            return ""

        device_list = ", ".join(devices) if devices else "none"
        return (
            f"{prompt}\n\n"
            "Additional required step before finalizing:\n"
            "- Live investigation produced evidence that may match prior cases.\n"
            f"- Recall signals: {', '.join(signals)}\n"
            f"- Candidate devices for recall: {device_list}\n"
            "- Call recall_similar_cases with the current issue description and the relevant devices.\n"
            "- Use historical results only as supporting context; keep live evidence primary.\n"
            "- If memory returns nothing useful, say so briefly and keep the answer grounded in live findings."
        )

    async def store_agent_memory_entry(self, prompt: str, final_text: str, agent_type: str = "troubleshoot") -> None:
        if not final_text:
            return
        try:
            try:
                from atlas.memory.agent_memory import store_memory
            except ImportError:
                from memory.agent_memory import store_memory  # type: ignore

            asyncio.create_task(store_memory(prompt, final_text, agent_type=agent_type))
        except Exception:
            pass


memory_manager = MemoryManager()
