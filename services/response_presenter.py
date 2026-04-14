"""Helpers to turn session/tool state into UI-facing response payloads."""
from __future__ import annotations

import re
from typing import Any

try:
    from atlas.services.request_preprocessor import extract_ips
except ImportError:
    from services.request_preprocessor import extract_ips  # type: ignore


_NETWORK_OPS_PATH_KEYWORDS = (
    "firewall",
    "policy",
    "rule",
    "path",
    "trace",
    "allow",
    "zone",
)

class ResponsePresenter:
    """Owns UI-facing response shaping for troubleshoot and network-ops flows."""

    def replace_markdown_section(self, text: str, header: str, body: str) -> str:
        if not text:
            return f"## {header}\n{body}"
        pattern = re.compile(rf"(?ms)^## {re.escape(header)}\n.*?(?=^## |\Z)")
        replacement = f"## {header}\n{body}\n"
        if pattern.search(text):
            return pattern.sub(replacement, text, count=1).rstrip()
        return f"{text.rstrip()}\n\n## {header}\n{body}"

    def group_interface_counters(self, entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        """Collapse per-interface counter rows into one payload per device."""
        if not entries:
            return []

        grouped: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            device = str(entry.get("device") or "Unknown device")
            current = grouped.setdefault(
                device,
                {
                    "device": device,
                    "window_s": 0,
                    "ssh_error": "",
                    "active_by_interface": {},
                    "clean_set": set(),
                },
            )

            current["window_s"] = max(current["window_s"], int(entry.get("window_s") or 0))
            if not current["ssh_error"] and entry.get("ssh_error"):
                current["ssh_error"] = str(entry["ssh_error"])

            for active in entry.get("active") or []:
                if not isinstance(active, dict):
                    continue
                interface = str(active.get("interface") or f"active-{len(current['active_by_interface'])}")
                current["active_by_interface"][interface] = active

            for clean in entry.get("clean") or []:
                if clean:
                    current["clean_set"].add(str(clean))

        normalized: list[dict[str, Any]] = []
        for device, payload in grouped.items():
            active = list(payload["active_by_interface"].values())
            active_interfaces = {
                str(item.get("interface"))
                for item in active
                if isinstance(item, dict) and item.get("interface")
            }
            clean = sorted(interface for interface in payload["clean_set"] if interface not in active_interfaces)
            normalized.append(
                {
                    "device": device,
                    "window_s": payload["window_s"],
                    "ssh_error": payload["ssh_error"],
                    "active": active,
                    "clean": clean,
                }
            )

        return sorted(normalized, key=lambda item: item.get("device", ""))

    def should_include_network_ops_path(self, prompt: str) -> bool:
        text = (prompt or "").lower()
        if re.search(
            r"\b(create|open|raise|close|update)\s+(an?\s+)?(incident|ticket|change request)\b",
            text,
            re.IGNORECASE,
        ):
            return False
        if re.search(r"\b(details?|status|show|get)\s+(about\s+)?(inc|chg)\d+\b", text, re.IGNORECASE):
            return False
        if re.search(r"\b(close|update)\s+(inc|chg)\d+\b", text, re.IGNORECASE):
            return False
        return any(keyword in text for keyword in _NETWORK_OPS_PATH_KEYWORDS)

    def build_troubleshoot_content(
        self,
        final_text: str,
        session_data: dict[str, Any],
        full_prompt: str,
        incident_summary: dict[str, Any] | None,
    ) -> dict[str, Any]:
        src_ip, dst_ip = extract_ips(full_prompt)
        path_hops = session_data.get("path_hops", [])
        reverse_path_hops = session_data.get("reverse_path_hops", [])
        interface_counters = self.group_interface_counters(session_data.get("interface_counters", []))
        connectivity_snapshot = session_data.get("connectivity_snapshot")
        servicenow_summary = session_data.get("servicenow_summary") or ""

        if final_text and servicenow_summary:
            final_text = self.replace_markdown_section(final_text, "ServiceNow", servicenow_summary)

        content: dict[str, Any] = {}
        if final_text:
            content["direct_answer"] = final_text
        if path_hops:
            content["path_hops"] = path_hops
            content["source"] = src_ip
            content["destination"] = dst_ip
        if reverse_path_hops:
            content["reverse_path_hops"] = reverse_path_hops
        if interface_counters:
            content["interface_counters"] = interface_counters
        if incident_summary:
            content["incident_summary"] = incident_summary
        if connectivity_snapshot:
            content["connectivity_snapshot"] = connectivity_snapshot
        return content

    def build_network_ops_content(
        self,
        final_text: str,
        session_data: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        src_ip, dst_ip = extract_ips(prompt)
        path_hops = session_data.get("path_hops", [])
        reverse_path_hops = session_data.get("reverse_path_hops", [])
        include_path = self.should_include_network_ops_path(prompt)

        content: dict[str, Any] = {}
        if final_text:
            content["direct_answer"] = final_text
        if include_path and path_hops:
            content["path_hops"] = path_hops
            content["source"] = src_ip
            content["destination"] = dst_ip
        if include_path and reverse_path_hops:
            content["reverse_path_hops"] = reverse_path_hops
        return content


response_presenter = ResponsePresenter()


def replace_markdown_section(text: str, header: str, body: str) -> str:
    return response_presenter.replace_markdown_section(text, header, body)


def group_interface_counters(entries: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return response_presenter.group_interface_counters(entries)


def should_include_network_ops_path(prompt: str) -> bool:
    return response_presenter.should_include_network_ops_path(prompt)


def build_troubleshoot_content(
    final_text: str,
    session_data: dict[str, Any],
    full_prompt: str,
    incident_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    return response_presenter.build_troubleshoot_content(
        final_text,
        session_data,
        full_prompt,
        incident_summary,
    )


def build_network_ops_content(
    final_text: str,
    session_data: dict[str, Any],
    prompt: str,
) -> dict[str, Any]:
    return response_presenter.build_network_ops_content(final_text, session_data, prompt)
