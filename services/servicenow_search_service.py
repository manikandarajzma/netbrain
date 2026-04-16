"""Owned Atlas-specific ServiceNow correlation/search workflow."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

try:
    from atlas.mcp_client import call_mcp_tool
except ImportError:
    from mcp_client import call_mcp_tool  # type: ignore

logger = logging.getLogger("atlas.servicenow_search")

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9\-_\.]*[A-Za-z0-9])?$")


class ServiceNowSearchService:
    """Owns Atlas-specific incident/change correlation for troubleshoot workflows."""

    @staticmethod
    def _is_device_name(value: str) -> bool:
        return bool(value and _HOSTNAME_RE.match(value) and not re.match(r"^\d+\.\d+\.\d+\.\d+$", value))

    @classmethod
    def _extract_devices_from_hops(cls, hops: list[dict[str, Any]]) -> list[str]:
        devices: list[str] = []
        for hop in hops or []:
            if not isinstance(hop, dict):
                continue
            for key in ("from_device", "to_device"):
                device = str(hop.get(key) or "").strip()
                if cls._is_device_name(device) and device not in devices:
                    devices.append(device)
        return devices

    async def resolve_devices(
        self,
        *,
        store: dict[str, Any],
        device_names: list[str],
        dest_ip: str = "",
    ) -> list[str]:
        discovered_devices: list[str] = []

        for device in device_names or []:
            device = str(device or "").strip()
            if self._is_device_name(device) and device not in discovered_devices:
                discovered_devices.append(device)

        for device in self._extract_devices_from_hops(store.get("path_hops") or []):
            if device not in discovered_devices:
                discovered_devices.append(device)
        for device in self._extract_devices_from_hops(store.get("reverse_path_hops") or []):
            if device not in discovered_devices:
                discovered_devices.append(device)

        routing_history = store.get("routing_history") or {}
        for device in (routing_history.get("historical_devices") or []):
            device = str(device or "").strip()
            if self._is_device_name(device) and device not in discovered_devices:
                discovered_devices.append(device)

        peer_hint = routing_history.get("peer_hint") or {}
        for key in ("from_device", "to_device"):
            device = str(peer_hint.get(key) or "").strip()
            if self._is_device_name(device) and device not in discovered_devices:
                discovered_devices.append(device)

        if not discovered_devices and dest_ip:
            try:
                try:
                    from atlas.db import fetch as _fetch
                except ImportError:
                    from db import fetch as _fetch  # type: ignore
                hist_rows = await _fetch(
                    """
                    SELECT DISTINCT device FROM routing_history
                    WHERE $1::inet << prefix
                      AND egress_interface IS NOT NULL
                      AND egress_interface NOT ILIKE 'management%'
                    ORDER BY device
                    """,
                    dest_ip,
                )
                for row in hist_rows or []:
                    device = str((row or {}).get("device") or "").strip()
                    if self._is_device_name(device) and device not in discovered_devices:
                        discovered_devices.append(device)
            except Exception:
                pass

        return discovered_devices

    @staticmethod
    def build_terms(
        *,
        devices: list[str],
        source_ip: str = "",
        dest_ip: str = "",
        port: str = "",
    ) -> list[str]:
        terms = [str(d).strip() for d in devices if str(d or "").strip()]
        for ip in (source_ip, dest_ip):
            if ip and ip.strip():
                terms.append(ip.strip())
        if port and port.strip():
            terms.append(port.strip())
        return list(dict.fromkeys(terms))

    async def search_summary(
        self,
        *,
        session_id: str,
        devices: list[str],
        source_ip: str = "",
        dest_ip: str = "",
        port: str = "",
        hours_back: int = 24,
    ) -> str:
        terms = self.build_terms(devices=devices, source_ip=source_ip, dest_ip=dest_ip, port=port)
        if not terms:
            return "ServiceNow skipped: no device names or IPs provided."

        query = " OR ".join(terms)
        chg_hours = max(hours_back * 30, 720)

        try:
            inc_result, chg_result = await asyncio.wait_for(
                asyncio.gather(
                    call_mcp_tool(
                        "search_servicenow_incidents",
                        {"query": query, "limit": 10, "updated_within_hours": hours_back},
                        timeout=4.0,
                    ),
                    call_mcp_tool(
                        "list_servicenow_change_requests",
                        {"query": query, "limit": 20, "updated_within_hours": chg_hours},
                        timeout=4.0,
                    ),
                ),
                timeout=5.0,
            )
        except asyncio.TimeoutError:
            return "ServiceNow timed out; continuing without ticket context."
        except Exception as exc:
            return f"ServiceNow unavailable: {exc}"

        inc_error = inc_result.get("error") if isinstance(inc_result, dict) else None
        chg_error = chg_result.get("error") if isinstance(chg_result, dict) else None
        if inc_error or chg_error:
            parts = []
            if inc_error:
                parts.append(f"incident search failed: {inc_error}")
            if chg_error:
                parts.append(f"change search failed: {chg_error}")
            return "ServiceNow unavailable: " + "; ".join(parts)

        def _cell(value: Any, length: int = 60) -> str:
            text = str(value or "—").strip().replace("|", "/").replace("\n", " ")
            return text[:length] if len(text) > length else text

        def _fmt_incidents(rows: list[dict[str, Any]]) -> str:
            if not rows:
                return "No incidents found."
            lines = []
            for row in rows:
                ci = row.get("cmdb_ci") or {}
                ci_name = (ci.get("display_value") or ci.get("value") or "—") if isinstance(ci, dict) else str(ci or "—")
                lines.append(
                    f"- **{_cell(row.get('number'))}**\n"
                    f"  CI: {_cell(ci_name)}\n"
                    f"  Description: {_cell(row.get('short_description'), 120)}\n"
                    f"  State: {_cell(row.get('state'))}\n"
                    f"  Priority: {_cell(row.get('priority'))}"
                )
            return "\n".join(lines)

        def _fmt_changes(rows: list[dict[str, Any]]) -> str:
            if not rows:
                return "No change requests found."
            lines = []
            for row in rows:
                ci = row.get("cmdb_ci") or {}
                ci_name = (ci.get("display_value") or ci.get("value") or "—") if isinstance(ci, dict) else str(ci or "—")
                lines.append(
                    f"- **{_cell(row.get('number'))}**\n"
                    f"  CI: {_cell(ci_name)}\n"
                    f"  Description: {_cell(row.get('short_description'), 120)}\n"
                    f"  State: {_cell(row.get('state'))}\n"
                    f"  Risk: {_cell(row.get('risk'))}\n"
                    f"  Scheduled: {_cell((row.get('start_date') or '—')[:16])}"
                )
            return "\n".join(lines)

        inc_rows = (inc_result or {}).get("result", [])
        chg_rows = (chg_result or {}).get("result", [])

        logger.info(
            "servicenow_search_completed: session=%s devices=%s terms=%s incidents=%s changes=%s",
            session_id,
            devices,
            terms,
            len(inc_rows),
            len(chg_rows),
        )

        return (
            f"Incidents found: {len(inc_rows)}\n"
            f"Change requests found: {len(chg_rows)}\n\n"
            "### Incidents\n"
            f"{_fmt_incidents(inc_rows)}\n\n"
            "### Change Requests\n"
            f"{_fmt_changes(chg_rows)}"
        )


servicenow_search_service = ServiceNowSearchService()
