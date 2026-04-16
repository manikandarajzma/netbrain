"""Owned active-test and interface diagnostic workflow helpers."""
from __future__ import annotations

import asyncio
import json
from typing import Any

try:
    from atlas.services.backend_contracts import backend_unavailable
    from atlas.services.nornir_client import nornir_client
    from atlas.services.session_store import session_store
except ImportError:
    from services.backend_contracts import backend_unavailable  # type: ignore
    from services.nornir_client import nornir_client  # type: ignore
    from services.session_store import session_store  # type: ignore


class DeviceDiagnosticsService:
    """Owns live active tests and interface diagnostic summaries."""

    @staticmethod
    def _resolve_ping_source_interface(
        store: dict[str, Any],
        *,
        device: str,
        source_interface: str = "",
    ) -> str:
        if source_interface:
            return source_interface

        path_meta = store.get("path_meta") or {}
        reverse_meta = store.get("reverse_path_meta") or {}
        if device == path_meta.get("first_hop_device"):
            return str(path_meta.get("first_hop_lan_interface") or "").strip()
        if device == reverse_meta.get("reverse_first_hop_device"):
            return str(reverse_meta.get("reverse_first_hop_lan_interface") or "").strip()
        if device == path_meta.get("last_hop_device"):
            return str(path_meta.get("last_hop_egress_interface") or "").strip()
        return ""

    async def ping_summary(
        self,
        *,
        session_id: str,
        device: str,
        destination: str,
        source_interface: str = "",
        vrf: str = "",
    ) -> str:
        store = session_store.get(session_id)
        resolved_source_interface = self._resolve_ping_source_interface(
            store,
            device=device,
            source_interface=source_interface,
        )

        try:
            result = await nornir_client.post(
                "/ping",
                {
                    "device": device,
                    "destination": destination,
                    "source_interface": resolved_source_interface,
                    "vrf": vrf,
                },
                timeout=60.0,
            )
        except Exception as exc:
            result = {"success": False, "error": str(exc)}

        store["ping_result"] = result
        store["ping_results"].append(
            {
                "device": device,
                "destination": destination,
                "source_interface": resolved_source_interface,
                "vrf": vrf or "default",
                **result,
            }
        )

        src_note = f", source {resolved_source_interface}" if resolved_source_interface else ""
        if result.get("success"):
            rtt = result.get("rtt_avg_ms")
            rtt_str = f", RTT avg {rtt}ms" if rtt else ""
            return f"✓ Ping {device} → {destination} (VRF: {vrf or 'default'}{src_note}): SUCCESS, 0% loss{rtt_str}"
        loss = result.get("loss_pct", 100)
        err = result.get("error", "")
        return f"✗ Ping {device} → {destination} (VRF: {vrf or 'default'}{src_note}): FAILED — {loss}% packet loss{(' — ' + err) if err else ''}"

    async def tcp_test_summary(
        self,
        *,
        session_id: str,
        device: str,
        destination: str,
        port: str,
        vrf: str = "",
    ) -> str:
        try:
            result = await nornir_client.post(
                "/tcp-test",
                {"device": device, "destination": destination, "port": int(port), "vrf": vrf},
                timeout=30.0,
            )
        except Exception as exc:
            result = {"reachable": False, "error": str(exc)}

        session_store.get(session_id)["tcp_result"] = result

        if result.get("reachable"):
            return f"✓ TCP {destination}:{port} is reachable from {device} — service is accepting connections."
        err = result.get("error", "connection refused or timed out")
        return f"✗ TCP {destination}:{port} is NOT reachable from {device} — {err}"

    async def routing_check_summary(
        self,
        *,
        devices: list[str],
        destination: str,
        vrf: str = "",
    ) -> str:
        try:
            result = await nornir_client.post(
                "/routing-check",
                {"devices": devices, "destination": destination, "vrf": vrf},
                timeout=60.0,
            )
        except Exception as exc:
            return backend_unavailable("Nornir", "routing check", exc)

        lines = [f"Routing check for {destination}:"]
        for device, info in (result.get("hops") or {}).items():
            if not info.get("found"):
                lines.append(f"  {device}: ✗ no route — {info.get('error', 'no match')}")
            else:
                lines.append(
                    f"  {device}: ✓ via {info.get('next_hop', 'directly connected')} "
                    f"egress {info.get('interface', '?')} ({info.get('protocol', '?')}, VRF {info.get('vrf', 'default')})"
                )
        return "\n".join(lines)

    async def interface_counters_summary(
        self,
        *,
        session_id: str,
        devices_and_interfaces: list[dict[str, Any]],
    ) -> str:
        valid = []
        for entry in devices_and_interfaces:
            if not isinstance(entry, dict):
                continue
            device = str(entry.get("device", "")).strip()
            interfaces = entry.get("interfaces")
            if not device or not interfaces:
                continue
            valid.append({"device": device, "interfaces": interfaces})
        if not valid:
            return "No interface data available for counter polling."

        async def _fetch_one(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            device = entry["device"]
            interfaces = entry["interfaces"]
            try:
                data = await nornir_client.post(
                    "/interface-counters",
                    {"device": device, "interfaces": interfaces},
                    timeout=15.0,
                )
            except Exception as exc:
                return f"{device}: unreachable ({exc})", {
                    "device": device,
                    "ssh_error": str(exc),
                    "active": [],
                    "clean": [],
                }

            if "error" in data:
                return f"{device}: {data['error']}", {
                    "device": device,
                    "ssh_error": data["error"],
                    "active": [],
                    "clean": [],
                }

            active = data.get("active_errors", [])
            clean = data.get("clean_interfaces", [])
            interval = data.get("poll_interval_s", 3)
            iters = data.get("iterations", 3)
            window = interval * (iters - 1)
            structured = {"device": device, "window_s": window, "active": active, "clean": clean}

            if not active:
                return f"{device}: all interfaces clean over {window}s", structured
            rows = []
            for counter in active:
                interface = counter.get("interface", "?")
                deltas = counter.get("delta_9s", {})
                parts = [f"{key}+{value}" for key, value in deltas.items() if value > 0]
                rows.append(f"  {interface}: ACTIVE — {', '.join(parts)} over {window}s")
            return f"{device}:\n" + "\n".join(rows), structured

        results = await asyncio.gather(*[_fetch_one(entry) for entry in valid])

        store = session_store.get(session_id)
        lines: list[str] = []
        for text, structured in results:
            if text:
                lines.append(text)
            if structured and structured.get("device"):
                store["interface_counters"].append(structured)

        return "Interface counters:\n" + "\n".join(lines) if lines else "No counter data."

    async def interface_detail_summary(
        self,
        *,
        session_id: str,
        device: str,
        interface: str,
    ) -> str:
        try:
            data = await nornir_client.post(
                "/interface-detail",
                {"device": device, "interface": interface},
                timeout=15.0,
            )
        except Exception as exc:
            return backend_unavailable("Nornir", "interface detail lookup", exc, subject=f"{device}/{interface}")

        session_store.get(session_id)["interface_details"][f"{device}:{interface}"] = data
        return json.dumps(data, indent=2)

    async def all_interfaces_summary(self, *, session_id: str, device: str) -> str:
        try:
            data = await nornir_client.post("/all-interfaces-status", {"device": device}, timeout=15.0)
        except Exception as exc:
            return backend_unavailable("Nornir", "interface inventory lookup", exc, subject=device)

        session_store.get(session_id)["all_interfaces"][device] = data
        interfaces = data.get("interfaces", [])
        if not interfaces:
            return f"No interface data returned for {device}."

        down = [interface for interface in interfaces if not interface.get("up")]
        lines = [f"{device}: {len(interfaces)} interfaces, {len(down)} DOWN"]
        for interface in interfaces:
            oper = interface.get("oper_status", "")
            ip = interface.get("primary_ip")
            prefix_len = interface.get("prefix_len")
            ip_text = f" ip {ip}/{prefix_len}" if ip and prefix_len is not None else ""
            desc_text = f" ({interface['description']})" if interface.get("description") else ""
            if interface.get("up"):
                if ip:
                    lines.append(f"  ✓ {interface['interface']} — up{ip_text}{desc_text}")
                continue
            status = "ADMIN-DOWN" if oper in ("disabled", "adminDown") else f"link-down ({oper})"
            lines.append(f"  ✗ {interface['interface']} — {status}{ip_text}{desc_text}")
        return "\n".join(lines)


device_diagnostics_service = DeviceDiagnosticsService()
