"""Owned routing and OSPF diagnostic workflow helpers."""
from __future__ import annotations

import asyncio
import datetime
import re
from typing import Any

try:
    from atlas.services.backend_contracts import backend_unavailable
    from atlas.services.nornir_client import nornir_client
    from atlas.services.observability import json_safe
    from atlas.services.session_store import session_store
except ImportError:
    from services.backend_contracts import backend_unavailable  # type: ignore
    from services.nornir_client import nornir_client  # type: ignore
    from services.observability import json_safe  # type: ignore
    from services.session_store import session_store  # type: ignore


class RoutingDiagnosticsService:
    """Owns OSPF/routing/syslog diagnostic workflow and structured side effects."""

    @staticmethod
    def _pick_interface(data: dict[str, Any], interface: str) -> dict[str, Any]:
        for row in data.get("interfaces", []) or []:
            if row.get("interface") == interface:
                return row
        return {}

    @staticmethod
    def _state_line(device: str, interface: str, row: dict[str, Any], detail: dict[str, Any]) -> str:
        ip = row.get("primary_ip")
        plen = row.get("prefix_len")
        ip_text = f" {ip}/{plen}" if ip and plen is not None else ""
        admin = detail.get("oper_status") or row.get("oper_status") or "unknown"
        proto = detail.get("line_protocol") or row.get("line_protocol") or "unknown"
        state = "UP" if row.get("up") else f"DOWN ({admin})"
        return f"  {device} {interface}:{ip_text} state={state}, line_protocol={proto}"

    async def device_syslog_summary(self, *, session_id: str, device: str, interface: str = "") -> str:
        try:
            data = await nornir_client.post("/show-logging", {"device": device, "lines": 100}, timeout=15.0)
        except Exception as exc:
            return backend_unavailable("Nornir", "syslog lookup", exc, subject=device)

        logs = data.get("logs", [])
        kw = ["link", "down", "flap", "err-disable", "lineproto", "ospf", "adjacency"]
        if interface:
            short = interface.replace("GigabitEthernet", "Gi").replace("Ethernet", "Et")
            relevant = [
                l
                for l in logs
                if interface.lower() in l.lower() or short.lower() in l.lower() or any(k in l.lower() for k in kw)
            ][-20:]
        else:
            relevant = [l for l in logs if any(k in l.lower() for k in kw)][-30:]

        if not relevant:
            return f"{device}: no relevant syslog events found."

        store = session_store.get(session_id)
        interface_inventory = (store.get("all_interfaces") or {}).get(device, {})
        interface_rows = interface_inventory.get("interfaces", []) if isinstance(interface_inventory, dict) else []
        ip_to_interface: dict[str, dict[str, Any]] = {}
        for row in interface_rows:
            ip = str(row.get("primary_ip") or "").strip()
            if ip:
                ip_to_interface[ip] = row

        correlations: list[dict[str, Any]] = []
        correlation_lines: list[str] = []
        for line in relevant:
            lower = line.lower()
            if "adjacency" not in lower and "ospf" not in lower:
                continue
            for ip in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", line):
                row = ip_to_interface.get(ip)
                if not row:
                    continue
                oper = row.get("oper_status", "unknown")
                state = "up" if row.get("up") else f"down ({oper})"
                correlation = {
                    "ip": ip,
                    "interface": row.get("interface"),
                    "up": row.get("up"),
                    "oper_status": oper,
                    "line": line,
                }
                if correlation not in correlations:
                    correlations.append(correlation)
                    correlation_lines.append(
                        f"  Correlated OSPF syslog IP {ip} -> {row.get('interface')} ({state})"
                    )

        store["syslog"][device] = {
            "logs": logs,
            "relevant": relevant,
            "interface": interface,
            "correlations": correlations,
        }
        body = [f"{device} syslog:"]
        body.extend(f"  {l}" for l in relevant)
        if correlation_lines:
            body.append("  OSPF interface correlation:")
            body.extend(correlation_lines)
        return "\n".join(body)

    async def inspect_ospf_peering_summary(
        self,
        *,
        session_id: str,
        device_a: str,
        interface_a: str,
        device_b: str,
        interface_b: str,
        ip_a: str = "",
        ip_b: str = "",
    ) -> str:
        store = session_store.get(session_id)

        async def _all_interfaces(device: str) -> dict[str, Any]:
            try:
                data = await nornir_client.post("/all-interfaces-status", {"device": device}, timeout=15.0)
            except Exception as exc:
                data = {"device": device, "interfaces": [], "error": str(exc)}
            store["all_interfaces"][device] = data
            return data

        async def _interface_detail(device: str, interface: str) -> dict[str, Any]:
            try:
                data = await nornir_client.post("/interface-detail", {"device": device, "interface": interface}, timeout=15.0)
            except Exception as exc:
                data = {"device": device, "interface": interface, "error": str(exc)}
            store["interface_details"][f"{device}:{interface}"] = data
            return data

        async def _interface_counters(device: str, interface: str) -> dict[str, Any]:
            try:
                data = await nornir_client.post("/interface-counters", {"device": device, "interfaces": [interface]}, timeout=20.0)
            except Exception as exc:
                return {"device": device, "ssh_error": str(exc), "active_errors": [], "clean_interfaces": []}

            structured = {
                "device": device,
                "window_s": data.get("poll_interval_s", 3) * max(0, data.get("iterations", 3) - 1),
                "active": data.get("active_errors", []),
                "clean": data.get("clean_interfaces", []),
            }
            store["interface_counters"].append(structured)
            return data

        async def _syslog(device: str) -> dict[str, Any]:
            try:
                data = await nornir_client.post("/show-logging", {"device": device, "lines": 100}, timeout=15.0)
            except Exception as exc:
                data = {"device": device, "logs": [], "relevant": [], "error": str(exc)}
            logs = data.get("logs", [])
            kw = ["link", "down", "flap", "err-disable", "lineproto", "ospf", "adjacency"]
            relevant = [l for l in logs if any(k in l.lower() for k in kw)][-30:]
            return {"logs": logs, "relevant": relevant, **({"error": data["error"]} if data.get("error") else {})}

        async def _ping(device: str, destination: str, source_interface: str) -> dict[str, Any]:
            try:
                result = await nornir_client.post(
                    "/ping",
                    {"device": device, "destination": destination, "source_interface": source_interface, "vrf": ""},
                    timeout=30.0,
                )
            except Exception as exc:
                result = {"success": False, "error": str(exc), "device": device, "destination": destination}
            store["ping_results"].append({
                "device": device,
                "destination": destination,
                "source_interface": source_interface,
                "vrf": "default",
                **result,
            })
            return result

        all_a, all_b, detail_a, detail_b, counters_a, counters_b, syslog_a, syslog_b = await asyncio.gather(
            _all_interfaces(device_a),
            _all_interfaces(device_b),
            _interface_detail(device_a, interface_a),
            _interface_detail(device_b, interface_b),
            _interface_counters(device_a, interface_a),
            _interface_counters(device_b, interface_b),
            _syslog(device_a),
            _syslog(device_b),
        )

        row_a = self._pick_interface(all_a, interface_a)
        row_b = self._pick_interface(all_b, interface_b)
        live_ip_a = str(row_a.get("primary_ip") or ip_a or "").strip()
        live_ip_b = str(row_b.get("primary_ip") or ip_b or "").strip()

        def _correlate_syslog(device: str, row: dict[str, Any], syslog_data: dict[str, Any]) -> list[str]:
            ip = str(row.get("primary_ip") or "").strip()
            if not ip:
                return []
            lines = []
            for line in syslog_data.get("relevant", []) or []:
                if ip in line and ("adjacency" in line.lower() or "ospf" in line.lower()):
                    oper = row.get("oper_status", "unknown")
                    state = "up" if row.get("up") else f"down ({oper})"
                    lines.append(f"{device}: syslog local IP {ip} belongs to {row.get('interface')} ({state})")
            return lines

        correlations = _correlate_syslog(device_a, row_a, syslog_a) + _correlate_syslog(device_b, row_b, syslog_b)
        store["syslog"][device_a] = {**syslog_a, "interface": interface_a, "correlations": correlations}
        store["syslog"][device_b] = {**syslog_b, "interface": interface_b, "correlations": correlations}

        ping_a = ping_b = None
        if live_ip_b and row_a.get("up"):
            ping_a = await _ping(device_a, live_ip_b, interface_a)
        if live_ip_a and row_b.get("up"):
            ping_b = await _ping(device_b, live_ip_a, interface_b)

        lines = [
            f"OSPF peering inspection for {device_a} {interface_a} <-> {device_b} {interface_b}:",
            self._state_line(device_a, interface_a, row_a, detail_a),
            self._state_line(device_b, interface_b, row_b, detail_b),
        ]
        for line in correlations:
            lines.append(f"  {line}")

        if ping_a is not None:
            lines.append(
                f"  Ping {device_a} {interface_a} -> {live_ip_b}: "
                f"{'SUCCESS' if ping_a.get('success') else 'FAILED'}"
            )
        if ping_b is not None:
            lines.append(
                f"  Ping {device_b} {interface_b} -> {live_ip_a}: "
                f"{'SUCCESS' if ping_b.get('success') else 'FAILED'}"
            )

        both_down = row_a.get("up") is False and row_b.get("up") is False
        one_down = row_a.get("up") is False or row_b.get("up") is False
        active_a = counters_a.get("active_errors", []) or []
        active_b = counters_b.get("active_errors", []) or []
        diagnosis_class = "unknown"
        recommended_action = ""

        if both_down:
            diagnosis_class = "peering_admin_down_both_sides"
            recommended_action = (
                f"Re-enable {interface_a} on {device_a} and {interface_b} on {device_b}, then verify the OSPF adjacency reforms "
                "and the route is readvertised upstream."
            )
            lines.append(
                f"  Evidence summary: both ends of the peering are down/admin-down ({device_a} {interface_a} and {device_b} {interface_b}). "
                "This is sufficient to explain the OSPF adjacency loss and route withdrawal."
            )
        elif one_down:
            diagnosis_class = "peering_admin_down_one_side"
            down_side = f"{device_a} {interface_a}" if row_a.get("up") is False else f"{device_b} {interface_b}"
            recommended_action = (
                f"Re-enable {down_side}, then verify the OSPF adjacency reforms and the route is readvertised upstream."
            )
            lines.append(
                f"  Evidence summary: {down_side} is down/admin-down. That is sufficient to explain the OSPF adjacency loss."
            )
        elif active_a or active_b:
            diagnosis_class = "peering_interface_errors"
            noisy_side = f"{device_a} {interface_a}" if active_a else f"{device_b} {interface_b}"
            recommended_action = (
                f"Treat {noisy_side} as a physical-layer suspect: inspect/reseat or replace the cable/optic/transceiver, "
                "check CRC/error counters and port health, then verify peer-IP reachability and confirm the OSPF adjacency reforms."
            )
            lines.append(
                f"  Evidence summary: {noisy_side} shows active interface errors while the peering is failing. "
                "This points to a physical/link-quality problem on the peering."
            )
        elif ping_a is not None and ping_b is not None and not ping_a.get("success") and not ping_b.get("success"):
            diagnosis_class = "peering_bidirectional_reachability_failure"
            recommended_action = (
                f"Treat the {device_a} {interface_a} <-> {device_b} {interface_b} peering as a physical/link-path suspect: "
                "inspect and reseat or replace the cable/optic/transceiver on both ends, verify interface counters and port health, "
                "then re-test direct peer-IP reachability before changing OSPF timers or policy."
            )
            lines.append(
                "  Evidence summary: both interfaces are up but bidirectional peer-IP reachability fails on the peering. "
                "This points to a peering/link problem rather than the destination LAN interface."
            )
        elif ping_a is not None and ping_a.get("success") is False:
            diagnosis_class = "peering_one_way_reachability_failure"
            recommended_action = (
                f"Start on {device_a} {interface_a} as the likely physical-layer suspect: inspect/reseat or replace the local cable/optic/transceiver, "
                f"check interface counters and port health, then re-test peer-IP reachability toward {live_ip_b} before changing OSPF settings."
            )
            lines.append(
                f"  Evidence summary: both peering interfaces are currently up, but {device_a} {interface_a} cannot reach peer IP {live_ip_b}. "
                "This localizes the failure to the peering itself and makes a generic route recommendation inappropriate."
            )
            lines.append(
                f"  OSPF impact: the OSPF adjacency on {device_a} {interface_a} <-> {device_b} {interface_b} is down/lost because peer-IP reachability failed on that peering."
            )
        elif ping_b is not None and ping_b.get("success") is False:
            diagnosis_class = "peering_one_way_reachability_failure"
            recommended_action = (
                f"Start on {device_b} {interface_b} as the likely physical-layer suspect: inspect/reseat or replace the local cable/optic/transceiver, "
                f"check interface counters and port health, then re-test peer-IP reachability toward {live_ip_a} before changing OSPF settings."
            )
            lines.append(
                f"  Evidence summary: both peering interfaces are currently up, but {device_b} {interface_b} cannot reach peer IP {live_ip_a}. "
                "This localizes the failure to the peering itself and makes a generic route recommendation inappropriate."
            )
            lines.append(
                f"  OSPF impact: the OSPF adjacency on {device_a} {interface_a} <-> {device_b} {interface_b} is down/lost because peer-IP reachability failed on that peering."
            )
        elif correlations:
            diagnosis_class = "peering_correlated_syslog_only"
            recommended_action = (
                f"Use the correlated OSPF-facing interfaces ({device_a} {interface_a} and {device_b} {interface_b}) as the investigation focus, "
                "then verify peer-IP reachability and current OSPF neighbor state on that exact peering."
            )
            lines.append(
                "  Evidence summary: syslog/IP correlation identifies the exact OSPF-facing interface(s); use that evidence directly in Root Cause."
            )
        else:
            diagnosis_class = "peering_unresolved_after_inspection"
            recommended_action = (
                f"Continue on the {device_a} {interface_a} <-> {device_b} {interface_b} peering with targeted OSPF neighbor/parameter checks; "
                "the current evidence identifies the failing adjacency but not yet the exact trigger."
            )

        lines.append(f"  Diagnosis class: {diagnosis_class}")
        lines.append(f"  Recommended next action: {recommended_action}")

        store["peering_inspections"].append({
            "device_a": device_a,
            "interface_a": interface_a,
            "device_b": device_b,
            "interface_b": interface_b,
            "ip_a": live_ip_a,
            "ip_b": live_ip_b,
            "row_a": row_a,
            "row_b": row_b,
            "detail_a": detail_a,
            "detail_b": detail_b,
            "correlations": correlations,
            "ping_a": ping_a,
            "ping_b": ping_b,
            "counters_a": counters_a,
            "counters_b": counters_b,
            "diagnosis_class": diagnosis_class,
            "recommended_action": recommended_action,
            "summary": lines[-1] if lines else "",
        })

        return "\n".join(lines)

    async def ospf_neighbors_summary(self, *, session_id: str, devices: list[str]) -> str:
        try:
            data = await nornir_client.post("/ospf-neighbors", {"devices": devices}, timeout=30.0)
        except Exception as exc:
            return backend_unavailable("Nornir", "OSPF neighbor lookup", exc)

        session_store.get(session_id)["ospf_neighbors"] = data

        neighbors = data.get("ospf_neighbors", {})
        lines = ["OSPF neighbors:"]
        for device, info in neighbors.items():
            count = info.get("count", 0)
            if count == 0:
                lines.append(f"  {device}: 0 OSPF neighbors")
            else:
                nbr_list = ", ".join(
                    f"{n['router_id']} via {n['interface']} ({n['state']})"
                    for n in info.get("neighbors", [])
                )
                lines.append(f"  {device}: {count} neighbor(s) — {nbr_list}")
        return "\n".join(lines)

    async def ospf_interfaces_summary(self, *, session_id: str, devices: list[str]) -> str:
        try:
            data = await nornir_client.post("/ospf-interfaces", {"devices": devices}, timeout=30.0)
        except Exception as exc:
            return backend_unavailable("Nornir", "OSPF interface lookup", exc)

        session_store.get(session_id)["ospf_interfaces"] = data

        intfs = data.get("ospf_interfaces", {})
        lines = ["OSPF interface configuration:"]
        for device, info in intfs.items():
            count = info.get("ospf_interface_count", 0)
            ifaces = info.get("interfaces", [])
            if count == 0:
                lines.append(
                    f"  {device}: ospf_interface_count=0 — "
                    f"no interfaces currently reported by 'show ip ospf interface brief'; "
                    f"correlate with interface state, syslog, and history before calling this a config issue"
                )
            else:
                lines.append(f"  {device}: {count} OSPF interface(s) — {', '.join(ifaces)}")
        return "\n".join(lines)

    async def ospf_history_summary(self, *, session_id: str, devices: list[str]) -> str:
        try:
            try:
                from atlas.db import fetch as _fetch
            except ImportError:
                from db import fetch as _fetch  # type: ignore
        except Exception as exc:
            return f"OSPF history DB unavailable: {exc}"

        results: dict[str, Any] = {}
        for device in devices:
            try:
                snapshots = await _fetch(
                    """
                    SELECT date_trunc('minute', collected_at) AS t, count(*) AS n
                    FROM ospf_history WHERE device=$1
                    GROUP BY t ORDER BY t DESC LIMIT 10
                    """,
                    device,
                )
                current = await _fetch(
                    "SELECT router_id, interface, state FROM ospf_neighbors WHERE device=$1",
                    device,
                )
                results[device] = {
                    "current_neighbor_count": len(current),
                    "history": [{"snapshot_time": r["t"].isoformat(), "neighbor_count": r["n"]} for r in snapshots],
                }
            except Exception as exc:
                results[device] = {"error": str(exc)}

        session_store.get(session_id)["ospf_history"] = results

        lines = ["OSPF neighbor history:"]
        for device, info in results.items():
            if "error" in info:
                lines.append(f"  {device}: DB error — {info['error']}")
                continue
            hist = info.get("history", [])
            curr = info.get("current_neighbor_count", 0)
            trend = " → ".join(str(s["neighbor_count"]) for s in reversed(hist))
            lines.append(f"  {device}: history [{trend}] → now: {curr}")
        return "\n".join(lines)

    async def routing_history_summary(self, *, session_id: str, destination_ip: str) -> str:
        try:
            try:
                from atlas.db import fetch as _fetch, fetchrow as _fetchrow
            except ImportError:
                from db import fetch as _fetch, fetchrow as _fetchrow  # type: ignore
        except Exception as exc:
            return f"Routing history DB unavailable: {exc}"

        try:
            hist_devs = await _fetch(
                """
                SELECT DISTINCT device FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface IS NOT NULL
                  AND egress_interface NOT ILIKE 'management%'
                """,
                destination_ip,
            )
            historical_devices = [r["device"] for r in hist_devs]

            last_route = await _fetchrow(
                """
                SELECT device, egress_interface, next_hop::text, protocol, prefix::text, collected_at
                FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface NOT ILIKE 'management%'
                ORDER BY collected_at DESC LIMIT 1
                """,
                destination_ip,
            )
            last_upstream_route = await _fetchrow(
                """
                SELECT device, egress_interface, next_hop::text, protocol, prefix::text, collected_at
                FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface NOT ILIKE 'management%'
                  AND next_hop IS NOT NULL
                ORDER BY collected_at DESC LIMIT 1
                """,
                destination_ip,
            )
        except Exception as exc:
            return f"Routing history query error: {exc}"

        peer_hint: dict[str, Any] | None = None
        peering_source = last_upstream_route or last_route
        if peering_source and peering_source.get("next_hop"):
            try:
                peer_data = await nornir_client.post(
                    "/find-device",
                    {"ip": str(peering_source["next_hop"]).split("/")[0]},
                    timeout=15.0,
                )
                if peer_data.get("found"):
                    peer_hint = {
                        "from_device": peering_source.get("device"),
                        "from_interface": peering_source.get("egress_interface"),
                        "next_hop_ip": peering_source.get("next_hop"),
                        "to_device": peer_data.get("device"),
                        "to_interface": peer_data.get("interface"),
                    }
            except Exception:
                peer_hint = None

        store = session_store.get(session_id)
        store["historical_devices"] = historical_devices
        store["routing_history"] = json_safe({
            "historical_devices": historical_devices,
            "last_route": dict(last_route) if last_route else None,
            "last_upstream_route": dict(last_upstream_route) if last_upstream_route else None,
            "peer_hint": peer_hint,
        })

        lines = [f"Routing history for {destination_ip}:"]
        if historical_devices:
            lines.append(f"  Historically known path devices: {', '.join(historical_devices)}")
            lines.append("  (Include these in OSPF checks even if not in current path)")
        else:
            lines.append("  No routing history found in DB.")

        if last_upstream_route:
            try:
                delta = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(str(last_upstream_route["collected_at"]))
                age = f"{int(delta.total_seconds() // 3600)}h ago"
            except Exception:
                age = "unknown age"
            lines.append(
                f"  Primary upstream clue ({age}): {last_upstream_route['device']} learned "
                f"{destination_ip} via {last_upstream_route['egress_interface']} next-hop "
                f"{last_upstream_route['next_hop']} ({last_upstream_route['protocol']})"
            )
            if peer_hint:
                lines.append(
                    f"  Primary OSPF peering to troubleshoot: {peer_hint['from_device']} {peer_hint['from_interface']} "
                    f"<-> {peer_hint['to_device']} {peer_hint['to_interface']} (via {peer_hint['next_hop_ip']})"
                )
                lines.append("  Troubleshoot this bilateral peering first. Do not stop at the destination gateway alone.")

        if last_route:
            try:
                delta = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(str(last_route["collected_at"]))
                age = f"{int(delta.total_seconds() // 3600)}h ago"
            except Exception:
                age = "unknown age"
            lines.append(
                f"  Last known route ({age}): {last_route['device']} egress "
                f"{last_route['egress_interface']} via {last_route['next_hop']} "
                f"({last_route['protocol']}, {last_route['prefix']})"
            )

        return "\n".join(lines)


routing_diagnostics_service = RoutingDiagnosticsService()
