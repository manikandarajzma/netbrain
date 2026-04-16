"""Owned connectivity snapshot workflow for troubleshoot investigations."""
from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

try:
    from atlas.services.nornir_client import nornir_client
    from atlas.services.observability import json_safe
    from atlas.services.path_trace_service import path_trace_service
except ImportError:
    from services.nornir_client import nornir_client  # type: ignore
    from services.observability import json_safe  # type: ignore
    from services.path_trace_service import path_trace_service  # type: ignore

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9\-_\.]*[A-Za-z0-9])?$")


class ConnectivitySnapshotService:
    """Owns device snapshot collection and structured connectivity evidence assembly."""

    @staticmethod
    def _is_device_name(value: str) -> bool:
        value = str(value or "").strip()
        return bool(value) and bool(_HOSTNAME_RE.match(value)) and not re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", value)

    @classmethod
    def _extract_devices_from_hops(cls, hops: list[dict[str, Any]]) -> list[str]:
        devices: list[str] = []
        for hop in hops or []:
            if not isinstance(hop, dict):
                continue
            for key in ("from_device", "to_device"):
                value = str(hop.get(key) or "").strip()
                if cls._is_device_name(value) and value not in devices:
                    devices.append(value)
        return devices

    @classmethod
    def _extract_relevant_interfaces(cls, hops: list[dict[str, Any]]) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for hop in hops or []:
            if not isinstance(hop, dict):
                continue
            from_device = str(hop.get("from_device") or "").strip()
            to_device = str(hop.get("to_device") or "").strip()
            out_interface = str(hop.get("out_interface") or "").strip()
            in_interface = str(hop.get("in_interface") or "").strip()
            if cls._is_device_name(from_device) and out_interface:
                result.setdefault(from_device, set()).add(out_interface)
            if cls._is_device_name(to_device) and in_interface:
                result.setdefault(to_device, set()).add(in_interface)
        return result

    @staticmethod
    def _pick_interface_row(data: dict[str, Any], interface: str) -> dict[str, Any]:
        for row in (data.get("interfaces") or []):
            if row.get("interface") == interface:
                return row
        return {}

    @staticmethod
    def _upsert_ping_result(store: dict[str, Any], entry: dict[str, Any]) -> None:
        results = store.setdefault("ping_results", [])
        if not isinstance(results, list):
            store["ping_results"] = [entry]
            return

        key = (
            str(entry.get("device") or ""),
            str(entry.get("destination") or ""),
            str(entry.get("source_interface") or ""),
            str(entry.get("vrf") or ""),
        )
        for idx, existing in enumerate(results):
            existing_key = (
                str(existing.get("device") or ""),
                str(existing.get("destination") or ""),
                str(existing.get("source_interface") or ""),
                str(existing.get("vrf") or ""),
            )
            if existing_key == key:
                results[idx] = entry
                return
        results.append(entry)

    async def _run_snapshot_ping(
        self,
        *,
        store: dict[str, Any],
        device: str,
        destination: str,
        source_interface: str = "",
        vrf: str = "",
    ) -> dict[str, Any]:
        try:
            result = await nornir_client.post(
                "/ping",
                {
                    "device": device,
                    "destination": destination,
                    "source_interface": source_interface,
                    "vrf": vrf,
                },
                timeout=60.0,
            )
        except Exception as exc:
            result = {"success": False, "error": str(exc)}

        entry = {
            "device": device,
            "destination": destination,
            "source_interface": source_interface,
            "vrf": vrf or "default",
            **result,
        }
        self._upsert_ping_result(store, entry)
        return entry

    @staticmethod
    def _format_ping_summary(result: dict[str, Any]) -> str:
        if not result:
            return "unavailable"
        status = "success" if result.get("success") else "failed"
        loss = result.get("loss_pct")
        source_interface = str(result.get("source_interface") or "").strip()
        src_note = f" source {source_interface}" if source_interface else ""
        vrf = str(result.get("vrf") or "default")
        base = (
            f"{result.get('device')} -> {result.get('destination')} "
            f"(vrf {vrf}{src_note}) {status}"
        )
        if loss is not None:
            base += f", loss {loss}%"
        error = str(result.get("error") or "").strip()
        if error:
            base += f" ({error})"
        return base

    def _infer_destination_side_device_from_store(
        self,
        store: dict[str, Any],
        route_to_dest: dict[str, Any] | None = None,
    ) -> str:
        if route_to_dest:
            hops = route_to_dest.get("hops") or {}
            for device, route in hops.items():
                if not isinstance(route, dict):
                    continue
                if not route.get("found"):
                    continue
                protocol = str(route.get("protocol") or "").lower()
                if protocol in {"connected", "local"} and self._is_device_name(str(device)):
                    return str(device)

        reverse_hops = store.get("reverse_path_hops") or []
        if reverse_hops and isinstance(reverse_hops, list):
            first = reverse_hops[0]
            if isinstance(first, dict):
                candidate = str(first.get("to_device") or "").strip()
                if self._is_device_name(candidate):
                    return candidate
        meta = store.get("path_meta") or {}
        candidate = str(meta.get("last_hop_device") or "").strip()
        return candidate if self._is_device_name(candidate) else ""

    async def build_snapshot_summary(
        self,
        *,
        session_id: str,
        store: dict[str, Any],
        source_ip: str,
        dest_ip: str,
        port: str = "",
        push_status: Callable[[str, str], Awaitable[None]],
    ) -> str:
        path_hops = store.get("path_hops") or []
        reverse_hops = store.get("reverse_path_hops") or []
        routing_history = store.get("routing_history") or {}
        peer_hint = routing_history.get("peer_hint") or {}

        devices: list[str] = []
        for device in (
            self._extract_devices_from_hops(path_hops)
            + self._extract_devices_from_hops(reverse_hops)
            + list(routing_history.get("historical_devices") or [])
            + [peer_hint.get("from_device"), peer_hint.get("to_device")]
        ):
            device = str(device or "").strip()
            if self._is_device_name(device) and device not in devices:
                devices.append(device)

        if not devices:
            return "Connectivity snapshot unavailable: no in-scope devices were discovered from live path or history."

        relevant_interfaces = self._extract_relevant_interfaces(path_hops)
        for device, interfaces in self._extract_relevant_interfaces(reverse_hops).items():
            relevant_interfaces.setdefault(device, set()).update(interfaces)
        if peer_hint.get("from_device") and peer_hint.get("from_interface"):
            relevant_interfaces.setdefault(str(peer_hint["from_device"]), set()).add(str(peer_hint["from_interface"]))
        if peer_hint.get("to_device") and peer_hint.get("to_interface"):
            relevant_interfaces.setdefault(str(peer_hint["to_device"]), set()).add(str(peer_hint["to_interface"]))

        syslog_devices: set[str] = set()
        for key in ("from_device", "to_device"):
            dev = str(peer_hint.get(key) or "").strip()
            if self._is_device_name(dev):
                syslog_devices.add(dev)
        if not syslog_devices:
            reverse_meta = store.get("reverse_path_meta") or {}
            for dev in (
                str((store.get("path_meta") or {}).get("first_hop_device") or "").strip(),
                str(reverse_meta.get("reverse_first_hop_device") or "").strip(),
            ):
                if self._is_device_name(dev):
                    syslog_devices.add(dev)

        async def _device_snapshot(device: str) -> dict[str, Any]:
            try:
                data = await nornir_client.post(
                    "/device-snapshot",
                    {
                        "device": device,
                        "source_ip": source_ip,
                        "dest_ip": dest_ip,
                        "relevant_interfaces": sorted(relevant_interfaces.get(device, set())),
                        "include_syslog": device in syslog_devices,
                    },
                    timeout=25.0,
                    retries=False,
                )
            except Exception as exc:
                data = {"device": device, "error": str(exc)}
            return data

        await push_status(
            session_id,
            f"Collecting parallel device snapshots for {len(devices)} device{'s' if len(devices) != 1 else ''}..."
        )
        snapshot_results_list = await asyncio.gather(*(_device_snapshot(device) for device in devices))
        device_snapshots = dict(zip(devices, snapshot_results_list))

        protocol_results: dict[str, Any] = {}
        interface_results: dict[str, Any] = {}
        syslog_results: dict[str, Any] = {}
        route_to_dest: dict[str, Any] = {"hops": {}}
        route_to_src: dict[str, Any] = {"hops": {}}
        ospf_neighbors: dict[str, Any] = {}
        ospf_interfaces: dict[str, Any] = {}
        counter_results_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        snapshot_errors: dict[str, str] = {}
        live_snapshot_devices: list[str] = []

        for device, snap in device_snapshots.items():
            if snap.get("error"):
                snapshot_errors[device] = str(snap.get("error") or "unknown snapshot error")
            else:
                live_snapshot_devices.append(device)

            proto = snap.get("protocol_discovery") or {
                "device": device,
                "routing_protocols": [],
                "configured_routing_protocols": [],
                "observed_route_types": [],
                "l2_control_plane": {
                    "spanning_tree_mode": "unknown",
                    "spanning_tree_enabled": None,
                    "summary_lines": [],
                },
                "errors": {"device_snapshot": snap.get("error")} if snap.get("error") else {},
            }
            protocol_results[device] = proto
            store["protocol_discovery"][device] = proto

            iface_payload = snap.get("all_interfaces") or {"device": device, "interfaces": [], "error": snap.get("error")}
            interface_results[device] = iface_payload
            store["all_interfaces"][device] = iface_payload

            syslog_payload = snap.get("syslog") or {"device": device, "logs": [], "relevant": [], "error": snap.get("error")}
            syslog_results[device] = syslog_payload
            store["syslog"][device] = syslog_payload

            route_to_dest["hops"][device] = snap.get("route_to_destination") or {"found": False, "error": snap.get("error", "missing")}
            route_to_src["hops"][device] = snap.get("route_to_source") or {"found": False, "error": snap.get("error", "missing")}

            if "ospf" in (proto.get("routing_protocols") or []):
                ospf_neighbors[device] = snap.get("ospf_neighbors") or {"device": device, "count": 0, "neighbors": []}
                ospf_interfaces[device] = snap.get("ospf_interfaces") or {"device": device, "ospf_enabled_interfaces": [], "ospf_interface_count": 0}

            snap_details = snap.get("interface_details") or {}
            for interface, detail in snap_details.items():
                store["interface_details"][f"{device}:{interface}"] = detail
                active = []
                if any((detail.get(k) or 0) > 0 for k in ("input_errors", "output_errors", "input_discards", "output_discards")):
                    active.append({
                        "interface": interface,
                        "input_errors": detail.get("input_errors", 0),
                        "output_errors": detail.get("output_errors", 0),
                        "input_discards": detail.get("input_discards", 0),
                        "output_discards": detail.get("output_discards", 0),
                    })
                counter_payload = {"active_errors": active, "clean_interfaces": [] if active else [interface]}
                counter_results_by_key[(device, interface)] = counter_payload
                store["interface_counters"].append({
                    "device": device,
                    "window_s": 0,
                    "active": active,
                    "clean": counter_payload["clean_interfaces"],
                })

        path_meta = store.get("path_meta") or {}
        reverse_meta = store.get("reverse_path_meta") or {}
        forward_ping_result: dict[str, Any] | None = None
        reverse_ping_result: dict[str, Any] | None = None

        forward_ping_device = str(path_meta.get("first_hop_device") or "").strip()
        forward_ping_source = str(path_meta.get("first_hop_lan_interface") or "").strip()
        forward_ping_vrf = str(path_meta.get("src_vrf") or "default").strip() or "default"

        reverse_ping_device = str(reverse_meta.get("reverse_first_hop_device") or "").strip()
        reverse_ping_source = str(reverse_meta.get("reverse_first_hop_lan_interface") or "").strip()
        reverse_ping_vrf = "default"
        if reverse_ping_device:
            reverse_ping_vrf = (
                str(await path_trace_service.infer_vrf(dest_ip, reverse_ping_device) or "default").strip()
                or "default"
            )

        ping_tasks: list[Awaitable[dict[str, Any]]] = []
        ping_keys: list[str] = []
        if forward_ping_device:
            ping_keys.append("forward")
            ping_tasks.append(
                self._run_snapshot_ping(
                    store=store,
                    device=forward_ping_device,
                    destination=dest_ip,
                    source_interface=forward_ping_source,
                    vrf=forward_ping_vrf,
                )
            )
        if reverse_ping_device:
            ping_keys.append("reverse")
            ping_tasks.append(
                self._run_snapshot_ping(
                    store=store,
                    device=reverse_ping_device,
                    destination=source_ip,
                    source_interface=reverse_ping_source,
                    vrf=reverse_ping_vrf,
                )
            )
        if ping_tasks:
            await push_status(session_id, "Validating forward and reverse reachability from path-adjacent devices...")
            ping_results = await asyncio.gather(*ping_tasks)
            ping_map = dict(zip(ping_keys, ping_results))
            forward_ping_result = ping_map.get("forward")
            reverse_ping_result = ping_map.get("reverse")

        destination_side_device = self._infer_destination_side_device_from_store(store, route_to_dest)
        service_snapshot: dict[str, Any] | None = None
        if port and destination_side_device:
            await push_status(session_id, f"Testing destination-side TCP from {destination_side_device} to {dest_ip}:{port}...")
            try:
                service_snapshot = await nornir_client.post(
                    "/tcp-test",
                    {"device": destination_side_device, "destination": dest_ip, "port": int(port), "vrf": ""},
                    timeout=30.0,
                )
                service_snapshot["device"] = destination_side_device
            except Exception as exc:
                service_snapshot = {
                    "device": destination_side_device,
                    "reachable": False,
                    "error": str(exc),
                    "destination": dest_ip,
                    "port": int(port),
                }

        await push_status(session_id, "Assembling structured connectivity findings...")
        findings: list[str] = []
        link_lines: list[str] = []
        if peer_hint:
            a_dev = str(peer_hint.get("from_device") or "")
            a_int = str(peer_hint.get("from_interface") or "")
            b_dev = str(peer_hint.get("to_device") or "")
            b_int = str(peer_hint.get("to_interface") or "")
            if a_dev and a_int and b_dev and b_int:
                a_row = self._pick_interface_row(interface_results.get(a_dev, {}), a_int)
                b_row = self._pick_interface_row(interface_results.get(b_dev, {}), b_int)
                a_state = "up" if a_row.get("up") else f"down ({a_row.get('oper_status', 'unknown')})"
                b_state = "up" if b_row.get("up") else f"down ({b_row.get('oper_status', 'unknown')})"
                a_ip = a_row.get("primary_ip") or "unknown"
                b_ip = b_row.get("primary_ip") or str(peer_hint.get("next_hop_ip") or "unknown")
                link_lines.append(
                    f"- {a_dev} {a_int} ({a_ip}) <-> {b_dev} {b_int} ({b_ip}) | states: {a_state} / {b_state}"
                )

                inspection = None
                for item in reversed(store.get("peering_inspections") or []):
                    if not isinstance(item, dict):
                        continue
                    direct = item.get("device_a") == a_dev and item.get("interface_a") == a_int and item.get("device_b") == b_dev and item.get("interface_b") == b_int
                    reverse = item.get("device_a") == b_dev and item.get("interface_a") == b_int and item.get("device_b") == a_dev and item.get("interface_b") == a_int
                    if direct or reverse:
                        inspection = item
                        break
                if inspection:
                    ping_forward = inspection.get("ping_a_success")
                    ping_reverse = inspection.get("ping_b_success")
                    link_lines.append(
                        f"  peer-IP reachability: {a_dev}->{b_ip}={'ok' if ping_forward else 'failed'}; "
                        f"{b_dev}->{a_ip}={'ok' if ping_reverse else 'failed'}"
                    )
                if a_row.get("up") is False or b_row.get("up") is False:
                    findings.append(
                        f"[interface] Primary peering {a_dev} {a_int} <-> {b_dev} {b_int} has an interface down/admin-down."
                    )

        for device in devices:
            route_info = (route_to_dest.get("hops") or {}).get(device, {})
            if route_info.get("error") and route_info.get("found") is not True:
                findings.append(f"[evidence] Live routing data for {device} was unavailable: {route_info.get('error')}.")
            elif route_info.get("found") is False:
                findings.append(f"[routing] {device} currently has no route to {dest_ip}.")
            for line in syslog_results.get(device, {}).get("relevant", []):
                low = line.lower()
                if "crc" in low or "fcs" in low or "discard" in low:
                    findings.append(f"[interface] {device} recent syslog mentions physical/interface errors: {line}")
                    break

        if service_snapshot:
            if service_snapshot.get("reachable") is True:
                findings.append(f"[service] TCP port {port} is reachable from destination-side device {service_snapshot['device']}.")
            else:
                err = str(service_snapshot.get("error") or service_snapshot.get("output") or "").lower()
                if "refused" in err:
                    findings.append(f"[service] TCP port {port} is actively refused from destination-side device {service_snapshot['device']}.")
                else:
                    findings.append(f"[service] TCP port {port} could not be validated from destination-side device {service_snapshot['device']}.")
        if forward_ping_result:
            if forward_ping_result.get("success"):
                findings.append(
                    f"[connectivity] Forward validation ping from {forward_ping_result['device']} to {dest_ip} succeeded."
                )
            else:
                findings.append(
                    f"[connectivity] Forward validation ping from {forward_ping_result['device']} to {dest_ip} failed."
                )
        if reverse_ping_result:
            if reverse_ping_result.get("success"):
                findings.append(
                    f"[connectivity] Reverse validation ping from {reverse_ping_result['device']} to {source_ip} succeeded."
                )
            else:
                findings.append(
                    f"[connectivity] Reverse validation ping from {reverse_ping_result['device']} to {source_ip} failed."
                )

        snapshot = {
            "source_ip": source_ip,
            "dest_ip": dest_ip,
            "port": port,
            "devices": devices,
            "routing_history": routing_history,
            "destination_side_device": destination_side_device,
            "findings": findings,
            "errors": snapshot_errors,
            "live_evidence_available": bool(path_hops or reverse_hops or live_snapshot_devices),
            "service": service_snapshot,
            "forward_ping": forward_ping_result,
            "reverse_ping": reverse_ping_result,
        }
        store["connectivity_snapshot"] = json_safe(snapshot)

        lines = [
            "Connectivity incident snapshot:",
            f"  source_ip: {source_ip}",
            f"  dest_ip: {dest_ip}",
        ]
        if port:
            lines.append(f"  requested_port: {port}")
        lines.extend([
            f"  forward_path_devices: {', '.join(self._extract_devices_from_hops(path_hops)) or 'none'}",
            f"  reverse_path_devices: {', '.join(self._extract_devices_from_hops(reverse_hops)) or 'none'}",
            f"  historical_devices: {', '.join(routing_history.get('historical_devices') or []) or 'none'}",
        ])
        if peer_hint:
            lines.append(
                f"  primary_historical_peering: {peer_hint.get('from_device')} {peer_hint.get('from_interface')} "
                f"<-> {peer_hint.get('to_device')} {peer_hint.get('to_interface')} via {peer_hint.get('next_hop_ip')}"
            )

        lines.append("Device summary:")
        for device in devices:
            proto = protocol_results.get(device, {})
            route_dest = (route_to_dest.get("hops") or {}).get(device, {})
            route_src = (route_to_src.get("hops") or {}).get(device, {})
            rel_intfs = sorted(relevant_interfaces.get(device, set()))
            route_dest_summary = (
                f"{route_dest.get('protocol')} via {route_dest.get('interface')} next-hop {route_dest.get('next_hop')}"
                if route_dest.get("found") else (
                    f"unavailable ({route_dest.get('error', 'unknown')})"
                    if route_dest.get("error")
                    else f"no route ({route_dest.get('error', 'unknown')})"
                )
            )
            route_src_summary = (
                f"{route_src.get('protocol')} via {route_src.get('interface')} next-hop {route_src.get('next_hop')}"
                if route_src.get("found") else (
                    f"unavailable ({route_src.get('error', 'unknown')})"
                    if route_src.get("error")
                    else f"no route ({route_src.get('error', 'unknown')})"
                )
            )
            lines.append(
                f"- {device}: protocols={', '.join(proto.get('routing_protocols') or []) or 'none'}; "
                f"route_to_destination={route_dest_summary}; route_to_source={route_src_summary}"
            )
            if "ospf" in (proto.get("routing_protocols") or []):
                neigh = ospf_neighbors.get(device, {})
                ospf_intf = ospf_interfaces.get(device, {})
                lines.append(
                    f"  ospf_neighbors={neigh.get('count', 0)}; "
                    f"ospf_interfaces={ospf_intf.get('ospf_interface_count', 0)}"
                )
            if rel_intfs:
                rel_bits: list[str] = []
                for interface in rel_intfs:
                    row = self._pick_interface_row(interface_results.get(device, {}), interface)
                    counters = counter_results_by_key.get((device, interface), {})
                    state = "up" if row.get("up") else f"down ({row.get('oper_status', 'unknown')})"
                    ip_text = f"{row.get('primary_ip')}/{row.get('prefix_len')}" if row.get("primary_ip") else "no_ip"
                    active_errors = counters.get("active_errors") or []
                    err_text = " active_errors" if active_errors else ""
                    rel_bits.append(f"{interface}={ip_text},{state}{err_text}")
                lines.append(f"  relevant_interfaces: {'; '.join(rel_bits)}")

        if link_lines:
            lines.append("Link summary:")
            lines.extend(link_lines)

        if service_snapshot:
            if service_snapshot.get("reachable") is True:
                lines.append(f"Service summary: tcp_port_{port} reachable from {service_snapshot.get('device')}")
            else:
                lines.append(
                    f"Service summary: tcp_port_{port} unreachable/refused from {service_snapshot.get('device')} "
                    f"({service_snapshot.get('error') or service_snapshot.get('output') or 'no detail'})"
                )

        if forward_ping_result or reverse_ping_result:
            lines.append("Ping summary:")
            if forward_ping_result:
                lines.append(f"  forward_ping: {self._format_ping_summary(forward_ping_result)}")
            if reverse_ping_result:
                lines.append(f"  reverse_ping: {self._format_ping_summary(reverse_ping_result)}")

        lines.append("Candidate issues:")
        if findings:
            for finding in findings[:10]:
                lines.append(f"  - {finding}")
        else:
            lines.append("  - No explicit candidate issue was derived; reason from the summarized device, link, and service evidence above.")

        return "\n".join(lines)


connectivity_snapshot_service = ConnectivitySnapshotService()
