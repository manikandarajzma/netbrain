"""Owned live path tracing workflow and path metadata extraction."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    from atlas.services.nornir_client import nornir_client
except ImportError:
    from services.nornir_client import nornir_client  # type: ignore

_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9]([A-Za-z0-9\-_\.]*[A-Za-z0-9])?$")
_IP_RE = re.compile(r"^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$")


class PathTraceService:
    """Owns hop-by-hop live trace logic and path metadata extraction."""

    async def live_path_trace(
        self,
        src_ip: str,
        dst_ip: str,
        *,
        session_id: str = "",
        store: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        flags: dict[str, Any] = {}
        store = store or {}

        async def _get_ip_owners() -> dict[str, Any]:
            owners = store.get("ip_owners") if isinstance(store, dict) else None
            if owners:
                return owners
            try:
                data = await nornir_client.cached_post(
                    session_id,
                    "/ip-owners",
                    {"devices": []},
                    timeout=8.0,
                    retries=False,
                )
                owners = data.get("owners") if isinstance(data, dict) else {}
            except Exception:
                owners = {}
            if isinstance(store, dict):
                store["ip_owners"] = owners or {}
            return owners or {}

        async def _find_device(ip: str) -> dict[str, Any]:
            owners = await _get_ip_owners()
            owner = owners.get(ip) if isinstance(owners, dict) else None
            if isinstance(owner, dict) and owner.get("device"):
                return {"found": True, **owner}
            try:
                result = await nornir_client.cached_post(
                    session_id,
                    "/find-device",
                    {"ip": ip},
                    timeout=5.0,
                    retries=False,
                )
                if isinstance(result, dict) and result.get("found") and result.get("device"):
                    if isinstance(store, dict):
                        cached = store.setdefault("ip_owners", {})
                        if isinstance(cached, dict):
                            cached[ip] = {
                                "device": result.get("device"),
                                "interface": result.get("interface"),
                                "host": result.get("host"),
                                "port": result.get("port"),
                                "ip": ip,
                            }
                    return result
            except Exception:
                pass
            return {"found": False, "ip": ip}

        async def _find_first_hop(src: str) -> tuple[str, str | None]:
            try:
                resp = await nornir_client.get("/devices", timeout=25.0, retries=False)
                devices = resp.get("devices", []) if isinstance(resp, dict) else []
            except Exception:
                devices = []
            if not devices:
                try:
                    import yaml

                    hosts_file = Path(__file__).resolve().parent.parent / "nornir" / "inventory" / "hosts.yaml"
                    with open(hosts_file) as handle:
                        devices = list(yaml.safe_load(handle).keys())
                except Exception:
                    pass
            for device in devices:
                try:
                    route = await nornir_client.cached_post(
                        session_id,
                        "/route",
                        {"device": device, "destination": src},
                        timeout=5.0,
                        retries=False,
                    )
                    if route.get("found") and route.get("protocol", "").lower() == "connected":
                        return device, route.get("egress_interface")
                except Exception:
                    continue
            return "", None

        current_device, gw_iface = await _find_first_hop(src_ip)
        if not current_device:
            return (
                f"Could not find a network device with a connected route to {src_ip} — check inventory.",
                [],
                flags,
            )

        await _get_ip_owners()

        text_hops: list[str] = []
        structured_hops: list[dict[str, Any]] = []
        seen: set[str] = set()
        max_hops = 15

        structured_hops.append(
            {
                "from_device": src_ip,
                "from_device_type": "host",
                "out_interface": None,
                "out_zone": None,
                "device_group": None,
                "to_device": current_device,
                "to_device_type": "switch",
                "in_interface": gw_iface,
                "in_zone": None,
            }
        )

        for _ in range(max_hops):
            if current_device in seen:
                text_hops.append(f"  !! Routing loop at {current_device}")
                break
            seen.add(current_device)

            try:
                route = await nornir_client.cached_post(
                    session_id,
                    "/route",
                    {"device": current_device, "destination": dst_ip},
                    timeout=5.0,
                    retries=False,
                )
            except Exception as exc:
                text_hops.append(f"  Hop {len(text_hops) + 1}: {current_device} — SSH error: {exc}")
                break

            if not route.get("found"):
                text_hops.append(f"  ⚠️  Hop {len(text_hops) + 1}: {current_device} — no route to {dst_ip}")
                flags["no_route_device"] = current_device
                break

            egress = route.get("egress_interface") or ""
            next_hop = route.get("next_hop")
            protocol = route.get("protocol") or ""
            prefix = route.get("prefix") or ""

            if egress.lower().startswith("management"):
                text_hops.append(
                    f"  ⚠️  Hop {len(text_hops) + 1}: **{current_device}** routing {dst_ip} via "
                    f"**{egress}** (default 0.0.0.0/0) — data-plane is likely DOWN."
                )
                structured_hops.append(
                    {
                        "from_device": current_device,
                        "from_device_type": "switch",
                        "out_interface": egress,
                        "out_zone": None,
                        "device_group": None,
                        "to_device": f"⚠️ Mgmt fallback ({egress})",
                        "to_device_type": "host",
                        "in_interface": None,
                        "in_zone": None,
                    }
                )
                flags["mgmt_routing_detected"] = True
                flags["mgmt_routing_device"] = current_device
                break

            text_hops.append(
                f"  Hop {len(text_hops) + 1}: {current_device} | Egress: {egress} | "
                f"Protocol: {protocol} | Prefix: {prefix} | Next-hop: {next_hop or 'directly connected'}"
            )

            if not next_hop and str(protocol).lower() in {"connected", "local"}:
                try:
                    arp = await nornir_client.post(
                        "/arp",
                        {"device": current_device, "ip": dst_ip},
                        timeout=10.0,
                        retries=False,
                    )
                except Exception:
                    arp = {}
                in_iface = arp.get("interface") if arp.get("found") else None
                text_hops.append(f"  Destination {dst_ip} reachable via ARP on {current_device}")
                structured_hops.append(
                    {
                        "from_device": current_device,
                        "from_device_type": "switch",
                        "out_interface": egress,
                        "out_zone": None,
                        "device_group": None,
                        "to_device": dst_ip,
                        "to_device_type": "host",
                        "in_interface": in_iface,
                        "in_zone": None,
                    }
                )
                break

            if not next_hop:
                text_hops.append(
                    f"  ⚠️  Hop {len(text_hops) + 1}: {current_device} returned route {prefix} via {egress} "
                    f"({protocol}) but no resolvable next-hop was provided"
                )
                flags["missing_next_hop_device"] = current_device
                flags["missing_next_hop_prefix"] = prefix
                break

            next_dev = await _find_device(next_hop)
            if not next_dev.get("found"):
                text_hops.append(f"  Next-hop {next_hop} not found — path ends here")
                flags["next_hop_resolution_failed"] = next_hop
                break

            next_device = next_dev["device"]
            in_iface = next_dev["interface"]
            structured_hops.append(
                {
                    "from_device": current_device,
                    "from_device_type": "switch",
                    "out_interface": egress,
                    "out_zone": None,
                    "device_group": None,
                    "to_device": next_device,
                    "to_device_type": "switch",
                    "in_interface": in_iface,
                    "in_zone": None,
                }
            )
            current_device = next_device

        devices_in_path = list(seen)
        text = (
            f"Path from {src_ip} to {dst_ip} (live SSH):\n"
            + "\n".join(text_hops)
            + f"\n\nAll devices in path: {', '.join(devices_in_path)}"
        )
        return text, structured_hops, flags

    @staticmethod
    def extract_path_metadata(hops: list[dict[str, Any]]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "path_devices": [],
            "first_hop_device": "",
            "first_hop_lan_interface": "",
            "first_hop_egress_interface": "",
            "last_hop_device": "",
            "last_hop_egress_interface": "",
            "has_firewalls": False,
            "firewall_hostnames": [],
            "path_hops_for_counters": [],
        }
        if not hops:
            return meta

        seen_devs: set[str] = set()
        devices: list[str] = []
        for hop in hops:
            for key in ("from_device", "to_device"):
                device = hop.get(key, "")
                if (
                    device
                    and device not in seen_devs
                    and not _IP_RE.match(device)
                    and _HOSTNAME_RE.match(device)
                ):
                    seen_devs.add(device)
                    devices.append(device)
        meta["path_devices"] = devices

        first_hop = hops[0]
        meta["first_hop_device"] = first_hop.get("to_device", "")
        meta["first_hop_lan_interface"] = first_hop.get("in_interface", "") or ""
        first_dev = meta["first_hop_device"]
        for hop in hops:
            if hop.get("from_device") == first_dev and hop.get("out_interface"):
                meta["first_hop_egress_interface"] = hop["out_interface"]
                break

        for hop in reversed(hops):
            if _IP_RE.match(hop.get("to_device", "")) and not _IP_RE.match(hop.get("from_device", "")):
                meta["last_hop_device"] = hop.get("from_device", "")
                meta["last_hop_egress_interface"] = hop.get("out_interface", "")
                break

        dev_intfs: dict[str, set[str]] = {}
        for hop in hops:
            for dev_key, intf_key in (("from_device", "out_interface"), ("to_device", "in_interface")):
                device = hop.get(dev_key, "")
                interface = hop.get(intf_key, "")
                if device and not _IP_RE.match(device) and interface:
                    dev_intfs.setdefault(device, set()).add(interface)
        meta["path_hops_for_counters"] = [
            {"device": device, "interfaces": sorted(interfaces)}
            for device, interfaces in dev_intfs.items()
            if interfaces
        ]
        return meta

    @staticmethod
    def extract_reverse_path_metadata(hops: list[dict[str, Any]]) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "reverse_first_hop_device": "",
            "reverse_first_hop_lan_interface": "",
            "reverse_first_hop_egress_interface": "",
            "reverse_last_hop_device": "",
            "reverse_last_hop_egress_interface": "",
        }
        if not hops:
            return meta

        first_hop = hops[0]
        meta["reverse_first_hop_device"] = first_hop.get("to_device", "")
        meta["reverse_first_hop_lan_interface"] = first_hop.get("in_interface", "") or ""
        first_dev = meta["reverse_first_hop_device"]
        for hop in hops:
            if hop.get("from_device") == first_dev and hop.get("out_interface"):
                meta["reverse_first_hop_egress_interface"] = hop["out_interface"]
                break

        for hop in reversed(hops):
            if _IP_RE.match(hop.get("to_device", "")) and not _IP_RE.match(hop.get("from_device", "")):
                meta["reverse_last_hop_device"] = hop.get("from_device", "")
                meta["reverse_last_hop_egress_interface"] = hop.get("out_interface", "")
                break
        return meta

    @staticmethod
    async def infer_vrf(src_ip: str, device: str) -> str:
        if not src_ip or not device:
            return "default"
        try:
            try:
                from atlas.persistence.db import fetchrow
            except ImportError:
                from persistence.db import fetchrow  # type: ignore
            row = await fetchrow(
                "SELECT vrf FROM routing_table WHERE device=$1 AND $2::inet << prefix "
                "ORDER BY masklen(prefix) DESC LIMIT 1",
                device,
                src_ip,
            )
            if row and row["vrf"]:
                return row["vrf"]
        except Exception:
            pass
        return "default"


path_trace_service = PathTraceService()
