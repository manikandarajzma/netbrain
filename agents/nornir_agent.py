"""
Nornir Path Agent — port 8006.

Traces hop-by-hop network paths by SSHing into live devices via Nornir/Netmiko
and running 'show ip route' and 'show ip arp' in real time.
Uses NetBox IPAM to identify the first-hop gateway for a given source IP.

Used as a NetBrain alternative when the user explicitly requests it.
"""
import json
import logging
import os
import pathlib
import sys
import uuid
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.tools import tool

logging.getLogger("atlas").setLevel(logging.INFO)
logger = logging.getLogger("atlas.agents.nornir")

app = FastAPI(title="Atlas Nornir Path Agent")

SSH_USER     = os.getenv("DEVICE_SSH_USER", "admin")
SSH_PASSWORD = os.getenv("DEVICE_SSH_PASSWORD", "admin")

# ---------------------------------------------------------------------------
# Nornir helpers
# ---------------------------------------------------------------------------

def _build_nornir(hostname: str, host: str, port: int):
    """Build a single-host Nornir instance for one device."""
    from nornir.core import Nornir
    from nornir.core.inventory import (
        Inventory, Hosts, Host, Groups, Defaults, ConnectionOptions,
    )
    from nornir.core.plugins.runners import RunnersPluginRegister
    from nornir.core.plugins.connections import ConnectionPluginRegister
    RunnersPluginRegister.auto_register()
    ConnectionPluginRegister.auto_register()
    runner_cls = RunnersPluginRegister.available["threaded"]

    hosts = {
        hostname: Host(
            name=hostname,
            hostname=host,
            port=port,
            username=SSH_USER,
            password=SSH_PASSWORD,
            platform="eos",
            connection_options={
                "netmiko": ConnectionOptions(
                    extras={"device_type": "arista_eos", "timeout": 30}
                )
            },
        )
    }
    inventory = Inventory(hosts=Hosts(hosts), groups=Groups(), defaults=Defaults())
    runner = runner_cls(num_workers=1)
    return Nornir(inventory=inventory, runner=runner)


def _run_privileged(hostname: str, host: str, port: int, command: str) -> str:
    """SSH into a device, enter enable mode, and run a privileged command (e.g. ping)."""
    from netmiko import ConnectHandler

    device = {
        "device_type": "arista_eos",
        "host": host,
        "port": port,
        "username": SSH_USER,
        "password": SSH_PASSWORD,
        "secret": SSH_PASSWORD,
        "timeout": 30,
    }
    with ConnectHandler(**device) as conn:
        conn.enable()
        return conn.send_command(command, read_timeout=30)


def _run_show(hostname: str, host: str, port: int, command: str) -> str:
    """SSH into a device and run a single show command. Returns raw output."""
    from nornir_netmiko.tasks import netmiko_send_command

    nr = _build_nornir(hostname, host, port)
    result = nr.run(task=netmiko_send_command, command_string=command)
    for _, multi in result.items():
        task_result = multi[0]
        # Raise if nornir captured an exception (avoids returning traceback as output)
        if task_result.failed or task_result.exception:
            exc = task_result.exception
            raise exc if exc else RuntimeError(f"Task failed on {hostname}: {task_result.result}")
        r = task_result.result
        logger.debug("_run_show %s %r → %r", hostname, command, (r or "")[:200])
        if r and not r.strip().startswith("Traceback"):
            return r
        if r and r.strip().startswith("Traceback"):
            raise RuntimeError(f"Task failed on {hostname}: {r[:200]}")
    raise RuntimeError(f"Empty output from {hostname} for '{command}'")


# ---------------------------------------------------------------------------
# Device registry (loaded from Nornir inventory)
# ---------------------------------------------------------------------------

def _load_device_registry() -> dict[str, dict]:
    """
    Return a dict of {hostname: {host, port}} from the Nornir hosts.yaml.
    Falls back to the hardcoded list used by collect_devices.py.
    """
    try:
        import yaml
        hosts_file = Path(__file__).parent.parent / "nornir" / "inventory" / "hosts.yaml"
        with open(hosts_file) as f:
            data = yaml.safe_load(f)
        registry = {}
        for name, cfg in (data or {}).items():
            registry[name] = {
                "host": cfg.get("hostname", "127.0.0.1"),
                "port": cfg.get("port", 22),
            }
        return registry
    except Exception as e:
        logger.warning("Could not load Nornir hosts.yaml: %s — using fallback", e)
        return {
            "arista1": {"host": "127.0.0.1", "port": 6022},
            "arista2": {"host": "127.0.0.1", "port": 6023},
            "arista3": {"host": "127.0.0.1", "port": 6024},
        }


_DEVICE_REGISTRY: dict[str, dict] | None = None


def _registry() -> dict[str, dict]:
    global _DEVICE_REGISTRY
    if _DEVICE_REGISTRY is None:
        _DEVICE_REGISTRY = _load_device_registry()
    return _DEVICE_REGISTRY


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def get_gateway(ip: str) -> dict:
    """
    Look up the gateway (VIP) for the prefix containing this IP using NetBox.
    Use this first to identify the first-hop router for a source IP.
    Returns: {gateway, prefix} or {error}.
    """
    try:
        from atlas.tools.netbox_tools import get_gateway_for_prefix
    except ImportError:
        from tools.netbox_tools import get_gateway_for_prefix
    return get_gateway_for_prefix.fn(ip)


@tool
def find_device_for_ip(ip: str) -> dict:
    """
    Find which device in the inventory owns a given IP address.
    Checks each device's interface IPs by running 'show interfaces | json'.
    Use this to resolve a gateway or next-hop IP to a device hostname and SSH details.
    Returns: {found, device, interface, host, port} or {found: false}.
    """
    registry = _registry()
    for hostname, conn in registry.items():
        try:
            raw = _run_show(hostname, conn["host"], conn["port"], "show interfaces | json")
            data = json.loads(raw)
            for intf_name, intf_data in data.get("interfaces", {}).items():
                for addr_entry in intf_data.get("interfaceAddress", []):
                    primary = addr_entry.get("primaryIp", {})
                    if primary.get("address") == ip:
                        return {
                            "found": True,
                            "device": hostname,
                            "interface": intf_name,
                            "host": conn["host"],
                            "port": conn["port"],
                        }
        except Exception as e:
            logger.debug("find_device_for_ip: skipping %s: %s", hostname, e)
    return {"found": False, "ip": ip}


@tool
def get_route(device: str, destination_ip: str) -> dict:
    """
    Run 'show ip route <destination_ip> | json' on a device via SSH.
    Returns the best matching route: next_hop, egress_interface, protocol, prefix.
    Use this to find where a device will forward traffic toward the destination.
    """
    conn = _registry().get(device)
    if not conn:
        return {"found": False, "error": f"Device '{device}' not in inventory"}
    try:
        raw = _run_show(
            device, conn["host"], conn["port"],
            f"show ip route {destination_ip} | json",
        )
        data = json.loads(raw)
        # Arista: vrfs -> default -> routes -> <prefix> -> vias
        for vrf_name, vrf_data in data.get("vrfs", {}).items():
            routes = vrf_data.get("routes", {})
            if not routes:
                continue
            # Pick the most specific prefix
            best_prefix = max(routes.keys(), key=lambda p: int(p.split("/")[1]) if "/" in p else 0)
            route = routes[best_prefix]
            vias = route.get("vias", [])
            if not vias:
                return {
                    "found": True,
                    "device": device,
                    "prefix": best_prefix,
                    "next_hop": None,
                    "egress_interface": route.get("interface"),
                    "protocol": route.get("routeType", "").lower(),
                    "vrf": vrf_name,
                }
            via = vias[0]
            return {
                "found": True,
                "device": device,
                "prefix": best_prefix,
                "next_hop": via.get("nexthopAddr") or None,
                "egress_interface": via.get("interface"),
                "protocol": route.get("routeType", "").lower(),
                "vrf": vrf_name,
            }
        return {"found": False, "device": device, "destination": destination_ip, "reason": "no route"}
    except Exception as e:
        return {"found": False, "device": device, "destination": destination_ip, "error": str(e)}


@tool
def get_arp(device: str, ip: str) -> dict:
    """
    Run 'show ip arp <ip> | json' on a device via SSH.
    Returns the MAC address and interface for the given IP.
    Use this for the final hop when the destination is directly connected.
    """
    conn = _registry().get(device)
    if not conn:
        return {"found": False, "error": f"Device '{device}' not in inventory"}
    try:
        raw = _run_show(device, conn["host"], conn["port"], f"show ip arp {ip} | json")
        data = json.loads(raw)
        for vrf_name, vrf_data in data.get("vrfs", {}).items():
            for entry in vrf_data.get("ipV4Neighbors", []):
                if entry.get("address") == ip:
                    return {
                        "found": True,
                        "device": device,
                        "ip": ip,
                        "mac": entry.get("hwAddress"),
                        "interface": entry.get("interface"),
                        "vrf": vrf_name,
                    }
        return {"found": False, "device": device, "ip": ip, "reason": "not in ARP table"}
    except Exception as e:
        return {"found": False, "device": device, "ip": ip, "error": str(e)}


@tool
def list_devices() -> dict:
    """
    List all devices in the Nornir inventory with their hostnames and connection details.
    Use this if you need to know what devices are available to query.
    """
    return {"devices": list(_registry().keys()), "details": _registry()}


@tool
def get_arp_table(device: str) -> dict:
    """
    Get the full ARP table from a device via SSH ('show ip arp vrf all | json').
    Use when the user asks to see ARP entries or all IPs known to a device.
    Returns a list of {ip, mac, interface, vrf} entries.
    """
    conn = _registry().get(device)
    if not conn:
        return {"found": False, "error": f"Device '{device}' not in inventory"}
    try:
        raw = _run_show(device, conn["host"], conn["port"], "show ip arp vrf all | json")
        data = json.loads(raw)
        entries = []
        for vrf_name, vrf_data in data.get("vrfs", {}).items():
            for entry in vrf_data.get("ipV4Neighbors", []):
                entries.append({
                    "ip": entry.get("address"),
                    "mac": entry.get("hwAddress"),
                    "interface": entry.get("interface"),
                    "vrf": vrf_name,
                })
        return {"device": device, "count": len(entries), "entries": entries}
    except Exception as e:
        return {"found": False, "device": device, "error": str(e)}


@tool
def get_routing_table(device: str) -> dict:
    """
    Get the full routing table from a device via SSH ('show ip route vrf all | json').
    Use when the user asks to see all routes on a device.
    Returns a list of {prefix, next_hop, interface, protocol, vrf} entries.
    """
    conn = _registry().get(device)
    if not conn:
        return {"found": False, "error": f"Device '{device}' not in inventory"}
    try:
        raw = _run_show(device, conn["host"], conn["port"], "show ip route vrf all | json")
        data = json.loads(raw)
        entries = []
        for vrf_name, vrf_data in data.get("vrfs", {}).items():
            for prefix, route in vrf_data.get("routes", {}).items():
                for via in route.get("vias", [{}]):
                    entries.append({
                        "prefix": prefix,
                        "next_hop": via.get("nexthopAddr"),
                        "interface": via.get("interface"),
                        "protocol": route.get("routeType", "").lower(),
                        "vrf": vrf_name,
                    })
        return {"device": device, "count": len(entries), "entries": entries}
    except Exception as e:
        return {"found": False, "device": device, "error": str(e)}


@tool
def get_mac_table(device: str) -> dict:
    """
    Get the MAC address table from a device via SSH ('show mac address-table | json').
    Use when the user asks which port a MAC or host is connected to.
    Returns a list of {mac, vlan, interface, entry_type} entries.
    """
    conn = _registry().get(device)
    if not conn:
        return {"found": False, "error": f"Device '{device}' not in inventory"}
    try:
        raw = _run_show(device, conn["host"], conn["port"], "show mac address-table | json")
        data = json.loads(raw)
        entries = [
            {
                "mac": e.get("macAddress"),
                "vlan": e.get("vlanId"),
                "interface": e.get("interface"),
                "entry_type": e.get("entryType"),
            }
            for e in data.get("unicastTable", {}).get("tableEntries", [])
        ]
        return {"device": device, "count": len(entries), "entries": entries}
    except Exception as e:
        return {"found": False, "device": device, "error": str(e)}


@tool
def get_interfaces(device: str) -> dict:
    """
    Get interface status and IP addresses from a device via SSH ('show interfaces | json').
    Use when the user asks about interface status, IPs, or errors on a device.
    Returns a list of {name, ip, prefix_len, status, description} entries.
    """
    conn = _registry().get(device)
    if not conn:
        return {"found": False, "error": f"Device '{device}' not in inventory"}
    try:
        raw = _run_show(device, conn["host"], conn["port"], "show interfaces | json")
        data = json.loads(raw)
        entries = []
        for name, intf in data.get("interfaces", {}).items():
            primary = {}
            for addr in intf.get("interfaceAddress", []):
                primary = addr.get("primaryIp", {})
                break
            entries.append({
                "name": name,
                "ip": primary.get("address"),
                "prefix_len": primary.get("maskLen"),
                "line_protocol": intf.get("lineProtocolStatus"),
                "oper_status": intf.get("interfaceStatus"),
                "description": intf.get("description", ""),
            })
        return {"device": device, "count": len(entries), "entries": entries}
    except Exception as e:
        return {"found": False, "device": device, "error": str(e)}


@tool
def get_ospf_neighbors(device: str) -> dict:
    """
    Get OSPF neighbor state from a device via SSH ('show ip ospf neighbor | json').
    Use when the user asks about OSPF adjacencies or neighbor status.
    """
    conn = _registry().get(device)
    if not conn:
        return {"found": False, "error": f"Device '{device}' not in inventory"}
    try:
        raw = _run_show(device, conn["host"], conn["port"], "show ip ospf neighbor | json")
        data = json.loads(raw)
        neighbors = []
        for instance in data.get("vrfs", {}).values():
            for iface in instance.get("instList", {}).values():
                for nbr_id, nbr in iface.get("ospfNeighborEntries", {}).items():
                    neighbors.append({
                        "neighbor_id": nbr_id,
                        "interface": nbr.get("interfaceAddress"),
                        "state": nbr.get("adjacencyState"),
                        "priority": nbr.get("routerPriority"),
                    })
        return {"device": device, "count": len(neighbors), "neighbors": neighbors}
    except Exception as e:
        return {"found": False, "device": device, "error": str(e)}


def _read_counters_via_conn(conn, intf: str) -> dict:
    """Read one snapshot of error counters for a single interface using an existing connection."""
    raw = conn.send_command(f"show interfaces {intf} | json", read_timeout=15)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Non-JSON for {intf}: {raw[:120]!r}") from exc
    intf_data = data.get("interfaces", {}).get(intf, {})
    c = intf_data.get("interfaceCounters", {})
    e = c.get("inputErrorsDetail", {})
    return {
        "inErrors":    c.get("inErrors", 0),
        "outErrors":   c.get("outErrors", 0),
        "inDiscards":  c.get("inDiscards", 0),
        "outDiscards": c.get("outDiscards", 0),
        "fcsErrors":   e.get("fcsErrors", 0),
        "runtFrames":  e.get("runtFrames", 0),
        "giantFrames": e.get("giantFrames", 0),
        "lastClear":   c.get("lastClear", "never"),
    }


@tool
def get_interface_counters(device: str, interfaces: list[str]) -> dict:
    """
    Poll interface error/discard counters 3 times (3-second intervals) and report
    only interfaces where counters are actively incrementing — distinguishing live
    faults from historical ones.

    Returns per-interface deltas: inErrors, outErrors, inDiscards, outDiscards,
    fcsErrors, runtFrames, giantFrames — only for interfaces showing active errors.

    Args:
        device:     Device hostname (must be in inventory).
        interfaces: List of interface names, e.g. ["Ethernet1", "Ethernet3"].
    """
    import time
    from netmiko import ConnectHandler

    conn_info = _registry().get(device)
    if not conn_info:
        return {"found": False, "error": f"Device '{device}' not in inventory"}

    POLL_INTERVAL = 3
    ITERATIONS    = 3
    COUNTER_KEYS  = ("inErrors", "outErrors", "inDiscards", "outDiscards",
                     "fcsErrors", "runtFrames", "giantFrames")

    # Open ONE SSH connection for all iterations — avoids rapid reconnect banner errors
    netmiko_conn = ConnectHandler(
        device_type="arista_eos",
        host=conn_info["host"],
        port=conn_info["port"],
        username=SSH_USER,
        password=SSH_PASSWORD,
        timeout=30,
    )

    # Collect snapshots: snapshots[iteration][intf] = counter dict
    snapshots: list[dict] = []
    try:
        for i in range(ITERATIONS):
            if i > 0:
                time.sleep(POLL_INTERVAL)
            snap = {}
            for intf in interfaces:
                try:
                    snap[intf] = _read_counters_via_conn(netmiko_conn, intf)
                except Exception as exc:
                    snap[intf] = {"error": str(exc)}
            snapshots.append(snap)
    finally:
        netmiko_conn.disconnect()

    # Report only interfaces with at least one incrementing counter
    active = []
    for intf in interfaces:
        first = snapshots[0].get(intf, {})
        last  = snapshots[-1].get(intf, {})
        if "error" in first or "error" in last:
            active.append({"interface": intf, "error": last.get("error", first.get("error"))})
            continue
        deltas = {k: last[k] - first[k] for k in COUNTER_KEYS if isinstance(last.get(k), (int, float))}
        if any(v > 0 for v in deltas.values()):
            active.append({
                "interface":  intf,
                "delta_9s":   deltas,           # change over ~9 seconds
                "last_clear": last.get("lastClear", "never"),
            })

    return {
        "device":           device,
        "poll_interval_s":  POLL_INTERVAL,
        "iterations":       ITERATIONS,
        "active_errors":    active,
        "clean_interfaces": [i for i in interfaces
                             if i not in {r["interface"] for r in active}],
    }


import re as _re


@tool
def ping_from_device(
    device: str, destination: str, source_interface: str = "", vrf: str = ""
) -> dict:
    """
    Run a ping from a network device to a destination IP.
    Optionally source from a specific interface (e.g. 'Ethernet1').
    Optionally specify a VRF (e.g. 'CORP', 'MGMT'). Defaults to global/default routing table.
    Returns {success, loss_pct, rtt_avg_ms, device, destination, vrf, output}.

    Args:
        device:           Device hostname (must be in inventory).
        destination:      Target IP address.
        source_interface: Egress interface to source the ping from (optional).
        vrf:              VRF name to ping within (optional; omit for default/global).
    """
    conn = _registry().get(device)
    if not conn:
        return {
            "success": False,
            "error": f"Device '{device}' not in inventory",
            "device": device,
            "destination": destination,
        }

    # Build ping command — VRF syntax varies by platform but Arista EOS and IOS both support:
    #   ping vrf <vrf> <dest> repeat 5 [source <intf>]
    if vrf and vrf.lower() not in ("", "default", "global"):
        cmd = f"ping vrf {vrf} {destination} repeat 5"
    else:
        cmd = f"ping {destination} repeat 5"
    if source_interface:
        cmd += f" source {source_interface}"

    try:
        raw = _run_privileged(device, conn["host"], conn["port"], cmd)
        loss_m = _re.search(r"(\d+)%\s+packet\s+loss", raw)
        rtt_m  = _re.search(r"min/avg/max.*?=\s*[\d.]+/([\d.]+)/", raw)
        loss_pct = int(loss_m.group(1)) if loss_m else 100
        rtt_avg  = float(rtt_m.group(1)) if rtt_m else None
        return {
            "success":          loss_pct == 0,
            "loss_pct":         loss_pct,
            "rtt_avg_ms":       rtt_avg,
            "device":           device,
            "destination":      destination,
            "vrf":              vrf or "default",
            "source_interface": source_interface or None,
            "output":           raw,
        }
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
            "device": device,
            "destination": destination,
        }


@tool  # also exposed as direct endpoint — see /tcp-test below
def test_tcp_port(
    device: str, destination: str, port: int, vrf: str = "", timeout: int = 5
) -> dict:
    """
    Test TCP connectivity from a network device to a destination IP and port.
    Uses telnet to attempt a TCP handshake — verifies whether a service
    (e.g. HTTPS on 443, SSH on 22) is accepting connections at the network level.
    Returns {reachable, device, destination, port, output}.

    Args:
        device:      Device hostname (must be in inventory).
        destination: Target IP address.
        port:        TCP port to test (e.g. 443, 80, 22).
        vrf:         VRF to source from (optional).
        timeout:     Seconds to wait (default 5).
    """
    conn = _registry().get(device)
    if not conn:
        return {
            "reachable": False,
            "error": f"Device '{device}' not in inventory",
            "device": device, "destination": destination, "port": port,
        }

    if vrf and vrf.lower() not in ("", "default", "global"):
        cmd = f"telnet vrf {vrf} {destination} {port}"
    else:
        cmd = f"telnet {destination} {port}"

    try:
        from netmiko import ConnectHandler
        device_cfg = {
            "device_type": "arista_eos",
            "host": conn["host"], "port": conn["port"],
            "username": SSH_USER, "password": SSH_PASSWORD,
            "secret": SSH_PASSWORD, "timeout": 30,
        }
        with ConnectHandler(**device_cfg) as nc:
            nc.enable()
            raw = nc.send_command_timing(cmd, delay_factor=timeout)
            if "Connected" in raw or "Escape" in raw:
                nc.send_command_timing("quit", delay_factor=2)
                reachable = True
            elif "refused" in raw.lower():
                reachable = False
            else:
                reachable = False
        return {"reachable": reachable, "device": device, "destination": destination, "port": port, "output": raw[:300]}
    except Exception as exc:
        return {"reachable": False, "error": str(exc), "device": device, "destination": destination, "port": port}


@tool
def check_routing_on_hops(
    devices: list[str], destination: str, vrf: str = ""
) -> dict:
    """
    Check the routing table on each device in the path for a specific destination.
    Returns per-device route info: found, vrf, protocol, next_hop, egress interface.
    Use this to find where routing breaks when a ping fails.

    Args:
        devices:     List of device hostnames to check.
        destination: Destination IP address to look up.
        vrf:         VRF to look in. If empty, searches all VRFs and reports which one matched.
    """
    hops: dict[str, dict] = {}
    for device in devices:
        conn = _registry().get(device)
        if not conn:
            hops[device] = {"error": "Not in inventory"}
            continue
        try:
            if vrf and vrf.lower() not in ("", "default", "global"):
                cmd = f"show ip route vrf {vrf} {destination} | json"
            else:
                cmd = f"show ip route {destination} | json"
            raw = _run_show(device, conn["host"], conn["port"], cmd)
            data = json.loads(raw)

            # Search the specified VRF, or scan all VRFs to find where the route lives
            matched_vrf = None
            route = None
            search_vrfs = (
                {vrf: data.get("vrfs", {}).get(vrf, {})}
                if vrf and vrf.lower() not in ("", "default", "global")
                else data.get("vrfs", {})
            )
            for vrf_name, vrf_data in search_vrfs.items():
                routes = vrf_data.get("routes", {})
                r = (
                    routes.get(f"{destination}/32")
                    or next((v for k, v in routes.items() if k.split("/")[0] == destination), None)
                    or (next(iter(routes.values()), None) if routes else None)
                )
                if r:
                    route = r
                    matched_vrf = vrf_name
                    break

            if route:
                via = (route.get("vias") or [{}])[0]
                hops[device] = {
                    "found":     True,
                    "vrf":       matched_vrf or "default",
                    "protocol":  route.get("routeType", "unknown"),
                    "next_hop":  via.get("nexthopAddr"),
                    "interface": via.get("interface"),
                }
            else:
                hops[device] = {"found": False, "vrf": vrf or "all", "error": "No route to destination"}
        except json.JSONDecodeError:
            hops[device] = {"error": f"Non-JSON output: {raw[:80]!r}"}
        except Exception as exc:
            hops[device] = {"error": str(exc)}

    return {"destination": destination, "vrf_requested": vrf or "all", "hops": hops}


NORNIR_TOOLS = [
    get_gateway,
    find_device_for_ip,
    get_route,
    get_arp,
    get_arp_table,
    get_routing_table,
    get_mac_table,
    get_interfaces,
    get_interface_counters,
    get_ospf_neighbors,
    list_devices,
    ping_from_device,
    test_tcp_port,
    check_routing_on_hops,
]

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "nornir_agent.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Atlas Nornir Agent",
    "description": (
        "Queries live network devices via SSH using Nornir. "
        "Supports path tracing, ARP tables, routing tables, MAC tables, "
        "interface status, and OSPF neighbors. "
        "Use when the user wants live data directly from a device."
    ),
    "url": "http://localhost:8006",
    "version": "1.0.0",
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "live_path_trace",
            "name": "Live Path Trace",
            "description": "Trace network path by querying live devices via SSH using Nornir.",
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Trace path from 10.0.100.100 to 10.0.200.200 without NetBrain",
                "Find the route from 10.0.100.55 to 10.0.200.200",
            ],
        }
    ],
}


@app.post("/ping")
async def handle_ping(request: Request) -> JSONResponse:
    """Direct endpoint: {device, destination, source_interface?, vrf?} → ping result."""
    body = await request.json()
    device           = body.get("device", "")
    destination      = body.get("destination", "")
    source_interface = body.get("source_interface", "")
    vrf              = body.get("vrf", "")
    if not device or not destination:
        return JSONResponse({"error": "device and destination required"}, status_code=400)
    result = ping_from_device.invoke({
        "device": device,
        "destination": destination,
        "source_interface": source_interface,
        "vrf": vrf,
    })
    return JSONResponse(result)


@app.post("/tcp-test")
async def handle_tcp_test(request: Request) -> JSONResponse:
    """Direct endpoint: {device, destination, port, vrf?} → TCP reachability result."""
    body = await request.json()
    device      = body.get("device", "")
    destination = body.get("destination", "")
    port        = body.get("port")
    vrf         = body.get("vrf", "")
    if not device or not destination or not port:
        return JSONResponse({"error": "device, destination and port required"}, status_code=400)
    result = test_tcp_port.invoke({
        "device": device, "destination": destination, "port": int(port), "vrf": vrf,
    })
    return JSONResponse(result)


@app.post("/routing-check")
async def handle_routing_check(request: Request) -> JSONResponse:
    """Direct endpoint: {devices: [str], destination, vrf?} → per-device routing result."""
    body = await request.json()
    devices     = body.get("devices", [])
    destination = body.get("destination", "")
    vrf         = body.get("vrf", "")
    if not devices or not destination:
        return JSONResponse({"error": "devices and destination required"}, status_code=400)
    result = check_routing_on_hops.invoke({
        "devices": devices,
        "destination": destination,
        "vrf": vrf,
    })
    return JSONResponse(result)


@app.post("/interface-counters")
async def handle_interface_counters(request: Request) -> JSONResponse:
    """Direct endpoint: {device, interfaces: [str]} → counter data, no LLM loop."""
    body = await request.json()
    device = body.get("device", "")
    interfaces = body.get("interfaces", [])
    if not device or not interfaces:
        return JSONResponse({"error": "device and interfaces required"}, status_code=400)
    result = get_interface_counters.invoke({"device": device, "interfaces": interfaces})
    return JSONResponse(result)


@app.get("/.well-known/agent.json")
async def agent_card():
    return JSONResponse(AGENT_CARD)


# ---------------------------------------------------------------------------
# A2A Task endpoint
# ---------------------------------------------------------------------------

@app.post("/")
async def handle_task(request: Request) -> JSONResponse:
    body = await request.json()
    task_id = body.get("id") or str(uuid.uuid4())
    message = body.get("message", {})
    parts = message.get("parts", [])
    text = next((p.get("text", "") for p in parts if p.get("type") == "text"), "")

    if not text:
        return _error_response(task_id, "No task text provided.")

    logger.info("Nornir path agent task: %s", text)

    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    try:
        result = await run_agent_loop(
            task=text,
            system_prompt=_load_skill(),
            tools=NORNIR_TOOLS,
            max_iterations=20,
        )
    except Exception as e:
        logger.exception("Nornir path agent loop error")
        return _error_response(task_id, f"Agent error: {e}")

    return _success_response(task_id, result)


def _success_response(task_id: str, text: str) -> JSONResponse:
    return JSONResponse({
        "id": task_id,
        "status": {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": text}]}],
    })


def _error_response(task_id: str, message: str) -> JSONResponse:
    return JSONResponse({
        "id": task_id,
        "status": {"state": "failed", "message": message},
        "artifacts": [],
    }, status_code=200)


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    uvicorn.run("nornir_agent:app", host="0.0.0.0", port=8006, reload=False)
