"""Agent-facing workflow tools for active tests and interface diagnostics."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.services.device_diagnostics_service import device_diagnostics_service
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from services.device_diagnostics_service import device_diagnostics_service  # type: ignore
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


@tool
async def ping_device(
    device: str,
    destination: str,
    config: RunnableConfig,
    source_interface: str = "",
    vrf: str = "",
) -> str:
    """
    Send ICMP pings from a network device to a destination IP via live SSH.

    Forward ping:  device=first_hop_device, destination=dest_ip, source_interface=first_hop_lan_interface, vrf=src_vrf
    Reverse ping:  device=reverse_first_hop_device,  destination=src_ip,  source_interface=reverse_first_hop_lan_interface,  vrf=dst_vrf

    Returns: success/failure, packet loss %, RTT.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Pinging {device} → {destination}...")
    return await device_diagnostics_service.ping_summary(
        session_id=session_id,
        device=device,
        destination=destination,
        source_interface=source_interface,
        vrf=vrf,
    )


@tool
async def test_tcp_port(
    device: str,
    destination: str,
    port: str,
    config: RunnableConfig,
    vrf: str = "",
) -> str:
    """
    Test TCP reachability to destination:port from a network device via live SSH.
    Use last_hop_device from trace_path as the device — it is closest to the destination.
    Call this when ping passes but application connectivity is still failing.

    Returns: reachable (True/False) and error details if unreachable.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Testing TCP {device} → {destination}:{port}...")
    return await device_diagnostics_service.tcp_test_summary(
        session_id=session_id,
        device=device,
        destination=destination,
        port=port,
        vrf=vrf,
    )


@tool
async def check_routing(
    devices: list[str],
    destination: str,
    config: RunnableConfig,
    vrf: str = "",
) -> str:
    """
    Check the routing table on multiple devices for a destination IP via live SSH.
    Call this when ping fails to identify which hop loses the route.
    Pass all path devices from trace_path.

    Returns: per-device route info (next-hop, egress interface, VRF, protocol).
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Checking routing on {', '.join(devices)}...")
    return await device_diagnostics_service.routing_check_summary(
        devices=devices,
        destination=destination,
        vrf=vrf,
    )


@tool
async def get_interface_counters(
    devices_and_interfaces: list[dict],
    config: RunnableConfig,
) -> str:
    """
    Poll interface error and discard counters 3× over ~9 seconds on path interfaces.
    Returns ONLY actively incrementing counters — clean interfaces are suppressed.
    Call in parallel with search_servicenow after trace_path.

    Args:
        devices_and_interfaces: List of {"device": str, "interfaces": [str]} dicts.
            Use path_hops_for_counters from the trace_path output.
            Example: [{"device": "arista1", "interfaces": ["Ethernet1", "Ethernet3"]}]
    """
    session_id = sid_from_config(config)
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

    await push_status(session_id, f"Polling interface counters: {', '.join(e['device'] for e in valid)}...")
    return await device_diagnostics_service.interface_counters_summary(
        session_id=session_id,
        devices_and_interfaces=valid,
    )


@tool
async def get_interface_detail(
    device: str,
    interface: str,
    config: RunnableConfig,
) -> str:
    """
    Fetch full operational status and error counters for a specific interface via live SSH.
    Use when path trace shows an interface DOWN or when you need oper_status details.

    Returns: line-protocol, oper_status, input/output errors, description.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Checking interface {device}/{interface}...")
    return await device_diagnostics_service.interface_detail_summary(
        session_id=session_id,
        device=device,
        interface=interface,
    )


@tool
async def get_all_interfaces(device: str, config: RunnableConfig) -> str:
    """
    List all non-management interfaces, their up/down state, and primary IP on a device.
    Use for device health queries or when you need to map an OSPF/syslog interface IP to
    a concrete interface and determine whether that interface is down.

    Returns: per-interface oper_status, line-protocol, description, and primary IP.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Getting all interfaces on {device}...")
    return await device_diagnostics_service.all_interfaces_summary(
        session_id=session_id,
        device=device,
    )


DEVICE_TOOL_CAPABILITIES = (
    (ping_device, ("workflow.connectivity.ping",)),
    (test_tcp_port, ("workflow.connectivity.tcp_test",)),
    (check_routing, ("workflow.routing.check",)),
    (get_interface_counters, ("workflow.interfaces.counters",)),
    (get_interface_detail, ("workflow.interfaces.detail",)),
    (get_all_interfaces, ("workflow.interfaces.inventory",)),
)
