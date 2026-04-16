"""Agent-facing workflow tools for routing, OSPF, and syslog diagnostics."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.services.routing_diagnostics_service import routing_diagnostics_service
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from services.routing_diagnostics_service import routing_diagnostics_service  # type: ignore
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


@tool
async def get_device_syslog(device: str, config: RunnableConfig, interface: str = "") -> str:
    """
    Fetch recent syslog messages from a device via live SSH.
    Filters to link/down/flap events. Pass interface to scope to a specific port.
    Use when you need to determine WHEN an interface went down or OSPF dropped.

    Returns: timestamped syslog lines.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Fetching syslog from {device}...")
    return await routing_diagnostics_service.device_syslog_summary(
        session_id=session_id,
        device=device,
        interface=interface,
    )


@tool
async def inspect_ospf_peering(
    device_a: str,
    interface_a: str,
    device_b: str,
    interface_b: str,
    config: RunnableConfig,
    ip_a: str = "",
    ip_b: str = "",
) -> str:
    """
    Inspect a specific OSPF peering end-to-end on both devices via live SSH.
    Use this when routing history identifies a concrete peering pair such as
    `ai3 Ethernet2 <-> ai4 Ethernet2`.

    Returns:
      - interface state and primary IP on both sides
      - interface counters on both sides
      - interface detail on both sides
      - recent syslog with OSPF/IP correlation on both sides
      - bilateral ping results across the peering IPs when provided
      - an explicit diagnosis class and recommended next action
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Inspecting OSPF peering {device_a}/{interface_a} <-> {device_b}/{interface_b}...")
    return await routing_diagnostics_service.inspect_ospf_peering_summary(
        session_id=session_id,
        device_a=device_a,
        interface_a=interface_a,
        device_b=device_b,
        interface_b=interface_b,
        ip_a=ip_a,
        ip_b=ip_b,
    )


@tool
async def check_ospf_neighbors(devices: list[str], config: RunnableConfig) -> str:
    """
    Check current OSPF neighbor adjacency state on each device via live SSH.
    Call in parallel with check_ospf_interfaces and lookup_ospf_history.
    Pass all path devices plus any historically known devices from lookup_routing_history.

    Returns: per-device neighbor count, router-IDs, interfaces, states.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Checking OSPF neighbors on {', '.join(devices)}...")
    return await routing_diagnostics_service.ospf_neighbors_summary(session_id=session_id, devices=devices)


@tool
async def check_ospf_interfaces(devices: list[str], config: RunnableConfig) -> str:
    """
    Check which interfaces are currently reported by 'show ip ospf interface brief' on each device.
    A device with ospf_interface_count=0 has no interfaces currently reported by that command, but
    this is NOT by itself proof of misconfiguration: an interface-down condition can also result in 0.
    Correlate with get_all_interfaces, get_device_syslog, and lookup_ospf_history before concluding
    whether the issue is misconfiguration or a physical/link failure.
    Call in parallel with check_ospf_neighbors and lookup_ospf_history.

    Returns: per-device ospf_interface_count and the list of interfaces currently reported by OSPF.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Checking OSPF interface config on {', '.join(devices)}...")
    return await routing_diagnostics_service.ospf_interfaces_summary(session_id=session_id, devices=devices)


@tool
async def lookup_ospf_history(devices: list[str], config: RunnableConfig) -> str:
    """
    Compare each device's current OSPF neighbor count against its last 10 historical snapshots.
    Use to confirm whether a device previously had OSPF neighbors before they were lost.
    Call in parallel with check_ospf_neighbors and check_ospf_interfaces.

    Returns: per-device historical neighbor trend and current count.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Looking up OSPF history for {', '.join(devices)}...")
    return await routing_diagnostics_service.ospf_history_summary(session_id=session_id, devices=devices)


@tool
async def lookup_routing_history(destination_ip: str, config: RunnableConfig) -> str:
    """
    Query the routing history database for:
    1. All devices that historically had a data-plane route to destination_ip.
    2. The last known good route (egress interface, next-hop, protocol, prefix).

    Use after trace_path to find devices that SHOULD be in the path but aren't
    (e.g. because they're down) — include these in OSPF checks.
    Call in parallel with search_servicenow and get_interface_counters.

    Returns: historically known path devices + last known route details.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Looking up routing history for {destination_ip}...")
    return await routing_diagnostics_service.routing_history_summary(
        session_id=session_id,
        destination_ip=destination_ip,
    )


ROUTING_TOOL_CAPABILITIES = (
    (get_device_syslog, ("workflow.device.syslog",)),
    (inspect_ospf_peering, ("workflow.ospf.peering.inspect",)),
    (check_ospf_neighbors, ("workflow.ospf.neighbors",)),
    (check_ospf_interfaces, ("workflow.ospf.interfaces",)),
    (lookup_ospf_history, ("workflow.ospf.history",)),
    (lookup_routing_history, ("workflow.routing.history",)),
)
