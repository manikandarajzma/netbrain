"""Agent-facing workflow tools for live path tracing."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.services.path_trace_service import path_trace_service
    from atlas.services.session_store import session_store
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from services.path_trace_service import path_trace_service  # type: ignore
    from services.session_store import session_store  # type: ignore
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


@tool
async def trace_path(source_ip: str, dest_ip: str, config: RunnableConfig) -> str:
    """
    Trace the hop-by-hop network path from source_ip to dest_ip via live SSH.
    Always call this FIRST for any connectivity troubleshooting query.
    Returns the path text including all device names, egress interfaces, and
    any anomalies (management-routing fallback, interface down, no route).

    The result tells you:
    - All device hostnames in the path (use these for ServiceNow, OSPF checks, etc.)
    - The first-hop device (use for ping source)
    - The last-hop device (use for TCP test source)
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Tracing path {source_ip} → {dest_ip} via live SSH...")

    store = session_store.get(session_id)
    text, hops, flags = await path_trace_service.live_path_trace(
        source_ip,
        dest_ip,
        session_id=session_id,
        store=store,
    )

    store["path_hops"] = hops
    store["path_flags"] = flags
    meta = path_trace_service.extract_path_metadata(hops)
    store["path_meta"] = meta
    meta["src_vrf"] = await path_trace_service.infer_vrf(source_ip, meta.get("first_hop_device", ""))

    if meta["first_hop_device"]:
        text += f"\n\nfirst_hop_device: {meta['first_hop_device']}"
        text += f"\nfirst_hop_lan_interface: {meta['first_hop_lan_interface']}"
        text += f"\nfirst_hop_egress_interface: {meta['first_hop_egress_interface']}"
    if meta["last_hop_device"]:
        text += f"\nlast_hop_device: {meta['last_hop_device']}"
        text += f"\nlast_hop_egress_interface: {meta['last_hop_egress_interface']}"
    text += f"\npath_devices: {', '.join(meta['path_devices'])}"
    text += f"\nsrc_vrf: {meta['src_vrf']}"
    return text


@tool
async def trace_reverse_path(source_ip: str, dest_ip: str, config: RunnableConfig) -> str:
    """
    Trace the return path from dest_ip back to source_ip via live SSH.
    Call in parallel with search_servicenow and get_interface_counters after trace_path.
    Maps the return path for comparison with the forward trace.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Tracing return path {dest_ip} → {source_ip}...")

    store = session_store.get(session_id)
    text, hops, _ = await path_trace_service.live_path_trace(
        dest_ip,
        source_ip,
        session_id=session_id,
        store=store,
    )

    if hops:
        store["reverse_path_hops"] = hops
        meta = path_trace_service.extract_reverse_path_metadata(hops)
        store["reverse_path_meta"] = meta
        if meta["reverse_first_hop_device"]:
            text += f"\n\nreverse_first_hop_device: {meta['reverse_first_hop_device']}"
            text += f"\nreverse_first_hop_lan_interface: {meta['reverse_first_hop_lan_interface']}"
            text += f"\nreverse_first_hop_egress_interface: {meta['reverse_first_hop_egress_interface']}"
        if meta["reverse_last_hop_device"]:
            text += f"\nreverse_last_hop_device: {meta['reverse_last_hop_device']}"
            text += f"\nreverse_last_hop_egress_interface: {meta['reverse_last_hop_egress_interface']}"

    return text


PATH_TOOL_CAPABILITIES = (
    (trace_path, ("workflow.path.trace",)),
    (trace_reverse_path, ("workflow.path.reverse_trace",)),
)
