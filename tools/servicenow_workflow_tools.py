"""Agent-facing workflow tool for Atlas-specific ServiceNow correlation."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.services.session_store import session_store
    from atlas.services.servicenow_search_service import servicenow_search_service
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from services.session_store import session_store  # type: ignore
    from services.servicenow_search_service import servicenow_search_service  # type: ignore
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


@tool
async def search_servicenow(
    device_names: list[str],
    config: RunnableConfig,
    source_ip: str = "",
    dest_ip: str = "",
    port: str = "",
    hours_back: int = 24,
) -> str:
    """
    Search ServiceNow for incidents AND change requests related to devices in the path.
    ALWAYS call this after trace_path. Pass every device hostname from the path.
    Change requests look back further than incidents because relevant changes often
    predate the resulting outage by more than a week.

    Args:
        device_names: All device hostnames from trace_path (e.g. ["arista1", "arista2"])
        source_ip:    Source IP of the traffic flow
        dest_ip:      Destination IP of the traffic flow
        port:         Destination port if known (improves relevance filtering)
        hours_back:   Window for incidents (default 24h)
    """
    session_id = sid_from_config(config)
    store = session_store.get(session_id)
    discovered_devices = await servicenow_search_service.resolve_devices(
        store=store,
        device_names=device_names,
        dest_ip=dest_ip,
    )
    await push_status(session_id, f"Checking ServiceNow for {', '.join(discovered_devices)}...")
    summary = await servicenow_search_service.search_summary(
        session_id=session_id,
        devices=discovered_devices,
        source_ip=source_ip,
        dest_ip=dest_ip,
        port=port,
        hours_back=hours_back,
    )
    return session_store.set_servicenow_summary(session_id, summary)


SERVICENOW_WORKFLOW_TOOL_CAPABILITIES = (
    (search_servicenow, ("servicenow.search",)),
)
