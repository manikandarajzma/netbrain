"""Agent-facing workflow tools for connectivity evidence bundles."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.services.connectivity_snapshot_service import connectivity_snapshot_service
    from atlas.services.observability import json_safe
    from atlas.services.nornir_client import nornir_client
    from atlas.services.path_trace_service import path_trace_service
    from atlas.services.session_store import session_store
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from services.connectivity_snapshot_service import connectivity_snapshot_service  # type: ignore
    from services.observability import json_safe  # type: ignore
    from services.nornir_client import nornir_client  # type: ignore
    from services.path_trace_service import path_trace_service  # type: ignore
    from services.session_store import session_store  # type: ignore
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


@tool
async def collect_connectivity_snapshot(
    source_ip: str,
    dest_ip: str,
    config: RunnableConfig,
    port: str = "",
) -> str:
    """
    Collect a holistic connectivity snapshot for the current incident.

    This is a discovery-first evidence bundle for the agent:
    - current forward and reverse topology clues
    - historical devices / primary peering hint
    - per-device routing protocol discovery
    - spanning-tree mode when relevant
    - route status toward source/destination
    - relevant interface state/detail/counters
    - protocol-specific OSPF evidence only when OSPF is actually discovered
    - destination-side TCP check when a port is provided

    Use this before writing the final report when you need to reason about
    multiple simultaneous issues across the path.
    """
    session_id = sid_from_config(config)
    store = session_store.get(session_id)

    if not store.get("path_hops"):
        text, hops, flags = await path_trace_service.live_path_trace(
            source_ip,
            dest_ip,
            session_id=session_id,
            store=store,
        )
        store["path_hops"] = hops
        store["path_flags"] = flags
        meta = path_trace_service.extract_path_metadata(hops)
        meta["src_vrf"] = await path_trace_service.infer_vrf(source_ip, meta.get("first_hop_device", ""))
        store["path_meta"] = meta
        store["path_text"] = text
    if not store.get("reverse_path_hops"):
        text, hops, _ = await path_trace_service.live_path_trace(
            dest_ip,
            source_ip,
            session_id=session_id,
            store=store,
        )
        store["reverse_path_hops"] = hops
        store["reverse_path_meta"] = path_trace_service.extract_reverse_path_metadata(hops)
        store["reverse_path_text"] = text

    if not store.get("routing_history"):
        try:
            try:
                from atlas.db import fetch as _fetch, fetchrow as _fetchrow
            except ImportError:
                from db import fetch as _fetch, fetchrow as _fetchrow  # type: ignore

            hist_devs = await _fetch(
                """
                SELECT DISTINCT device FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface IS NOT NULL
                  AND egress_interface NOT ILIKE 'management%'
                """,
                dest_ip,
            )
            last_route = await _fetchrow(
                """
                SELECT device, egress_interface, next_hop::text, protocol, prefix::text, collected_at
                FROM routing_history
                WHERE $1::inet << prefix
                  AND egress_interface NOT ILIKE 'management%'
                ORDER BY collected_at DESC LIMIT 1
                """,
                dest_ip,
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
                dest_ip,
            )
            peer_hint = None
            peering_source = last_upstream_route or last_route
            if peering_source and peering_source.get("next_hop"):
                try:
                    peer_data = await nornir_client.cached_post(
                        session_id,
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
            store["routing_history"] = json_safe(
                {
                    "historical_devices": [row["device"] for row in hist_devs],
                    "last_route": dict(last_route) if last_route else None,
                    "last_upstream_route": dict(last_upstream_route) if last_upstream_route else None,
                    "peer_hint": peer_hint,
                }
            )
        except Exception as exc:
            store["routing_history"] = {"error": str(exc)}

    return await connectivity_snapshot_service.build_snapshot_summary(
        session_id=session_id,
        store=store,
        source_ip=source_ip,
        dest_ip=dest_ip,
        port=port,
        push_status=push_status,
    )


CONNECTIVITY_WORKFLOW_TOOL_CAPABILITIES = (
    (collect_connectivity_snapshot, ("workflow.connectivity.snapshot",)),
)
