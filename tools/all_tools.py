"""Compatibility export layer for Atlas workflow tools.

The final design keeps agent-facing workflow tools split by domain:
- `path_agent_tools.py`
- `device_agent_tools.py`
- `routing_agent_tools.py`
- `connectivity_agent_tools.py`
- `servicenow_workflow_tools.py`

This module remains as a thin aggregator so existing imports continue to work
while `ToolRegistry` can still assemble one uniform workflow capability set.
"""
from __future__ import annotations

from tools.connectivity_agent_tools import (
    CONNECTIVITY_WORKFLOW_TOOL_CAPABILITIES,
    collect_connectivity_snapshot,
)
from tools.device_agent_tools import (
    DEVICE_TOOL_CAPABILITIES,
    check_routing,
    get_all_interfaces,
    get_interface_counters,
    get_interface_detail,
    ping_device,
    test_tcp_port,
)
from tools.path_agent_tools import PATH_TOOL_CAPABILITIES, trace_path, trace_reverse_path
from tools.routing_agent_tools import (
    ROUTING_TOOL_CAPABILITIES,
    check_ospf_interfaces,
    check_ospf_neighbors,
    get_device_syslog,
    inspect_ospf_peering,
    lookup_ospf_history,
    lookup_routing_history,
)
from tools.servicenow_workflow_tools import (
    SERVICENOW_WORKFLOW_TOOL_CAPABILITIES,
    search_servicenow,
)
from tools.tool_runtime import push_status as _push_status
from tools.tool_runtime import sid_from_config as _sid


WORKFLOW_TOOL_CAPABILITIES = (
    PATH_TOOL_CAPABILITIES
    + DEVICE_TOOL_CAPABILITIES
    + ROUTING_TOOL_CAPABILITIES
    + CONNECTIVITY_WORKFLOW_TOOL_CAPABILITIES
    + SERVICENOW_WORKFLOW_TOOL_CAPABILITIES
)


__all__ = [
    "_push_status",
    "_sid",
    "WORKFLOW_TOOL_CAPABILITIES",
    "trace_path",
    "trace_reverse_path",
    "ping_device",
    "test_tcp_port",
    "check_routing",
    "get_interface_counters",
    "get_interface_detail",
    "get_all_interfaces",
    "get_device_syslog",
    "inspect_ospf_peering",
    "check_ospf_neighbors",
    "check_ospf_interfaces",
    "lookup_ospf_history",
    "lookup_routing_history",
    "collect_connectivity_snapshot",
    "search_servicenow",
]
