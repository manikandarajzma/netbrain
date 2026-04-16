"""Atlas tool modules.

- `all_tools.py` is a thin compatibility export layer for workflow tools.
- `path_agent_tools.py`, `device_agent_tools.py`, `routing_agent_tools.py`,
  `connectivity_agent_tools.py`, and `servicenow_workflow_tools.py` contain the
  real workflow entrypoints exposed to agents.
- `services/path_trace_service.py` owns live forward/reverse path tracing and
  path metadata extraction used by the workflow layer.
- `services/connectivity_snapshot_service.py` owns the heavyweight
  connectivity evidence bundle used by the workflow layer.
- `services/routing_diagnostics_service.py` owns OSPF, routing-history, and
  syslog-oriented diagnostic workflow used by the workflow layer.
- `services/servicenow_search_service.py` owns Atlas-specific ServiceNow
  incident/change correlation used by workflow tools.
- `knowledge_agent_tools.py` contains vendor knowledge lookup tools.
- `memory_agent_tools.py` contains memory-facing recall tools for agents.
- `servicenow_agent_tools.py` contains thin product-facing ServiceNow adapters
  for agent CRUD/detail actions.
- MCP server domain modules are imported separately by `mcp_server.py` to
  register backend tools on the shared FastMCP instance.
"""
