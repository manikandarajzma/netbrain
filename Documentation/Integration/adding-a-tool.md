# Adding a New Tool

This guide covers adding a new **agent-facing tool** in the current Atlas architecture.

Atlas no longer registers tools by stuffing everything into one monolithic module or by teaching `application/chat_service.py` about every tool individually.

The current pattern is:

1. put backend logic in an owned service or client
2. expose a thin agent-facing tool in the correct tool module
3. register a capability in that module's manifest
4. add the capability to the right `ToolRegistry` profile
5. add tests

---

## Decide What Kind of Tool You Are Adding

There are two tool categories in Atlas:

### 1. Product-facing tools

These already match a user task directly.

Examples:
- `get_incident_details(...)`
- `create_servicenow_incident(...)`
- `update_servicenow_change_request(...)`

These should stay thin.

Good home:
- `tools/servicenow_agent_tools.py`
- another dedicated `*_agent_tools.py` module if you add a new product-facing integration

### 2. Workflow tools

These support Atlas-specific investigation workflows and often compose multiple backend calls or write session-side effects.

Examples:
- `trace_path(...)`
- `collect_connectivity_snapshot(...)`
- `search_servicenow(...)`

These belong in the workflow tool modules:
- `tools/path_agent_tools.py`
- `tools/device_agent_tools.py`
- `tools/routing_agent_tools.py`
- `tools/connectivity_agent_tools.py`
- `tools/servicenow_workflow_tools.py`

---

## Current Registration Model

Atlas uses capability-based registration through `ToolRegistry`.

That means:
- the tool module owns the tool implementation
- the tool module also exports the capability manifest
- `ToolRegistry` resolves profiles to capabilities and then to concrete tool objects

You do **not** update `application/chat_service.py` to register a tool.

---

## Checklist

- [ ] 1. Put backend logic in the right owned service/client
- [ ] 2. Add the agent-facing tool in the right tool module
- [ ] 3. Add or update the capability manifest in that module
- [ ] 4. Add the capability to the correct profile in `tools/tool_registry.py`
- [ ] 5. Add tests
- [ ] 6. Update docs if the tool changes the public architecture or workflows

---

## Step 1: Put Backend Logic in the Right Owner

Do not put transport, retry, caching, or backend orchestration directly into the tool wrapper unless the wrapper is truly trivial.

Current examples:
- `services/nornir_client.py`
- `services/path_trace_service.py`
- `services/device_diagnostics_service.py`
- `services/routing_diagnostics_service.py`
- `services/connectivity_snapshot_service.py`
- `services/servicenow_search_service.py`

Rule of thumb:
- if the logic talks to a backend, owns caching, or shapes backend results repeatedly, it belongs in a service/client
- if the logic is only a small agent-facing adapter, it can stay in the tool module

---

## Step 2: Add the Tool in the Correct Module

Example: adding a new product-facing ServiceNow detail tool.

```python
# tools/servicenow_agent_tools.py
from langchain_core.tools import tool

from atlas.integrations.mcp_client import call_mcp_tool
from atlas.services.backend_contracts import lookup_error, not_found


@tool
def get_problem_details(problem_number: str) -> str:
    """
    Retrieve details for a ServiceNow problem record.

    Use for:
    - "show me PRB0012345"
    - "details about PRB0012345"
    """
    result = call_mcp_tool("get_servicenow_problem", {"number": problem_number})
    if isinstance(result, dict) and result.get("error"):
        return lookup_error("ServiceNow", result["error"])
    if not result:
        return not_found("Problem", problem_number)
    return f"Problem {problem_number}: {result}"
```

Keep the tool surface task-level and small.

---

## Step 3: Add the Capability Manifest Entry

Each tool module owns its own capability map.

Example:

```python
# tools/servicenow_agent_tools.py
SERVICENOW_TOOL_CAPABILITIES = {
    "servicenow.problem.read": get_problem_details,
}
```

Workflow modules do the same thing with their own manifest maps.

This places the registration metadata next to the tool implementation.

---

## Step 4: Add the Capability to the Right Tool Profile

`ToolRegistry` owns which agents get which capabilities.

Example:

```python
# tools/tool_registry.py
"network_ops": {
    "servicenow.incident.read",
    "servicenow.change.read",
    "servicenow.problem.read",
}
```

That is the main step that makes the tool available to an agent.

---

## Step 5: Keep Error Contracts Consistent

Use the shared backend contract helpers instead of inventing ad hoc error strings.

Current helpers live in:
- `services/backend_contracts.py`

Examples:
- `backend_unavailable(...)`
- `lookup_error(...)`
- `not_found(...)`
- `operation_failed(...)`
- `unexpected_response(...)`
- `verification_failed(...)`

This applies the shared backend failure contract to the tool.

---

## Step 6: Add Tests

At minimum, add:

1. Tool behavior test
- success path
- failure path

2. Registry/profile test
- capability resolves correctly
- agent profile includes it when expected

Good examples:
- `tests/test_servicenow_agent_tools.py`
- `tests/test_application_owners.py`

If the tool affects a workflow, add workflow-level coverage too.

---

## Step 7: Update Docs Only Where It Matters

Update documentation if the tool changes:
- the agent-facing capability set
- a documented workflow
- diagnostics/health surfaces
- architecture ownership

Typical places:
- `README.md`
- `Documentation/Architecture/design.md`
- `Documentation/End-to-End-flow/troubleshooting-query-flow.md`

---

## Summary

In the current architecture, adding a tool means:

1. decide whether it is product-facing or workflow-facing
2. put backend logic in the correct owner
3. add the thin tool wrapper in the right tool module
4. register its capability in that module
5. add the capability to a `ToolRegistry` profile
6. test it

That is the current Atlas design, and it is intentionally different from the older “register everything in chat_service and one big tool module” approach.
