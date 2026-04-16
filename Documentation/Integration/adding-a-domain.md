# Adding a New Domain / Integration

This guide covers connecting a brand new backend domain or external system to Atlas.

Examples:
- a new ITSM system
- a new inventory platform
- a new vendor knowledge source
- a new network-control backend

The current Atlas architecture is owner-based, so adding a new domain means adding a small set of clear owners, not scattering logic across the app.

---

## What a New Domain Usually Needs

For most integrations, you will add some combination of:

- a backend client or owned service
- one or more agent-facing tool modules
- capability registrations
- health/diagnostics reporting
- tests

You should **not** add a new integration by putting everything into `chat_service.py`, `graph_nodes.py`, or a giant shared tool file.

---

## Checklist

- [ ] 1. Decide what kind of owner the integration needs
- [ ] 2. Add configuration / credentials
- [ ] 3. Create the owned backend client or service
- [ ] 4. Add agent-facing tools
- [ ] 5. Register capabilities in the tool manifest and `ToolRegistry`
- [ ] 6. Add health/diagnostics support if this is a live dependency
- [ ] 7. Add tests
- [ ] 8. Update docs if the architecture or user-visible workflow changed

---

## Step 1: Decide the Owner Shape

Use the smallest owner that cleanly matches the integration.

Typical options:

### Backend client

Use this when the integration is mostly transport, retry, timeout, and caching behavior.

Example:
- `services/nornir_client.py`

### Domain service

Use this when the integration needs Atlas-specific orchestration or correlation behavior.

Examples:
- `services/servicenow_search_service.py`
- `services/connectivity_snapshot_service.py`
- `services/routing_diagnostics_service.py`

### Thin product-facing tool adapter

Use this when the backend already exposes a task-level action and Atlas mostly needs to present it as an agent tool.

Example:
- `tools/servicenow_agent_tools.py`

---

## Step 2: Add Configuration and Credentials

Centralize configuration and credential access.

Current pattern:
- shared backend constants in `tools/shared.py` where applicable
- Azure Key Vault or environment variables for secrets
- backend/service owners import configuration, rather than each tool reloading it

Keep credentials out of:
- prompts
- graph nodes
- workflow services that do not own that backend

If the new backend needs health checks, add enough config to support a simple readiness probe as well.

---

## Step 3: Create the Owned Backend Layer

Example shape:

```python
# services/example_backend_client.py
class ExampleBackendClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    async def get_record(self, record_id: str) -> dict:
        ...


example_backend_client = ExampleBackendClient(base_url=EXAMPLE_URL)
```

Responsibilities that belong here:
- transport details
- retries
- timeout behavior
- caching
- backend-specific request/response normalization

Responsibilities that do **not** belong here:
- agent prompts
- graph routing
- UI payload shaping

---

## Step 4: Add Agent-Facing Tools

Expose the integration through one or more tool modules depending on whether the actions are:
- product-facing
- workflow-facing

Examples from Atlas:
- product-facing:
  - `tools/servicenow_agent_tools.py`
  - `tools/memory_agent_tools.py`
  - `tools/knowledge_agent_tools.py`
- workflow-facing:
  - `tools/path_agent_tools.py`
  - `tools/device_agent_tools.py`
  - `tools/routing_agent_tools.py`
  - `tools/connectivity_agent_tools.py`
  - `tools/servicenow_workflow_tools.py`

If the new integration introduces a new category, add a new dedicated tool module rather than overloading an unrelated one.

---

## Step 5: Register Capabilities

Each tool module owns its capability manifest.

Examples:
- `WORKFLOW_TOOL_CAPABILITIES`
- `SERVICENOW_TOOL_CAPABILITIES`
- `MEMORY_TOOL_CAPABILITIES`
- `KNOWLEDGE_TOOL_CAPABILITIES`

Then add those capabilities to the correct agent profile in:
- `tools/tool_registry.py`

This is what actually makes the integration available to an agent.

---

## Step 6: Add Health / Diagnostics Support

If the new integration is a live runtime dependency, wire it into:
- `services/health_service.py`
- `services/diagnostics_service.py`

That ensures:
- header health status stays meaningful
- Diagnostics shows whether the backend is reachable
- failures are visible without running a full prompt

If it is not a runtime-critical backend, this step can be skipped.

---

## Step 7: Use Shared Failure Contracts

Do not invent backend errors ad hoc.

Use the helpers in:
- `services/backend_contracts.py`

This keeps failures uniform across:
- product-facing tools
- workflow tools
- backend services

---

## Step 8: Add Tests

Recommended test layers:

1. Owner/client tests
- backend success
- backend failure

2. Tool tests
- success path
- error contract path

3. Registry/profile tests
- capability is registered
- right agent profile sees it

4. Diagnostics/health tests if the integration is live

Current examples:
- `tests/test_servicenow_search_service.py`
- `tests/test_servicenow_agent_tools.py`
- `tests/test_application_owners.py`
- `tests/test_diagnostics_service.py`

---

## Summary

In the current Atlas design, adding a domain means:

1. create a clear owner for backend behavior
2. expose task-level tools through the right tool module
3. register capabilities through `ToolRegistry`
4. wire health/diagnostics if needed
5. test the owner and the tool surface

That keeps the architecture consistent with the current owner-based, capability-driven design instead of drifting back toward a monolithic integration model.
