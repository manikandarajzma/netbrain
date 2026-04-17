# Atlas Code Map

This document is the file-by-file reference for the Atlas codebase.

Use it when you want to answer questions like:

- which file owns intent routing
- which file builds the prompts
- which file decides tool exposure
- which file calls MCP
- which file talks to Nornir
- which file shapes the frontend payload

It is organized by layer, starting from the browser entrypoint and moving downward into agents, tools, services, and backend integrations.

## 1. Root Entry Files

### [`app.py`](<app.py>)

Primary FastAPI application.

Owns:

- HTTP routes
- `POST /api/chat`
- `GET /health`
- `GET /api/internal/diagnostics`
- authentication/session enforcement
- SSE lifecycle for chat responses
- login/logout/auth callback routes
- startup background jobs such as ServiceNow memory sync kickoff

### [`run_web.py`](<run_web.py>)

Recommended local launcher for the Atlas web app.

Owns:

- local uvicorn startup
- log configuration bootstrap
- development launch path for the backend

### [`mcp_server.py`](<mcp_server.py>)

Local MCP server entrypoint.

Owns:

- starting the MCP tool server that exposes backend-facing ServiceNow tools

## 2. Application Layer

### [`application/chat_service.py`](<application/chat_service.py>)

Thin application entrypoint between `app.py` and `AtlasApplication`.

Owns:

- delegating a chat request into the top-level application owner
- a small shared IP/CIDR regex used by routing/helpers

### [`application/atlas_application.py`](<application/atlas_application.py>)

Top-level owner for the Atlas application.

Owns:

- `AtlasRuntime`
- `AgentFactory`
- `MemoryManager`
- `ToolRegistry`
- `ResponsePresenter`
- workflow owners and diagnostics owner

Entry method:

- `AtlasApplication.process_query(...)`

### [`application/status_bus.py`](<application/status_bus.py>)

Per-session SSE queue registry.

Owns:

- registering status queues
- deregistering status queues
- pushing UI status messages into the active session queue

## 3. Graph Layer

### [`graph/graph_builder.py`](<graph/graph_builder.py>)

LangGraph topology owner.

Owns:

- graph construction
- graph compilation
- node wiring
- conditional routing map

### [`graph/graph_nodes.py`](<graph/graph_nodes.py>)

Graph boundary logic.

Owns:

- `classify_intent(...)`
- thin delegation to troubleshoot and network-ops workflow services
- dismiss/exit handling

This is where coarse intent routing happens.

### [`graph/graph_state.py`](<graph/graph_state.py>)

Typed LangGraph state definition.

Owns:

- request prompt
- conversation history
- username
- session id
- request id
- intent
- final response payload

## 4. Agent Layer

### [`agents/agent_factory.py`](<agents/agent_factory.py>)

Shared ReAct agent builder.

Owns:

- building `create_react_agent(...)` instances
- attaching the system prompt
- attaching the allowed tool list
- role-specific LLM builders for:
  - router
  - selector
  - network ops
  - troubleshoot

### [`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>)

Troubleshoot agent builder.

Owns:

- loading [`skills/troubleshooter.md`](<skills/troubleshooter.md>)
- loading one selected scenario prompt from `skills/troubleshooting_scenarios/`
- choosing the appropriate troubleshoot tool profile from `ToolRegistry`
- building the troubleshoot ReAct agent

### [`agents/network_ops_agent.py`](<agents/network_ops_agent.py>)

Network-ops agent builder.

Owns:

- loading [`skills/network_ops.md`](<skills/network_ops.md>)
- loading one selected scenario prompt from `skills/network_ops_scenarios/` when present
- requesting the `network_ops` tool profile from `ToolRegistry`
- building the network-ops ReAct agent

## 5. Prompt Files

### [`skills/troubleshooter.md`](<skills/troubleshooter.md>)

Core troubleshoot system prompt.

Defines:

- troubleshoot role
- diagnostic principles
- evidence expectations
- general behavioral instructions

### [`skills/network_ops.md`](<skills/network_ops.md>)

Core network-ops system prompt.

Defines:

- incident/change/task-oriented operating mode
- expected behavior for network-ops requests

### `skills/network_ops_scenarios/*`

Scenario-specific network-ops addenda.

Examples:

- `incident_record.md`
- `record_lookup.md`
- `change_record.md`
- `change_update.md`
- `access_change.md`

### [`skills/troubleshooting_scenarios/connectivity.md`](<skills/troubleshooting_scenarios/connectivity.md>)

Connectivity-specific troubleshoot instructions.

Defines:

- required connectivity evidence expectations
- how to interpret path, routing, and service-level findings

### [`skills/troubleshooting_scenarios/performance.md`](<skills/troubleshooting_scenarios/performance.md>)

Performance scenario addendum.

### [`skills/troubleshooting_scenarios/intermittent.md`](<skills/troubleshooting_scenarios/intermittent.md>)

Intermittent-issue scenario addendum.

## 6. Workflow Orchestration Services

These services sit above the pure agents and below the graph nodes.

### [`services/intent_routing_service.py`](<services/intent_routing_service.py>)

Owns the LLM-backed coarse lane selector for normal requests.

Responsibilities:

- asking the router model to choose `troubleshoot`, `network_ops`, or `dismiss`
- parsing the router JSON response
- rejecting invalid router output

### [`services/troubleshoot_scenario_service.py`](<services/troubleshoot_scenario_service.py>)

Owns LLM-backed troubleshooting scenario selection.

Responsibilities:

- choosing `connectivity`, `performance`, `intermittent`, or `general`
- keeping scenario selection outside the pure agent file

### [`services/network_ops_scenario_service.py`](<services/network_ops_scenario_service.py>)

Owns LLM-backed network-ops scenario selection.

Responsibilities:

- choosing `incident_record`, `record_lookup`, `change_record`, `change_update`, `access_change`, or `general`
- keeping scenario selection outside the pure agent file

### [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>)

Owns troubleshoot-side execution around the pure agent.

Responsibilities:

- session/tool-state reset
- pending-context recovery
- incident prompt expansion handoff
- required evidence follow-up for connectivity
- optional memory follow-up when justified by evidence
- contradiction correction for specific conclusion mistakes
- presenter handoff

### [`services/network_ops_workflow_service.py`](<services/network_ops_workflow_service.py>)

Owns network-ops execution around the pure agent.

Responsibilities:

- pending clarification handling
- network-ops agent invocation
- session state merge
- presenter handoff

### [`services/workflow_state_service.py`](<services/workflow_state_service.py>)

Owns shared workflow guards and session-data merging.

Responsibilities:

- merging session-side effects
- checking whether required connectivity evidence is missing
- checking whether required path visuals are missing
- generating guarded follow-up prompts when the workflow must enforce an evidence boundary

## 7. Diagnostic and Domain Services

These services own domain logic below the tool wrappers.

### [`services/path_trace_service.py`](<services/path_trace_service.py>)

Owns:

- forward path walking
- reverse path walking
- path metadata extraction
- first-hop LAN/source-interface metadata for connectivity testing

### [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>)

Owns:

- ping execution
- source-interface selection for pings
- TCP reachability tests
- routing checks across devices
- interface counters
- interface detail lookups
- interface inventory summaries

### [`services/connectivity_snapshot_service.py`](<services/connectivity_snapshot_service.py>)

Owns:

- connectivity evidence bundle assembly
- relevant interface selection
- parallel device snapshot collection
- destination-side TCP validation
- structured connectivity snapshot summaries

### [`services/routing_diagnostics_service.py`](<services/routing_diagnostics_service.py>)

Owns:

- routing history lookup
- OSPF neighbors
- OSPF interface checks
- OSPF history summaries
- syslog-driven interface correlation
- peering inspection workflow

### [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>)

Owns Atlas-specific troubleshooting correlation for ServiceNow.

Responsibilities:

- combining incident and change searches
- correlating by devices, IPs, and port context
- producing deterministic ServiceNow summary structures for troubleshoot runs

### [`services/memory_manager.py`](<services/memory_manager.py>)

Owns:

- pending clarification state
- long-term memory write policy
- recall gating
- evidence-driven recall signals
- recall follow-up prompt construction

### [`services/pending_context.py`](<services/pending_context.py>)

Owns the pending-context storage object used by memory/workflow flows.

### [`services/session_store.py`](<services/session_store.py>)

Owns transient per-session tool side effects.

Examples of stored side effects:

- `path_hops`
- `reverse_path_hops`
- `interface_counters`
- `connectivity_snapshot`
- `servicenow_summary`
- memory recall signals

## 8. Runtime, Diagnostics, and Observability Services

### [`services/graph_runtime.py`](<services/graph_runtime.py>)

Owns:

- initial graph state creation
- graph config assembly
- graph invocation
- final response extraction

### [`services/checkpointer_runtime.py`](<services/checkpointer_runtime.py>)

Owns:

- LangGraph checkpointer lifecycle
- Redis-backed graph persistence readiness state

### [`services/status_service.py`](<services/status_service.py>)

Owns the service-style interface used by workflows and tools to push UI status updates.

### [`services/response_presenter.py`](<services/response_presenter.py>)

Owns final UI payload shaping.

Responsibilities:

- grouping interface counters
- ServiceNow section replacement
- path visual payload inclusion
- network-ops payload shaping
- deterministic troubleshoot payload shaping

### [`services/diagnostics_service.py`](<services/diagnostics_service.py>)

Owns the internal diagnostics snapshot used by the diagnostics UI and `/api/internal/diagnostics`.

### [`services/health_service.py`](<services/health_service.py>)

Owns:

- overall backend health status
- dependency health checks
- checkpointer health labeling

### [`services/observability.py`](<services/observability.py>)

Owns:

- request id creation
- structured event logging helpers
- elapsed-time helpers

### [`services/metrics.py`](<services/metrics.py>)

Owns:

- in-process counters
- timing aggregates
- metrics snapshots for diagnostics

### [`services/backend_contracts.py`](<services/backend_contracts.py>)

Owns shared backend failure/result wording such as:

- backend unavailable
- not found
- lookup error
- unexpected response
- verification failed

## 9. Tool Layer

The agent never sees raw backend clients. It sees Atlas tools.

### [`tools/tool_registry.py`](<tools/tool_registry.py>)

Owns:

- capability registration
- profile-to-capability mapping
- profile-to-tool resolution

Default profiles:

- `troubleshoot.general`
- `troubleshoot.connectivity`
- `network_ops`

### [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)

Agent-facing workflow tools for:

- forward path tracing
- reverse path tracing

Delegates to:

- `PathTraceService`

### [`tools/device_agent_tools.py`](<tools/device_agent_tools.py>)

Agent-facing workflow tools for:

- ping
- TCP reachability
- routing checks
- interface counters
- interface details
- interface inventory

Delegates to:

- `DeviceDiagnosticsService`

### [`tools/routing_agent_tools.py`](<tools/routing_agent_tools.py>)

Agent-facing workflow tools for:

- routing history
- OSPF neighbors
- OSPF interface checks
- OSPF history
- peering inspection
- syslog lookups

Delegates to:

- `RoutingDiagnosticsService`

### [`tools/connectivity_agent_tools.py`](<tools/connectivity_agent_tools.py>)

Agent-facing workflow tools for:

- bundled connectivity snapshot collection

Delegates to:

- `ConnectivitySnapshotService`

### [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)

Agent-facing workflow tools for:

- troubleshoot-side ServiceNow correlation and summary

Delegates to:

- `ServiceNowSearchService`

### [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>)

Thin product-facing agent tools for:

- incident detail lookup
- change detail lookup
- incident creation
- change creation
- change update

These are Atlas tools that call MCP under the hood.

### [`tools/memory_agent_tools.py`](<tools/memory_agent_tools.py>)

Agent-facing memory tools for:

- recall of similar prior cases

Delegates to:

- `MemoryManager`
- long-term memory storage
- `SessionStore`

### [`tools/knowledge_agent_tools.py`](<tools/knowledge_agent_tools.py>)

Agent-facing knowledge lookup tools for:

- vendor KB lookup

### [`tools/servicenow_tools.py`](<tools/servicenow_tools.py>)

Backend-facing MCP tool implementation module.

Owns:

- raw ServiceNow MCP tools
- REST calls to ServiceNow Table API
- retries/circuit behavior for the MCP server side

This is not the agent-facing tool layer. It is backend/MCP infrastructure.

### [`tools/tool_runtime.py`](<tools/tool_runtime.py>)

Shared helper functions used by the agent-facing tool wrappers.

### [`tools/shared.py`](<tools/shared.py>)

Shared MCP/logging helpers used by backend-facing tool implementations.

### [`tools/resilience.py`](<tools/resilience.py>)

Shared retry and circuit-breaker helpers for backend-facing tool implementations.

### [`tools/all_tools.py`](<tools/all_tools.py>)

Thin compatibility export layer.

It exists so older imports still resolve, but the real workflow tools now live in the split `*_agent_tools.py` modules.

## 10. Integrations and Backend Clients

### [`integrations/mcp_client.py`](<integrations/mcp_client.py>)

Atlas-side MCP client transport.

Owns:

- calling MCP tools from Atlas services/tool adapters

### [`integrations/kv_helper.py`](<integrations/kv_helper.py>)

Key Vault helper integration.

### [`integrations/servicenowauth.py`](<integrations/servicenowauth.py>)

ServiceNow credential and auth helper integration.

### [`services/nornir_client.py`](<services/nornir_client.py>)

Atlas-side client for the local Nornir service.

Owns:

- HTTP requests to Nornir
- retry behavior
- run-scoped cache
- cache clear

### [`nornir/server.py`](<nornir/server.py>)

Local Nornir FastAPI service.

Owns live network operations such as:

- route lookups
- device discovery
- ping
- interface counters
- all-interfaces status
- device snapshots

### [`nornir/tasks.py`](<nornir/tasks.py>)

Task implementations used by the Nornir service.

### [`nornir/config.yaml`](<nornir/config.yaml>)
### [`nornir/inventory/defaults.yaml`](<nornir/inventory/defaults.yaml>)
### [`nornir/inventory/groups.yaml`](<nornir/inventory/groups.yaml>)
### [`nornir/inventory/hosts.yaml`](<nornir/inventory/hosts.yaml>)

Nornir configuration and inventory files.

## 11. Persistence and Security

### [`persistence/db.py`](<persistence/db.py>)

Database access helpers.

### [`persistence/chat_history.py`](<persistence/chat_history.py>)

Chat history persistence and retrieval helpers.

### [`persistence/schema.sql`](<persistence/schema.sql>)

Database schema reference.

### [`security/auth.py`](<security/auth.py>)

Authentication and RBAC owner.

Owns:

- session creation/destruction
- current-user resolution
- group/category access
- OIDC-related auth helpers

## 12. Memory Layer

### [`memory/agent_memory.py`](<memory/agent_memory.py>)

Long-term memory storage and similarity retrieval.

### [`memory/servicenow_memory_sync.py`](<memory/servicenow_memory_sync.py>)

ServiceNow-to-memory sync job.

Owns:

- pulling closed incidents
- storing them into long-term memory

## 13. Frontend Files Involved in the Runtime Path

### [`frontend/src/stores/chatStore.js`](<frontend/src/stores/chatStore.js>)

Owns the client-side chat lifecycle:

- sending messages
- status-step tracking
- in-memory chat state

### [`frontend/src/utils/api.js`](<frontend/src/utils/api.js>)

Owns frontend API calls for:

- `/api/chat`
- diagnostics
- history

### [`frontend/src/App.jsx`](<frontend/src/App.jsx>)

Top-level frontend composition.

### [`frontend/src/components/layout/ChatLayout.jsx`](<frontend/src/components/layout/ChatLayout.jsx>)

High-level page layout for the chat experience.

### [`frontend/src/components/layout/AppHeader.jsx`](<frontend/src/components/layout/AppHeader.jsx>)

Header, navigation, and health badge.

### [`frontend/src/components/layout/AppSidebar.jsx`](<frontend/src/components/layout/AppSidebar.jsx>)

Sidebar, history, and example-query surface.

### [`frontend/src/components/chat/ChatInput.jsx`](<frontend/src/components/chat/ChatInput.jsx>)

User input box.

### [`frontend/src/components/chat/ChatMessages.jsx`](<frontend/src/components/chat/ChatMessages.jsx>)

Message-list container.

### [`frontend/src/components/chat/StatusMessage.jsx`](<frontend/src/components/chat/StatusMessage.jsx>)

Status timeline row rendering.

### [`frontend/src/components/chat/WelcomeState.jsx`](<frontend/src/components/chat/WelcomeState.jsx>)

Initial landing/empty-state content.

### [`frontend/src/components/messages/AssistantMessage.jsx`](<frontend/src/components/messages/AssistantMessage.jsx>)

Main assistant-message renderer.

Owns:

- rendering markdown responses
- rendering path payloads
- rendering interface counters
- rendering fallback structured blocks

### [`frontend/src/components/messages/MarkdownContent.jsx`](<frontend/src/components/messages/MarkdownContent.jsx>)

Markdown rendering owner.

### [`frontend/src/components/messages/InterfaceCounters.jsx`](<frontend/src/components/messages/InterfaceCounters.jsx>)

Interface-counter rendering owner.

### [`frontend/src/components/messages/ErrorMessage.jsx`](<frontend/src/components/messages/ErrorMessage.jsx>)

User-visible error bubble rendering.

### [`frontend/src/components/messages/JsonFallback.jsx`](<frontend/src/components/messages/JsonFallback.jsx>)

Fallback renderer for unexpected structured content.

### [`frontend/src/components/path/PathVisualization.jsx`](<frontend/src/components/path/PathVisualization.jsx>)

Forward/reverse path rendering owner.

### [`frontend/src/components/path/PathItem.jsx`](<frontend/src/components/path/PathItem.jsx>)
### [`frontend/src/components/path/PathConnectors.jsx`](<frontend/src/components/path/PathConnectors.jsx>)
### [`frontend/src/components/path/DeviceIcon.jsx`](<frontend/src/components/path/DeviceIcon.jsx>)
### [`frontend/src/components/path/PathFullscreen.jsx`](<frontend/src/components/path/PathFullscreen.jsx>)
### [`frontend/src/components/path/FirewallDetails.jsx`](<frontend/src/components/path/FirewallDetails.jsx>)

Supporting path-visualization components.

### [`frontend/src/components/diagnostics/Diagnostics.jsx`](<frontend/src/components/diagnostics/Diagnostics.jsx>)

Internal diagnostics screen.

### [`frontend/src/components/dashboard/Dashboard.jsx`](<frontend/src/components/dashboard/Dashboard.jsx>)

Dashboard screen.

### [`frontend/src/components/tables/DataTable.jsx`](<frontend/src/components/tables/DataTable.jsx>)
### [`frontend/src/components/tables/VerticalTable.jsx`](<frontend/src/components/tables/VerticalTable.jsx>)

Shared table components used by structured views.

### [`frontend/src/hooks/useHealth.js`](<frontend/src/hooks/useHealth.js>)

Frontend health polling hook.

### [`frontend/src/hooks/useTheme.js`](<frontend/src/hooks/useTheme.js>)

Frontend theme hook.

### [`frontend/src/utils/responseClassifier.js`](<frontend/src/utils/responseClassifier.js>)

Classifies returned content for the frontend renderer.

### [`frontend/src/utils/formatters.js`](<frontend/src/utils/formatters.js>)
### [`frontend/src/utils/exampleQueries.js`](<frontend/src/utils/exampleQueries.js>)
### [`frontend/src/utils/deviceIcons.js`](<frontend/src/utils/deviceIcons.js>)
### [`frontend/src/utils/csvExport.js`](<frontend/src/utils/csvExport.js>)

Supporting frontend utility modules.

### CSS module files under `frontend/src/**`

These own styling only. They do not affect routing, prompts, tool choice, or backend behavior.

## 14. What to Read First

If you are trying to understand the system quickly, read in this order:

1. [`README.md`](<README.md>)
2. [`Documentation/Architecture/agent-routing-and-tooling.md`](<Documentation/Architecture/agent-routing-and-tooling.md>)
3. [`Documentation/End-to-End-flow/troubleshooting-query-flow.md`](<Documentation/End-to-End-flow/troubleshooting-query-flow.md>)
4. this file
5. [`graph/graph_nodes.py`](<graph/graph_nodes.py>)
6. [`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>)
7. [`tools/tool_registry.py`](<tools/tool_registry.py>)
8. [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>)

That sequence gives you:

- the overall architecture
- the routing rules
- the prompt model
- the tool model
- the file ownership map
