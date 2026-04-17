# Atlas Component Reference

This document is the component-level reference for the Atlas codebase.

Use it when you need to answer questions such as:

- which file owns a specific runtime decision
- what a component receives and returns
- which layer is responsible for orchestration versus execution
- where a backend protocol is hidden behind an Atlas owner
- which frontend files are involved in the chat runtime path

It is organized by layer, starting at the web entrypoint and moving downward into graph routing, workflow services, agents, tools, backend integrations, and frontend rendering.

## 1. Root Entry Files

### [`app.py`](<app.py>)

Primary FastAPI application.

Role:

- owns the HTTP API surface for Atlas
- authenticates requests
- starts the SSE chat stream
- persists conversation history after the run completes

Receives:

- browser requests such as `POST /api/chat`
- authenticated session and user context

Returns:

- SSE events for status updates and final chat responses
- health and diagnostics responses
- login/logout/auth callback responses

Collaborates with:

- `application/chat_service.py`
- `application/status_bus.py`
- `security/auth.py`
- `services/diagnostics_service.py`
- `services/health_service.py`

Runtime notes:

- local development is typically served on `:8001` through `run_web.py`
- direct `uvicorn` execution in this file uses `:8000`

### [`run_web.py`](<run_web.py>)

Recommended local backend launcher.

Role:

- starts the Atlas web application for local development
- applies log bootstrap and dev-friendly startup behavior

Receives:

- local environment configuration

Returns:

- a running uvicorn process for the Atlas backend

Collaborates with:

- `app.py`

### [`mcp_server.py`](<mcp_server.py>)

Local MCP server entrypoint.

Role:

- starts the FastMCP server that exposes backend-facing ServiceNow tools

Receives:

- MCP tool registrations from backend-facing tool modules

Returns:

- MCP HTTP endpoints for Atlas-side MCP clients

Collaborates with:

- `tools/shared.py`
- `tools/servicenow_tools.py`
- `integrations/mcp_client.py`

Runtime notes:

- default MCP server port is `8765`

## 2. Application Layer

### [`application/chat_service.py`](<application/chat_service.py>)

Thin application entrypoint between `app.py` and `AtlasApplication`.

Role:

- hands a chat request into the top-level application owner
- exposes a shared IP/CIDR regex reused by other runtime code

Receives:

- prompt
- conversation history
- username
- session id

Returns:

- the final application response from `AtlasApplication`

Collaborates with:

- `application/atlas_application.py`
- `graph/graph_nodes.py`

### [`application/atlas_application.py`](<application/atlas_application.py>)

Top-level runtime owner for the Atlas application.

Role:

- coordinates the request lifecycle above the graph
- owns the major runtime collaborators as application-level dependencies
- records top-level metrics and request logging

Receives:

- prompt
- conversation history
- session id
- username

Returns:

- the final application response payload produced by the runtime

Collaborates with:

- `services/graph_runtime.py`
- `agents/agent_factory.py`
- `services/memory_manager.py`
- `tools/tool_registry.py`
- `services/response_presenter.py`
- workflow services

Primary method:

- `AtlasApplication.process_query(...)`

### [`application/status_bus.py`](<application/status_bus.py>)

Per-session SSE queue registry.

Role:

- stores live status queues for connected browser sessions
- provides the queue surface used by status updates during a run

Receives:

- queue registration requests from `app.py`
- status events from `services/status_service.py`

Returns:

- queued SSE status messages to the active session

Collaborates with:

- `app.py`
- `services/status_service.py`

## 3. Graph Layer

### [`graph/graph_builder.py`](<graph/graph_builder.py>)

LangGraph topology owner.

Role:

- defines the graph shape
- compiles the graph
- maps `intent` values to the next node

Receives:

- graph nodes
- graph state type
- optional checkpointer

Returns:

- compiled LangGraph application

Collaborates with:

- `graph/graph_nodes.py`
- `graph/graph_state.py`
- `services/checkpointer_runtime.py`

Key responsibility:

- reads `state["intent"]` and routes to:
  - `call_troubleshoot_agent`
  - `call_network_ops_agent`
  - `build_final_response`

### [`graph/graph_nodes.py`](<graph/graph_nodes.py>)

Graph boundary logic.

Role:

- owns graph-entry intent classification
- handles small fast-path dismissals and pending-context continuations
- delegates into the workflow services

Receives:

- `AtlasState`

Returns:

- updated graph state with:
  - `intent`
  - optional `final_response`

Collaborates with:

- `services/intent_routing_service.py`
- `services/troubleshoot_workflow_service.py`
- `services/network_ops_workflow_service.py`
- `services/memory_manager.py`

Important distinction:

- this is where lane selection happens
- it is not where the tool loop runs

### [`graph/graph_state.py`](<graph/graph_state.py>)

Typed LangGraph state definition.

Role:

- defines the shared state object passed between graph nodes

Fields include:

- `prompt`
- `conversation_history`
- `username`
- `session_id`
- `request_id`
- `intent`
- `rbac_error`
- `final_response`

Why it matters:

- it is the handoff format between the application/runtime layer and the workflow layer

## 4. Agent Layer

### [`agents/agent_factory.py`](<agents/agent_factory.py>)

Shared ReAct agent factory.

Role:

- constructs role-specific chat models
- instantiates request-time ReAct agent objects
- attaches the system prompt and visible tools

Receives:

- selected model role
- tool tuple
- system prompt
- agent name

Returns:

- live ReAct agent object ready for `.ainvoke(...)`

Collaborates with:

- `langgraph.prebuilt.create_react_agent`
- `tools/tool_registry.py`
- `agents/troubleshoot_agent.py`
- `agents/network_ops_agent.py`

Model roles:

- router
- selector
- network ops
- troubleshoot

Important distinction:

- this file does not create new agent types
- it instantiates request-specific runtime agents from the existing troubleshoot or network-ops definitions

### [`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>)

Troubleshoot agent definition and factory wrapper.

Role:

- loads the troubleshoot system prompt
- loads the selected troubleshooting scenario prompt
- chooses the correct troubleshoot tool profile
- instantiates the request-specific troubleshoot ReAct agent

Receives:

- selected scenario
- optional model override

Returns:

- live troubleshoot ReAct agent object

Collaborates with:

- `skills/troubleshooter.md`
- `skills/troubleshooting_scenarios/*`
- `tools/tool_registry.py`
- `agents/agent_factory.py`

Tool exposure:

- `troubleshoot.connectivity`
- `troubleshoot.general`

### [`agents/network_ops_agent.py`](<agents/network_ops_agent.py>)

Network-ops agent definition and factory wrapper.

Role:

- loads the network-ops system prompt
- loads the selected network-ops scenario prompt when one applies
- requests the `network_ops` tool profile
- instantiates the request-specific network-ops ReAct agent

Receives:

- selected scenario
- optional model override

Returns:

- live network-ops ReAct agent object

Collaborates with:

- `skills/network_ops.md`
- `skills/network_ops_scenarios/*`
- `tools/tool_registry.py`
- `agents/agent_factory.py`

## 5. Prompt Files

### [`skills/troubleshooter.md`](<skills/troubleshooter.md>)

Core troubleshoot system prompt.

Defines:

- the troubleshoot role
- evidence expectations
- reporting behavior
- tool-use expectations

Used by:

- `agents/troubleshoot_agent.py`

### [`skills/network_ops.md`](<skills/network_ops.md>)

Core network-ops system prompt.

Defines:

- incident and change oriented operating mode
- clarification behavior
- tool-use rules for operational requests

Used by:

- `agents/network_ops_agent.py`

### `skills/troubleshooting_scenarios/*`

Scenario-specific troubleshoot prompt addenda.

Examples:

- `connectivity.md`
- `performance.md`
- `intermittent.md`

Role:

- narrow the behavior of the troubleshoot agent for a specific class of request

Selected by:

- `services/troubleshoot_scenario_service.py`

Loaded by:

- `agents/troubleshoot_agent.py`

### `skills/network_ops_scenarios/*`

Scenario-specific network-ops prompt addenda.

Examples:

- `incident_record.md`
- `record_lookup.md`
- `change_record.md`
- `change_update.md`
- `access_change.md`

Role:

- narrow the behavior of the network-ops agent for a specific operational pattern

Selected by:

- `services/network_ops_scenario_service.py`

Loaded by:

- `agents/network_ops_agent.py`

## 6. Workflow Orchestration Services

These services sit below the graph nodes and above the pure agents.

### [`services/intent_routing_service.py`](<services/intent_routing_service.py>)

LLM-backed lane selector for normal requests.

Role:

- asks the router model to choose `troubleshoot`, `network_ops`, or `dismiss`
- parses and validates the returned JSON

Receives:

- original user prompt

Returns:

- validated router decision dict
- or `None` if the response is invalid or unavailable

Collaborates with:

- `agents/agent_factory.py`
- `graph/graph_nodes.py`

### [`services/troubleshoot_scenario_service.py`](<services/troubleshoot_scenario_service.py>)

LLM-backed troubleshoot scenario selector.

Role:

- chooses `connectivity`, `performance`, `intermittent`, or `general`

Receives:

- the effective troubleshoot prompt after pending-context or incident expansion

Returns:

- selected troubleshoot scenario

Collaborates with:

- `agents/agent_factory.py`
- `services/troubleshoot_workflow_service.py`

### [`services/network_ops_scenario_service.py`](<services/network_ops_scenario_service.py>)

LLM-backed network-ops scenario selector.

Role:

- chooses `incident_record`, `record_lookup`, `change_record`, `change_update`, `access_change`, or `general`

Receives:

- the operational request prompt

Returns:

- selected network-ops scenario

Collaborates with:

- `agents/agent_factory.py`
- `services/network_ops_workflow_service.py`

### [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>)

Troubleshoot-side orchestration owner.

Role:

- prepares the request for troubleshoot execution
- resets run-scoped state
- asks the scenario selector for the current scenario
- instantiates and invokes the troubleshoot agent
- enforces required evidence follow-ups when needed
- hands the final result to the presenter

Receives:

- graph state

Returns:

- `final_response` payload for the graph

Collaborates with:

- `services/request_preprocessor.py`
- `services/troubleshoot_scenario_service.py`
- `agents/troubleshoot_agent.py`
- `services/workflow_state_service.py`
- `services/memory_manager.py`
- `services/session_store.py`
- `services/response_presenter.py`
- `services/status_service.py`

Important behaviors:

- incident prompts can be rewritten into a concrete troubleshoot prompt
- required connectivity evidence can trigger a follow-up pass
- path visuals can be backfilled if the session data is missing them
- memory recall can trigger an additional pass when evidence justifies it

### [`services/network_ops_workflow_service.py`](<services/network_ops_workflow_service.py>)

Network-ops orchestration owner.

Role:

- handles pending clarification continuation
- selects the network-ops scenario
- instantiates and invokes the network-ops agent
- sends the result to the presenter

Receives:

- graph state

Returns:

- `final_response` payload for the graph

Collaborates with:

- `services/network_ops_scenario_service.py`
- `agents/network_ops_agent.py`
- `services/memory_manager.py`
- `services/session_store.py`
- `services/response_presenter.py`
- `services/status_service.py`

### [`services/workflow_state_service.py`](<services/workflow_state_service.py>)

Shared workflow guard owner.

Role:

- merges session-side effects from multiple tool passes
- decides whether required connectivity evidence is missing
- decides whether path visuals are missing
- provides guarded follow-up prompts and checks

Receives:

- accumulated session-side effects
- prompt-derived context such as source and destination IPs

Returns:

- merge results
- boolean guards

Collaborates with:

- workflow services
- `services/session_store.py`

## 7. Diagnostic and Domain Services

These services own domain logic below the tool wrappers.

### [`services/path_trace_service.py`](<services/path_trace_service.py>)

Path tracing owner.

Role:

- walks the forward path through the network
- walks the reverse path
- extracts path metadata used later by connectivity testing

Receives:

- source IP
- destination IP

Returns:

- hop-by-hop path data
- metadata such as first-hop and reverse-first-hop interface details

Collaborates with:

- `services/nornir_client.py`
- `tools/path_agent_tools.py`
- `services/connectivity_snapshot_service.py`

### [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>)

Active device diagnostic owner.

Role:

- runs pings
- resolves ping source-interface behavior
- runs TCP reachability tests
- gathers interface counters and details
- performs route checks across devices

Receives:

- device identifiers
- IPs
- ports
- optional interface or VRF context

Returns:

- human-readable summaries
- structured diagnostic data written into `SessionStore`

Collaborates with:

- `services/nornir_client.py`
- `tools/device_agent_tools.py`
- `services/connectivity_snapshot_service.py`

### [`services/connectivity_snapshot_service.py`](<services/connectivity_snapshot_service.py>)

Connectivity evidence bundle owner.

Role:

- assembles the full connectivity evidence bundle used for endpoint-to-endpoint troubleshooting
- collects per-device snapshots in parallel
- runs the primary forward and reverse validation tests
- runs the destination-side TCP validation

Receives:

- source IP
- destination IP
- optional port

Returns:

- structured connectivity snapshot
- summary text and side effects written into `SessionStore`

Collaborates with:

- `services/path_trace_service.py`
- `services/device_diagnostics_service.py`
- `services/routing_diagnostics_service.py`
- `tools/connectivity_agent_tools.py`

### [`services/routing_diagnostics_service.py`](<services/routing_diagnostics_service.py>)

Routing and protocol diagnostic owner.

Role:

- looks up routing history
- checks OSPF neighbors and interfaces
- inspects peering state
- correlates syslog with routing and interface evidence

Receives:

- destination IP
- device names
- interface names

Returns:

- routing and protocol summaries used by the agent and presenter

Collaborates with:

- `services/nornir_client.py`
- `tools/routing_agent_tools.py`

### [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>)

Troubleshoot-side ServiceNow correlation owner.

Role:

- searches incidents and changes relevant to a troubleshooting run
- correlates by devices, IPs, and port context
- produces deterministic summary structures for the presenter and the agent

Receives:

- devices
- source IP
- destination IP
- optional port

Returns:

- structured ServiceNow correlation summary

Collaborates with:

- `tools/servicenow_workflow_tools.py`
- `integrations/mcp_client.py`

### [`services/memory_manager.py`](<services/memory_manager.py>)

Memory and pending-context owner.

Role:

- stores and retrieves pending clarification context
- decides when long-term memory recall should be used
- builds recall follow-up prompts
- stores troubleshoot runs into long-term memory

Receives:

- prompt
- session id
- session-side effects
- final troubleshoot text

Returns:

- pending-context state
- recall decisions
- recall follow-up prompts

Collaborates with:

- `services/pending_context.py`
- `memory/agent_memory.py`
- `tools/memory_agent_tools.py`

### [`services/pending_context.py`](<services/pending_context.py>)

Pending-context storage owner.

Role:

- stores and retrieves pending follow-up context for multi-turn flows

Used by:

- `services/memory_manager.py`

### [`services/session_store.py`](<services/session_store.py>)

Transient side-effect store.

Role:

- stores structured side effects produced by tools during a single run
- provides the workflow and presenter with structured evidence outside the LLM text

Examples of stored data:

- `path_hops`
- `reverse_path_hops`
- `interface_counters`
- `connectivity_snapshot`
- `servicenow_summary`
- memory recall signals

Collaborates with:

- workflow services
- tool wrappers
- `services/response_presenter.py`

## 8. Runtime, Diagnostics, and Observability Services

### [`services/graph_runtime.py`](<services/graph_runtime.py>)

Graph execution owner.

Role:

- creates the initial graph state
- builds graph config
- ensures checkpointer readiness
- invokes the compiled graph
- extracts the final response

Receives:

- prompt
- conversation history
- username
- session id

Returns:

- graph result and final response

Collaborates with:

- `application/atlas_application.py`
- `services/checkpointer_runtime.py`
- `graph/graph_builder.py`

### [`services/checkpointer_runtime.py`](<services/checkpointer_runtime.py>)

LangGraph checkpointer lifecycle owner.

Role:

- creates and tracks the Redis-backed LangGraph checkpointer
- recompiles the graph when the checkpointer becomes available

Receives:

- runtime initialization requests

Returns:

- checkpointer readiness state
- compiled graph with or without persistent checkpointer support

Collaborates with:

- `services/graph_runtime.py`
- `services/diagnostics_service.py`

### [`services/status_service.py`](<services/status_service.py>)

Status update owner.

Role:

- provides a service-style interface for pushing status messages into the active session

Collaborates with:

- workflow services
- tool wrappers
- `application/status_bus.py`

### [`services/response_presenter.py`](<services/response_presenter.py>)

Final payload shaping owner.

Role:

- turns final agent text plus structured side effects into the frontend payload
- shapes troubleshoot and network-ops payloads
- includes path data, counters, ServiceNow sections, and other structured content

Receives:

- final LLM text
- session-side effects
- prompt context

Returns:

- final assistant content object used by the frontend

Collaborates with:

- workflow services
- `services/session_store.py`

### [`services/diagnostics_service.py`](<services/diagnostics_service.py>)

Internal diagnostics snapshot owner.

Role:

- assembles the read-only runtime summary used by `/api/internal/diagnostics`

Includes:

- owner summary
- tool registry mappings
- registered tool metadata
- checkpointer status
- metrics snapshot
- model assignment snapshot

### [`services/health_service.py`](<services/health_service.py>)

Health snapshot owner.

Role:

- checks local dependencies and runtime readiness
- returns a summarized backend health view

Checks include:

- Atlas runtime readiness
- Nornir reachability
- MCP configuration and connectivity

### [`services/observability.py`](<services/observability.py>)

Structured runtime logging owner.

Role:

- creates request ids
- writes structured runtime events
- provides elapsed-time helpers for instrumentation

### [`services/metrics.py`](<services/metrics.py>)

In-process metrics owner.

Role:

- stores counters and timing aggregates for the running process

Used by:

- workflow services
- graph routing
- diagnostics

### [`services/backend_contracts.py`](<services/backend_contracts.py>)

Shared backend result and failure-contract owner.

Role:

- standardizes wording and shapes for backend failures and backend lookups

Examples:

- backend unavailable
- not found
- lookup error
- verification failed

## 9. Tool Layer

The agent never sees raw backend clients. It sees Atlas tools.

### [`tools/tool_registry.py`](<tools/tool_registry.py>)

Tool visibility owner.

Role:

- registers capabilities
- maps agent profiles to capability sets
- resolves capability sets into visible tool tuples

Default profiles:

- `troubleshoot.general`
- `troubleshoot.connectivity`
- `network_ops`

Why it matters:

- this is where code decides which tools the LLM can see

### [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)

Agent-facing path workflow tools.

Role:

- expose forward and reverse path tracing to the agent layer

Delegates to:

- `services/path_trace_service.py`

### [`tools/device_agent_tools.py`](<tools/device_agent_tools.py>)

Agent-facing device diagnostic tools.

Role:

- expose ping, TCP test, routing check, interface counters, and interface detail actions

Delegates to:

- `services/device_diagnostics_service.py`

### [`tools/routing_agent_tools.py`](<tools/routing_agent_tools.py>)

Agent-facing routing and protocol tools.

Role:

- expose routing history, OSPF checks, peering inspection, and syslog-related diagnostics

Delegates to:

- `services/routing_diagnostics_service.py`

### [`tools/connectivity_agent_tools.py`](<tools/connectivity_agent_tools.py>)

Agent-facing connectivity snapshot tools.

Role:

- expose bundled connectivity evidence collection

Delegates to:

- `services/connectivity_snapshot_service.py`

### [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)

Agent-facing troubleshoot-side ServiceNow tools.

Role:

- expose correlation searches and deterministic summaries for troubleshoot runs

Delegates to:

- `services/servicenow_search_service.py`

### [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>)

Agent-facing product tools for ServiceNow records.

Role:

- expose incident and change detail reads
- expose incident and change create/update operations

Delegates to:

- `integrations/mcp_client.py`

### [`tools/memory_agent_tools.py`](<tools/memory_agent_tools.py>)

Agent-facing memory tools.

Role:

- expose recall of similar prior cases
- mark memory recall usage in session state

Delegates to:

- `services/memory_manager.py`
- `services/session_store.py`

### [`tools/knowledge_agent_tools.py`](<tools/knowledge_agent_tools.py>)

Agent-facing knowledge lookup tools.

Role:

- expose vendor knowledge-base lookups

### [`tools/servicenow_tools.py`](<tools/servicenow_tools.py>)

Backend-facing MCP tool implementation module.

Role:

- implements raw ServiceNow MCP tools
- performs REST calls to ServiceNow from the MCP server side

This is backend infrastructure, not the agent-facing tool layer.

### [`tools/tool_runtime.py`](<tools/tool_runtime.py>)

Shared tool-wrapper runtime helpers.

Role:

- provides helpers reused by the agent-facing tool wrappers

### [`tools/shared.py`](<tools/shared.py>)

Shared MCP infrastructure owner.

Role:

- defines the shared FastMCP instance
- provides MCP configuration constants
- provides shared Ollama and logging helpers

### [`tools/resilience.py`](<tools/resilience.py>)

Shared backend resilience helpers.

Role:

- provides retry and circuit-breaker helpers for backend-facing tool implementations

### [`tools/all_tools.py`](<tools/all_tools.py>)

Compatibility export layer.

Role:

- preserves older imports while the real tool implementations live in the split `*_agent_tools.py` files

## 10. Integrations and Backend Clients

### [`integrations/mcp_client.py`](<integrations/mcp_client.py>)

Atlas-side MCP client transport.

Role:

- calls MCP tools by name from Atlas runtime code

Receives:

- MCP tool name
- tool arguments

Returns:

- MCP tool result normalized for Atlas callers

Collaborates with:

- `mcp_server.py`
- ServiceNow-facing Atlas tools and services

### [`integrations/kv_helper.py`](<integrations/kv_helper.py>)

Key Vault helper integration.

Role:

- retrieves secrets and related configuration from Azure Key Vault

### [`integrations/servicenowauth.py`](<integrations/servicenowauth.py>)

ServiceNow credential and auth helper integration.

Role:

- handles ServiceNow credential and authentication setup for backend calls

### [`services/nornir_client.py`](<services/nornir_client.py>)

Atlas-side Nornir client.

Role:

- sends HTTP requests to the local Nornir service
- applies retries and a run-scoped cache

Receives:

- endpoint path
- JSON payload

Returns:

- normalized Nornir response data

Collaborates with:

- `nornir/server.py`
- path, device, routing, and connectivity services

Runtime notes:

- default Nornir base URL is `http://localhost:8006`

### [`nornir/server.py`](<nornir/server.py>)

Local Nornir FastAPI service.

Role:

- owns the live SSH-backed network execution surface used by Atlas

Provides endpoints for:

- route lookups
- device discovery
- ping
- TCP tests
- interface counters
- inventory and status
- device snapshots

Runtime notes:

- default Nornir service port is `8006`

### [`nornir/tasks.py`](<nornir/tasks.py>)

Task implementation module for Nornir operations.

### [`nornir/config.yaml`](<nornir/config.yaml>)
### [`nornir/inventory/defaults.yaml`](<nornir/inventory/defaults.yaml>)
### [`nornir/inventory/groups.yaml`](<nornir/inventory/groups.yaml>)
### [`nornir/inventory/hosts.yaml`](<nornir/inventory/hosts.yaml>)

Nornir configuration and inventory files.

Role:

- define inventory and runtime behavior for the local Nornir service

## 11. Persistence and Security

### [`persistence/db.py`](<persistence/db.py>)

Database helper layer.

Role:

- provides PostgreSQL access helpers used by Atlas persistence code

### [`persistence/chat_history.py`](<persistence/chat_history.py>)

Chat history persistence owner.

Role:

- stores and retrieves saved chat history

Used by:

- `app.py`

### [`persistence/schema.sql`](<persistence/schema.sql>)

Database schema reference.

### [`security/auth.py`](<security/auth.py>)

Authentication and RBAC owner.

Role:

- creates and destroys sessions
- resolves the current user and user group
- enforces category and role access
- contains OIDC-related auth helpers

Used by:

- `app.py`

## 12. Memory Layer

### [`memory/agent_memory.py`](<memory/agent_memory.py>)

Long-term memory owner.

Role:

- stores troubleshooting findings into vector-backed memory
- supports semantic similarity retrieval for recall

### [`memory/servicenow_memory_sync.py`](<memory/servicenow_memory_sync.py>)

ServiceNow-to-memory sync job.

Role:

- pulls closed incidents from ServiceNow
- stores them into long-term memory

Used by:

- startup and scheduled sync paths in `app.py`

## 13. Frontend Files in the Runtime Path

### [`frontend/src/stores/chatStore.js`](<frontend/src/stores/chatStore.js>)

Client-side chat lifecycle owner.

Role:

- sends chat messages
- maintains live status-step state
- stores client-side chat history in memory

Collaborates with:

- `frontend/src/utils/api.js`

### [`frontend/src/utils/api.js`](<frontend/src/utils/api.js>)

Frontend API transport owner.

Role:

- opens the `/api/chat` SSE request
- handles diagnostics and history API calls

### [`frontend/src/App.jsx`](<frontend/src/App.jsx>)

Top-level frontend composition owner.

### [`frontend/src/components/layout/ChatLayout.jsx`](<frontend/src/components/layout/ChatLayout.jsx>)

High-level page layout owner for the chat experience.

### [`frontend/src/components/layout/AppHeader.jsx`](<frontend/src/components/layout/AppHeader.jsx>)

Header, navigation, and health badge owner.

### [`frontend/src/components/layout/AppSidebar.jsx`](<frontend/src/components/layout/AppSidebar.jsx>)

Sidebar and history surface owner.

Role:

- renders history
- renders example queries
- exposes chat reset actions

### [`frontend/src/components/chat/ChatInput.jsx`](<frontend/src/components/chat/ChatInput.jsx>)

Chat input owner.

Role:

- captures user input
- submits the next request

### [`frontend/src/components/chat/ChatMessages.jsx`](<frontend/src/components/chat/ChatMessages.jsx>)

Message-list container.

Role:

- orders and renders the message stream

### [`frontend/src/components/chat/StatusMessage.jsx`](<frontend/src/components/chat/StatusMessage.jsx>)

Status timeline row renderer.

Role:

- renders step-by-step runtime status updates

### [`frontend/src/components/chat/WelcomeState.jsx`](<frontend/src/components/chat/WelcomeState.jsx>)

Initial landing and empty-state renderer.

### [`frontend/src/components/messages/AssistantMessage.jsx`](<frontend/src/components/messages/AssistantMessage.jsx>)

Primary assistant-message renderer.

Role:

- renders markdown answers
- renders path visuals
- renders interface counters
- renders fallback structured content

### [`frontend/src/components/messages/MarkdownContent.jsx`](<frontend/src/components/messages/MarkdownContent.jsx>)

Markdown rendering owner.

### [`frontend/src/components/messages/InterfaceCounters.jsx`](<frontend/src/components/messages/InterfaceCounters.jsx>)

Interface-counter rendering owner.

### [`frontend/src/components/messages/ErrorMessage.jsx`](<frontend/src/components/messages/ErrorMessage.jsx>)

User-visible error bubble renderer.

### [`frontend/src/components/messages/JsonFallback.jsx`](<frontend/src/components/messages/JsonFallback.jsx>)

Fallback renderer for unexpected structured assistant content.

### [`frontend/src/components/path/PathVisualization.jsx`](<frontend/src/components/path/PathVisualization.jsx>)

Forward and reverse path visualization owner.

### [`frontend/src/components/path/PathItem.jsx`](<frontend/src/components/path/PathItem.jsx>)
### [`frontend/src/components/path/PathConnectors.jsx`](<frontend/src/components/path/PathConnectors.jsx>)
### [`frontend/src/components/path/DeviceIcon.jsx`](<frontend/src/components/path/DeviceIcon.jsx>)
### [`frontend/src/components/path/PathFullscreen.jsx`](<frontend/src/components/path/PathFullscreen.jsx>)
### [`frontend/src/components/path/FirewallDetails.jsx`](<frontend/src/components/path/FirewallDetails.jsx>)

Supporting path visualization components.
