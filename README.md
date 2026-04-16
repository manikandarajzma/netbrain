# Atlas

Atlas is an AI-assisted network operations application with two primary workflows:

- **Troubleshooting** for live investigations such as connectivity, performance, and intermittent issues
- **Network operations** for operational actions such as ServiceNow incident/change work and firewall or policy review

The architecture is built around a small LangGraph router, two specialized ReAct agents, a uniform tool layer, an owned `NornirClient`, and a separate Nornir HTTP service for live network collection.

## What Atlas Does

Atlas supports work in these broad categories:

- **Connectivity troubleshooting**
  - live forward and reverse path tracing
  - route and interface inspection
  - interface counter collection
  - routing-history correlation
  - ServiceNow correlation
  - deterministic structured response payloads for path visuals and counters
- **Performance and intermittent troubleshooting**
  - live diagnostics plus selective long-term memory recall when live evidence suggests history is relevant
- **Network operations**
  - create and retrieve ServiceNow incidents
  - create, update, close, and retrieve ServiceNow change requests
  - policy checks
  - controlled path lookups for operational context

## Architecture Overview

Atlas has clear owners for the main application responsibilities:

- [`atlas_application.py`](<atlas_application.py>)
  - top-level application owner
- [`services/graph_runtime.py`](<services/graph_runtime.py>)
  - graph execution owner
- [`agents/agent_factory.py`](<agents/agent_factory.py>)
  - agent construction owner
- [`services/response_presenter.py`](<services/response_presenter.py>)
  - final response payload owner
- [`services/memory_manager.py`](<services/memory_manager.py>)
  - pending-context and recall-policy owner
- [`services/path_trace_service.py`](<services/path_trace_service.py>)
  - live path tracing and path metadata owner for troubleshoot workflows
- [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>)
  - active tests and interface diagnostic workflow owner for troubleshoot workflows
- [`services/session_store.py`](<services/session_store.py>)
  - transient per-session tool side-effect store owner
- [`services/status_service.py`](<services/status_service.py>)
  - UI status-bus owner
- [`services/workflow_state_service.py`](<services/workflow_state_service.py>)
  - session-side-effect merge and workflow guard owner
- [`services/connectivity_snapshot_service.py`](<services/connectivity_snapshot_service.py>)
  - connectivity evidence bundle owner for troubleshoot workflows
- [`services/routing_diagnostics_service.py`](<services/routing_diagnostics_service.py>)
  - routing history, OSPF, and syslog diagnostic workflow owner for troubleshoot workflows
- [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>)
  - Atlas-specific incident/change correlation owner for troubleshoot workflows
- [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>)
  - troubleshoot orchestration owner above the pure agent layer
- [`services/network_ops_workflow_service.py`](<services/network_ops_workflow_service.py>)
  - network-ops orchestration owner above the pure agent layer
- [`tools/tool_registry.py`](<tools/tool_registry.py>)
  - tool-set owner
- [`services/observability.py`](<services/observability.py>)
  - structured request and tool observability owner
- [`services/metrics.py`](<services/metrics.py>)
  - lightweight in-process metrics owner
- [`services/diagnostics_service.py`](<services/diagnostics_service.py>)
  - internal diagnostics snapshot owner

### Request Flow

1. The React UI sends the user request to `POST /api/chat` over SSE.
2. `app.py` authenticates the request, creates the SSE lifecycle, and delegates to `chat_service.py`.
3. `chat_service.py` hands the request to `AtlasApplication`.
4. `AtlasApplication` calls `AtlasRuntime`, which prepares graph state, request metadata, and optional checkpointer context.
5. LangGraph classifies the intent and routes to either the troubleshoot workflow, the network-ops workflow, or a dismiss path.
6. The selected workflow service invokes the appropriate pure ReAct agent.
7. The agent resolves its allowed tools through `ToolRegistry`.
8. Workflow tools call owned services and backends such as Nornir or MCP-backed systems.
9. Tool side effects are written into the per-session store.
10. `ResponsePresenter` turns the final text and session state into the UI payload.
11. `app.py` streams the final response back to the frontend, which renders markdown, path visuals, counters, and status updates.

### Agent Execution

1. The workflow service picks the agent type.
2. The agent receives a prompt plus a restricted profile of tools from `ToolRegistry`.
3. The agent chooses tools from a uniform Atlas tool surface.
4. Those tools delegate to workflow owners or thin product-facing adapters.
5. Backends return live data or record data.
6. The workflow merges side effects, enforces required evidence when needed, and hands the result to `ResponsePresenter`.

### Layer Boundaries

- **Frontend**
  - sends the query
  - renders the SSE status timeline
  - renders structured payloads such as path visuals and interface counters
- **FastAPI**
  - authentication
  - SSE lifecycle
  - chat-history persistence
  - cache flush after write operations
  - internal diagnostics endpoint
- **Chat entrypoint**
  - `chat_service.py` is intentionally thin
  - delegates to `AtlasApplication`
- **Application/runtime**
  - `AtlasApplication` owns the top-level processing flow
  - `AtlasRuntime` owns graph invocation, initial state, config, and final-response extraction
  - `services/observability.py` owns shared request ids and structured event logging
  - `services/metrics.py` owns lightweight counters and timing aggregates
  - workflow services own troubleshoot and network-ops execution around the pure agents
- **Graph**
  - coarse routing and thin node delegation only: `troubleshoot`, `network_ops`, `dismiss`
- **Agents**
  - pure specialized ReAct agents with minimal wrappers
- **Tools**
  - the only layer allowed to interact with backends
  - return human-readable tool output plus structured side effects
- **Presenter**
  - converts session/tool state into the UI-facing payload

## Core Modules

### Entry and runtime

- [`app.py`](<app.py>)
  - FastAPI routes, SSE streaming, chat persistence, built-frontend serving, and internal diagnostics
- [`run_web.py`](<run_web.py>)
  - recommended development launcher for the web app
- `chat_service.py`
  - thin entrypoint from HTTP into the application
- [`atlas_application.py`](<atlas_application.py>)
  - application owner that wires runtime, memory, presenter, tools, and agents
- [`services/graph_runtime.py`](<services/graph_runtime.py>)
  - graph execution owner
- [`services/checkpointer_runtime.py`](<services/checkpointer_runtime.py>)
  - Redis-backed LangGraph checkpointer lifecycle
- [`services/diagnostics_service.py`](<services/diagnostics_service.py>)
  - read-only runtime diagnostics snapshot for internal inspection
- [`services/status_service.py`](<services/status_service.py>)
  - status-bus owner reused by workflow services and tool runtime
- [`services/workflow_state_service.py`](<services/workflow_state_service.py>)
  - merge/guard owner for transient workflow state

### Graph

- [`graph_builder.py`](<graph_builder.py>)
  - graph structure
- [`graph_nodes.py`](<graph_nodes.py>)
  - routing node and thin delegation nodes
- [`graph_state.py`](<graph_state.py>)
  - typed graph state

### Agents

- [`agents/agent_factory.py`](<agents/agent_factory.py>)
  - minimal shared ReAct agent factory
- [`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>)
  - troubleshooting agent builder
- [`agents/network_ops_agent.py`](<agents/network_ops_agent.py>)
  - network-ops agent builder

### Services

- [`services/memory_manager.py`](<services/memory_manager.py>)
  - pending clarification state and evidence-driven recall signals
- [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>)
  - troubleshoot agent orchestration, mandatory evidence follow-up, and fail-closed execution
- [`services/network_ops_workflow_service.py`](<services/network_ops_workflow_service.py>)
  - network-ops agent orchestration and follow-up handling
- [`services/path_trace_service.py`](<services/path_trace_service.py>)
  - owned forward/reverse path tracing and path metadata extraction
- [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>)
  - owned ping, TCP, routing-check, and interface diagnostic workflow
- [`services/request_preprocessor.py`](<services/request_preprocessor.py>)
  - incident expansion, IP/port extraction, clarification helpers
- [`services/response_presenter.py`](<services/response_presenter.py>)
  - deterministic payload shaping for troubleshoot and network-ops answers
- [`services/connectivity_snapshot_service.py`](<services/connectivity_snapshot_service.py>)
  - owned connectivity snapshot collection and summarization workflow
- [`services/routing_diagnostics_service.py`](<services/routing_diagnostics_service.py>)
  - owned routing-history, OSPF, and syslog diagnostic workflow
- [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>)
  - Atlas-specific ServiceNow incident/change correlation and summary formatting

### Tools and external integrations

- [`tools/all_tools.py`](<tools/all_tools.py>)
  - thin compatibility export layer for workflow tools
- [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)
  - path tracing workflow entrypoints
- [`tools/device_agent_tools.py`](<tools/device_agent_tools.py>)
  - active test and interface diagnostic workflow entrypoints
- [`tools/routing_agent_tools.py`](<tools/routing_agent_tools.py>)
  - routing, OSPF, and syslog workflow entrypoints
- [`tools/connectivity_agent_tools.py`](<tools/connectivity_agent_tools.py>)
  - connectivity snapshot workflow entrypoints
- [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)
  - troubleshooting-oriented ServiceNow correlation workflow entrypoints
- [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>)
  - thin product-facing ServiceNow adapters for incident/change CRUD and record lookup
- [`tools/tool_registry.py`](<tools/tool_registry.py>)
  - capability registration and agent profile resolution
- [`mcp_client.py`](<mcp_client.py>)
  - MCP calls for systems such as ServiceNow
- [`services/nornir_client.py`](<services/nornir_client.py>)
  - owned HTTP + retry + run-cache client for the local Nornir service
- [`nornir/server.py`](<nornir/server.py>)
  - live network collection service on port `8006`

### Frontend

- [`frontend/src/stores/chatStore.js`](<frontend/src/stores/chatStore.js>)
  - chat lifecycle and status timeline
- [`frontend/src/utils/api.js`](<frontend/src/utils/api.js>)
  - `/api/chat`, diagnostics, and history helpers
- [`frontend/src/components/messages/AssistantMessage.jsx`](<frontend/src/components/messages/AssistantMessage.jsx>)
  - payload-driven assistant rendering
- [`frontend/src/components/path/PathVisualization.jsx`](<frontend/src/components/path/PathVisualization.jsx>)
  - forward and reverse path diagrams

## Intent Routing

Atlas uses **coarse routing** in [`graph_nodes.py`](<graph_nodes.py>) `classify_intent(...)`.

This routing is intentionally simple and deterministic:

- `troubleshoot`
  - live investigations
  - connectivity, routing, OSPF, reachability, packet loss, latency
- `network_ops`
  - incidents
  - change requests
  - policy review
  - firewall/path review workflows
- `dismiss`
  - acknowledgements or unsupported requests

Once Atlas chooses the agent, the **LLM** decides which tools to use inside that agent.

That means:

- regex decides **which agent**
- the ReAct agent decides **which tools**

## Tool Model

Atlas exposes one uniform agent-facing tool model through [`tools/tool_registry.py`](<tools/tool_registry.py>), with a few distinct owner-backed implementation styles:

- workflow tools split across:
  - [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)
  - [`tools/device_agent_tools.py`](<tools/device_agent_tools.py>)
  - [`tools/routing_agent_tools.py`](<tools/routing_agent_tools.py>)
  - [`tools/connectivity_agent_tools.py`](<tools/connectivity_agent_tools.py>)
  - [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)
- live path ownership in [`services/path_trace_service.py`](<services/path_trace_service.py>)
- active tests and interface diagnostics in [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>)
- thin product-facing ServiceNow adapters in [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>)

Every agent-facing tool follows the same model:

- accept typed arguments the LLM can fill
- accept hidden runtime config for `session_id`
- optionally push status updates
- write structured side effects into the per-session store

## Observability

Atlas uses a shared observability helper in [`services/observability.py`](<services/observability.py>).

Current behavior:
- every query gets a `request_id`
- the `request_id` flows through:
  - `AtlasApplication`
  - `AtlasRuntime`
  - `graph_nodes.py`
- major events are logged as structured JSON messages, including:
  - query start / completion
  - graph invoke start / completion
  - intent classification
  - agent completion / failure
  - Nornir request timing and cache hits

This makes it easier to correlate one UI request with:
- graph routing
- backend execution
- final rendered output

Atlas has a lightweight metrics owner in [`services/metrics.py`](<services/metrics.py>).

Current metrics include:
- query started / completed counters
- query duration timing
- graph invocation counters and timing
- agent completion / failure counters and timing
- Nornir request timing
- Nornir cache hit / store counters
- return a human-readable string for the LLM

These metrics are inspectable through the internal diagnostics surface instead of only appearing in logs.

## Internal Diagnostics

Authenticated users can inspect a lightweight runtime snapshot at:

- `GET /api/internal/diagnostics`

It returns:

- owner summary
- checkpointer readiness
- tool profiles and tool-capability mappings
- in-process metric counters and timing aggregates

This is an internal/admin endpoint for debugging and architecture inspection, not an end-user workflow.

### Agent profiles

`ToolRegistry` resolves named agent profiles to capabilities and then to concrete tools:

- `troubleshoot.general`
  - full troubleshooting surface
- `troubleshoot.connectivity`
  - restricted connectivity surface
  - deliberately excludes memory recall
- `network_ops`
  - operational record/change surface
  - includes product-facing ServiceNow tools
  - may use `trace_path(...)` for CI selection

## State and Memory

Atlas uses three distinct state layers:

### 1. LangGraph conversation state

- owned by `AtlasRuntime`
- optionally persisted to Redis through `AsyncRedisSaver`
- keyed by browser `session_id` as LangGraph `thread_id`

### 2. Per-session tool side-effect store

Owned by [`services/session_store.py`](<services/session_store.py>) and cleared between runs.

Examples:

- `path_hops`
- `reverse_path_hops`
- `interface_counters`
- `routing_history`
- `connectivity_snapshot`
- `servicenow_summary`

### 3. Run-scoped Redis cache

Also used by the workflow tool layer for read-only backend results during a run:

- route lookups
- find-device lookups
- owner maps

This cache is scoped to the session/run and explicitly cleared.

### 4. MemoryManager responsibilities

[`services/memory_manager.py`](<services/memory_manager.py>) owns:

- pending clarification state
- recall-signal evaluation
- long-term memory store hook

Long-term recall is evidence-gated rather than always-on background context. It is triggered by signals such as:

- path anomalies
- interface failures
- service reachability failures
- unresolved connectivity findings

## Response Shaping

Atlas does not let the LLM freely shape every UI payload.

[`services/response_presenter.py`](<services/response_presenter.py>) owns:

- deterministic `ServiceNow` section replacement
- interface counter grouping
- network-ops path visibility rules
- fail-closed troubleshoot output when live evidence is unavailable

That means the final payload can safely contain:

- `direct_answer`
- `path_hops`
- `reverse_path_hops`
- `interface_counters`
- `connectivity_snapshot`
- `incident_summary`

without depending on the LLM to keep those structures consistent.

## Running Atlas

### Prerequisites

- Python environment in `.venv`
- Node.js for the frontend
- Ollama reachable at the configured `OLLAMA_BASE_URL`
- Redis recommended for LangGraph persistence and run cache
- ServiceNow / other backend credentials as needed

### Backend web app

Recommended development launcher:

```bash
cd /Users/manig/Documents/coding/atlas
.venv/bin/python run_web.py
```

This starts the FastAPI app on port `8001` with reload enabled.

### Nornir live network service

```bash
cd /Users/manig/Documents/coding/atlas
.venv/bin/python nornir/server.py
```

This starts the Nornir HTTP service on port `8006`.

### Frontend development server

```bash
cd frontend
npm install
npm run dev
```

Vite serves the frontend on port `5173`.

### Frontend production build

```bash
cd frontend
npm run build
```

FastAPI serves the built app automatically from `frontend/dist` when it exists.

## Key HTTP Endpoints

- `POST /api/chat`
  - SSE chat stream
  - emits `status` events and one `done` event
- `GET /api/chat/history`
- `GET /api/chat/conversations`
- `GET /api/chat/conversations/{conversation_id}`
- `GET /api/internal/diagnostics`
  - authenticated internal diagnostics snapshot

## Troubleshooting the Application

See:

- [`Documentation/General/troubleshooting/troubleshooting.md`](<Documentation/General/troubleshooting/troubleshooting.md>)
- [`Documentation/End-to-End-flow/troubleshooting-query-flow.md`](<Documentation/End-to-End-flow/troubleshooting-query-flow.md>)
