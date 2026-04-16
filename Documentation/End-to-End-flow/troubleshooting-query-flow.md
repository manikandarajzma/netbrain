# Troubleshooting Query Flow

This document describes the current end-to-end flow for a troubleshooting request in Atlas, from the moment the user clicks send in the browser to the moment the UI renders the final structured response.

It is based on the owner hierarchy:

- Entry and runtime:
  - [`frontend/src/stores/chatStore.js`](<frontend/src/stores/chatStore.js>)
  - [`frontend/src/utils/api.js`](<frontend/src/utils/api.js>)
  - [`app.py`](<app.py>)
  - [`application/chat_service.py`](<application/chat_service.py>)
  - [`application/atlas_application.py`](<application/atlas_application.py>)
  - [`services/graph_runtime.py`](<services/graph_runtime.py>)
- Graph boundary:
  - [`graph/graph_builder.py`](<graph/graph_builder.py>)
  - [`graph/graph_nodes.py`](<graph/graph_nodes.py>)
  - [`graph/graph_state.py`](<graph/graph_state.py>)
- Workflow owners:
  - [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>)
  - [`services/network_ops_workflow_service.py`](<services/network_ops_workflow_service.py>)
  - [`services/path_trace_service.py`](<services/path_trace_service.py>)
  - [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>)
  - [`services/connectivity_snapshot_service.py`](<services/connectivity_snapshot_service.py>)
  - [`services/routing_diagnostics_service.py`](<services/routing_diagnostics_service.py>)
  - [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>)
- Shared owners:
  - [`services/memory_manager.py`](<services/memory_manager.py>)
  - [`services/response_presenter.py`](<services/response_presenter.py>)
  - [`services/session_store.py`](<services/session_store.py>)
  - [`services/status_service.py`](<services/status_service.py>)
  - [`services/workflow_state_service.py`](<services/workflow_state_service.py>)
  - [`services/observability.py`](<services/observability.py>)
  - [`services/metrics.py`](<services/metrics.py>)
  - [`services/diagnostics_service.py`](<services/diagnostics_service.py>)
- Agent-facing tool surface:
  - [`tools/tool_registry.py`](<tools/tool_registry.py>)
  - [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)
  - [`tools/device_agent_tools.py`](<tools/device_agent_tools.py>)
  - [`tools/routing_agent_tools.py`](<tools/routing_agent_tools.py>)
  - [`tools/connectivity_agent_tools.py`](<tools/connectivity_agent_tools.py>)
  - [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)
  - [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>)
- Backend runtime:
  - [`services/nornir_client.py`](<services/nornir_client.py>)
  - [`nornir/server.py`](<nornir/server.py>)

## High-Level Sequence

1. The user submits a troubleshooting request in the React UI.
2. The frontend opens `POST /api/chat` as an SSE stream.
3. `app.py` owns authentication, session lookup, the status queue, and the SSE lifecycle.
4. `application/chat_service.py` delegates the request to `AtlasApplication`.
5. `AtlasApplication` invokes `AtlasRuntime`.
6. `AtlasRuntime` builds graph state and calls the compiled LangGraph graph.
7. LangGraph classifies the intent and routes to the troubleshoot workflow.
8. `TroubleshootWorkflowService` invokes the pure troubleshoot ReAct agent.
9. The agent gets its tool profile from `ToolRegistry`.
10. Those tools call owned services and backends such as:
    - `PathTraceService`
    - `DeviceDiagnosticsService`
    - `RoutingDiagnosticsService`
    - `ConnectivitySnapshotService`
    - `ServiceNowSearchService`
    - Nornir on `:8006`
    - MCP-backed systems
11. Tool side effects accumulate in `SessionStore`.
12. `ResponsePresenter` turns final text plus session side effects into the structured frontend payload.
13. `app.py` streams the completed answer back to the browser, where the UI renders markdown, path visuals, counters, and status steps.

## Agent View

1. `AtlasRuntime` passes prompt, session id, request id, and conversation context into the graph.
2. The graph selects the troubleshoot agent node.
3. The troubleshoot workflow builds a pure ReAct agent and gives it a restricted tool profile.
4. The agent sees one uniform Atlas tool interface.
5. Under that interface:
   - workflow tools handle path tracing, snapshots, routing diagnostics, and ServiceNow correlation
   - memory tools handle historical recall when live evidence justifies it
   - product-facing ServiceNow tools handle incident and change CRUD/detail operations
6. Tool results return both human-readable output and structured side effects.
7. The workflow and presenter turn those results into the final response payload.

## 1. Frontend Submission

The frontend entrypoint is [`frontend/src/stores/chatStore.js`](<frontend/src/stores/chatStore.js>).

When the user submits a troubleshoot prompt:

1. `sendMessage(text)` appends the user message to chat state.
2. The store resets live status tracking:
   - `currentStatus = "Routing request"`
   - `statusSteps = []`
   - `_stepStart = performance.now()`
3. It launches the real chat request:
   - `sendChat(...)`

The actual routing happens inside LangGraph after `/api/chat` starts.

## 2. Frontend Opens the SSE Chat Stream

[`frontend/src/utils/api.js`](<frontend/src/utils/api.js>) sends:

- `POST /api/chat`

with:

- `message`
- `conversation_history`
- optional conversation metadata

The response is treated as **Server-Sent Events**, not plain JSON.

The frontend listens for:

- `status`
- `done`

### Status handling

Status events update the timeline in the UI.

Current early neutral labels are:

- `Routing request`
- `Analyzing your query...`

Later statuses come from graph nodes and tools.

## Internal Diagnostics Surface

Authenticated internal diagnostics endpoint:

- `GET /api/internal/diagnostics`

That endpoint does not run the graph. It returns a read-only snapshot from `DiagnosticsService` containing:

- owner summary
- checkpointer readiness
- tool registry profile mappings
- registered tool/capability metadata
- in-process metrics counters and timings

This is useful when debugging the runtime architecture itself rather than a user workflow.

## 3. FastAPI Owns the SSE Lifecycle

The main endpoint is [`app.py`](<app.py>) `api_chat(...)`.

Responsibilities:

1. authenticate the user
2. load `session_id`
3. register a status queue via [`application/status_bus.py`](<application/status_bus.py>)
4. start `process_message(...)` in a background task
5. stream:
   - `status` events from the queue
   - keep-alive heartbeats
   - final `done` event
6. cancel the backend task if the browser disconnects
7. save conversation history after the result is ready

Important operational behavior:

- stopping the browser request cancels the in-flight backend task
- after write-like requests, `app.py` also clears relevant Redis caches so later reads do not serve stale ticket data

## 4. `application/chat_service.py` Is Intentionally Thin

It does one thing:

- delegate to `atlas_application.process_query(...)`

It also exports `_IP_OR_CIDR_RE` so the graph node can reuse the same IP/CIDR detection logic.

## 5. `AtlasApplication` Owns Top-Level Query Processing

[`application/atlas_application.py`](<application/atlas_application.py>) is the top-level owner.

It owns these collaborating objects:

- `AtlasRuntime`
- `AgentFactory`
- `MemoryManager`
- `ResponsePresenter`
- `ToolRegistry`

For each query it:

1. calls `self.runtime.invoke_atlas_graph(...)`
2. calls `self.runtime.extract_final_response(...)`

This keeps the entrypoint clean and gives the application a single high-level owner.

## 6. `AtlasRuntime` Builds State, Ensures the Checkpointer, and Invokes the Graph

[`services/graph_runtime.py`](<services/graph_runtime.py>) owns graph execution.

It is responsible for:

- building initial LangGraph state
- building per-run graph config
- ensuring Redis-backed checkpointer setup
- invoking `atlas_graph`
- extracting the final payload

### Initial state

`build_initial_state(...)` creates:

- `prompt`
- `conversation_history`
- `username`
- `session_id`
- `request_id`
- `intent = None`
- `rbac_error = None`
- `final_response = None`

### Graph config

`build_graph_config(...)` always sets:

- `recursion_limit = 50`

and, when a `session_id` exists:

- `configurable.thread_id = session_id`

That means LangGraph conversation state is keyed to the browser session.

`DiagnosticsService` reads the same runtime/checkpointer owners so their current readiness can be inspected without sending a synthetic user prompt.

## 6a. Observability and Request IDs

[`services/observability.py`](<services/observability.py>) owns shared observability helpers.

Current behavior:

- every incoming query gets a `request_id`
- the `request_id` is carried through:
  - `AtlasApplication`
  - `AtlasRuntime`
  - `graph/graph_nodes.py`
  - diagnostics-visible runtime snapshots
- major runtime events are logged as structured JSON messages, including:
  - `query_started`
  - `query_completed`
  - `graph_invoke_started`
  - `graph_invoke_completed`
  - `intent_classified`
  - `troubleshoot_agent_completed`
  - `network_ops_agent_completed`
  - Nornir request timing and cache events

- lightweight in-process metrics are also recorded through [`services/metrics.py`](<services/metrics.py>), including:
  - query counters and duration
  - graph invocation counters and duration
  - agent success/failure counters and duration
  - Nornir request duration
  - Nornir cache hit/store counters

This makes it much easier to correlate:

- one frontend request
- one graph run
- the selected agent path
- live backend activity

## 7. Redis Checkpointer Is Optional but Supported

[`services/checkpointer_runtime.py`](<services/checkpointer_runtime.py>) owns checkpointer lifecycle.

On first use:

1. it attempts to create `AsyncRedisSaver`
2. it recompiles `atlas_graph` with that checkpointer
3. if Redis is unavailable, Atlas logs a warning and continues without persistent graph state

This is important:

- **Atlas runs if Redis is unavailable**
- but multi-turn LangGraph persistence degrades gracefully

## 8. LangGraph Performs Coarse Routing

The graph itself is defined in [`graph/graph_builder.py`](<graph/graph_builder.py>).

Graph shape:

```text
classify_intent
  ├─► call_troubleshoot_agent
  ├─► call_network_ops_agent
  └─► build_final_response
```

This is a deliberately small graph.

Atlas does **not** use the graph for deep reasoning. The graph only owns:

- coarse routing
- thin node delegation
- early exit

## 9. `classify_intent()` Decides Which Agent Runs

[`graph/graph_nodes.py`](<graph/graph_nodes.py>) `classify_intent(...)` performs deterministic coarse routing.

Possible values:

- `troubleshoot`
- `network_ops`
- `dismiss`

### Important behavior

- bare acknowledgements can dismiss if nothing is pending
- pending clarification context can continue the previous flow
- diagnostic phrasing overrides ops keywords

That means:

- regex chooses the **agent**
- the chosen agent chooses the **tools**

## 10. Troubleshoot Node Starts a Fresh Live Investigation

`call_troubleshoot_agent(...)` in [`graph/graph_nodes.py`](<graph/graph_nodes.py>) is a thin delegation node. The orchestration lives in [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>).

Before the agent runs, `TroubleshootWorkflowService` explicitly resets run-scoped live state:

- `nornir_client.clear_session_cache(session_id)`
- `pop_session_data(session_id)`

This prevents stale live evidence from a prior run from contaminating the current one.

`TroubleshootWorkflowService` then:

1. pushes `Investigating...`
2. recovers pending clarification context if present
3. optionally expands `INC...` prompts into a more explicit troubleshooting prompt
4. decides the effective `issue_type`
5. builds the troubleshoot agent
6. invokes the agent
7. enforces any required evidence follow-up
8. delegates final payload shaping to `ResponsePresenter`

## 11. Incident-Based Troubleshooting Is Rewritten Before Agent Execution

When the prompt references an incident such as:

```text
help me troubleshoot INC0010043
```

the graph uses [`services/request_preprocessor.py`](<services/request_preprocessor.py>) to:

- fetch the incident details
- extract IPs and other context
- rewrite the prompt into a concrete troubleshooting prompt

That prevents incident troubleshooting from drifting into a vague generic path.

## 12. `AgentFactory` Builds Thin ReAct Agents

[`agents/agent_factory.py`](<agents/agent_factory.py>) owns agent creation.

It provides:

- `build_default_llm()`
- `create_specialized_agent(...)`

The factory is intentionally minimal:

- it creates `ChatOpenAI(...)`
- it wires `create_react_agent(...)`
- it applies the system prompt

It does **not** own:

- checkpointer setup
- session state
- response formatting
- status updates

Those responsibilities stay outside the agent layer.

## 13. The Troubleshoot Agent Selects the Prompt and Tool Set

[`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>) decides:

- which troubleshooting system prompt to use
- which tool collection to expose

### Prompt composition

It loads:

- core prompt from `skills/troubleshooter.md`
- scenario prompt from one of:
  - `skills/troubleshooting_scenarios/connectivity.md`
  - `skills/troubleshooting_scenarios/performance.md`
  - `skills/troubleshooting_scenarios/intermittent.md`

### Tool selection

- connectivity scenario → profile `troubleshoot.connectivity`
- other troubleshoot cases → profile `troubleshoot.general`

That keeps connectivity runs constrained and reduces accidental tool sprawl.

## 14. ToolRegistry Owns the Tool Sets

[`tools/tool_registry.py`](<tools/tool_registry.py>) owns:

- capability registration
- profile registration
- profile → capability → tool resolution

This gives the application one place to define which tools are allowed for each agent without relying on hand-maintained static lists.

Current intent:

- profile `troubleshoot.connectivity`
  - live evidence and correlation tools only
  - no `recall_similar_cases(...)`
- profile `network_ops`
  - ServiceNow + operational review tools
  - may use path lookup for CI context

## 15. The Tool Layer Does the Real Backend Work

Atlas keeps one uniform agent-facing tool interface, with a few distinct owner-backed implementation styles:

- workflow entrypoints split across:
  - [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)
  - [`tools/device_agent_tools.py`](<tools/device_agent_tools.py>)
  - [`tools/routing_agent_tools.py`](<tools/routing_agent_tools.py>)
  - [`tools/connectivity_agent_tools.py`](<tools/connectivity_agent_tools.py>)
  - [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)
- [`services/path_trace_service.py`](<services/path_trace_service.py>) for live forward/reverse path walking and derived path metadata
- [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>) for ping, TCP, routing-check, and interface diagnostic workflow
- [`services/connectivity_snapshot_service.py`](<services/connectivity_snapshot_service.py>) for the heavy connectivity evidence bundle
- [`services/routing_diagnostics_service.py`](<services/routing_diagnostics_service.py>) for routing-history, OSPF, and syslog diagnostic workflow
- [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>) for Atlas-specific ServiceNow incident/change correlation
- [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>) for thin product-facing ServiceNow adapters

Each tool may do all of the following:

- push a status message
- call a backend
- store structured side effects
- return a string for the LLM

### Example connectivity tools

- `trace_path(...)`
- `trace_reverse_path(...)`
- `check_routing(...)`
- `trace_path(...)` and `trace_reverse_path(...)` delegate live hop-by-hop path assembly to `PathTraceService`
- `collect_connectivity_snapshot(...)` (delegates evidence assembly to `ConnectivitySnapshotService`)
- `search_servicenow(...)` (delegates correlation/query formatting to `ServiceNowSearchService`)
- `lookup_vendor_kb(...)`

## 16. Atlas Uses Two Different Runtime State Layers for Tool Output

### 16.1 In-memory per-session side-effect store

[`services/session_store.py`](<services/session_store.py>) owns the transient structured run data written by tools, including:

- `path_hops`
- `reverse_path_hops`
- `interface_counters`
- `routing_history`
- `interface_details`
- `syslog`
- `connectivity_snapshot`
- `servicenow_summary`

These are not the agent’s natural language answer. They are structured artifacts later used by the presenter.

### 16.2 Redis run cache

The workflow tool layer also uses Redis for read-only run-scoped caching, such as:

- Nornir route lookups
- next-hop owner lookups
- device-owner maps

This cache is explicitly cleared at the beginning and end of the run.

## 17. Nornir Is a Separate Live Collection Service

Live network collection does **not** happen inside the Atlas web process.

Atlas calls the Nornir HTTP service at:

- `http://localhost:8006`

implemented in:

- [`nornir/server.py`](<nornir/server.py>)

This service is responsible for:

- live route lookup
- next-hop ownership lookup
- device snapshots
- interface counters
- TCP tests
- other live SSH-backed network actions

This separation is important because:

- Atlas remains the reasoning/orchestration layer
- Nornir remains the live collection layer

## 18. ServiceNow and Other Systems Are Accessed Through the Tool Layer

Not every tool goes through Nornir.

For example:

- ServiceNow work often uses the MCP path
- other external systems can be reached through dedicated tool integrations

The important invariant is:

- the agent does not talk to those systems directly
- tools do

## 19. `MemoryManager` Owns Pending Context and Recall Policy

[`services/memory_manager.py`](<services/memory_manager.py>) owns two important memory concerns:

### Pending clarification context

It keeps track of whether the prior answer was asking the user for more details.

That allows follow-up answers to continue the correct flow instead of being treated as brand-new queries.

### Long-term memory recall policy

Recall is evidence-gated rather than always-on background context.

`MemoryManager` evaluates live evidence signals such as:

- path anomalies
- active interface errors
- interface lookup failures
- recent syslog signals
- OSPF instability
- peer diagnosis anomalies
- failed pings
- unresolved connectivity snapshot findings

Only when those signals justify recall should historical memory be used.

## 20. Troubleshoot Node Enforces Required Evidence

After the first troubleshoot agent pass, `call_troubleshoot_agent(...)` may run additional enforcement logic:

- if no `connectivity_snapshot` exists but the prompt is a connectivity case, Atlas forces a snapshot follow-up
- if path visuals are missing, Atlas directly runs `trace_path(...)` and `trace_reverse_path(...)`

This is application logic, not agent creativity.

The goal is to avoid answers that omit mandatory live evidence artifacts.

## 21. Presenter Owns the Final Payload Shape

[`services/response_presenter.py`](<services/response_presenter.py>) owns the final UI-facing payload.

It does not just pass through the agent’s text.

It also:

- deterministically replaces the `ServiceNow` section with the tool-generated summary
- groups duplicate interface-counter rows by device
- decides whether network-ops answers should include path visuals
- fails closed when live troubleshoot evidence is unavailable

### Fail-closed behavior

If a troubleshoot run has:

- no live path
- no reverse path
- no grouped interface counters
- and `connectivity_snapshot.live_evidence_available == False`

then Atlas does **not** claim a specific OSPF or routing root cause.

Instead, it returns a deterministic “unable to determine current root cause from live evidence” style answer.

## 22. The Final LangGraph State Returns `final_response`

The graph node returns:

```python
{"final_response": {"role": "assistant", "content": payload}}
```

where `payload` may include:

- `direct_answer`
- `path_hops`
- `reverse_path_hops`
- `interface_counters`
- `connectivity_snapshot`
- `incident_summary`

`AtlasRuntime.extract_final_response(...)` then pulls that payload out of the graph result.

## 23. FastAPI Persists Chat History and Emits the Final SSE `done`

Back in [`app.py`](<app.py>):

1. the completed result is read from the task
2. if the user message was a write-like operation, relevant caches are flushed
3. the assistant content is stored in chat history
4. the endpoint sends one final SSE `done` event

That event contains the structured response payload.

## 24. Frontend Renders Structured Troubleshooting Output

The frontend assistant-message layer inspects the returned payload:

- markdown-like `direct_answer`
- `path_hops`
- `reverse_path_hops`
- `interface_counters`

Then it renders:

- the main answer text
- the forward path panel
- the reverse path panel
- grouped interface counters
- the status timeline recorded during execution

## 25. Summary of Responsibility Boundaries

### `app.py`

- HTTP
- SSE
- auth
- history persistence

### `application/chat_service.py`

- thin entrypoint only

### `AtlasApplication`

- top-level owner for query processing

### `AtlasRuntime`

- graph state/config/invocation/final extraction

### `graph/graph_nodes.py`

- routing and thin node delegation

### `TroubleshootWorkflowService`

- troubleshoot agent orchestration
- mandatory evidence follow-up
- fail-closed troubleshooting execution

### `NetworkOpsWorkflowService`

- network-ops agent orchestration
- clarification follow-up handling

### agents

- reasoning + tool use only

### Workflow tool entrypoints

- `tools/path_agent_tools.py`
- `tools/device_agent_tools.py`
- `tools/routing_agent_tools.py`
- `tools/connectivity_agent_tools.py`
- `tools/servicenow_workflow_tools.py`
- `tools/all_tools.py` serves as a compatibility export layer

### `ConnectivitySnapshotService`

- owned connectivity evidence collection and summarization
- keeps heavyweight snapshot logic out of the workflow module

### `MemoryManager`

- pending context + recall policy

### `ResponsePresenter`

- deterministic output shaping

### `nornir/server.py`

- live SSH-backed network collection

## 26. Practical Example

For a prompt like:

```text
help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443
```

the typical modern flow is:

1. UI opens `/api/chat` SSE
2. `process_message(...)` delegates to `AtlasApplication`
3. `AtlasRuntime` builds graph state and invokes `atlas_graph`
4. `classify_intent()` returns `troubleshoot`
5. `TroubleshootWorkflowService` clears run-scoped live state
6. troubleshoot agent runs the connectivity scenario
7. tools collect live path/snapshot/ServiceNow evidence
8. `TroubleshootWorkflowService` enforces any required follow-up
9. presenter injects deterministic `ServiceNow` and grouped counter output
10. FastAPI sends the final structured payload
11. UI renders markdown, path visuals, counters, and the status timeline

This is the architecture implemented in Atlas.
