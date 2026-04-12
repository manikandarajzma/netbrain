# Troubleshooting Query Flow

This document explains the end-to-end flow for a troubleshooting query in Atlas, starting from the moment a user submits a query in the UI and ending when the final response, path visualizations, and status timeline are rendered back in the browser.

The description below is based on the current implementation in:

- [`/Users/manig/Documents/coding/atlas/frontend/src/utils/api.js`](</Users/manig/Documents/coding/atlas/frontend/src/utils/api.js>)
- [`/Users/manig/Documents/coding/atlas/frontend/src/stores/chatStore.js`](</Users/manig/Documents/coding/atlas/frontend/src/stores/chatStore.js>)
- [`/Users/manig/Documents/coding/atlas/frontend/src/components/messages/AssistantMessage.jsx`](</Users/manig/Documents/coding/atlas/frontend/src/components/messages/AssistantMessage.jsx>)
- [`/Users/manig/Documents/coding/atlas/frontend/src/components/path/PathVisualization.jsx`](</Users/manig/Documents/coding/atlas/frontend/src/components/path/PathVisualization.jsx>)
- [`/Users/manig/Documents/coding/atlas/app.py`](</Users/manig/Documents/coding/atlas/app.py>)
- [`/Users/manig/Documents/coding/atlas/chat_service.py`](</Users/manig/Documents/coding/atlas/chat_service.py>)
- [`/Users/manig/Documents/coding/atlas/graph_nodes.py`](</Users/manig/Documents/coding/atlas/graph_nodes.py>)
- [`/Users/manig/Documents/coding/atlas/agents/troubleshoot_agent.py`](</Users/manig/Documents/coding/atlas/agents/troubleshoot_agent.py>)
- [`/Users/manig/Documents/coding/atlas/tools/all_tools.py`](</Users/manig/Documents/coding/atlas/tools/all_tools.py>)
- [`/Users/manig/Documents/coding/atlas/status_bus.py`](</Users/manig/Documents/coding/atlas/status_bus.py>)
- [`/Users/manig/Documents/coding/atlas/nornir/server.py`](</Users/manig/Documents/coding/atlas/nornir/server.py>)

## 1. User submits a troubleshooting query in the UI

The frontend entrypoint is the Zustand store in [`/Users/manig/Documents/coding/atlas/frontend/src/stores/chatStore.js`](</Users/manig/Documents/coding/atlas/frontend/src/stores/chatStore.js>).

When the user presses send:

1. `sendMessage(text)` is called.
2. The store:
   - appends the user message to the visible chat
   - appends the same text to `conversationHistory`
   - creates an `AbortController`
   - initializes status tracking:
     - `isLoading = true`
     - `currentStatus = "Identifying query"`
     - `statusSteps = []`
     - `_stepStart = performance.now()`
3. The store kicks off two async operations:
   - `discoverTool(...)`
   - `sendChat(...)`

Important detail:

- `discoverTool(...)` is only used to improve status text early in the run.
- The actual troubleshooting result comes from `sendChat(...)`.

## 2. Frontend calls `/api/chat` and opens an SSE stream

The network helper is in [`/Users/manig/Documents/coding/atlas/frontend/src/utils/api.js`](</Users/manig/Documents/coding/atlas/frontend/src/utils/api.js>).

`sendChat(message, conversationHistory, signal, ...)`:

1. Sends a `POST /api/chat` request.
2. Includes:
   - `message`
   - `conversation_history`
   - optional `conversation_id`
   - optional `parent_conversation_id`
3. Expects the response to be an SSE stream, not a normal JSON response.

The code then:

1. Reads the response stream chunk-by-chunk.
2. Parses SSE `data:` lines.
3. Handles two event types:
   - `status`
   - `done`

Status events update the live progress list in the chat UI.

The `done` event carries the final assistant payload.

## 3. FastAPI `/api/chat` creates the SSE workflow

The backend HTTP endpoint is in [`/Users/manig/Documents/coding/atlas/app.py`](</Users/manig/Documents/coding/atlas/app.py>) under:

- `@app.post("/api/chat")`

This route:

1. Validates authentication.
2. Reads the session cookie via `get_session_id(request)`.
3. Pulls:
   - `message`
   - `conversation_history`
   - optional conversation IDs
4. Registers a per-session status queue with [`/Users/manig/Documents/coding/atlas/status_bus.py`](</Users/manig/Documents/coding/atlas/status_bus.py>):
   - `queue = status_bus.register(sid)`
5. Starts the real backend work as a task:
   - `process_message(...)`
6. Streams status events from the queue while the task is running.
7. Sends SSE heartbeats (`: keep-alive`) during idle periods.
8. Cancels the task if the browser disconnects.
9. When the task finishes:
   - drains any remaining status events
   - attaches/saves chat history
   - emits one final SSE `done` event

Important operational behavior:

- Browser disconnects now cancel the running backend task.
- The route always tries to deregister the session queue in `finally`.

## 4. Status events are pushed through the per-session status bus

[`/Users/manig/Documents/coding/atlas/status_bus.py`](</Users/manig/Documents/coding/atlas/status_bus.py>) is intentionally small.

It is just:

- `_queues: dict[str, asyncio.Queue]`
- `register(session_id)`
- `deregister(session_id)`
- `push(session_id, message)`

Tools and graph nodes do not write to the frontend directly.
They call `_push_status(...)`, which eventually calls:

- `status_bus.push(session_id, "some status text")`

That status is then picked up by the SSE generator in `app.py`.

## 5. `process_message()` enters LangGraph

The main backend entrypoint is [`/Users/manig/Documents/coding/atlas/chat_service.py`](</Users/manig/Documents/coding/atlas/chat_service.py>).

`process_message(...)` does three important things:

1. Ensures the LangGraph Redis checkpointer is initialized:
   - `_ensure_checkpointer()`
2. Builds the initial graph state:
   - `prompt`
   - `conversation_history`
   - `username`
   - `session_id`
   - `intent`
   - `rbac_error`
   - `final_response`
3. Invokes the compiled graph:
   - `atlas_graph.ainvoke(initial_state, config=config)`

Important detail:

- `session_id` is also used as LangGraph `thread_id`
- that means graph checkpoints are keyed per browser session in Redis

This is separate from the tool-side run cache described later.

## 6. LangGraph classifies the query intent

The first graph node is [`/Users/manig/Documents/coding/atlas/graph_nodes.py`](</Users/manig/Documents/coding/atlas/graph_nodes.py>) `classify_intent(...)`.

This decides whether the prompt is:

- `troubleshoot`
- `network_ops`
- `dismiss`

For a connectivity question like:

```text
help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443
```

the graph routes to:

- `call_troubleshoot_agent(...)`

## 7. `call_troubleshoot_agent()` prepares a fresh troubleshooting run

This function lives in [`/Users/manig/Documents/coding/atlas/graph_nodes.py`](</Users/manig/Documents/coding/atlas/graph_nodes.py>).

Before invoking the agent, it resets run-scoped state:

1. `clear_session_cache(session_id)`
2. `pop_session_data(session_id)`

This is important because Atlas uses two different state layers:

### 7.1 Redis run cache

In [`/Users/manig/Documents/coding/atlas/tools/all_tools.py`](</Users/manig/Documents/coding/atlas/tools/all_tools.py>), there is a session-scoped Redis cache for read-only Nornir results.

This cache stores things like:

- route lookups
- find-device lookups
- interface ownership maps

The cache is keyed by:

- `session_id`
- endpoint
- request payload

and is explicitly cleared:

- at the start of a troubleshoot run
- at the end of the troubleshoot run

This means cached data is reused only within the same run/session lifecycle, not indefinitely.

### 7.2 In-memory session side-effect store

Also in [`/Users/manig/Documents/coding/atlas/tools/all_tools.py`](</Users/manig/Documents/coding/atlas/tools/all_tools.py>), `_store(session_id)` holds structured side effects from tools:

- `path_hops`
- `reverse_path_hops`
- `interface_counters`
- `routing_history`
- `ping_results`
- `all_interfaces`
- `interface_details`
- `syslog`
- `protocol_discovery`
- `connectivity_snapshot`
- `ip_owners`

The LLM does not return these directly.
Tools populate them as side effects, and `graph_nodes.py` later packages them into the final response.

## 8. The troubleshoot agent is built with the connectivity runbook

The agent is built in [`/Users/manig/Documents/coding/atlas/agents/troubleshoot_agent.py`](</Users/manig/Documents/coding/atlas/agents/troubleshoot_agent.py>).

`build_agent(prompt, issue_type)` does this:

1. Picks the system prompt:
   - core prompt from `skills/troubleshooter.md`
   - scenario prompt from `skills/troubleshooting_scenarios/connectivity.md`
2. Chooses the tool list:
   - for connectivity: `CONNECTIVITY_TOOLS`
3. Creates a LangGraph ReAct agent using:
   - `ChatOpenAI(...)`
   - `create_react_agent(...)`

Important detail:

- connectivity runs use `CONNECTIVITY_TOOLS`, not the entire `ALL_TOOLS` list
- this was done so the agent cannot fan out into every legacy low-level tool by default

## 9. The connectivity agent executes the investigation

The agent runs inside LangGraph, but the actual useful work is done by tool calls in [`/Users/manig/Documents/coding/atlas/tools/all_tools.py`](</Users/manig/Documents/coding/atlas/tools/all_tools.py>).

The current intended connectivity flow is:

1. `trace_path(source_ip, dest_ip)`
2. `trace_reverse_path(source_ip, dest_ip)`
3. `lookup_routing_history(dest_ip)`
4. `search_servicenow(...)`
5. `collect_connectivity_snapshot(source_ip, dest_ip, port)`
6. LLM synthesis from the accumulated evidence

The graph also has enforcement logic:

- if no `connectivity_snapshot` exists after the first agent pass, it forces a follow-up pass
- if path visualizations are missing, it directly calls `trace_path` and `trace_reverse_path` itself

That logic is in [`/Users/manig/Documents/coding/atlas/graph_nodes.py`](</Users/manig/Documents/coding/atlas/graph_nodes.py>).

## 10. Live path tracing

The core of live path tracing is `_live_path_trace(...)` in [`/Users/manig/Documents/coding/atlas/tools/all_tools.py`](</Users/manig/Documents/coding/atlas/tools/all_tools.py>).

This function returns three things:

- human-readable text
- `structured_hops`
- `flags`

### 10.1 First-hop discovery

The trace begins by finding the first L3 gateway for the source IP.

`_find_first_hop(src_ip)`:

1. gets device list from `GET /devices` on the Nornir server
2. loops over devices
3. asks each device for a route to the source IP:
   - `POST /route`
4. the first device that reports:
   - `found = true`
   - `protocol = connected`
   is treated as the source-side first hop

The host-facing interface on that device becomes:

- `first_hop_lan_interface`

### 10.2 Hop walking

Once the first hop is known, `_live_path_trace(...)` loops hop-by-hop:

1. query current device route to destination via:
   - `POST /route`
2. inspect:
   - `egress_interface`
   - `next_hop`
   - `protocol`
   - `prefix`
3. if route is `connected`/`local` and no next hop exists:
   - use `POST /arp`
   - treat that device as the final router before the destination host
4. otherwise resolve `next_hop` to a device

### 10.3 Next-hop owner resolution

This is where the path trace turns an IP next hop like `169.254.0.5` into a device/interface like:

- `arista-ai4 / Ethernet2`

Current logic:

1. preload an owner map once per run from:
   - `POST /ip-owners`
2. if the owner map has the next-hop IP:
   - use it directly
3. otherwise:
   - do a one-off exact live `POST /find-device`
   - cache the result into the session store

This was introduced to avoid repeated inventory-wide next-hop-owner probing on every hop.

### 10.4 Structured path hops

Each hop is stored as a dict like:

```json
{
  "from_device": "arista-ai2",
  "from_device_type": "switch",
  "out_interface": "Ethernet2",
  "to_device": "arista-ai3",
  "to_device_type": "switch",
  "in_interface": "Ethernet1"
}
```

These become:

- `path_hops`
- `reverse_path_hops`

in the per-session store.

### 10.5 Path metadata extraction

After the hop list is built:

- `_extract_path_metadata(hops)`
- `_extract_reverse_path_metadata(hops)`

derive:

- `first_hop_device`
- `first_hop_lan_interface`
- `first_hop_egress_interface`
- `last_hop_device`
- `last_hop_egress_interface`
- `reverse_first_hop_device`
- `reverse_first_hop_lan_interface`
- `reverse_first_hop_egress_interface`
- `path_devices`
- `path_hops_for_counters`

That metadata drives later actions like:

- source interface selection for pings
- destination-side TCP test selection
- interface counter selection

## 11. Holistic connectivity snapshot

The snapshot entrypoint is:

- `collect_connectivity_snapshot(...)`

in [`/Users/manig/Documents/coding/atlas/tools/all_tools.py`](</Users/manig/Documents/coding/atlas/tools/all_tools.py>).

This is not the live path walker.
It is the structured evidence collection pass that happens after path discovery.

### 11.1 Device set selection

It builds the device set from:

- forward path devices
- reverse path devices
- routing-history devices
- historical peering hint devices

### 11.2 One snapshot per device

For each device in scope, Atlas calls:

- `POST /device-snapshot`

on the Nornir server.

Each `/device-snapshot` call is intended to use one SSH session for that device and gather:

- route to source
- route to destination
- interface inventory
- protocol discovery
- limited syslog
- OSPF detail only if OSPF is discovered
- relevant interface state/detail/counters

The device snapshots are collected in parallel across devices.

### 11.3 Additional service check

If a port is provided, the snapshot path also performs:

- destination-side TCP test

The intended source is the destination-side device inferred from live route data, typically:

1. a device with `connected` or `local` route to destination
2. else reverse-path first hop
3. else fallback metadata

### 11.4 Snapshot output

The snapshot builds:

- structured object stored in `store["connectivity_snapshot"]`
- compact text summary returned to the agent

The LLM reasons over the compact summary, not the full raw device dumps.

## 12. ServiceNow lookup

The connectivity flow may also call `search_servicenow(...)`.

Current design intent:

- best-effort only
- time-bounded
- must not block core connectivity diagnosis

This is important because ServiceNow context is supplemental, not foundational.

## 13. Graph-level post-processing and packaging

After the agent finishes, [`/Users/manig/Documents/coding/atlas/graph_nodes.py`](</Users/manig/Documents/coding/atlas/graph_nodes.py>) does not just blindly return the raw LLM output.

It:

1. extracts final assistant text from the agent messages
2. reads tool side effects via:
   - `pop_session_data(session_id)`
3. merges side effects across passes with:
   - `_merge_session_data(...)`
4. enforces mandatory evidence when needed

### 13.1 What goes into the final response payload

The final `content` dict may contain:

- `direct_answer`
- `path_hops`
- `reverse_path_hops`
- `source`
- `destination`
- `interface_counters`
- `incident_summary`
- `connectivity_snapshot`

This is the payload the frontend ultimately receives in the SSE `done` event.

## 14. SSE `done` event returns to the frontend

Back in [`/Users/manig/Documents/coding/atlas/app.py`](</Users/manig/Documents/coding/atlas/app.py>), once `process_message(...)` finishes:

1. the result is read from the task
2. conversation history is saved
3. noisy L2 text may be stripped from path content
4. a final SSE event is sent:

```json
{
  "type": "done",
  "result": {
    "role": "assistant",
    "content": { ... }
  }
}
```

## 15. Frontend converts the SSE payload into a renderable assistant message

In [`/Users/manig/Documents/coding/atlas/frontend/src/stores/chatStore.js`](</Users/manig/Documents/coding/atlas/frontend/src/stores/chatStore.js>):

1. the `done` event is received
2. `rawContent = data.content ?? data.message ?? 'No response'`
3. if `data.path_hops` exists, the frontend wraps everything into a structured object:

```js
{
  text: rawContent,
  path_hops: data.path_hops,
  reverse_path_hops: data.reverse_path_hops
}
```

4. that object is appended as the assistant message content

This is the bridge that allows the UI to show:

- path visualization
- return path visualization
- markdown explanation text

in the same assistant bubble.

## 16. Assistant message rendering

The main renderer is [`/Users/manig/Documents/coding/atlas/frontend/src/components/messages/AssistantMessage.jsx`](</Users/manig/Documents/coding/atlas/frontend/src/components/messages/AssistantMessage.jsx>).

If the response is classified as a path-style response:

1. render `Forward Path`
2. render `PathVisualization` for `content.path_hops`
3. if `reverse_path_hops` exists:
   - render `Return Path`
   - render a second `PathVisualization`
4. render the markdown/text explanation below the diagrams

## 17. Path visualization rendering

[`/Users/manig/Documents/coding/atlas/frontend/src/components/path/PathVisualization.jsx`](</Users/manig/Documents/coding/atlas/frontend/src/components/path/PathVisualization.jsx>) turns `path_hops` into display nodes.

### 17.1 How nodes are built

It:

1. starts from the first hop
2. builds a `Map` of nodes in order
3. updates `out` interfaces once a node later appears as `from_device`
4. marks:
   - the first node as source
   - the last node as destination only if the last hop ends on a host

This last rule is important:

- if the path merely stops on a router/switch, the UI should not pretend that device is the destination host

### 17.2 Path item display

[`/Users/manig/Documents/coding/atlas/frontend/src/components/path/PathItem.jsx`](</Users/manig/Documents/coding/atlas/frontend/src/components/path/PathItem.jsx>) renders:

- node name
- `A` and `B` badges for source/destination
- source/destination IPs only when appropriate
- `In:` and `Out:` interfaces

## 18. Status timeline rendering

The running status timeline comes from:

- the SSE `status` events
- stored in `chatStore.statusSteps`
- rendered by [`/Users/manig/Documents/coding/atlas/frontend/src/components/chat/StatusMessage.jsx`](</Users/manig/Documents/coding/atlas/frontend/src/components/chat/StatusMessage.jsx>)

The store records:

- label
- duration

using `performance.now()` timing.

## 19. Summary of the full lifecycle

For a troubleshooting query, the flow is:

1. user types message in chat
2. frontend store sends `/api/chat`
3. backend opens SSE stream
4. `process_message()` invokes LangGraph
5. `classify_intent()` routes to troubleshooting
6. `call_troubleshoot_agent()` clears run-scoped cache/state
7. troubleshoot agent runs connectivity runbook with `CONNECTIVITY_TOOLS`
8. tools push statuses through `status_bus`
9. live path trace builds `path_hops`
10. reverse trace builds `reverse_path_hops`
11. snapshot collection builds `connectivity_snapshot`
12. graph merges all tool side effects
13. graph packages final payload
14. SSE `done` event returns the final result
15. frontend wraps `path_hops` + text into a renderable object
16. assistant message renders:
   - forward path
   - return path
   - markdown summary
   - counters / extra details

## 20. Architectural separation of responsibilities

The system is split like this:

### Frontend

- submit query
- stream statuses
- display path diagrams and analysis

### FastAPI app

- authentication
- SSE orchestration
- cancellation handling
- conversation history persistence

### Chat service

- initializes LangGraph checkpointer
- invokes graph with `session_id`

### Graph nodes

- classify request
- choose troubleshoot vs network ops
- enforce mandatory evidence
- package final response payload

### Agent

- reasons over the connectivity runbook
- chooses tools
- writes the final narrative

### Tools

- perform the actual work
- push statuses
- store structured side effects
- return human-readable summaries to the agent

### Nornir server

- executes live SSH-backed network commands
- provides low-level route / path / interface / snapshot primitives

## 21. Current design intent for production scale

The intended shape of this architecture for large networks is:

- live path trace stays live and minimal
- device evidence collection happens once per device in parallel
- repeated read-only lookups are cached only within the current run/session
- the agent reasons over a structured incident snapshot, not over many scattered raw calls

That is the direction the current code is trying to implement, even though some parts have recently been under active debugging.

