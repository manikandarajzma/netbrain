# LangGraph in Atlas

## Why LangGraph

Atlas uses LangGraph because Atlas has a small but explicit application flow that needs:

- one graph entrypoint for every request
- a typed shared state object passed between steps
- a single place to define lane routing
- a fixed transition from routing to workflow execution
- a clean graph exit that returns the final payload

In Atlas, LangGraph is the runtime layer that holds that flow together.

It is used for:

- entering the request through `classify_intent`
- reading and writing `AtlasState`
- routing to `call_troubleshoot_agent`, `call_network_ops_agent`, or `build_final_response`
- terminating at `END`

It is not used for:

- backend transport
- tool implementation
- tool selection inside the ReAct loop
- troubleshooting orchestration
- response presentation

Those responsibilities stay below the graph in workflow services, agents, tools, and owned backend services.

---

## LangGraph Role

Atlas uses LangGraph for one narrow, important job: lane selection and controlled execution flow.

The application entrypoint does not own:
- intent routing
- early exits for unsupported prompts
- troubleshoot vs network-ops branching
- final response assembly boundaries

LangGraph owns that control flow.

What LangGraph does **not** do in this design:
- it is not a giant tool-selection loop
- it is not where backend logic lives
- it is not where workflow orchestration lives

Those responsibilities live in owned services and pure agents.

---

## The Current Graph Shape

Atlas uses a small graph:

```text
classify_intent
    ├─► call_troubleshoot_agent
    ├─► call_network_ops_agent
    └─► build_final_response
```

- `classify_intent` performs graph-entry lane selection
- the two agent nodes are thin delegation nodes
- `build_final_response` is the graph exit point

The heavy work happens below the graph:
- `TroubleshootWorkflowService`
- `NetworkOpsWorkflowService`
- pure ReAct agents built by `AgentFactory`
- capability-based tools from `ToolRegistry`

---

## The Three Graph Files

### `graph/graph_state.py`

Defines `AtlasState`, the typed shared state passed between graph nodes.

The state is small because the graph has a narrow routing scope.

Important fields:

| Field | Purpose |
|-------|---------|
| `prompt` | Raw user prompt for this turn |
| `conversation_history` | Prior chat messages used for context |
| `username` | Authenticated user identity |
| `session_id` | Per-browser/session identifier |
| `request_id` | Per-request observability id |
| `intent` | Set by `classify_intent` to `troubleshoot`, `network_ops`, or `dismiss` |
| `rbac_error` | Optional user-facing authorization error |
| `final_response` | Final assistant payload returned to the caller |

`AtlasState` does not carry a large pile of tool-selection loop fields such as selected-tool bookkeeping, accumulated tool outputs, or retry-loop internals.

That logic stays out of the graph layer.

---

### `graph/graph_builder.py`

Compiles the graph and wires the nodes together.

Current routing behavior:

- `classify_intent` routes to one of:
  - `call_troubleshoot_agent`
  - `call_network_ops_agent`
  - `build_final_response`
- both agent nodes always flow into `build_final_response`
- `build_final_response` flows to `END`

`graph/graph_builder.py` owns:
- graph topology
- entry point
- conditional routing map

It does **not** own workflow logic.

#### How the route is chosen

The graph route is chosen in two explicit steps:

1. `classify_intent(...)` writes `state["intent"]`
2. `graph_builder.py` reads that value and picks the next node from the conditional edge map

The actual routing map is:

```python
g.add_conditional_edges(
    "classify_intent",
    self._route_intent,
    {
        "troubleshoot": "call_troubleshoot_agent",
        "network_ops": "call_network_ops_agent",
        "dismiss": "build_final_response",
    },
)
```

And the resolver is:

```python
def _route_intent(self, state: AtlasState) -> str:
    return state.get("intent") or "dismiss"
```

So when we say “the graph routes,” what really happens is:

- `classify_intent(...)` decides the `intent`
- the graph reads `state["intent"]`
- the graph follows the matching edge

---

### `graph/graph_nodes.py`

Implements the graph nodes, but only at the graph boundary.

Current node responsibilities:

| Node | What it does |
|------|--------------|
| `classify_intent` | Handles fast-path acknowledgements/pending follow-ups in code, then uses the router LLM for normal `troubleshoot`, `network_ops`, or `dismiss` lane selection |
| `call_troubleshoot_agent` | Thin delegation node that calls `TroubleshootWorkflowService.run(...)` |
| `call_network_ops_agent` | Thin delegation node that calls `NetworkOpsWorkflowService.run(...)` |
| `build_final_response` | Returns any final graph-owned response such as an RBAC error or previously prepared payload |

`graph/graph_nodes.py` is responsible for:
- routing
- graph boundary delegation
- minimal graph exit handling

It is not responsible for:
- tool execution
- retry loops
- tool chaining
- backend integration
- response formatting
- workflow-specific orchestration

---

## Where the Real Work Happens Now

LangGraph is the router, not the whole application.

The owned services below it do the rest:

### `TroubleshootWorkflowService`

Owns troubleshoot-side orchestration:
- run-scoped state reset
- pending-context recovery
- agent execution
- required follow-up enforcement
- session-side-effect merge
- presenter handoff

### `NetworkOpsWorkflowService`

Owns network-ops orchestration:
- network-ops agent execution
- follow-up clarification handling
- payload shaping for incident/change/ticket workflows

### `AgentFactory`

Builds the pure ReAct agents.

The graph does not build agent internals directly.

### `ToolRegistry`

Owns the uniform agent-facing tool surface:
- capability registration
- profile-to-capability mapping
- capability-to-tool resolution

### `ResponsePresenter`

Owns final payload shaping such as:
- path visualization payloads
- ServiceNow section replacement
- interface-counter grouping

---

## End-to-End Flow

### Troubleshoot Flow

```text
application/chat_service.py
    ▼
AtlasApplication
    ▼
AtlasRuntime
    ▼
atlas_graph
    ▼
classify_intent
    ▼
call_troubleshoot_agent
    ▼
TroubleshootWorkflowService
    ▼
Troubleshoot ReAct agent
    ▼
ToolRegistry tools
    ▼
owned services / backend clients
    ▼
ResponsePresenter
    ▼
build_final_response
```

### Network Ops Flow

```text
application/chat_service.py
    ▼
AtlasApplication
    ▼
AtlasRuntime
    ▼
atlas_graph
    ▼
classify_intent
    ▼
call_network_ops_agent
    ▼
NetworkOpsWorkflowService
    ▼
Network Ops ReAct agent
    ▼
ToolRegistry tools
    ▼
owned services / backend adapters
    ▼
build_final_response
```

---

## Summary

LangGraph in Atlas is small.

It is used to:
- classify the request
- choose the right workflow lane
- terminate cleanly

It is **not** the main place for:
- backend logic
- tool-chaining internals
- retry orchestration
- formatting

That work lives in the owners below the graph.
