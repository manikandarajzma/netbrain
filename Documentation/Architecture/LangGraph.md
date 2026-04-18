# LangGraph in Atlas

## What LangGraph Is

LangGraph is a graph-based orchestration library for LLM applications.

At a high level, it models execution as:

- a shared state object
- a set of named nodes
- edges between those nodes
- a runtime that moves from node to node until the graph reaches `END`

Instead of writing one large request handler with many nested conditionals, a LangGraph application defines:

- what data is carried in state
- which node runs first
- which node runs next
- when the workflow stops

In Atlas, LangGraph is used as the application-flow runtime above the agents and below the web entrypoint.

## Core LangGraph Components

These are the main concepts that matter for understanding the Atlas graph.

### State

State is the shared data object passed between graph nodes.

A node reads the current state, performs its work, and returns a partial update to state.

Examples of the kinds of values a LangGraph state may carry:

- the current user prompt
- prior conversation history
- routing decisions
- final response payloads
- runtime metadata such as request ids

In Atlas, this shared object is `AtlasState`.

### Nodes

Nodes are the named execution steps in the graph.

A node is usually a Python function that:

- receives the current state
- performs one bounded piece of work
- returns updated state fields

Examples in Atlas:

- `classify_intent`
- `call_troubleshoot_agent`
- `call_network_ops_agent`
- `build_final_response`

### Edges

Edges define where execution goes next after a node completes.

There are two common kinds:

- fixed edges
- conditional edges

Fixed edges always go to the same next node.

Conditional edges read the current state and choose the next node dynamically.

### Entry Point

The entry point is the first node the runtime executes.

In Atlas, the graph starts at:

- `classify_intent`

### END

`END` is the graph termination marker.

When execution reaches `END`, LangGraph stops and returns the accumulated state.

### Compiled Graph

After the nodes and edges are defined, the graph is compiled into the runnable graph application.

That compiled graph is what Atlas invokes at runtime.

### Checkpointer

LangGraph can use an optional checkpointer to persist graph state across turns or recoverable execution boundaries.

In Atlas, the checkpointer is backed by Redis when available.

#### What LangGraph persistence is used for in Atlas

In Atlas, LangGraph persistence is used for thread-level conversation continuity across requests in the same browser session.

The runtime sets:

```python
config["configurable"] = {"thread_id": session_id}
```

That means the LangGraph thread id is the browser session id.

When the Redis-backed checkpointer is available, LangGraph persists graph thread state for that session.

This mainly supports:

- keeping the same LangGraph thread for a session
- enabling multi-turn conversation continuity across requests in that session

It is not the main place Atlas stores:

- connectivity snapshots
- path hops
- interface counters
- pending clarification context
- long-term semantic memory

Those are owned separately by `SessionStore`, `MemoryManager`, and the memory layer.

### Prebuilt ReAct Agents

LangGraph also provides prebuilt agent patterns such as `create_react_agent(...)`.

Atlas uses those prebuilt ReAct agents below the graph layer for the actual tool loop.

So there are two distinct LangGraph-related layers in Atlas:

- the application graph for routing and flow control
- the prebuilt ReAct agent runtime for tool-based reasoning

## Generic Execution Pattern

A typical LangGraph flow looks like this:

```text
state enters graph
    ▼
entry node runs
    ▼
state is updated
    ▼
edge or conditional edge chooses the next node
    ▼
next node runs
    ▼
graph eventually reaches END
```

That is the general execution model Atlas is using.

## Why Atlas Uses LangGraph

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

Workflow services, agents, tools, and owned backend services handle execution below the graph.

---

## Atlas LangGraph Role

Atlas uses LangGraph for one narrow, important job: lane selection and controlled execution flow.

The application entrypoint does not own:
- intent routing
- early exits for unsupported prompts
- troubleshoot vs network-ops branching
- final response assembly boundaries

LangGraph owns that control flow.

What LangGraph does **not** do in this design:
- workflow services, agents, tools, and backend services own execution below the graph.

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

#### How `classify_intent(...)` engages the LLM

`classify_intent(...)` does not perform ordinary lane selection with static graph metadata alone. For normal requests, it explicitly calls the router LLM through `IntentRoutingService`.

The sequence is:

1. `classify_intent(...)` receives the current `AtlasState`
2. code handles a few fast paths first:
   - bare acknowledgements
   - pending clarification follow-ups
3. if those do not apply, `classify_intent(...)` calls:

```python
llm_decision = await intent_routing_service.route_prompt(prompt)
```

4. `IntentRoutingService` builds the router model through `AgentFactory`:

```python
llm = agent_factory.build_router_llm()
```

5. it sends two messages to the model:

```python
response = await llm.ainvoke(
    [
        SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
)
```

6. `_ROUTER_SYSTEM_PROMPT` instructs the model to return exactly one of:
   - `troubleshoot`
   - `network_ops`
   - `dismiss`
7. the router response is parsed and validated as JSON
8. `classify_intent(...)` writes the validated result into graph state as:

```python
{"intent": "troubleshoot"}
```

or:

```python
{"intent": "network_ops"}
```

or:

```python
{"intent": "dismiss"}
```

9. the conditional edge in `graph_builder.py` reads `state["intent"]` and chooses the next node

So the control chain is:

```text
classify_intent(...)
    -> IntentRoutingService.route_prompt(...)
    -> router LLM
    -> parsed JSON decision
    -> state["intent"]
    -> graph conditional edge
```

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

---

## FAQ

### What is LangGraph persistence used for in Atlas?

LangGraph persistence is used for thread-level conversation state across requests in the same browser session.

In Atlas, the graph runtime sets:

```python
config["configurable"] = {"thread_id": session_id}
```

and the checkpointer uses that `thread_id` to persist graph thread state in Redis when the checkpointer is enabled.

This mainly supports:

- keeping the same LangGraph thread for a session
- enabling multi-turn context continuity

Example:

Turn 1:

```text
help troubleshoot connectivity between 10.0.100.100 and 10.0.200.200 on tcp 443
```

Turn 2 in the same session:

```text
what recent changes happened on those devices?
```

Because both requests use the same:

```text
thread_id = session_id
```

LangGraph treats them as part of the same thread rather than two unrelated runs.

Another example:

Turn 1:

```text
create a change request for arista-ai1 route map update
```

Turn 2:

```text
justification is to restore routing for business x
```

With the same session thread, Atlas can continue the same conversation instead of treating the second message as a brand-new standalone request.

If the Redis checkpointer is unavailable:

- Atlas still runs
- LangGraph thread persistence is disabled
- other state systems such as `SessionStore` and `MemoryManager` still continue to do their own work

### Where in the agent prompt is Atlas telling the model to reason and act?

It happens in two layers.

First, the prompt files tell the model to investigate by using tools.

For troubleshooting, [`skills/troubleshooter.md`](<skills/troubleshooter.md>) says:

```text
You are a network troubleshooting agent. Investigate network problems by calling tools, reasoning about findings, and writing a precise root cause analysis.
```

and:

```text
- Always call tools yourself. Never tell the user to run a command or check something themselves. You have the tools — use them.
- Only report what tools returned.
```

For network operations, [`skills/network_ops.md`](<skills/network_ops.md>) gives tool-directed instructions such as:

```text
For explicit requests to create/open/raise an incident or ticket:
- use `create_servicenow_incident(...)`
```

So the prompt layer tells the model:

- what kind of agent it is
- when it should use tools
- how it should interpret the returned evidence

Second, the actual Reason → Act → Observe loop is provided by LangGraph's prebuilt ReAct runtime in [`agents/agent_factory.py`](<agents/agent_factory.py>):

```python
return create_react_agent(
    llm,
    tools,
    prompt=SystemMessage(content=system_prompt),
    name=agent_name,
)
```

That means:

- the prompt provides the domain instructions
- `create_react_agent(...)` provides the actual ReAct tool-calling runtime

Atlas does not write its own literal prompt format such as:

```text
Thought:
Action:
Observation:
```

Instead, Atlas relies on LangGraph's prebuilt ReAct agent and supplies:

- the system prompt
- the visible tool list
- the user message

The combined effect is:

- the prompt tells the model to use tools and reason over evidence
- the ReAct runtime gives the model the mechanism to call a tool, observe the result, and continue

That work lives in the owners below the graph.
