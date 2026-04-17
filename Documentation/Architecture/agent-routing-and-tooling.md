# Agent Routing, Prompts, and Tooling

This document explains how Atlas decides which agent runs, where the prompts come from, how tools are selected, where regex is still used, and what the autonomy boundaries are.

It is the reference document for the questions:

- where does intent routing happen
- where do prompts live
- does the agent use regex to choose tools
- does the agent call MCP tools directly
- what is the actual autonomy level of the agents

For the file-by-file ownership reference, see:

- [`Documentation/Architecture/code-map.md`](<Documentation/Architecture/code-map.md>)

## Scope

This document covers the runtime path for both Atlas agents:

- troubleshooting
- network operations

It describes the live code in:

- [`app.py`](<app.py>)
- [`application/chat_service.py`](<application/chat_service.py>)
- [`application/atlas_application.py`](<application/atlas_application.py>)
- [`services/graph_runtime.py`](<services/graph_runtime.py>)
- [`graph/graph_builder.py`](<graph/graph_builder.py>)
- [`graph/graph_nodes.py`](<graph/graph_nodes.py>)
- [`agents/agent_factory.py`](<agents/agent_factory.py>)
- [`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>)
- [`agents/network_ops_agent.py`](<agents/network_ops_agent.py>)
- [`tools/tool_registry.py`](<tools/tool_registry.py>)
- the `tools/*_agent_tools.py` modules
- the owned services below the tool layer

## 1. Top-Level Request Path

The request path is:

1. The frontend sends the user message to `POST /api/chat`.
2. [`app.py`](<app.py>) authenticates the request, creates the SSE stream, and delegates to [`application/chat_service.py`](<application/chat_service.py>).
3. [`application/chat_service.py`](<application/chat_service.py>) calls [`application/atlas_application.py`](<application/atlas_application.py>) `AtlasApplication.process_query(...)`.
4. `AtlasApplication` delegates to [`services/graph_runtime.py`](<services/graph_runtime.py>) `AtlasRuntime`.
5. `AtlasRuntime` invokes the LangGraph graph compiled in [`graph/graph_builder.py`](<graph/graph_builder.py>).
6. The graph routes to either:
   - `call_troubleshoot_agent(...)`
   - `call_network_ops_agent(...)`
   - or a dismiss path
7. The selected workflow service runs the corresponding pure ReAct agent.
8. The agent chooses tools from the profile it was given.
9. Tool side effects accumulate in [`services/session_store.py`](<services/session_store.py>).
10. [`services/response_presenter.py`](<services/response_presenter.py>) turns final text plus structured side effects into the frontend payload.

## 2. Where Intent Routing Happens

Intent routing happens before the agent runs.

The files involved are:

- [`graph/graph_builder.py`](<graph/graph_builder.py>)
  - defines the graph topology
- [`graph/graph_nodes.py`](<graph/graph_nodes.py>)
  - implements `classify_intent(...)`

The graph shape is intentionally small:

```text
classify_intent
  ├─► call_troubleshoot_agent
  ├─► call_network_ops_agent
  └─► build_final_response
```

`classify_intent(...)` is the graph entrypoint for lane selection.

For normal requests it sets one of:

- `troubleshoot`
- `network_ops`
- `dismiss`

The graph decides which agent path runs.
The agent does not decide which agent it is.

## 3. What `classify_intent(...)` Actually Uses

[`graph/graph_nodes.py`](<graph/graph_nodes.py>) does two different things:

1. small code-owned fast paths for:
   - bare acknowledgements and dismissals
   - pending clarification follow-ups
   - IP/CIDR detection used to decide whether a short follow-up should stay attached to pending context
2. normal requests are passed to [`services/intent_routing_service.py`](<services/intent_routing_service.py>), which asks the router LLM to choose:
   - `troubleshoot`
   - `network_ops`
   - `dismiss`

That means coarse routing is currently:

- LLM-driven for normal requests
- code-owned only for short-circuit acknowledgement and pending-context behavior

## 4. Where the Prompts Live

Atlas prompts are authored in the `skills` directory.

Core prompt files:

- [`skills/troubleshooter.md`](<skills/troubleshooter.md>)
- [`skills/network_ops.md`](<skills/network_ops.md>)

Scenario prompt files:

- [`skills/troubleshooting_scenarios/connectivity.md`](<skills/troubleshooting_scenarios/connectivity.md>)
- [`skills/troubleshooting_scenarios/performance.md`](<skills/troubleshooting_scenarios/performance.md>)
- [`skills/troubleshooting_scenarios/intermittent.md`](<skills/troubleshooting_scenarios/intermittent.md>)

Prompt loading happens in:

- [`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>)
  - `load_system_prompt(...)`
- [`agents/network_ops_agent.py`](<agents/network_ops_agent.py>)
  - `load_system_prompt(...)`
- [`services/troubleshoot_scenario_service.py`](<services/troubleshoot_scenario_service.py>)
  - LLM-based troubleshoot scenario selection
- [`services/network_ops_scenario_service.py`](<services/network_ops_scenario_service.py>)
  - LLM-based network-ops scenario selection

Prompt attachment to the actual ReAct agent happens in:

- [`agents/agent_factory.py`](<agents/agent_factory.py>)

`AgentFactory.create_specialized_agent(...)` builds the agent with:

- an LLM
- a tool list
- a `SystemMessage` prompt

## 5. How the Troubleshoot Prompt Is Built

The troubleshoot prompt is assembled in two stages.

1. [`services/troubleshoot_scenario_service.py`](<services/troubleshoot_scenario_service.py>) asks the selector LLM which scenario applies:
   - `connectivity`
   - `performance`
   - `intermittent`
   - `general`
2. [`agents/troubleshoot_agent.py`](<agents/troubleshoot_agent.py>) loads:
   - the core prompt from [`skills/troubleshooter.md`](<skills/troubleshooter.md>)
   - the selected scenario prompt from [`skills/troubleshooting_scenarios/`](<skills/troubleshooting_scenarios/>)

The agent file no longer performs regex scenario picking itself.

## 6. How the Network Ops Prompt Is Built

The network-ops prompt follows the same pattern.

1. [`services/network_ops_scenario_service.py`](<services/network_ops_scenario_service.py>) asks the selector LLM which scenario applies:
   - `incident_record`
   - `record_lookup`
   - `change_record`
   - `change_update`
   - `access_change`
   - `general`
2. [`agents/network_ops_agent.py`](<agents/network_ops_agent.py>) loads:
   - [`skills/network_ops.md`](<skills/network_ops.md>)
   - the selected scenario prompt from `skills/network_ops_scenarios/` when one exists
3. It asks [`tools/tool_registry.py`](<tools/tool_registry.py>) for the `network_ops` profile and builds the pure ReAct agent

## 7. How Tool Selection Actually Works

Tool selection is done by the LLM inside the ReAct agent.

The tool-selection path is:

1. The workflow chooses which agent profile to use.
2. [`tools/tool_registry.py`](<tools/tool_registry.py>) resolves that profile to a tool tuple.
3. [`agents/agent_factory.py`](<agents/agent_factory.py>) builds the ReAct agent with those tools.
4. The LLM decides which tool to call next from the provided tool set.

So the system decides:

- which lane
- which scenario
- which prompt
- which tool profile

The LLM decides:

- which tool to call
- in what order
- whether to call another tool or answer

## 8. Does the Agent Use Regex to Choose Tools

No.

Regex is used in Atlas, but not for the agent’s ReAct tool choice.

Regex is used in:

- [`graph/graph_nodes.py`](<graph/graph_nodes.py>)
  - short follow-up / acknowledgement handling
- [`services/request_preprocessor.py`](<services/request_preprocessor.py>)
  - extracting IPs, ports, incident references, and related context
- a few validation/presentation helpers

The ReAct tool choice itself comes from:

- the system prompt
- tool descriptions/docstrings
- the current message state
- the LLM’s reasoning

If regex starts deciding which tool must be called next, that part of the system is no longer really agentic. It has become a hardcoded workflow.

## 9. The Agent-Facing Tool Surface

Atlas exposes one uniform tool surface to the agents.

That surface is owned by:

- [`tools/tool_registry.py`](<tools/tool_registry.py>)

The registry has two jobs:

1. register concrete tools and their capabilities
2. map profiles to capabilities

Current default profiles include:

- `troubleshoot.general`
- `troubleshoot.connectivity`
- `network_ops`

The agent does not see:

- raw backend client methods
- raw HTTP endpoints
- raw MCP schemas

The agent sees only Atlas tools.

## 10. What Kinds of Tools Atlas Has

Atlas currently exposes three practical categories of agent-facing tools.

### Workflow tools

These are Atlas-specific task tools.

Examples:

- `trace_path(...)`
- `trace_reverse_path(...)`
- `check_routing(...)`
- `collect_connectivity_snapshot(...)`
- `search_servicenow(...)`

They live in:

- [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)
- [`tools/device_agent_tools.py`](<tools/device_agent_tools.py>)
- [`tools/routing_agent_tools.py`](<tools/routing_agent_tools.py>)
- [`tools/connectivity_agent_tools.py`](<tools/connectivity_agent_tools.py>)
- [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)

### Product-facing ServiceNow tools

These already match user tasks directly.

Examples:

- `get_incident_details(...)`
- `get_change_request_details(...)`
- `create_servicenow_incident(...)`
- `create_servicenow_change_request(...)`
- `update_servicenow_change_request(...)`

They live in:

- [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>)

### Memory and knowledge tools

These are still Atlas tools, but they support reasoning rather than live collection.

They live in:

- [`tools/memory_agent_tools.py`](<tools/memory_agent_tools.py>)
- [`tools/knowledge_agent_tools.py`](<tools/knowledge_agent_tools.py>)

## 11. Does the Agent Call MCP Tools Directly

No.

The agent calls Atlas tools.
Some Atlas tools call MCP under the hood.

That distinction matters.

The layering is:

- ReAct agent chooses an Atlas tool
- Atlas tool delegates to:
  - an owned service
  - a backend client
  - MCP
  - or the local Nornir HTTP service

Examples:

- [`tools/servicenow_agent_tools.py`](<tools/servicenow_agent_tools.py>)
  - thin product-facing adapters
  - call [`integrations/mcp_client.py`](<integrations/mcp_client.py>)
- [`tools/servicenow_workflow_tools.py`](<tools/servicenow_workflow_tools.py>)
  - workflow-level correlation tool
  - delegates to [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>)
- [`tools/path_agent_tools.py`](<tools/path_agent_tools.py>)
  - do not use MCP
  - delegate to owned services and then to [`services/nornir_client.py`](<services/nornir_client.py>)

So:

- the agent does not reason about backend protocols
- it reasons about Atlas task-level tools

## 12. What Each Service Owns Below the Tool Layer

The heavy logic does not live in the tools themselves.

Current owners include:

- [`services/path_trace_service.py`](<services/path_trace_service.py>)
  - forward/reverse path walking and path metadata
- [`services/device_diagnostics_service.py`](<services/device_diagnostics_service.py>)
  - ping, TCP tests, interface detail/counters, interface inventory, routing checks
- [`services/connectivity_snapshot_service.py`](<services/connectivity_snapshot_service.py>)
  - holistic connectivity evidence bundle
- [`services/routing_diagnostics_service.py`](<services/routing_diagnostics_service.py>)
  - OSPF, routing history, syslog, peering inspection
- [`services/servicenow_search_service.py`](<services/servicenow_search_service.py>)
  - Atlas-specific incident/change correlation for troubleshoot flows
- [`services/nornir_client.py`](<services/nornir_client.py>)
  - HTTP transport, retry, and run-scoped caching for the local Nornir service
- [`integrations/mcp_client.py`](<integrations/mcp_client.py>)
  - MCP transport

## 13. What the Workflow Services Do

The workflow services sit above the pure agents.

### [`services/troubleshoot_workflow_service.py`](<services/troubleshoot_workflow_service.py>)

Owns:

- session reset for live troubleshoot state
- pending clarification recovery
- incident prompt expansion flow
- agent invocation
- required evidence follow-up
- memory follow-up
- contradiction correction when needed
- final presenter handoff

### [`services/network_ops_workflow_service.py`](<services/network_ops_workflow_service.py>)

Owns:

- network-ops agent invocation
- follow-up clarification handling
- final response shaping for network-ops payloads

The workflow services are where bounded orchestration lives.
The agents themselves stay relatively pure.

## 14. What the Presenter Does

[`services/response_presenter.py`](<services/response_presenter.py>) owns the final response payload shape.

It is responsible for things like:

- inserting deterministic ServiceNow sections
- grouping interface counters by device
- deciding whether path visuals belong in the payload
- fail-closed messaging when live evidence is unavailable

It is not supposed to be the main source of reasoning.
It is the payload-shaping layer.

## 15. Autonomy Boundaries

Atlas is not fully autonomous.
It is a bounded semi-autonomous system.

The system decides:

- which tools are visible
- which tool profile is exposed
- which workflow boundaries and guardrails apply

The LLM decides:

- which lane to use for normal requests
- which scenario applies inside that lane
- which tool to call next
- how to interpret returned evidence within the prompt
- when to stop and answer

Atlas does not currently allow the agent to:

- create new tools on the fly
- bypass `ToolRegistry`
- route itself to a different agent
- directly talk to backend protocols without Atlas tools

That is why Atlas is best described as:

- bounded
- semi-autonomous
- agentic inside a defined domain

## 16. Practical Summary

If you want the shortest accurate mental model:

- [`graph/graph_nodes.py`](<graph/graph_nodes.py>) handles fast-path acknowledgements and delegates normal lane selection to the router LLM
- the scenario services decide which scenario prompt applies
- `skills/*.md` define the system prompts
- [`tools/tool_registry.py`](<tools/tool_registry.py>) defines what tools each agent can see
- the LLM inside the ReAct agent chooses tools
- those tools call owned services and backend clients
- the presenter shapes the final UI payload

In one sentence:

Atlas uses LLM-driven lane selection and LLM-driven scenario selection above pure ReAct agents with a bounded Atlas tool layer, while hiding backend protocols such as MCP and Nornir behind owned services and agent-facing tools.
