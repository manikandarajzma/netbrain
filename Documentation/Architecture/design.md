# Core Design Principles — Atlas Network Operations AI

**Last Updated:** April 17, 2026
**Version:** 1.1

This document defines the fundamental design principles of Atlas. All future development (new tools, agents, features, or changes) must adhere to these principles.

---

## 1. Overall Architecture

Atlas follows a **Hierarchical Specialized ReAct Agent** pattern:

- **LangGraph** acts as the **Supervisor / High-level Router**
- **Specialized ReAct Agents** perform the actual reasoning and work
- **Uniform agent-facing tools** are the only way agents interact with external systems

```
User Prompt
    │
    ▼
classify_intent  (graph_nodes.py)
    │
    ├──► call_troubleshoot_agent  ──► TroubleshootWorkflowService
    │                                      │
    ├──► call_network_ops_agent   ──► NetworkOpsWorkflowService
    │                                      │
    └──► build_final_response              ▼
                                   create_react_agent(llm, tools)
                                      Reason → Act → Observe
                                          │
                                          ▼
                            ToolRegistry → agent-facing tools
                         (workflow tools + product-facing adapters)
                                          │
                                          ▼
                              MCP / HTTP / DB backends
```

---

## 2. Core Principles

### 2.1 Tools Are the Single Source of Truth

- All external capabilities must be implemented as agent-facing tools
- Workflow/composed Atlas tools are split across:
  - `tools/path_agent_tools.py`
  - `tools/device_agent_tools.py`
  - `tools/routing_agent_tools.py`
  - `tools/connectivity_agent_tools.py`
  - `tools/servicenow_workflow_tools.py`
- `tools/all_tools.py` serves as a thin compatibility export layer
- Live path tracing and path metadata extraction live in `services/path_trace_service.py`
- Active tests and interface diagnostics live in `services/device_diagnostics_service.py`
- Transient per-session tool side effects live in `services/session_store.py`
- OSPF, routing-history, and syslog diagnostic workflow lives in `services/routing_diagnostics_service.py`
- Atlas-specific ServiceNow correlation for troubleshoot flows lives in `services/servicenow_search_service.py`
- Product-facing ServiceNow adapters live in `tools/servicenow_agent_tools.py`
- `tools/tool_registry.py` is the agent-facing source of truth for tool exposure
- Tools use MCP (Model Context Protocol) or direct HTTP to communicate with backends (Nornir, ServiceNow, and other external systems)
- Agent files must never contain direct API calls, HTTP logic, XML parsing, or backend-specific code

### 2.2 Exactly Two Specialized Agents

| Agent | Responsibility | Examples |
|-------|---------------|---------|
| `troubleshoot_agent` | Pure diagnosis | Connectivity failures, packet loss, OSPF issues, root cause analysis, layered troubleshooting |
| `network_ops_agent` | Constructive output | Firewall access request forms, change planning, policy review, spreadsheet generation |

No agent may perform both diagnostic and constructive work. If a task is ambiguous, `classify_intent` in `graph_nodes.py` decides.

### 2.3 No Agent-to-Agent Communication

- No A2A (agent-to-agent) HTTP calls
- Agents must never call each other directly
- All shared capabilities are accessed exclusively through the tool layer

> **Why?** A2A is fragile, complex, and breaks the clean ReAct pattern. Agents calling each other via HTTP leads to unstructured text passing, error-prone coordination, tight coupling, and debugging nightmares. Shared tools keep the architecture simple, reliable, and maintainable — each specialized agent stays focused on its own strength using the ReAct loop.

### 2.4 Pure ReAct Agents

- Both agents use `create_react_agent` from `langgraph.prebuilt`
- The LLM fully drives the Reason → Act → Observe loop
- Agent files (`troubleshoot_agent.py`, `network_ops_agent.py`) expose only `build_agent()` — no orchestration logic
- Infrastructure concerns (status bus, session handling, INC→IP resolution, response formatting) belong in workflow/services layers, not in agent files

### 2.5 Routing Belongs in LangGraph

- Intent classification and routing are the exclusive responsibility of `graph_nodes.py` and `graph_builder.py`
- Agents do not perform routing or decide which agent should handle a query
- `classify_intent` owns graph-entry lane selection:
  - small code-owned fast paths for acknowledgements and pending clarifications
  - LLM-driven lane selection for normal requests
- Lane selection routes to `troubleshoot`, `network_ops`, or `dismiss`

### 2.6 Prompt Discipline

- Each agent has one focused system prompt: `skills/troubleshooter.md` and `skills/network_ops.md`
- `troubleshooter.md` contains only: role definition, core principles, and the layered diagnosis framework
- Scenario-specific sequences and report formats live in `skills/troubleshooting_scenarios/` to prevent prompt bloat
- Network-ops scenario-specific addenda live in `skills/network_ops_scenarios/`
- No scenario-specific rules belong in the general `troubleshooter.md`

### 2.7 Responsibility Boundary: What the LLM Should Do vs What Code Should Do

Atlas is designed as a bounded agent system, not as a code-only workflow engine and not as an unconstrained autonomous agent. The boundary must stay explicit.

#### What the LLM should do

- Understand the user’s intent for normal requests
- Choose the best semantic lane when the request is genuinely about:
  - troubleshooting
  - network operations
  - dismissal
- Choose the most appropriate scenario inside the selected lane
- Decide which visible tools to call, and in what order, inside the allowed tool set
- Interpret tool results, combine evidence, and update the working hypothesis
- Ask for clarification when the request is underspecified
- Write the final natural-language answer

#### What code should do

- Own authentication, authorization, session state, and request lifecycle
- Own the graph boundary and the workflow boundary
- Decide which tools are exposed to the LLM
- Own all backend communication, transport logic, retries, caching, and protocol details
- Enforce safety-critical evidence requirements when the workflow cannot safely finalize without them
- Store structured side effects and runtime state outside the LLM
- Shape the final structured UI payload
- Fail closed when required evidence or backend availability is missing

#### What the LLM should not do

- Reach backends directly
- Discover hidden tools or call tools outside the profile it was given
- Own authorization or trust decisions
- Be the only enforcement layer for safety-critical evidence
- Be trusted as the sole source of structured application state

#### What code should not do

- Micromanage every reasoning step when the LLM can make the decision safely
- Recreate a hardcoded workflow for every investigation branch
- Embed backend-specific logic inside agent files
- Replace tool selection with regex or deterministic branching when semantic agent choice is the intended behavior
- Force the LLM to act as a templating engine only

#### Principle

The LLM should own:

- semantic judgment
- tool choice inside bounded visibility
- evidence interpretation
- answer writing

Code should own:

- system boundaries
- backend execution
- safety enforcement
- state management
- deterministic payload shaping

---

## 3. Development Rules

### Adding a New Tool
1. Decide the tool type:
   - path tracing workflow entrypoint → implement in `tools/path_agent_tools.py`
   - active test / interface diagnostic workflow entrypoint → implement in `tools/device_agent_tools.py`
   - routing / OSPF / syslog workflow entrypoint → implement in `tools/routing_agent_tools.py`
   - connectivity snapshot workflow entrypoint → implement in `tools/connectivity_agent_tools.py`
   - troubleshoot-side ServiceNow correlation workflow entrypoint → implement in `tools/servicenow_workflow_tools.py`
   - live path tracing / path metadata owner → implement in `services/path_trace_service.py`
   - active test / interface diagnostic workflow owner → implement in `services/device_diagnostics_service.py`
   - routing / OSPF / syslog diagnostic workflow owner → implement in `services/routing_diagnostics_service.py`
   - Atlas-specific ServiceNow incident/change correlation → implement in `services/servicenow_search_service.py`
   - product-facing ServiceNow CRUD/detail action → implement in `tools/servicenow_agent_tools.py`
2. Use MCP or the owned backend clients (for example `services/nornir_client.py`) for external calls
3. Register capabilities in `tools/tool_registry.py`
4. Add or update the relevant agent profile in `tools/tool_registry.py`
5. Write a clear docstring — the LLM reads it to decide when to call the tool

### Adding a New Workflow
1. Determine the nature: diagnostic → `troubleshoot_agent`, constructive → `network_ops_agent`
2. Extend the appropriate agent's system prompt or add a scenario file
3. Prefer reusing existing tools over adding new ones

### Adding a New Agent
1. It must be highly specialized with a narrow, well-defined responsibility
2. It must use `create_react_agent` from `langgraph.prebuilt`
3. The agent file exposes only `build_agent()` — no wrapper logic
4. Add routing logic to `graph_nodes.py` and `graph_builder.py`

### Forbidden Patterns

| Pattern | Why |
|---------|-----|
| A2A HTTP calls between agents | Creates tight coupling; use shared tools instead |
| Backend/API logic inside agent files | Tools are the single source of truth |
| Mixing diagnostic and constructive logic in one agent | Violates specialization principle |
| Custom agent loops outside `create_react_agent` | LangGraph prebuilt is the standard |
| Scenario-specific rules in `troubleshooter.md` | Belongs in `troubleshooting_scenarios/` |
| Infrastructure logic (status, formatting) in agent files | Belongs in workflow/services layers |

---

## 4. Memory Architecture

Atlas uses two distinct memory layers backed by Redis.

### Short-Term Memory (Conversation / Thread)

Implemented via **LangGraph's `AsyncRedisSaver`** checkpointer, wired in `application/chat_service.py`.

- Stores the full message history (user messages, tool calls, tool results, agent replies) for a single session
- Keyed by `thread_id = session_id` — each browser session gets its own slot
- Enables multi-turn conversations: the agent remembers what was said and which tools were called earlier in the same chat
- Very fast (sub-millisecond Redis reads/writes)
- No TTL — persists until the session expires

### Long-Term Memory (Cross-Session / Semantic)

Implemented via **RedisVL vector store** in `memory/agent_memory.py`.

| Mechanism | What it stores | How it's written | How it's read |
|-----------|---------------|-----------------|---------------|
| `store_memory()` | Past troubleshooting findings (query + root cause) | Called automatically in `TroubleshootWorkflowService` after each successful troubleshoot run | `recall_similar_cases` tool |
| `store_incident_memory()` | Closed ServiceNow incidents | Nightly sync via `memory/servicenow_memory_sync.py` | `recall_similar_cases` tool |

**Agent usage:** `recall_similar_cases` is available through the `troubleshoot.general` profile. It is evidence-gated and should only be used after live results suggest recurrence or unresolved historical patterns.

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-only)
**TTL:** 30 days (configurable via `AGENT_MEMORY_TTL_DAYS`)

## 5. File Structure

```
atlas/
├── application/
│   ├── atlas_application.py     # Top-level owner
│   ├── chat_service.py          # Thin request entrypoint
│   └── status_bus.py            # SSE status bus
├── graph/
│   ├── graph_nodes.py           # LangGraph supervisor: routing and thin node delegation
│   ├── graph_builder.py         # LangGraph graph definition
│   └── graph_state.py           # Typed graph state
├── agents/
│   ├── troubleshoot_agent.py   # build_agent() only — pure create_react_agent
│   └── network_ops_agent.py    # build_agent() only — pure create_react_agent
├── memory/
│   ├── agent_memory.py         # Long-term memory: RedisVL vector store (store + recall)
│   └── servicenow_memory_sync.py  # Nightly job: closed SNOW incidents → long-term memory
├── persistence/
│   ├── db.py                   # PostgreSQL access helpers
│   └── chat_history.py         # Stored conversation history helpers
├── integrations/
│   ├── mcp_client.py           # MCP client transport and tool calls
│   ├── kv_helper.py            # Key Vault integration helpers
│   └── servicenowauth.py       # ServiceNow auth/session helpers
├── security/
│   └── auth.py                 # Auth and RBAC helpers
├── services/
│   ├── intent_routing_service.py # LLM-backed lane selector
│   ├── troubleshoot_scenario_service.py # LLM-backed troubleshoot scenario selector
│   ├── network_ops_scenario_service.py # LLM-backed network-ops scenario selector
│   ├── path_trace_service.py   # Live forward/reverse path tracing and path metadata extraction
│   ├── device_diagnostics_service.py # Ping, TCP, routing-check, and interface diagnostic workflow owner
│   ├── connectivity_snapshot_service.py  # Connectivity evidence bundle owner
│   ├── routing_diagnostics_service.py # Routing history, OSPF, and syslog diagnostic workflow owner
│   ├── servicenow_search_service.py # Atlas-specific ServiceNow correlation owner
│   ├── troubleshoot_workflow_service.py # Troubleshoot orchestration owner
│   ├── network_ops_workflow_service.py # Network-ops orchestration owner
│   ├── workflow_state_service.py # Session-side-effect merge and workflow guard owner
│   └── status_service.py # Status bus owner
├── tools/
│   ├── all_tools.py                 # Thin compatibility export layer for workflow tools
│   ├── path_agent_tools.py          # Path tracing workflow entrypoints
│   ├── device_agent_tools.py        # Active test and interface diagnostic workflow entrypoints
│   ├── routing_agent_tools.py       # Routing / OSPF / syslog workflow entrypoints
│   ├── connectivity_agent_tools.py  # Connectivity snapshot workflow entrypoints
│   ├── servicenow_workflow_tools.py # Troubleshoot-side ServiceNow correlation workflow entrypoints
│   ├── servicenow_agent_tools.py    # Thin product-facing ServiceNow adapters
│   └── tool_registry.py             # Capability registry and agent profiles
├── skills/
│   ├── troubleshooter.md       # Core principles + diagnosis framework (no rules)
│   ├── network_ops.md          # Network ops agent prompt
│   ├── network_ops_scenarios/
│   │   ├── incident_record.md
│   │   ├── record_lookup.md
│   │   ├── change_record.md
│   │   ├── change_update.md
│   │   └── access_change.md
│   └── troubleshooting_scenarios/
│       ├── connectivity.md     # Sequence + root cause patterns + report format
│       ├── performance.md
│       └── intermittent.md
└── mcp_server.py           # FastMCP server — ServiceNow tools over MCP
```

---

## 5. Goal

A **clean, maintainable, and truly agentic** network operations system where:

- The right agent handles the right type of work
- The LLM intelligently drives tool usage through the ReAct loop
- The architecture remains simple and scalable as new tools and capabilities are added
- Infrastructure concerns are cleanly separated from agent reasoning

---

*All contributors must follow this document when making changes to Atlas.*
