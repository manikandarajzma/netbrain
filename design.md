# Core Design Principles — Atlas Network Operations AI

**Last Updated:** April 10, 2026
**Version:** 1.0

This document defines the fundamental design principles of Atlas. All future development (new tools, agents, features, or changes) must adhere to these principles.

---

## 1. Overall Architecture

Atlas follows a **Hierarchical Specialized ReAct Agent** pattern:

- **LangGraph** acts as the **Supervisor / High-level Router**
- **Specialized ReAct Agents** perform the actual reasoning and work
- **Centralized Tools** (`tools/all_tools.py`) are the only way agents interact with external systems

```
User Prompt
    │
    ▼
classify_intent  (graph_nodes.py)
    │
    ├──► call_troubleshoot_agent  ──► troubleshoot_agent.build_agent()
    │                                      │
    ├──► call_network_ops_agent   ──► network_ops_agent.build_agent()
    │                                      │
    └──► build_final_response              ▼
                                   create_react_agent(llm, tools)
                                      Reason → Act → Observe
                                          │
                                          ▼
                                    tools/all_tools.py
                                (MCP / HTTP / DB backends)
```

---

## 2. Core Principles

### 2.1 Tools Are the Single Source of Truth

- All external capabilities must be implemented as tools in `tools/all_tools.py`
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
- All shared capabilities are accessed exclusively through tools in `all_tools.py`

> **Why?** A2A is fragile, complex, and breaks the clean ReAct pattern. Agents calling each other via HTTP leads to unstructured text passing, error-prone coordination, tight coupling, and debugging nightmares. Shared tools keep the architecture simple, reliable, and maintainable — each specialized agent stays focused on its own strength using the ReAct loop.

### 2.4 Pure ReAct Agents

- Both agents use `create_react_agent` from `langgraph.prebuilt`
- The LLM fully drives the Reason → Act → Observe loop
- Agent files (`troubleshoot_agent.py`, `network_ops_agent.py`) expose only `build_agent()` — no orchestration logic
- Infrastructure concerns (status bus, session handling, INC→IP resolution, response formatting) belong in `graph_nodes.py`, not in agent files

### 2.5 Routing Belongs in LangGraph

- Intent classification and routing are the exclusive responsibility of `graph_nodes.py` and `graph_builder.py`
- Agents do not perform routing or decide which agent should handle a query
- The classifier (`classify_intent`) routes to `troubleshoot_agent` or `network_ops_agent` based on prompt intent

### 2.6 Prompt Discipline

- Each agent has one focused system prompt: `skills/troubleshooter.md` and `skills/network_ops.md`
- `troubleshooter.md` contains only: role definition, core principles, and the layered diagnosis framework
- Scenario-specific sequences and report formats live in `skills/troubleshooting_scenarios/` to prevent prompt bloat
- No scenario-specific rules belong in the general `troubleshooter.md`

---

## 3. Development Rules

### Adding a New Tool
1. Implement it in `tools/all_tools.py`
2. Use MCP or the existing `_nornir_post()` / CB+retry pattern for external calls
3. Add it to `ALL_TOOLS` (and `NETWORK_OPS_TOOLS` if the network-ops agent needs it)
4. Write a clear docstring — the LLM reads it to decide when to call the tool

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
| Infrastructure logic (status, formatting) in agent files | Belongs in `graph_nodes.py` |

---

## 4. Memory Architecture

Atlas uses two distinct memory layers backed by Redis.

### Short-Term Memory (Conversation / Thread)

Implemented via **LangGraph's `AsyncRedisSaver`** checkpointer, wired in `chat_service.py`.

- Stores the full message history (user messages, tool calls, tool results, agent replies) for a single session
- Keyed by `thread_id = session_id` — each browser session gets its own slot
- Enables multi-turn conversations: the agent remembers what was said and which tools were called earlier in the same chat
- Very fast (sub-millisecond Redis reads/writes)
- No TTL — persists until the session expires

### Long-Term Memory (Cross-Session / Semantic)

Implemented via **RedisVL vector store** in `agent_memory.py`.

| Mechanism | What it stores | How it's written | How it's read |
|-----------|---------------|-----------------|---------------|
| `store_memory()` | Past troubleshooting findings (query + root cause) | Called automatically in `graph_nodes.py` after each successful troubleshoot run | `recall_similar_cases` tool |
| `store_incident_memory()` | Closed ServiceNow incidents | Nightly sync via `servicenow_memory_sync.py` | `recall_similar_cases` tool |

**Agent usage:** The `recall_similar_cases` tool is in `ALL_TOOLS`. The troubleshoot agent calls it early in its investigation — semantically similar past cases surface root causes before live tools run.

**Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU-only)
**TTL:** 30 days (configurable via `AGENT_MEMORY_TTL_DAYS`)

## 5. File Structure

```
atlas/
├── graph_nodes.py          # LangGraph supervisor: routing, status bus, response formatting
├── graph_builder.py        # LangGraph graph definition
├── agents/
│   ├── troubleshoot_agent.py   # build_agent() only — pure create_react_agent
│   └── network_ops_agent.py    # build_agent() only — pure create_react_agent
├── tools/
│   └── all_tools.py            # ALL tools — the only way agents touch external systems
├── skills/
│   ├── troubleshooter.md       # Core principles + diagnosis framework (no rules)
│   ├── network_ops.md          # Network ops agent prompt
│   └── troubleshooting_scenarios/
│       ├── connectivity.md     # Sequence + root cause patterns + report format
│       ├── performance.md
│       └── intermittent.md
├── mcp_server.py           # FastMCP server — ServiceNow tools over MCP
├── agent_memory.py         # Long-term memory: RedisVL vector store (store + recall)
└── servicenow_memory_sync.py  # Nightly job: closed SNOW incidents → long-term memory
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
