# Panorama Query — End-to-End Flow

This document traces the complete lifecycle of a Panorama query through Atlas, from user input to rendered response. It covers both query paths: direct MCP tool calls and the A2A multi-agent path.

---

## Architecture Overview

Atlas uses **LangGraph** to route queries through a graph of nodes. The path a Panorama query takes depends on intent classification:

```
Browser (React + Zustand)
    │  POST /api/chat
    ▼
FastAPI (app.py)
    │  Session + RBAC validation
    ▼
chat_service.py → atlas_graph (LangGraph)
    │
    ├── intent: "network"  ──────────────────────────────────────────────┐
    │   (e.g. "what group is 11.0.0.1 in?")                             │
    │   fetch_mcp_tools → tool_selector → check_rbac → tool_executor    │
    │       └── MCP Server → panorama_tools.py → Panorama API           │
    │                                                                    ▼
    └── intent: "risk"                                        build_final_response
        (e.g. "is 11.0.0.1 suspicious?")                         │
        risk_orchestrator                                         ▼
            ├── Panorama A2A agent (port 8003)              FastAPI → JSON → React
            │       └── agent_loop.py (ReAct)
            │           └── panorama_tools via MCP
            └── Splunk A2A agent (port 8002)
                    └── agent_loop.py (ReAct)
                        └── splunk_tools via MCP
            │
            └── Ollama synthesis (risk_synthesis.md skill)
```

---

## Intent Classification

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `classify_intent()`

Every query enters the LangGraph at `classify_intent`. The node inspects the prompt and returns one of several intents:

| Condition | Intent | Route |
|---|---|---|
| One IP + risk keywords (suspicious, threat, etc.) | `risk` | `risk_orchestrator` |
| Two IPs, or path keywords (path, route, trace) | `netbrain` | `netbrain_agent` |
| Documentation question | `doc` | `doc_tool_caller` |
| Everything else | `network` | `fetch_mcp_tools` → tool selector |

Panorama queries like "what group is 11.0.0.1 in?" or "show unused address objects" fall through to `network`.

---

## Path 1: Direct MCP Query (network intent)

Used for: group lookups, member listings, unused object queries.

### 1. Skills loaded into system prompt

**File:** [`chat_service.py`](../../chat_service.py), [`skills/`](../../skills/)

Before the LLM is called, skill files are loaded and concatenated into the system prompt:

- `skills/base.md` — role statement ("You are a network infrastructure assistant…")
- `skills/panorama_agent.md` — Panorama domain concepts (address objects, groups, device groups, zones)

### 2. Tool selection (LLM)

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `tool_selector()`

An Ollama LLM receives the system prompt + conversation history + available MCP tool schemas. It selects the appropriate tool and extracts arguments:

- "What group is 11.0.0.1 in?" → `query_panorama_ip_object_group(ip_address="11.0.0.1")`
- "What IPs are in group leander_web?" → `query_panorama_address_group_members(address_group_name="leander_web")`
- "Show unused address objects" → `find_unused_panorama_objects()`

Tool schemas (name, description, parameters) come directly from the `@mcp.tool()` docstrings in [`tools/panorama_tools.py`](../../tools/panorama_tools.py).

### 3. RBAC check

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `check_rbac()`

Verifies the user's allowed categories include `panorama`. If blocked, returns an error response without calling the tool.

### 4. Tool execution

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `tool_executor()`

Calls the selected tool via the MCP client ([`mcp_client.py`](../../mcp_client.py)), which sends the request to the MCP server process ([`mcp_server.py`](../../mcp_server.py)).

**Deterministic chaining:** if `query_panorama_ip_object_group` returns a result and the original prompt asked for members or policies, `tool_executor` automatically chains to `query_panorama_address_group_members` without returning to the LLM. This bypasses LLM unreliability for sequential lookups.

### 5. MCP Server → Panorama API

**Files:** [`mcp_server.py`](../../mcp_server.py), [`tools/panorama_tools.py`](../../tools/panorama_tools.py), [`panoramaauth.py`](../../panoramaauth.py)

The MCP server dispatches to the tool function in `panorama_tools.py`:
- `panoramaauth.py` retrieves and caches the Panorama API key (username + password → keygen endpoint)
- XML API calls are made to the Panorama host over HTTPS (certificate verification disabled)
- Results are parsed from XML into Python dicts

### 6. Response normalization and rendering

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `enrich_with_insights()`

The raw result is normalized and optionally enriched with LLM-generated insights. The frontend renders the result using `AssistantMessage.jsx`, which classifies the response structure and picks the appropriate component (data table, vertical table, markdown, etc.).

---

## Path 2: A2A Risk Assessment (risk intent)

Used for: "is 11.0.0.1 suspicious?", "are there any risks with 10.0.0.1?"

### 1. risk_orchestrator node

**File:** [`graph_nodes.py`](../../graph_nodes.py) — `risk_orchestrator()`, [`agents/orchestrator.py`](../../agents/orchestrator.py)

The orchestrator extracts the IP from the prompt and fans out to two A2A agents **in parallel**:

```python
panorama_task = "Assess the Panorama security posture for IP 11.0.0.1. Find which address group it belongs to, list the group members, and show all referencing security policies."

splunk_task = "Analyze Splunk firewall data for IP 11.0.0.1. Get recent deny events, a traffic summary broken down by action, and destination spread."

panorama_result, splunk_result = await asyncio.gather(
    _call_agent(PANORAMA_AGENT_URL, panorama_task),
    _call_agent(SPLUNK_AGENT_URL, splunk_task),
)
```

### 2. Panorama A2A agent

**File:** [`agents/panorama_agent.py`](../../agents/panorama_agent.py) — port 8003

The Panorama agent is a FastAPI service that receives the natural language task and runs a **ReAct-style LLM loop** ([`agents/agent_loop.py`](../../agents/agent_loop.py)):

```
HumanMessage: "Assess the Panorama security posture for IP 11.0.0.1..."
    ↓
LLM (Ollama) + tools bound
    ↓
tool_call: panorama_ip_object_group(ip_address="11.0.0.1")
    ↓
ToolMessage: {"address_groups": [{"name": "leander_web", "device_group": "leander"}]}
    ↓
tool_call: panorama_address_group_members(address_group_name="leander_web", device_group="leander")
    ↓
ToolMessage: {"members": [...], "policies": [...]}
    ↓
AIMessage: "11.0.0.1 belongs to address group leander_web in device group leander..."
```

**Available tools in the Panorama agent:**

| Tool | Purpose |
|---|---|
| `panorama_ip_object_group` | Find which groups contain an IP |
| `panorama_address_group_members` | Get members and policies for a group |
| `panorama_unused_objects` | Find orphaned/unused objects |
| `panorama_firewall_zones` | Get security zones for firewall interfaces |
| `panorama_firewall_device_group` | Get device group for a firewall |

**System prompt:** `skills/panorama_agent.md` — Panorama domain concepts loaded at agent startup.

The LLM decides which tools to call and in what order based on the task. The agent returns a **natural language summary**.

### 3. Splunk A2A agent

**File:** [`agents/splunk_agent.py`](../../agents/splunk_agent.py) — port 8002

Same ReAct loop pattern. The Splunk agent calls all three Splunk tools:
- `splunk_recent_denies` — firewall deny events for the IP
- `splunk_traffic_summary` — total traffic by action (allow/deny)
- `splunk_unique_destinations` — unique destination IPs and ports

Returns a natural language summary of the IP's traffic behavior.

### 4. Synthesis

**File:** [`agents/orchestrator.py`](../../agents/orchestrator.py), [`skills/risk_synthesis.md`](../../skills/risk_synthesis.md)

The orchestrator passes both agent summaries to Ollama with the `risk_synthesis.md` skill as the system prompt. The synthesis LLM produces a structured risk assessment:

```
**Verdict:** <one sentence>

**Panorama**
- Group: `leander_web` (8 members, device group: `leander`)
- Referencing policies:
| Policy | Action | Source | Destination |
...

**Splunk**
- Deny events (24h): 0
- Total traffic events: 42 (38 allow, 4 deny)
- Destination spread: 3 unique IPs, 2 unique ports

**Recommendation**
No action required.
```

This is returned as `{"direct_answer": synthesis}` and rendered by `DirectAnswerBadge.jsx` using ReactMarkdown with `remark-gfm` for table support.

---

## Skills System

Skills are Markdown files in [`skills/`](../../skills/) loaded as system prompts. Each agent or LLM call has its own skill file:

| File | Loaded by | Purpose |
|---|---|---|
| `skills/base.md` | Main LangGraph (all queries) | Role statement + short-reply context hint |
| `skills/panorama_agent.md` | Panorama A2A agent | Panorama domain knowledge (concepts, terminology, zones) |
| `skills/splunk_agent.md` | Splunk A2A agent | Splunk domain knowledge (deny events, risk signals) |
| `skills/netbrain_agent.md` | NetBrain A2A agent | Path query concepts, Panorama enrichment instructions |
| `skills/risk_synthesis.md` | Risk orchestrator synthesis | Output format + risk signal guidance |

**Design principle:** skills contain only domain knowledge. Tool selection logic lives in tool docstrings (`@mcp.tool()` descriptions). Sequential chaining logic lives in code (`tool_executor` deterministic chaining, `agent_loop.py`).

---

## Panorama A2A Agent vs Direct MCP Tool

| | Direct MCP (network intent) | A2A agent (risk intent) |
|---|---|---|
| Trigger | Group/member lookups, unused objects | Risk assessment queries |
| Tool selection | Ollama LLM picks from all MCP tools | Ollama LLM within agent picks from Panorama-only tools |
| Chaining | Deterministic code in `tool_executor` | LLM-driven ReAct loop |
| Output | Structured JSON → table/visualization | Natural language summary |
| Port | Via MCP server (internal) | HTTP 8003 |

---

## Key Files

| File | Role |
|---|---|
| [`graph_nodes.py`](../../graph_nodes.py) | LangGraph nodes: intent classification, tool selection, execution, risk orchestration |
| [`graph_builder.py`](../../graph_builder.py) | LangGraph graph construction and routing |
| [`graph_state.py`](../../graph_state.py) | State schema shared across all graph nodes |
| [`agents/panorama_agent.py`](../../agents/panorama_agent.py) | Panorama A2A agent (FastAPI, port 8003) |
| [`agents/agent_loop.py`](../../agents/agent_loop.py) | Shared ReAct tool-calling loop |
| [`agents/orchestrator.py`](../../agents/orchestrator.py) | Risk fan-out: parallel A2A calls + Ollama synthesis |
| [`tools/panorama_tools.py`](../../tools/panorama_tools.py) | Panorama MCP tool implementations + Panorama API calls |
| [`skills/panorama_agent.md`](../../skills/panorama_agent.md) | Panorama agent system prompt |
| [`skills/risk_synthesis.md`](../../skills/risk_synthesis.md) | Risk synthesis system prompt |
| [`mcp_client.py`](../../mcp_client.py) | Client that calls the MCP server |
| [`mcp_server.py`](../../mcp_server.py) | MCP server process (FastMCP) |
| [`panoramaauth.py`](../../panoramaauth.py) | Panorama API key management |
