# LangGraph in Atlas

## Why LangGraph?

Without LangGraph, `chat_service.py` would need a long chain of if/else logic to handle routing — "if intent is network, do this; if risk, do that; if tool fails, retry; if RBAC blocks, return error." This becomes hard to reason about and test. LangGraph lets you define that logic as an explicit graph with named nodes and edges, making the flow visible and each step independently testable.

The specific reason Atlas needs it: queries don't always follow a straight line. After a tool executes, the LLM might decide to call another tool (chaining), or retry on failure, or stop. That back-edge (`tool_executor → tool_selector`) is trivial in a graph, hard to express cleanly with linear code.

---

## The Three Graph Files

### graph_state.py

Defines `AtlasState` — a typed dictionary that is the shared memory passed between every node in the graph. Every node reads from it and writes back to it.

| Field | Purpose |
|-------|---------|
| `prompt`, `conversation_history`, `username`, `session_id` | Set once at entry, never mutated |
| `intent` | Set by `classify_intent`, used by the router to pick the next node |
| `messages` | The growing LangChain message list passed to the LLM |
| `selected_tool_name`, `selected_tool_args` | Written by `tool_selector`, read by `tool_executor` |
| `tool_raw_result`, `accumulated_results` | Written by `tool_executor` as results come in |
| `tool_error`, `iteration` | Used by the retry loop |
| `rbac_error` | Set by `check_rbac` to block execution before a tool runs |
| `final_response` | Once set, the graph routes to `build_final_response` and ends |
| `prefilled_tool_name/params` | Shortcut path when the answer to a follow-up confirmation is already known |

---

### graph_builder.py

Wires all nodes together into a compiled graph. Key routing decisions:

- **`classify_intent` branches** to one of 7 paths: `prefilled`, `doc`, `network`, `dismiss`, `risk`, `netbrain`, `troubleshoot`
- **`tool_selector` → `check_rbac` or done** — if the LLM stopped producing tool calls, the result is already in `accumulated_results`
- **`tool_executor` → `tool_selector`** (back-edge) — after a tool runs, the result is fed back to the LLM which decides whether to call another tool or produce a final answer. This is the tool-chaining loop
- **`tool_executor` → `synthesize_error`** — if all retry attempts are exhausted
- All terminal paths converge at `build_final_response → END`

The compiled graph (`atlas_graph`) is a singleton instantiated once on import and reused for every request.

---

### graph_nodes.py

Implements every node as an async function. Each node receives `AtlasState`, does its work, and returns a dict of fields to update in the state.

| Node | What it does |
|------|-------------|
| `classify_intent` | Calls the LLM with a strict system prompt to classify the query into one of 7 intents. Also handles follow-up confirmations ("yes"/"no") by checking the previous `follow_up_action` in conversation history |
| `tool_selector` | Calls the LLM with all MCP tool descriptions bound and forces it to pick a tool (`tool_choice="required"` on first iteration, `"auto"` on retries). Returns the tool name and args |
| `check_rbac` | Checks whether the user's role permits the selected tool. Sets `rbac_error` if not |
| `tool_executor` | Calls the MCP tool. Also contains deterministic chaining logic — e.g. if `query_panorama_ip_object_group` ran and the user asked for members, it immediately calls `query_panorama_address_group_members` without going back to the LLM |
| `prefilled_tool_executor` | Skips LLM tool selection entirely — runs a known tool directly. Used when the user confirms a follow-up offer ("yes, show me the members") |
| `enrich_with_insights` | After the main result is ready, adds proactive hints — e.g. "this group has no policies referencing it" |
| `troubleshoot_orchestrator` | Checks if the troubleshoot query has enough context (port, issue type). If not, asks a clarifying question and saves the original prompt in memory keyed by session ID. The next message from the same session is combined with it |
| `synthesize_error` | If the tool failed after retries, asks the LLM to produce a friendly error message |
| `netbrain_agent` | Delegates path queries to the NetBrain A2A agent over HTTP — no MCP involved |
| `risk_orchestrator` | Delegates risk queries to the risk orchestrator which fans out to Panorama and Splunk A2A agents in parallel |

---

## Query Flow (network intent)

```
classify_intent
    │ intent = "network"
    ▼
fetch_mcp_tools
    ▼
tool_selector  ←──────────────────────────────┐
    │ tool selected                            │
    ▼                                          │ (tool result fed back — LLM may chain)
check_rbac                                     │
    │ allowed                                  │
    ▼                                          │
tool_executor ─────────────────────────────────┘
    │ success (no more tool calls)
    ▼
enrich_with_insights
    ▼
build_final_response
```

## Query Flow (netbrain intent)

```
classify_intent
    │ intent = "netbrain"
    ▼
netbrain_agent  ──► HTTP POST localhost:8004 (NetBrain agent)
                        └── ask_panorama_agent ──► HTTP POST localhost:8003 (Panorama agent)
    ▼
build_final_response
```
