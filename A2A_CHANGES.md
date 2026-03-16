# A2A Multi-Agent Changes

## New Files

### Agents (`agents/`)
| File | Purpose | Port |
|---|---|---|
| `agents/agent_loop.py` | Shared ReAct-style LLM tool-calling loop used by all agents | — |
| `agents/panorama_agent.py` | Panorama lookups (address groups, members, policies, zones, device groups) | 8003 |
| `agents/splunk_agent.py` | Splunk firewall event analysis (denies, traffic summary, destination spread) | 8002 |
| `agents/netbrain_agent.py` | Network path queries; calls Panorama agent when a PA firewall is found in the path | 8004 |
| `agents/orchestrator.py` | Risk assessment fan-out — calls Panorama + Splunk agents in parallel, synthesizes with Ollama | — |

### Skills (`skills/`)
| File | Used by |
|---|---|
| `skills/panorama_agent.md` | Panorama agent system prompt (Panorama concepts, zones, device groups) |
| `skills/splunk_agent.md` | Splunk agent system prompt (deny events, traffic patterns, risk signals) |
| `skills/netbrain_agent.md` | NetBrain agent system prompt (path hops, firewall enrichment instructions) |
| `skills/risk_synthesis.md` | Ollama synthesis prompt for risk assessments |

---

## Modified Files

### Graph / Routing
| File | Change |
|---|---|
| `graph_state.py` | Added `"risk"` and `"netbrain"` to the intent Literal |
| `graph_nodes.py` | Added risk keyword detection → `risk` intent; two-IP / path keyword detection → `netbrain` intent; added `risk_orchestrator` and `netbrain_agent` nodes |
| `graph_builder.py` | Wired `risk_orchestrator` and `netbrain_agent` nodes into the LangGraph with edges |

### Tools
| File | Change |
|---|---|
| `tools/netbrain_tools.py` | Removed internal `_add_panorama_zones_to_hops` and `_add_panorama_device_groups_to_hops` calls — Panorama enrichment now handled by the NetBrain agent via A2A |
| `tools/splunk_tools.py` | Added `get_splunk_traffic_summary` and `get_splunk_unique_destinations` tools |

### Backend
| File | Change |
|---|---|
| `chat_service.py` | Updated skill file reference: `panorama_lookup.md` → `panorama_agent.md` |
| `skills/base.md` | Stripped down to role statement only — tool selection rules moved to tool docstrings |

### Frontend
| File | Change |
|---|---|
| `frontend/src/components/messages/DirectAnswerBadge.jsx` | Added `ReactMarkdown` + `remark-gfm` for GFM table rendering |
| `frontend/src/components/messages/DirectAnswerBadge.module.css` | Added table, `th`, `td` styles |
| `frontend/package.json` | Added `remark-gfm` dependency |

---

## Deleted Files
| File | Reason |
|---|---|
| `skills/panorama_lookup.md` | Renamed to `skills/panorama_agent.md` for consistency |

---

## How to Start the Agents

```bash
# From the atlas repo root, in separate terminals:
.venv/bin/python -m uvicorn agents.panorama_agent:app --port 8003
.venv/bin/python -m uvicorn agents.splunk_agent:app --port 8002
.venv/bin/python -m uvicorn agents.netbrain_agent:app --port 8004
```

`atlas-web` (the main LangGraph service) must also be restarted after any changes to `graph_*.py` or `chat_service.py`.

---

## Intent Routing

| Query pattern | Intent | Handler |
|---|---|---|
| One IP + risk keywords (suspicious, threat, etc.) | `risk` | `risk_orchestrator` → Panorama + Splunk agents |
| Two IPs, or path keywords (path, route, trace, hops) | `netbrain` | `netbrain_agent` → NetBrain + optionally Panorama agent |
| Documentation questions | `doc` | `doc_tool_caller` |
| Everything else | `network` | LangGraph tool selector (direct MCP tools) |
