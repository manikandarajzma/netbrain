# Atlas — AI-Powered Network Infrastructure Assistant

Atlas is a natural language interface for querying network infrastructure. Ask questions in plain English — Atlas routes the query to the right system, runs the right tools, and returns a structured answer.

---

## What Atlas can answer

- **Panorama** — "What address group is 11.0.0.1 in?", "Show members of leander_web", "Find unused address objects"
- **NetBrain** — "Find path from 10.0.0.1 to 10.0.1.1", "Is traffic from X to Y on TCP 443 allowed?"
- **Splunk** — "Show recent deny events for 10.0.0.250"

---

## Architecture

Atlas uses **LangGraph** to route queries through one of two paths:

- **Direct MCP path (`network` intent)** — the LLM picks a tool, the MCP server executes it against the target API, and the result is returned directly.
- **A2A path (`netbrain` intent)** — the NetBrain agent traces the network path and calls the Panorama agent mid-reasoning to enrich firewall hops with zones and device group.

```
Browser
  │  POST /api/discover   (tool pre-selection — loading indicator only)
  │  POST /api/chat       (full query)
  ▼
FastAPI (app.py)
  │  Session cookie validation + RBAC (auth.py)
  ▼
LangGraph (graph_builder.py, graph_nodes.py)
  │
  ├── network intent ──► MCP Server (port 8765) ──► Panorama / Splunk API
  │
  └── netbrain intent ──► NetBrain agent (port 8004)
                              └── ask_panorama_agent ──► Panorama agent (port 8003)
                                                             └── MCP Server ──► Panorama API
```

**LLM:** Ollama (llama3.1:8b by default) — runs locally, no external AI API calls.

---

## Project Structure

```
atlas/
├── app.py                     # FastAPI web server (port 8000)
├── run_web.py                 # Dev launcher
├── auth.py                    # OIDC authentication + session management
├── chat_service.py            # LLM tool selection, MCP execution, normalization
├── chat_history.py            # Conversation persistence (disk-based, per user)
├── graph_builder.py           # LangGraph graph definition
├── graph_nodes.py             # LangGraph node functions (intent classifier, tool executor, etc.)
├── graph_state.py             # LangGraph state schema
├── mcp_client.py              # MCP client (connects to MCP server)
├── mcp_server.py              # MCP server (port 8765) — exposes tools
├── panoramaauth.py            # Panorama API key retrieval (Azure Key Vault)
├── netbrainauth.py            # NetBrain authentication
├── splunkauth.py              # Splunk authentication
├── kv_helper.py               # Azure Key Vault helper
├── status_bus.py              # SSE status updates
│
├── agents/                    # A2A agents (each is a standalone FastAPI service)
│   ├── agent_loop.py          # Shared tool-calling loop used by all agents
│   ├── panorama_agent.py      # Panorama agent (port 8003)
│   ├── netbrain_agent.py      # NetBrain agent (port 8004)
│   ├── splunk_agent.py        # Splunk agent (port 8002)
│   └── orchestrator.py        # Risk orchestrator (fans out to Panorama + Splunk)
│
├── tools/                     # MCP tool implementations
│   ├── panorama_tools.py      # Panorama XML API tools
│   ├── netbrain_tools.py      # NetBrain API tools
│   ├── splunk_tools.py        # Splunk API tools
│   ├── docs_tool.py           # Documentation search tool
│   └── shared.py              # Shared config (Ollama URL, model, etc.)
│
├── skills/                    # LLM system prompts (domain knowledge per agent)
│   ├── base.md                # Loaded for all queries
│   ├── panorama_agent.md      # Panorama domain knowledge
│   ├── netbrain_agent.md      # NetBrain path query knowledge
│   ├── splunk_agent.md        # Splunk domain knowledge
│   └── risk_synthesis.md      # Risk assessment output format
│
└── frontend/                  # React 18 SPA (Vite)
    ├── src/
    │   ├── stores/            # State management (user, chat)
    │   ├── components/        # UI components (layout, chat, messages, tables)
    │   └── utils/             # API client, formatters, response classifier
    └── dist/                  # Production build (served by FastAPI)
```

---

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| Python 3.13+ | Runtime |
| Ollama | Local LLM (llama3.1:8b by default) |
| Node.js | Frontend build |
| Nginx | Reverse proxy / TLS termination |
| Azure Key Vault | Secrets (Panorama, NetBrain credentials) |
| Azure Entra ID | OIDC authentication |

---

## Installation

### 1. Install Ollama and pull the model

```bash
ollama serve
ollama pull llama3.1:8b
```

### 2. Install Python dependencies

```bash
cd atlas
uv sync
```

### 3. Build the frontend

```bash
cd atlas/frontend
npm install
npm run build
```

---

## Configuration

All configuration is done via a `.env` file in the project root.

```env
# Authentication
AUTH_MODE=oidc
AZURE_CLIENT_ID=...
AZURE_CLIENT_SECRET=...
AZURE_TENANT_ID=...
SESSION_SECRET=...

# Azure Key Vault (used to retrieve Panorama and NetBrain credentials)
AZURE_KEYVAULT_URL=https://your-vault.vault.azure.net/

# Panorama
PANORAMA_URL=https://192.168.15.247
PANORAMA_VERIFY_SSL=false

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
```

See [Documentation/auth-rbac.md](Documentation/auth-rbac.md) for full auth setup.

---

## Running the Application

Services are managed by **systemd** and start automatically on boot.

```bash
# Start
sudo systemctl start atlas-web
sudo systemctl start atlas-mcp

# Restart after code changes
sudo systemctl restart atlas-web
sudo systemctl restart atlas-mcp

# Logs
tail -f /var/log/atlas/atlas_web.log
tail -f /var/log/atlas/mcp_server.log
```

### Running manually (development)

```bash
# Terminal 1 — MCP server
cd /opt/atlas && .venv/bin/python mcp_server.py

# Terminal 2 — Web server
cd /opt/atlas && .venv/bin/python run_web.py
```

---

## Available Tools

| Tool | System | Description |
|------|--------|-------------|
| `query_panorama_ip_object_group` | Panorama | Find which address groups contain an IP |
| `query_panorama_address_group_members` | Panorama | List members of an address group |
| `find_unused_panorama_objects` | Panorama | Find orphaned/unused address objects |
| `query_network_path` | NetBrain | Trace hop-by-hop path between two IPs |
| `check_path_allowed` | NetBrain | Check if traffic between two IPs is allowed |
| `get_splunk_recent_denies` | Splunk | Recent firewall deny events for an IP |

---

## Troubleshooting

See [Documentation/General/troubleshooting/troubleshooting.md](Documentation/General/troubleshooting/troubleshooting.md).

---

## License

[Add your license here]
