# Atlas - AI-Powered Network Infrastructure Assistant

**Atlas** is an intelligent network infrastructure assistant built on the Model Context Protocol (MCP), providing natural language access to path queries (via NetBrain API), Panorama address groups, and Splunk firewall deny events.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Available Tools](#available-tools)
- [Usage Examples](#usage-examples)

---

## Architecture

The system uses a **client-server architecture** with AI-powered tool selection:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Browser                             │
│              (React 18 SPA + Zustand + Vite)                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (fetch)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                           │
│                   (app.py)                              │
│  • Authentication (auth.py, OIDC or local)                      │
│  • Chat API (/api/chat, /api/discover, /api/me)                 │
│  • Serves React build (frontend/dist/) in production            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Chat Service Layer                            │
│                  (chat_service.py)                              │
│  • Query parsing & routing                                      │
│  • Tool discovery & execution                                   │
│  • Conversation history management                              │
│  • Result normalization                                         │
└─────────────┬───────────────────────────────────┬───────────────┘
              │                                   │
              ▼                                   ▼
┌──────────────────────────┐    ┌──────────────────────────────┐
│  LLM Tool Selection      │    │    MCP Client                │
│  (mcp_client_tool_       │    │    (mcp_client.py)           │
│   selection.py)          │    │  • Session management        │
│  • Ollama/Qwen2.5:14b    │    │  • Tool execution            │
│  • Intent classification │    │  • Streaming responses       │
│  • Parameter extraction  │    │                              │
└──────────────────────────┘    └──────────────┬───────────────┘
                                               │ MCP Protocol
                                               ▼
                             ┌─────────────────────────────────────┐
                             │       MCP Server                    │
                             │      (mcp_server.py)                │
                             │  Exposes tools via MCP:             │
                             │  • Path queries (NetBrain API)      │
                             │  • Panorama address groups          │
                             │  • Splunk deny events               │
                             └─────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────┬───────────────────────┐
                    ▼                      ▼                       ▼
            ┌───────────────┐     ┌──────────────────┐     ┌────────────┐
            │  NetBrain API │     │ Panorama XML API │     │ Splunk API │
            │  (netbrainauth│     │  (panoramaauth   │     │            │
            │   .py)        │     │   .py)           │     │            │
            └───────────────┘     └──────────────────┘     └────────────┘
```

### Key Components

1. **FastAPI Web Server** (`app.py`)
   - Serves the React build from `frontend/dist/` in production
   - Handles authentication (local or Microsoft OIDC)
   - Provides `/api/chat`, `/api/discover`, `/api/me` endpoints
   - RBAC: role-based access to tool categories
   - Auto-reloads on code changes (development mode)

7. **React Frontend** (`frontend/`)
   - React 18 SPA with Zustand state management
   - Vite dev server with API proxying for development
   - Component-based architecture with CSS Modules
   - See [frontend/frontend.md](frontend/frontend.md) for full details

2. **Chat Service** (`chat_service.py`)
   - Orchestrates tool discovery and execution
   - Manages conversation context
   - Implements agentic loop (retry on failure)
   - Normalizes and formats results

3. **LLM Tool Selection** (`mcp_client_tool_selection.py`)
   - Uses Ollama (Qwen2.5:14b) for intelligent tool selection
   - Extracts parameters from natural language queries
   - Handles clarification questions
   - Structured output via Pydantic schemas

4. **MCP Client** (`mcp_client.py`)
   - Connects to MCP server via HTTP transport
   - Executes selected tools
   - Handles streaming responses
   - Session management

5. **MCP Server** (`mcp_server.py`)
   - Exposes network infrastructure tools via MCP protocol
   - Integrates with NetBrain API, Panorama, and Splunk APIs
   - Runs independently on port 8765
   - Must be started before the web client

6. **Authentication Modules**
   - `netbrainauth.py`: NetBrain API credentials
   - `panoramaauth.py`: Panorama firewall credentials
   - `auth.py`: Local web UI authentication

---

## Project Structure

```
atlas/
└── (project root)                     # Main application directory
    ├── app.py                         # FastAPI web server (port 8000)
    ├── run_web.py                     # Launcher: run with `uv run python run_web.py` (reload + file logging built in)
    ├── auth.py                        # Authentication (local + OIDC)
    ├── chat_service.py                # Tool orchestration & execution
    ├── mcp_client.py                  # MCP client library
    ├── mcp_client_tool_selection.py   # LLM-based tool selection
    ├── mcp_server.py                  # MCP server (port 8765)
    ├── netbrainauth.py                # NetBrain API credentials
    ├── panoramaauth.py                # Panorama API credentials
    ├── pyproject.toml                 # Python dependencies
    ├── uv.lock                        # Dependency lock file
    ├── README.md                      # This file
    │
    ├── frontend/                      # React 18 SPA (Vite + Zustand)
    │   ├── package.json               # Node dependencies
    │   ├── vite.config.js             # Vite config + API proxy
    │   ├── index.html                 # Vite entry
    │   ├── frontend.md                # Frontend documentation
    │   ├── dist/                      # Production build (git-ignored)
    │   └── src/
    │       ├── main.jsx               # React entry point
    │       ├── App.jsx                # Root component
    │       ├── stores/                # Zustand stores (user, chat)
    │       ├── hooks/                 # Custom hooks (theme, health)
    │       ├── components/            # React components
    │       │   ├── layout/            # Header, Sidebar, ChatLayout
    │       │   ├── chat/              # Input, Messages, Status
    │       │   ├── messages/          # AssistantMessage, badges
    │       │   ├── path/              # Path visualization
    │       │   ├── tables/            # DataTable, BatchResults
    │       │   └── particles/         # Background particles
    │       └── utils/                 # API, formatters, classifiers
    │
    ├── templates/                     # Jinja2 templates
    │   ├── index.html                 # Fallback UI (if no React build)
    │   └── login.html                 # Login page
    │
    ├── static/                        # Static assets (login page CSS)
    │
    ├── icons/                         # Device type icons (PNG)
    │   ├── paloalto_firewall.png
    │   ├── arista_switch.png
    │   └── ...
    │
    └── test_*.py                      # Test scripts
```

---

## Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.13+ | Runtime environment |
| **Node.js** | 18+ | Frontend build tooling |
| **Ollama** | Latest | LLM for tool selection (runs locally) |
| **uv** or **pip** | Latest | Python package management |
| **NetBrain** | N/A | Network path calculation API (optional) |
| **Panorama** | N/A | Palo Alto address group API (optional) |
| **Splunk** | N/A | Firewall log search API (optional) |

---

## Installation

### 1. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

```bash
# Start Ollama service
ollama serve

# Pull the required model (in a separate terminal)
ollama pull llama3.1:8b
```

### 2. Install Python Dependencies

**Using uv (recommended):**
```bash
cd atlas
uv sync
```

**Using pip:**
```bash
cd atlas
pip install -e .
```

### 3. Build the Frontend

After cloning the repo or whenever you don't have `node_modules` yet, run:

```bash
cd atlas/frontend
npm install
npm run build
```

This installs dependencies (including Vite) and creates `frontend/dist/`, which FastAPI serves automatically.

---

## Configuration

### 1. NetBrain Credentials

Edit `netbrainauth.py` to set your NetBrain username and password:

```python
USERNAME = "your_username"
PASSWORD = "your_password"
```

Optionally, set the NetBrain URL via environment variable:
```bash
# Linux/Mac
export NETBRAIN_URL="http://your-netbrain-server.com"

# Windows PowerShell
$env:NETBRAIN_URL="http://your-netbrain-server.com"
```

### 2. Panorama Configuration (Optional)

Edit `panoramaauth.py` to set your Panorama credentials:

```python
PANORAMA_HOST = "your-panorama-host.com"
API_KEY = "your_api_key"
```

### 3. Splunk Configuration (Optional)

Edit `mcp_server.py` to configure Splunk connection:

```python
SPLUNK_HOST = "your-splunk-host.com"
SPLUNK_PORT = 8089
SPLUNK_USERNAME = "your_username"
SPLUNK_PASSWORD = "your_password"
```

### 4. Web UI Authentication

Sign-in uses **Microsoft Entra ID (OIDC)** only. There are no local passwords. Configure Azure App Registration and set in your environment (typically from **Azure Key Vault** in production):

- `AUTH_MODE=oidc`
- `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`

See [Documentation/auth-rbac.md](Documentation/auth-rbac.md) for setup.

---

## Running the Application

You need **two terminal windows** running simultaneously (three for frontend development):

### Terminal 1: Start the MCP Server

```bash
cd atlas
uv run python mcp_server.py
```

**Output:**
```
INFO: MCP Server starting on http://127.0.0.1:8765
INFO: Server initialized with tools: check_path_allowed, query_network_path, query_panorama_ip_object_group, ...
```

**Important:**
- Leave this terminal running
- The server must be started **before** the web client
- Listens on `http://127.0.0.1:8765`
- Does **not** auto-reload (restart manually if you change `mcp_server.py`)

### Terminal 2: Start the Web Client

From the project root (`atlas/`):

```bash
cd atlas
uv run python run_web.py
```

This starts the server with **reload** and **file-only logging** (no console output). Logs: **MCP server** → `mcp_server.log`; **web app** → `atlas_web.log` (both in project root).

**Why `run_web`?** The app lives in the `atlas` package (`atlas.app`, `atlas.auth`, etc.). When you run from inside the project directory, Python does not see that directory as the `atlas` package, so `uvicorn atlas.app:app` would fail with “No module named 'atlas'”. `run_web.py` is a thin launcher: it adds the parent of the project directory to `sys.path`, then imports `atlas.app`. That way uvicorn can load `run_web:app` from the project root and the `atlas` package resolves correctly.

Alternatively, from the **parent** of the project directory: `uv run python -m atlas.app` (or run uvicorn manually with your own options).

**Output:**
```
INFO: Started server process [12345]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Features:**
- Web UI available at **http://localhost:8000**
- Serves the React build from `frontend/dist/` (run `npm run build` first)
- **Auto-reload enabled**: changes to Python files are picked up automatically
- No need to restart when editing `chat_service.py`, `app.py`, etc.

### Terminal 3 (Optional): Frontend Development

For frontend hot-reload during development:

```bash
cd atlas/frontend
npm run dev
```

- Use the app at **http://localhost:5173** (proxies API calls to port 8000)
- Login at `http://localhost:8000/login` first (session cookie is shared)
- Changes to `.jsx` and `.module.css` files are reflected instantly
- See [frontend/frontend.md](frontend/frontend.md) for full frontend documentation

### Alternative: Custom Host/Port

From project root:

```bash
uv run python run_web.py
```

---

## Available Tools

The MCP server exposes the following tools:

### 1. `check_path_allowed`
**Purpose:** Check if traffic between two IPs is allowed by firewall policies

**Parameters:**
- `source` (required): Source IP address
- `destination` (required): Destination IP address
- `protocol` (optional): TCP or UDP (default: TCP)
- `port` (optional): Port number (default: 0)

**Returns:** `allowed`, `denied`, or `unknown` with policy details

**Example Query:** *"Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 443 allowed?"*

---

### 2. `query_network_path`
**Purpose:** Get hop-by-hop network path between two endpoints

**Parameters:**
- `source` (required): Source IP or hostname
- `destination` (required): Destination IP or hostname
- `protocol` (required): TCP or UDP
- `port` (required): Port number

**Returns:** Path hops with device names, interfaces, and zones

**Example Query:** *"Show me the path from 10.0.0.1 to 10.0.1.1"*

---

### 3. `query_panorama_ip_object_group`
**Purpose:** Find which address group(s) contain an IP or CIDR

**Parameters:**
- `ip_address` (required): IP or CIDR (e.g., "10.0.0.1", "11.0.0.0/24")

**Returns:** Address groups and address objects containing the IP

**Example Query:** *"What address group is 11.0.0.1 part of?"*

---

### 4. `query_panorama_address_group_members`
**Purpose:** List all members of an address group

**Parameters:**
- `address_group_name` (required): Address group name

**Returns:** All IPs, CIDRs, and nested groups in the address group

**Example Query:** *"What IPs are in address group leander_web?"*

---

### 5. `get_splunk_recent_denies`
**Purpose:** Search Splunk for recent firewall deny events

**Parameters:**
- `ip_address` (required): IP to search for (source or destination)
- `limit` (optional): Max number of events (default: 100)
- `earliest_time` (optional): Time window (default: -24h)

**Returns:** Recent deny events with timestamps, source/dest, ports, etc.

**Example Queries:**
- *"List all the denies for 10.0.0.250"*
- *"Show me the latest 10 deny events for 192.168.1.1"*

---

## Usage Examples

### Web UI

1. Open **http://localhost:8000** in your browser
2. Log in with **admin** / **admin** (or your configured credentials)
3. Type natural language queries in the chat box:

#### Path Queries
```
Find path from 10.0.0.1 to 10.0.1.1
Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 80 allowed?
Query path from 192.168.1.10 to 192.168.2.20 using TCP port 443
```

#### Panorama Queries
```
What address group is 11.0.0.0/24 part of?
What IPs are in address group leander_web?
Which group contains 10.0.0.1?
```

#### Splunk Queries
```
List all the denies for 10.0.0.250
Show me the latest 10 deny events for 192.168.1.1
Recent deny events for 192.168.1.1 in the last 24 hours
```

---

## Troubleshooting

### 1. Chat Returns Nothing or "No result"

**Cause:** MCP server is not running

**Solution:** Start the MCP server first in Terminal 1 (see [Running the Application](#running-the-application))

---

### 2. "Tool selection failed" or "LLM did not select a tool"

**Cause:** Ollama is not running or model is not pulled

**Solution:**
```bash
# Start Ollama (if not running)
ollama serve

# Pull the model (if not already pulled)
ollama pull llama3.1:8b
```

---

### 3. Panorama/Splunk Tools Fail

**Cause:** API credentials or URLs are incorrect

**Solution:** Check environment variables and credential files:
- `netbrainauth.py` for NetBrain credentials
- `panoramaauth.py` for Panorama credentials
- `SPLUNK_HOST`, `SPLUNK_USER`, `SPLUNK_PASSWORD` for Splunk credentials

---

### 4. Browser Shows Old Cached UI

**Cause:** Browser is caching old HTML/CSS/JS files

**Solution:** Hard refresh the browser:
- **Windows/Linux:** Ctrl + Shift + R or Ctrl + F5
- **Mac:** Cmd + Shift + R

---

### 5. Import Errors

**Cause:** Dependencies not installed

**Solution:**
```bash
cd atlas
uv sync  # or: pip install -e .
```

---

## Development Notes

- **FastAPI auto-reload:** The web client (`app.py`) automatically reloads when you edit Python files. No need to restart Terminal 2.
- **MCP server does NOT auto-reload:** If you change `mcp_server.py`, you must restart Terminal 1.
- **Frontend development:** Run `npm run dev` in `frontend/` for hot-reload at `:5173`. For production, run `npm run build` and access via `:8000`.
- **Frontend architecture:** React 18 + Zustand + CSS Modules. See [frontend/frontend.md](frontend/frontend.md).
- **Logs:** Check `mcp_server.log` for detailed MCP server logs.
- **LLM prompts:** Edit `mcp_client_tool_selection.py` to customize tool selection behavior.

---

## License

[Add your license here]

---

## Contributing

[Add contribution guidelines here]
