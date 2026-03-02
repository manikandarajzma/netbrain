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
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTPS
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Nginx                                    │
│              (reverse proxy / TLS termination)                  │
│  • Forwards requests to FastAPI on port 8000                    │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (localhost:8000)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                           │
│                   (app.py)                                      │
│  • Authentication (auth.py, OIDC)                               │
│  • Chat API (/api/chat, /api/discover, /api/me)                 │
│  • RBAC: role-based access to tool categories                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Chat Service Layer                            │
│                  (chat_service.py)                              │
│  • Query parsing & routing                                      │
│  • Conversation history management                              │
│  • Result normalization                                         │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  LLM Tool Selection  (mcp_client_tool_selection.py)       │  │
│  │  • Ollama/Llama3.1:8b  • Intent classification            │  │
│  │  • Parameter extraction from natural language             │  │
│  └───────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │ tool name + parameters
                             ▼
             ┌──────────────────────────────────┐
             │    MCP Client  (mcp_client.py)   │
             │  • Session management            │
             │  • Tool execution                │
             │  • HTTP transport (port 8765)    │
             └──────────────────┬───────────────┘
                                │ streamable-http
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
| **Nginx** | Latest | Reverse proxy / TLS termination |
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

Both services are managed by **systemd** and run automatically on boot. Nginx handles all incoming traffic and proxies to the web server.

### Service Management

```bash
# Start services
sudo systemctl start atlas-mcp
sudo systemctl start atlas-web

# Stop services
sudo systemctl stop atlas-mcp
sudo systemctl stop atlas-web

# Restart services (e.g. after a code change)
sudo systemctl restart atlas-mcp
sudo systemctl restart atlas-web

# Check status
sudo systemctl status atlas-mcp
sudo systemctl status atlas-web
```

### Logs

```bash
# Live log tailing
tail -f /var/log/atlas/mcp_server.log
tail -f /var/log/atlas/atlas_web.log

# Or via journald
journalctl -u atlas-mcp -f
journalctl -u atlas-web -f
```

### Nginx

Nginx terminates TLS and proxies to the FastAPI web server on `localhost:8000`.

```bash
# Reload nginx config after changes
sudo systemctl reload nginx

# Check nginx status
sudo systemctl status nginx
```

### Running Manually (Development)

If you need to run the services directly outside of systemd:

```bash
# Terminal 1 — MCP server
cd /opt/atlas
.venv/bin/python mcp_server.py

# Terminal 2 — Web server
cd /opt/atlas
.venv/bin/python run_web.py
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

1. Open the Atlas URL in your browser (via Nginx, e.g. **https://atlas.yourcompany.com**)
2. Log in with your **Microsoft account** (OIDC via Azure Entra ID)
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

See [Documentation/General/troubleshooting/troubleshooting.md](Documentation/General/troubleshooting/troubleshooting.md) for common issues and solutions.

---

## Development Notes

- **Code changes:** After editing Python files, restart the relevant systemd service (`sudo systemctl restart atlas-web` or `atlas-mcp`).
- **MCP server does NOT auto-reload:** Changes to `mcp_server.py` or tool files require `sudo systemctl restart atlas-mcp`.
- **Logs:** `/var/log/atlas/atlas_web.log` (web) and `/var/log/atlas/mcp_server.log` (MCP). Falls back to the project directory if `/var/log/atlas/` is not writable.
- **Nginx config:** Located at `/etc/nginx/sites-available/atlas`. Reload with `sudo systemctl reload nginx` after changes.
- **LLM prompts:** Edit `mcp_client_tool_selection.py` to customize tool selection behavior.

---

## License

[Add your license here]

---

## Contributing

[Add contribution guidelines here]
