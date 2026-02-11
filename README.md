# NetAssist - AI-Powered Network Infrastructure Assistant

**NetAssist** is an intelligent network infrastructure assistant built on the Model Context Protocol (MCP), providing natural language access to NetBrain path queries, NetBox device/rack lookups, Panorama address groups, and Splunk firewall deny events.

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
│                  (React-style UI + WebSocket)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/WebSocket
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Web Server                           │
│                   (app_fastapi.py)                              │
│  • Authentication (auth.py)                                     │
│  • Chat API endpoint (/api/chat)                                │
│  • Static file serving (HTML, CSS, JS)                          │
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
                             │  • NetBrain path queries            │
                             │  • NetBox device/rack lookups       │
                             │  • Panorama address groups          │
                             │  • Splunk deny events               │
                             └─────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────┼───────────────────────┐
                    ▼                      ▼                       ▼
            ┌───────────────┐     ┌──────────────┐     ┌─────────────────┐
            │  NetBrain API │     │ NetBox API   │     │ Panorama/Splunk │
            │  (netbrainauth│     │              │     │   (panoramaauth)│
            │   .py)        │     │              │     │                 │
            └───────────────┘     └──────────────┘     └─────────────────┘
```

### Key Components

1. **FastAPI Web Server** (`app_fastapi.py`)
   - Serves the web UI (HTML/JS/CSS)
   - Handles authentication
   - Provides `/api/chat` endpoint for user queries
   - Auto-reloads on code changes (development mode)

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
   - Integrates with NetBrain, NetBox, Panorama, Splunk APIs
   - Runs independently on port 8765
   - Must be started before the web client

6. **Authentication Modules**
   - `netbrainauth.py`: NetBrain API credentials
   - `panoramaauth.py`: Panorama firewall credentials
   - `auth.py`: Local web UI authentication

---

## Project Structure

```
palo-netbrain/
└── netbrain/                          # Main application directory
    ├── app_fastapi.py                 # FastAPI web server (port 8000)
    ├── auth.py                        # Local authentication
    ├── chat_service.py                # Tool orchestration & execution
    ├── mcp_client.py                  # MCP client library
    ├── mcp_client_tool_selection.py   # LLM-based tool selection
    ├── mcp_server.py                  # MCP server (port 8765)
    ├── netbrainauth.py                # NetBrain API credentials
    ├── panoramaauth.py                # Panorama API credentials
    ├── main.py                        # Legacy CLI entry point
    ├── pyproject.toml                 # Project dependencies
    ├── uv.lock                        # Dependency lock file
    ├── README.md                      # This file
    │
    ├── templates/                     # Jinja2 HTML templates
    │   ├── index.html                 # Main chat UI
    │   └── login.html                 # Login page
    │
    ├── static/                        # Static assets
    │   └── (CSS, JS, images)
    │
    ├── icons/                         # Device type icons (SVG)
    │   ├── firewall.svg
    │   ├── switch.svg
    │   ├── router.svg
    │   └── ...
    │
    └── test_*.py                      # Test scripts
```

---

## Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.13+ | Runtime environment |
| **Ollama** | Latest | LLM for tool selection (runs locally) |
| **uv** or **pip** | Latest | Package management |
| **NetBrain** | N/A | Network path calculation API (optional) |
| **NetBox** | N/A | Device/rack management API (optional) |
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
ollama pull qwen2.5:14b
```

### 2. Install Python Dependencies

**Using uv (recommended):**
```bash
cd netbrain
uv sync
```

**Using pip:**
```bash
cd netbrain
pip install -e .
```

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

### 2. NetBox Configuration

Set NetBox URL and token via environment variables:

```bash
# Linux/Mac
export NETBOX_URL="http://192.168.15.109:8080"
export NETBOX_TOKEN="your_netbox_api_token"
export NETBOX_VERIFY_SSL="false"  # Optional: disable SSL verification

# Windows PowerShell
$env:NETBOX_URL="http://192.168.15.109:8080"
$env:NETBOX_TOKEN="your_netbox_api_token"
$env:NETBOX_VERIFY_SSL="false"
```

### 3. Panorama Configuration (Optional)

Edit `panoramaauth.py` to set your Panorama credentials:

```python
PANORAMA_HOST = "your-panorama-host.com"
API_KEY = "your_api_key"
```

### 4. Splunk Configuration (Optional)

Edit `mcp_server.py` to configure Splunk connection:

```python
SPLUNK_HOST = "your-splunk-host.com"
SPLUNK_PORT = 8089
SPLUNK_USERNAME = "your_username"
SPLUNK_PASSWORD = "your_password"
```

### 5. Web UI Authentication (Optional)

Set custom login users via environment variable:

```bash
# Linux/Mac
export NETBRAIN_USERS="admin:secret,operator:pass"

# Windows PowerShell
$env:NETBRAIN_USERS="admin:secret,operator:pass"
```

Default credentials: `admin` / `admin`

---

## Running the Application

You need **two terminal windows** running simultaneously:

### Terminal 1: Start the MCP Server

```bash
cd netbrain
uv run python mcp_server.py
```

**Output:**
```
INFO: MCP Server starting on http://127.0.0.1:8765
INFO: Server initialized with tools: check_path_allowed, query_network_path, get_device_rack_location, ...
```

**Important:**
- Leave this terminal running
- The server must be started **before** the web client
- Listens on `http://127.0.0.1:8765`
- Does **not** auto-reload (restart manually if you change `mcp_server.py`)

### Terminal 2: Start the Web Client

```bash
cd netbrain
uv run python -m netbrain.app_fastapi
```

**Output:**
```
INFO: Started server process [12345]
INFO: Waiting for application startup.
INFO: Application startup complete.
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Features:**
- Web UI available at **http://localhost:8000**
- **Auto-reload enabled**: changes to Python files are picked up automatically
- No need to restart when editing `chat_service.py`, `app_fastapi.py`, etc.

### Alternative: Custom Host/Port

```bash
uv run uvicorn netbrain.app_fastapi:app --host 0.0.0.0 --port 8000 --reload
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

### 3. `get_device_rack_location`
**Purpose:** Look up device rack location and details in NetBox

**Parameters:**
- `device_name` (required): Device name to look up
- `format` (optional): Output format (table, json, minimal)
- `expected_rack` (optional): For yes/no questions like "is device X in rack Y?"

**Returns:** Rack location, position, site, device type, IPs, etc.

**Example Queries:**
- *"Where is leander-dc-border-leaf1 racked?"*
- *"Is roundrock-dc-leaf1 in rack A4?"*

---

### 4. `get_rack_details`
**Purpose:** Get details about a specific rack

**Parameters:**
- `rack_name` (required): Rack name (e.g., "A4", "B12")
- `site_name` (optional): Site name to disambiguate

**Returns:** Rack height, occupied units, space utilization, mounted devices

**Example Query:** *"Show me details for rack A4 at Leander"*

---

### 5. `list_racks`
**Purpose:** List all racks at a site or globally

**Parameters:**
- `site_name` (optional): Filter by site name

**Returns:** Table of all racks with utilization metrics

**Example Query:** *"List all racks at Leander DC"*

---

### 6. `query_panorama_ip_object_group`
**Purpose:** Find which address group(s) contain an IP or CIDR

**Parameters:**
- `ip_address` (required): IP or CIDR (e.g., "10.0.0.1", "11.0.0.0/24")

**Returns:** Address groups and address objects containing the IP

**Example Query:** *"What address group is 11.0.0.1 part of?"*

---

### 7. `query_panorama_address_group_members`
**Purpose:** List all members of an address group

**Parameters:**
- `address_group_name` (required): Address group name

**Returns:** All IPs, CIDRs, and nested groups in the address group

**Example Query:** *"What IPs are in address group leander_web?"*

---

### 8. `get_splunk_recent_denies`
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

#### NetBrain Path Queries
```
Find path from 10.0.0.1 to 10.0.1.1
Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 80 allowed?
Query path from 192.168.1.10 to 192.168.2.20 using TCP port 443
```

#### NetBox Device Queries
```
Where is leander-dc-border-leaf1 racked?
Rack location of roundrock-dc-leaf1
Is roundrock-dc-leaf1 in rack A4?
```

#### NetBox Rack Queries
```
Rack details for A4
Show rack A1 at Leander
List all racks at Leander DC
What is the space utilization of rack A4?
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
ollama pull qwen2.5:14b
```

---

### 3. NetBox/Panorama/Splunk Tools Fail

**Cause:** API credentials or URLs are incorrect

**Solution:** Check environment variables and credential files:
- `NETBOX_URL` and `NETBOX_TOKEN`
- `netbrainauth.py` for NetBrain credentials
- `panoramaauth.py` for Panorama credentials
- `mcp_server.py` for Splunk credentials

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
cd netbrain
uv sync  # or: pip install -e .
```

---

## Development Notes

- **FastAPI auto-reload:** The web client (`app_fastapi.py`) automatically reloads when you edit Python files. No need to restart Terminal 2.
- **MCP server does NOT auto-reload:** If you change `mcp_server.py`, you must restart Terminal 1.
- **Frontend changes (HTML/CSS/JS):** Hard refresh your browser to see changes.
- **Logs:** Check `mcp_server.log` for detailed MCP server logs.
- **LLM prompts:** Edit `mcp_client_tool_selection.py` to customize tool selection behavior.

---

## License

[Add your license here]

---

## Contributing

[Add contribution guidelines here]
