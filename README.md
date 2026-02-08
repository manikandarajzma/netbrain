# Network Assistant - MCP Server and Client

A Model Context Protocol (MCP) server and client for network infrastructure management: NetBrain path queries, NetBox device/rack lookups, Panorama address groups, and Splunk deny events, with AI-powered natural language understanding via Ollama.

---

## Quick start

1. **Install dependencies** (from the project root directory):
   ```bash
   uv sync
   ```

2. **Start Ollama** (for tool selection and analysis):
   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ```

3. **Start the MCP server** (first terminal):
   ```bash
   uv run python netbrain/mcp_server.py
   ```
   Leave this running. It listens on `http://127.0.0.1:8765` by default.

4. **Start the web client** (second terminal, from project root):
   ```bash
   uv run python -m netbrain.app_fastapi
   ```
   The FastAPI app runs with **auto-reload**: edits to Python files (e.g. `chat_service.py`, `app_fastapi.py`) are picked up automatically; you don’t need to restart. (The MCP server in step 3 does not auto-reload—restart it if you change `mcp_server.py`.)

5. **Open the app**: go to **http://localhost:8000**, log in with **admin** / **admin**, and type a query (e.g. *Find path from 10.0.0.1 to 10.0.1.1* or *Where is leander-dc-border-leaf1 racked?*).

---

## Overview

This project provides an MCP server that integrates with:
- **NetBrain API**: Query network paths between source and destination endpoints
- **NetBox API**: Look up device rack locations, positions, and device details
- **Panorama**: Address group membership and IP-to-group lookups
- **Splunk**: Recent deny events for an IP (Palo Alto firewall logs)

The server uses Ollama (running locally) for natural language tool selection. The **FastAPI** web client provides a chat UI with **local authentication** (SAML can be added later).

## Features

- **Network Path Querying (NetBrain)**: Query network paths between source and destination with protocol and port specifications
- **Device Rack Location Lookup (NetBox)**: Query device rack location, position, site, and full device details
- **Natural Language Understanding**: AI-powered query parsing that understands user intent and automatically selects the right tool
- **Tool Discovery**: Dynamic tool selection based on available MCP tools and their descriptions
- **AI-Enhanced Analysis**: Uses Ollama LLM (llama3.1:8b) for intelligent analysis of network path and device information
- **Flexible Output Formats**: Support for table, JSON, list, and minimal formats for device queries
- **Chat interface**: FastAPI app with conversation history
- **Local authentication**: Username/password login (SAML can be added later)
- **NetBrain/NetBox auth**: Credentials for backend APIs

## Prerequisites

| Requirement | Purpose |
|-------------|---------|
| **Python 3.13+** | Runtime |
| **Ollama** | LLM for tool selection; run `ollama serve` and `ollama pull llama3.1:8b` |
| **NetBrain API** | Path queries (URL + credentials in `netbrainauth.py`) |
| **NetBox API** | Device/rack lookups (URL + token in `mcp_server.py`) |
| **Panorama** (optional) | Address group tools (see `panoramaauth.py`) |
| **Splunk** (optional) | Recent denies tool (host/port/user/pass in `mcp_server.py`) |

## Installation

**From the project root** (the directory that contains the `netbrain` folder):

1. Install dependencies with **uv** (recommended):
   ```bash
   uv sync
   ```
   Or with **pip**:
   ```bash
   pip install -e .
   ```

2. Ensure **Ollama** is installed and running:
   - https://ollama.ai
   - `ollama serve`
   - `ollama pull llama3.1:8b` (or the model referenced in tool selection config)

## Configuration

### NetBrain Credentials

Edit `netbrainauth.py` directly to set your NetBrain credentials:

```python
# NetBrain username for authentication
USERNAME = "your_username"

# NetBrain password for authentication
PASSWORD = "your_password"
```

### NetBrain URL (Optional)

You can optionally set the NetBrain URL via environment variable:

```bash
# Linux/Mac
export NETBRAIN_URL="http://your-netbrain-server.com"

# Windows PowerShell
$env:NETBRAIN_URL="http://your-netbrain-server.com"
```

If not set, it defaults to `http://localhost`.

### NetBox API (Device Rack Location Tool)

NetBox is used for device rack location and rack details. Default URL is **192.168.15.109** (port 8080). You can override with environment variables:

- **NetBox URL**: `NETBOX_URL` (default: `http://192.168.15.109:8080`)
- **NetBox Token**: `NETBOX_TOKEN` (or set in `mcp_server.py` as fallback)
- **SSL verification**: `NETBOX_VERIFY_SSL` — set to `false` to disable (e.g. self-signed certs)

The NetBox integration provides:
- Device rack location lookup
- Device details (type, manufacturer, model, serial, IP addresses, etc.)
- Site and rack information
- Position and face (front/rear) information

## Usage

You need **two processes**: the **MCP server** (backend tools) and the **web client** (UI). Run them in two terminals.

### Step 1: Start the MCP server

In a terminal, from the **project root**:

```bash
uv run python netbrain/mcp_server.py
```

- The server starts on **http://127.0.0.1:8765** (streamable-http transport).
- Leave this terminal open. The client will connect to this port.

If you are **inside** the `netbrain` directory:

```bash
uv run python mcp_server.py
```

### Step 2: Start the web client

In a **second** terminal, from the **project root**:

```bash
uv run python -m netbrain.app_fastapi
```

- The app listens on **http://0.0.0.0:8000**.
- Open a browser at **http://localhost:8000**.

**Alternative** (custom host/port):

```bash
uv run uvicorn netbrain.app_fastapi:app --host 0.0.0.0 --port 8000
```

### Step 3: Log in and use the chat

1. At **http://localhost:8000** you’ll see the login page.
2. Default credentials: **Username** `admin`, **Password** `admin`.
3. After login you’ll see the chat interface. Example queries:
   - *Find path from 10.0.0.1 to 10.0.1.1*
   - *Where is leander-dc-border-leaf1 racked?*
   - *What address group is 11.0.0.0/24 part of?*
   - *List all the denies for 10.0.0.250*
   - *Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 80 allowed?*

**If the chat returns nothing or “No result”:**

- Ensure the **MCP server** is running first (Step 1). The web app talks to it at `http://127.0.0.1:8765`. If it’s not running, requests can hang or fail.
- For *Where is &lt;device&gt; racked?*: ensure **NetBox** is reachable at `NETBOX_URL` (default `http://192.168.15.109:8080`) and `NETBOX_TOKEN` is set. Check the terminal where the MCP server runs for errors.

#### Custom login users

Set `NETBRAIN_USERS` to a comma-separated list of `username:password`:

```bash
# Linux / macOS
export NETBRAIN_USERS="admin:secret,operator:pass"

# Windows PowerShell
$env:NETBRAIN_USERS="admin:secret,operator:pass"
```

Then start the FastAPI app as above.

### Example queries

| Category | Examples |
|----------|----------|
| **NetBrain path** | *Find path from 10.0.0.1 to 10.0.1.1* • *Query path from 192.168.1.10 to 192.168.2.20 using TCP port 443* |
| **NetBrain policy** | *Is traffic from 10.0.0.1 to 10.0.1.1 on TCP port 80 allowed?* |
| **NetBox device** | *Where is leander-dc-border-leaf1 racked?* • *Rack location of roundrock-dc-leaf1* |
| **NetBox rack** | *Rack details for A4* • *Show rack A1* |
| **Panorama** | *What address group is 11.0.0.0/24 part of?* • *What IPs are in address group leander_web?* |
| **Splunk** | *List all the denies for 10.0.0.250* • *Recent deny events for 192.168.1.1* |

The FastAPI client uses the Ollama LLM to choose the right tool and extract parameters from your question.

**Agent loop and final answer:** The chat service runs an agent loop (up to 3 iterations by default). Each iteration: discover tool → execute. If a tool fails, the error is added to the conversation context and the LLM can try again (e.g. a different tool or parameters). After the last attempt, or when the tool returns an error, the service forces a **synthesized final answer**: the LLM summarizes what was tried, why it failed, and suggests what the user can do next, instead of returning only the raw error message. You can change the iteration limit via `MAX_AGENT_ITERATIONS` in `chat_service.py` or by passing `max_iterations` to `process_message`.

### Tool selection: "path allowed" vs path hops vs Panorama

| Query intent | Correct tool | What it does |
|--------------|--------------|---------------|
| **"Is path allowed from X to Y?"** / **"Is traffic allowed?"** / **"Check if traffic from A to B is allowed"** | **`check_path_allowed`** (NetBrain) | Runs NetBrain path calculation with policy enforcement; returns **allowed** / **denied** / **unknown**. This is the only tool that answers "is traffic allowed?" between two IPs. |
| **"What is the path from X to Y?"** / **"Show path"** / **"Trace route"** | **`query_network_path`** (NetBrain) | Returns path hops (topology); can continue past policy denial. Use when you want to see the route, not a yes/no allow verdict. |
| **"What address group is IP in?"** / **"Which group contains 10.0.0.1?"** | **Panorama** (`query_panorama_ip_object_group`) | Looks up whether an IP appears in Panorama address objects/groups. Does **not** simulate path or policy allow/deny. |
| **"What IPs are in address group X?"** | **Panorama** (`query_panorama_address_group_members`) | Lists members of a Panorama address group. |

If you see Panorama output (e.g. "IP not found in any address objects or address groups") for a question like **"Is path allowed from 10.0.0.1 to 10.0.1.1?"**, the agent chose a Panorama tool instead of **check_path_allowed**. Tool selection is driven by the LLM and the tool descriptions in `mcp_server.py`; there is no keyword-based routing in code. To get NetBrain path+policy for "is path allowed" queries, the tool descriptions must lead the LLM to select **check_path_allowed**.

## Project Structure

```
netbrain/
├── app_fastapi.py         # FastAPI app (login, chat UI, /api/chat)
├── auth.py                # Local authentication (swap for SAML later)
├── chat_service.py        # Tool discovery + execution for chat API
├── mcp_server.py          # MCP server (tools for path, device, Panorama, Splunk)
├── mcp_client.py          # MCP client library (get_mcp_session, execute_*); FastAPI is the UI
├── mcp_client_tool_selection.py  # LLM tool selection
├── netbrainauth.py        # NetBrain authentication
├── panoramaauth.py        # Panorama authentication
├── templates/             # Login and chat HTML
│   ├── login.html
│   └── index.html
├── pyproject.toml
└── README.md
```

## MCP Server Tools

### `check_path_allowed`

**Use for:** "Is path allowed from X to Y?", "Is traffic allowed?", "Check if traffic from A to B is allowed". Answers whether traffic between two IPs is **allowed**, **denied**, or **unknown** using NetBrain path calculation with policy enforcement (stops at first policy denial).

**Parameters:** `source`, `destination`, `protocol` (e.g. TCP), `port` (e.g. 443).

**Returns:** `status` ("allowed" / "denied" / "unknown"), `reason`, `path_exists`, `policy_details`, etc. This is **not** a Panorama object lookup; it uses NetBrain.

### `query_network_path`

Queries the network path (hop-by-hop) between source and destination. Use when you want to see the route; for a yes/no "is traffic allowed?" use **check_path_allowed** instead.

**Parameters:**
- `source` (str, required): Source IP address or hostname
- `destination` (str, required): Destination IP address or hostname
- `protocol` (str, required): Protocol (TCP or UDP)
- `port` (str, required): Port number

**Returns:**
- Dictionary containing:
  - `source`: Source endpoint
  - `destination`: Destination endpoint
  - `protocol`: Protocol used
  - `port`: Port number
  - `path_info`: Network path information from NetBrain API
  - `ai_analysis`: AI-enhanced analysis (if LLM is available)
  - `error`: Error message if query fails

### `get_device_rack_location`

Looks up a device in NetBox and returns its rack location and device details.

**Parameters:**
- `device_name` (str, required): Device name to look up in NetBox
- `format` (str, optional): Output format - "table", "json", "list", "minimal", "summary", or None for natural language summary
- `conversation_history` (list, optional): Conversation history for context-aware responses

**Returns:**
- Dictionary containing:
  - `device`: Device name
  - `rack`: Rack name (if assigned)
  - `position`: Rack unit position (formatted as U1, U2, etc.)
  - `face`: Rack face (front/rear)
  - `site`: Site name (if available)
  - `status`: Device status (if available)
  - `device_type`: Device type/model
  - `device_role`: Device role
  - `manufacturer`: Manufacturer name
  - `model`: Device model
  - `serial`: Serial number
  - `primary_ip`: Primary IP address
  - `primary_ip4`: Primary IPv4 address
  - `ai_analysis`: AI-generated summary (if LLM is available)
  - `error`: Error message if lookup fails

**Format Options:**
- `"table"`: Full device details displayed in a table format
- `"minimal"` or `"summary"`: Only rack location fields (site, rack, position)
- `"json"`: JSON format output
- `"list"`: List format output
- `None`: Natural language summary with AI analysis

## API Endpoints

### NetBrain API

The server communicates with the following NetBrain API endpoints:

- **Authentication**: `POST /ServicesAPI/API/V1/Session`
- **Path Calculation**: `POST /ServicesAPI/API/V1/CMDB/Path/Calculation`
- **Path Details**: `GET /ServicesAPI/API/V1/CMDB/Path/Calculation/{taskID}/OverView`
- **Device Gateway Resolution**: `GET /ServicesAPI/API/V1/CMDB/Path/Gateways`

**Note:** The Path Calculation API returns a `taskID` which can be used with the GetPath API to retrieve detailed hop-by-hop path information. See the [NetBrain API documentation](https://github.com/NetBrainAPI/NetBrain-REST-API-R11/blob/main/REST%20APIs%20Documentation/Path%20Management/Calculate%20Path%20API.md) for more details.

### NetBox API

The server communicates with the following NetBox API endpoints:

- **Device Lookup**: `GET /api/dcim/devices/` (with name or search query)
- **Device Details**: Returns full device information including rack, position, site, and device attributes

**Note:** The NetBox integration supports both exact name matching and search queries to find devices.

## Dependencies

- `mcp>=1.0.0`: Model Context Protocol SDK
- `fastmcp>=0.9.0`: FastMCP for building MCP servers
- `fastapi>=0.115.0`, `uvicorn`, `jinja2`: Web app and auth
- `ollama>=0.1.0`, `langchain`, `langchain-ollama`: LLM tool selection
- `aiohttp`, `requests`, `urllib3`: HTTP clients
- `pandas`, `matplotlib`, `networkx`: Data and visualizations

## Troubleshooting

### Authentication Errors

If you encounter authentication errors:
- Verify your username and password in `netbrainauth.py` are correct
- Ensure the NetBrain server URL is accessible
- Check that your account is not locked or expired

### LLM Not Available

If Ollama is not running or the model is not available:
- The server will continue to function but without AI-enhanced analysis
- Ensure Ollama is running: `ollama serve`
- Pull the required model: `ollama pull llama3.1:8b`

### Connection Errors

If you encounter network errors:
- Verify the NetBrain server URL is correct and accessible
- Check firewall and network connectivity
- Ensure SSL certificates are properly configured (self-signed certificates are handled)

### Import Errors

If you encounter import errors:
- Make sure all dependencies are installed: `uv sync` or `pip install -e .`
- Verify you're using the correct Python environment
- Check that `fastmcp` and `langchain-ollama` packages are installed

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
