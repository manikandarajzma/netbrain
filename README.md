# Network Assistant - MCP Server and Client

A Model Context Protocol (MCP) server and client implementation for network infrastructure management, integrating NetBrain API for network path queries and NetBox API for device rack location lookups, with AI-enhanced analysis using Ollama.

## Overview

This project provides an MCP server that integrates with:
- **NetBrain API**: Query network paths between source and destination endpoints
- **NetBox API**: Look up device rack locations, positions, and device details

The server uses Ollama (running locally) for AI-enhanced analysis and natural language understanding. A Streamlit-based client provides an intuitive chat interface that understands natural language queries and automatically selects the appropriate tool.

## Features

- **Network Path Querying (NetBrain)**: Query network paths between source and destination with protocol and port specifications
- **Device Rack Location Lookup (NetBox)**: Query device rack location, position, site, and full device details
- **Natural Language Understanding**: AI-powered query parsing that understands user intent and automatically selects the right tool
- **Tool Discovery**: Dynamic tool selection based on available MCP tools and their descriptions
- **AI-Enhanced Analysis**: Uses Ollama LLM (llama3.2:latest) for intelligent analysis of network path and device information
- **Flexible Output Formats**: Support for table, JSON, list, and minimal formats for device queries
- **Chat Interface**: Streamlit-based client with conversation history and context-aware responses
- **Session-based Authentication**: Secure authentication with NetBrain API using username/password

## Prerequisites

1. **Python 3.13+**: This project requires Python 3.13 or higher
2. **Ollama**: Make sure Ollama is installed and running locally
   - Download from: https://ollama.ai
   - Start Ollama service: `ollama serve`
   - Pull the required model: `ollama pull llama3.2:latest`
3. **NetBrain API Access** (for network path queries): 
   - NetBrain server URL
   - Username and password for NetBrain authentication
4. **NetBox API Access** (for device rack location queries):
   - NetBox server URL (currently hardcoded to `http://192.168.15.136:8080`)
   - NetBox API token

## Installation

1. Clone or download this repository

2. Install dependencies using `uv` (recommended):
   ```bash
   uv sync
   ```

   Or using pip:
   ```bash
   pip install -e .
   ```

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

NetBox URL and API token are currently hardcoded in `mcp_server.py`:
- **NetBox URL**: `http://192.168.15.136:8080`
- **NetBox Token**: Configured in `mcp_server.py`

The NetBox integration provides:
- Device rack location lookup
- Device details (type, manufacturer, model, serial, IP addresses, etc.)
- Site and rack information
- Position and face (front/rear) information

To modify the NetBox configuration, edit `mcp_server.py` and update:
- `NETBOX_URL`: NetBox server URL
- `NETBOX_TOKEN`: NetBox API token
- `NETBOX_VERIFY_SSL`: Set to `False` to disable SSL verification (use only for self-signed certs)

## Usage

### Running the MCP Server

The MCP server runs via stdio transport and communicates with NetBrain API:

```bash
python mcp_server.py
```

Or using uv:

```bash
uv run python mcp_server.py
```

The server will:
- Authenticate with NetBrain API using username/password
- Expose the `query_network_path` tool
- Use Ollama for AI-enhanced analysis (if available)

### Running the Client

The Streamlit client provides a chat-based web interface that understands natural language queries:

```bash
streamlit run mcp_client.py
```

Or using uv:

```bash
uv run streamlit run mcp_client.py
```

This will start a web server (typically at `http://localhost:8501`) with a chat interface where you can:

#### Network Path Queries (NetBrain)
Ask questions like:
- "Find path from 10.0.0.1 to 10.0.1.1"
- "Query path from 192.168.1.10 to 192.168.2.20 using TCP port 443"
- "Show me the network path from 10.10.3.253 to 172.24.32.225 UDP port 53 with live data"

#### Device Rack Location Queries (NetBox)
Ask questions like:
- "leander-dc-border-leaf1" (shows full device details in table format)
- "give me just the rack location for roundrock-dc-leaf1"
- "leander-dc-border-leaf1 just the rack location in table format"
- "where is roundrock-dc-leaf1"
- "device details for leander-dc-border-leaf1"

The client uses AI-powered natural language understanding to:
- Automatically detect query intent (network path vs. device lookup)
- Extract parameters from natural language
- Select the appropriate tool based on available MCP tools
- Format responses according to user preferences (table, JSON, list, or minimal)

## Project Structure

```
.
├── mcp_server.py          # MCP server implementation
├── mcp_client.py           # Streamlit client interface
├── netbrainauth.py         # NetBrain authentication module
├── pyproject.toml          # Project dependencies and configuration
├── README.md               # This file
└── main.py                 # Original project entry point
```

## MCP Server Tools

### `query_network_path`

Queries the network path between source and destination.

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
- `ollama>=0.1.0`: Ollama Python client
- `streamlit>=1.28.0`: Web interface framework (for client)
- `langchain>=0.1.0`: LLM integration framework
- `langchain-core>=0.1.0`: LangChain core functionality
- `langchain-community>=0.0.20`: LangChain community integrations
- `langchain-ollama>=0.1.0`: LangChain Ollama integration
- `jsonpatch>=1.32`: JSON patch support (required by langchain-core)
- `aiohttp>=3.9.0`: Async HTTP client
- `requests>=2.31.0`: HTTP library
- `urllib3>=2.0.0`: HTTP client library

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
- Pull the required model: `ollama pull llama3.2:latest`

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
