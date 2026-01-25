# AI NetBrain - MCP Server and Client

A Model Context Protocol (MCP) server and client implementation for querying network paths using the NetBrain API with Ollama running locally.

## Overview

This project provides an MCP server that integrates with NetBrain API to query network paths between source and destination endpoints. The server uses Ollama (running locally) for AI-enhanced analysis of network path information. A Streamlit-based client provides a simple web interface for submitting network queries.

## Features

- **Network Path Querying**: Query network paths between source and destination with protocol and port specifications
- **AI-Enhanced Analysis**: Uses Ollama LLM (llama3.2:latest) for intelligent analysis of network path information
- **Simple Web Interface**: Streamlit-based client with intuitive form inputs
- **Session-based Authentication**: Secure authentication with NetBrain API using username/password
- **Rack Location Lookup (NetBox)**: Query device rack/position/site via NetBox API

## Prerequisites

1. **Python 3.13+**: This project requires Python 3.13 or higher
2. **Ollama**: Make sure Ollama is installed and running locally
   - Download from: https://ollama.ai
   - Start Ollama service: `ollama serve`
   - Pull the required model: `ollama pull llama3.2:latest`
3. **NetBrain API Access**: 
   - NetBrain server URL
   - Username and password for NetBrain authentication

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

### NetBox API (Rack Location Tool)

NetBox URL is currently hardcoded in `mcp_server.py`:

```
http://192.168.15.136:8080
```

Configure these environment variables to enable rack location lookups:

```bash
# Linux/Mac
export NETBOX_TOKEN="your_netbox_api_token"
export NETBOX_VERIFY_SSL="true"

# Windows PowerShell
$env:NETBOX_TOKEN="your_netbox_api_token"
$env:NETBOX_VERIFY_SSL="true"
```

If `NETBOX_VERIFY_SSL` is set to `false`, SSL verification is disabled (use only for self-signed certs).

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

The Streamlit client provides a web interface for network queries:

```bash
streamlit run mcp_client.py
```

Or using uv:

```bash
uv run streamlit run mcp_client.py
```

This will start a web server (typically at `http://localhost:8501`) where you can:

1. Enter **Source** (IP address or hostname)
2. Select **Protocol** from dropdown (TCP or UDP)
3. Enter **Destination** (IP address or hostname)
4. Enter **Port** number
5. Click **Query** to submit the network path query

All fields marked with `*` are required.

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

Looks up a device in NetBox and returns its rack location details.

**Parameters:**
- `device_name` (str, required): Device name to look up in NetBox

**Returns:**
- Dictionary containing:
  - `device`: Device name
  - `rack`: Rack name (if assigned)
  - `position`: Rack unit position (if available)
  - `face`: Rack face (front/rear)
  - `site`: Site name (if available)
  - `status`: Device status (if available)
  - `error`: Error message if lookup fails

## API Endpoints

The server communicates with the following NetBrain API endpoints:

- **Authentication**: `POST /ServicesAPI/API/V1/Session`
- **Path Calculation**: `POST /ServicesAPI/API/V1/CMDB/Path/Calculation`

**Note:** The Path Calculation API returns a `taskID` which can be used with the GetPath API to retrieve detailed hop-by-hop path information. See the [NetBrain API documentation](https://github.com/NetBrainAPI/NetBrain-REST-API-R11/blob/main/REST%20APIs%20Documentation/Path%20Management/Calculate%20Path%20API.md) for more details.

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
