# AI NetBrain - MCP Server and Client

A simple Model Context Protocol (MCP) server and client implementation for querying network paths using the NetBrain API with Ollama running locally.

## Overview

This project provides an MCP server that integrates with NetBrain API to query network paths between source and destination endpoints. The server uses Ollama (running locally) for AI-enhanced analysis of network path information. A Streamlit-based client provides a simple web interface for submitting network queries.

## Features

- **Network Path Querying**: Query network paths between source and destination with protocol and port specifications
- **AI-Enhanced Analysis**: Uses Ollama LLM for intelligent analysis of network path information
- **Simple Web Interface**: Streamlit-based client with intuitive form inputs
- **OAuth2 Authentication**: Secure authentication with NetBrain API using OAuth2 client credentials flow

## Prerequisites

1. **Python 3.13+**: This project requires Python 3.13 or higher
2. **Ollama**: Make sure Ollama is installed and running locally
   - Download from: https://ollama.ai
   - Start Ollama service: `ollama serve`
   - Pull the required model: `ollama pull llama3.3:latest`
3. **NetBrain API Access**: 
   - NetBrain server URL
   - OAuth2 Client ID and Client Secret configured in NetBrain

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

Set the following environment variables before running:

```bash
# NetBrain API Configuration
export NETBRAIN_URL="https://your-netbrain-server.com"
export NETBRAIN_CLIENT_ID="your_client_id"
export NETBRAIN_CLIENT_SECRET="your_client_secret"
```

On Windows (PowerShell):
```powershell
$env:NETBRAIN_URL="https://your-netbrain-server.com"
$env:NETBRAIN_CLIENT_ID="your_client_id"
$env:NETBRAIN_CLIENT_SECRET="your_client_secret"
```

Alternatively, you can edit `netbrainauth.py` directly to set these values.

## Usage

### Running the MCP Server

The MCP server runs via stdio transport and communicates with NetBrain API:

```bash
python mcp_server.py
```

The server will:
- Authenticate with NetBrain API using OAuth2
- Expose the `query_network_path` tool
- Use Ollama for AI-enhanced analysis (if available)

### Running the Client

The Streamlit client provides a web interface for network queries:

```bash
streamlit run mcp_client.py
```

This will start a web server (typically at `http://localhost:8501`) where you can:

1. Enter **Source** (IP address or hostname)
2. Select **Protocol** from dropdown (TCP, UDP, ICMP, HTTP, HTTPS, SSH, FTP, SMTP, DNS, SNMP)
3. Enter **Destination** (IP address or hostname)
4. Enter **Port** number
5. Click **Query** to submit the network path query

All fields marked with `*` are required.

## Project Structure

```
.
├── mcp_server.py          # MCP server implementation
├── mcp_client.py          # Streamlit client interface
├── netbrainauth.py        # NetBrain OAuth2 authentication module
├── pyproject.toml         # Project dependencies and configuration
├── README.md              # This file
└── main.py                # Original project entry point
```

## MCP Server Tools

### `query_network_path`

Queries the network path between source and destination.

**Parameters:**
- `source` (str, required): Source IP address or hostname
- `destination` (str, required): Destination IP address or hostname
- `protocol` (str, required): Protocol (TCP, UDP, ICMP, HTTP, HTTPS, SSH, FTP, SMTP, DNS, SNMP)
- `port` (str, required): Port number

**Returns:**
- Dictionary containing:
  - `source`: Source endpoint
  - `destination`: Destination endpoint
  - `protocol`: Protocol used
  - `port`: Port number
  - `path_info`: Network path information from NetBrain API
  - `ai_analysis`: AI-enhanced analysis (if LLM is available)

## API Endpoints

The server communicates with the following NetBrain API endpoints:

- **Authentication**: `POST /api/aaa/oauth2/token`
- **Network Path Query**: `POST /api/network/path`

## Dependencies

- `mcp>=1.0.0`: Model Context Protocol SDK
- `ollama>=0.1.0`: Ollama Python client
- `streamlit`: Web interface framework (for client)
- `langchain`: LLM integration
- `aiohttp`: Async HTTP client
- `requests`: HTTP library

## Troubleshooting

### Authentication Errors

If you encounter authentication errors:
- Verify your `NETBRAIN_CLIENT_ID` and `NETBRAIN_CLIENT_SECRET` are correct
- Ensure the OAuth client is properly configured in NetBrain
- Check that the NetBrain server URL is accessible

### LLM Not Available

If Ollama is not running or the model is not available:
- The server will continue to function but without AI-enhanced analysis
- Ensure Ollama is running: `ollama serve`
- Pull the required model: `ollama pull llama3.3:latest`

### Connection Errors

If you encounter network errors:
- Verify the NetBrain server URL is correct and accessible
- Check firewall and network connectivity
- Ensure SSL certificates are properly configured (self-signed certificates are handled)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

