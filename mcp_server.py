"""
MCP Server for NetBrain Network Query
FastMCP-based server that provides network path querying capabilities.

This module:
- Exposes a query_network_path tool via MCP protocol
- Connects to NetBrain API for network path information
- Optionally uses Ollama LLM for AI-enhanced analysis
- Handles authentication and error handling
"""

# Import aiohttp for asynchronous HTTP client operations
# This allows making async HTTP requests to the NetBrain API
import aiohttp

# Import asyncio for asynchronous programming support
# Used for async/await syntax and async function definitions
import asyncio

# Import json for JSON serialization/deserialization
# Used to parse API responses and format LLM prompts
import json

# Import ssl for SSL/TLS context configuration
# Used to disable SSL verification for self-signed certificates
import ssl

# Import requests for synchronous HTTP requests (used in netbrainauth)
# Note: This is imported but not directly used in this file
import requests

# Import type hints for better code documentation and IDE support
# Optional: indicates a value can be None
# Dict, Any, List: type hints for dictionaries, any type, and lists
from typing import Optional, Dict, Any, List

# Import datetime utilities for time-based operations
# Currently imported but not actively used in this file
from datetime import datetime, timedelta

# Import the local netbrainauth module for OAuth2 authentication
# This module handles getting access tokens from NetBrain API
import netbrainauth

# Import FastMCP for creating MCP servers easily
# FastMCP provides decorators and utilities for MCP server development
from fastmcp import FastMCP

# Import ChatOllama for LLM integration with Ollama
# This allows AI-enhanced analysis of network path data
from langchain_ollama import ChatOllama

# Import os for environment variable access
# Used to read NETBRAIN_URL from environment variables
import os

# Disable SSL warnings from urllib3
# This suppresses warnings about unverified SSL certificates
import urllib3

# Disable all SSL warnings globally
# This prevents console spam when using self-signed certificates
urllib3.disable_warnings()

# Initialize MCP server instance
# FastMCP("netbrain-mcp-server") creates a new MCP server with the given name
# This name is used for identification in MCP protocol communication
mcp = FastMCP("netbrain-mcp-server")

# Initialize LLM for AI analysis
# Wrap in try-except to gracefully handle cases where Ollama is not available
try:
    # Create a ChatOllama instance for LLM interactions
    # model_name: specifies which Ollama model to use (llama3.2:latest)
    # temperature: controls randomness (0.7 = moderate creativity)
    llm = ChatOllama(
        model_name="llama3.2:latest",  # Ollama model identifier
        temperature=0.7,  # Sampling temperature (0.0 = deterministic, 1.0 = very random)
    )
    # Attach the LLM instance to the MCP server object
    # This allows the server to use the LLM for AI analysis if needed
    mcp.llm = llm
except Exception:
    # If LLM initialization fails (e.g., Ollama not running), set to None
    # The server will still work but without AI-enhanced analysis
    mcp.llm = None

# NetBrain API configuration
# Get NETBRAIN_URL from environment variable, default to localhost if not set
# os.getenv() reads environment variables, second parameter is the default value
# This should match the URL used in netbrainauth.py
NETBRAIN_URL = os.getenv("NETBRAIN_URL", "http://localhost")

# Register a tool with the MCP server using the @mcp.tool() decorator
# This makes the function callable via MCP protocol from clients
@mcp.tool()
async def query_network_path(
    source: str, 
    destination: str, 
    protocol: str, 
    port: str,
    source_gw_ip: Optional[str] = None,
    source_gw_dev: Optional[str] = None,
    source_gw_intf: Optional[str] = None,
    is_live: int = 0
):
    """
    Query network path between source and destination using NetBrain Path Calculation API.
    
    This function:
    1. Authenticates with NetBrain API
    2. Sends a network path calculation request to NetBrain API
    3. Processes the response (returns taskID which can be used with GetPath API)
    4. Optionally enhances results with AI analysis
    
    Args:
        source: Source IP address (e.g., "192.168.1.1")
        destination: Destination IP address (e.g., "192.168.1.100")
        protocol: Network protocol to query (e.g., "TCP" or "UDP")
        port: Port number to query (e.g., "80", "443", "22")
        source_gw_ip: Gateway IP address (optional, defaults to source IP)
        source_gw_dev: Gateway device hostname (optional, defaults to source)
        source_gw_intf: Gateway interface name (optional, defaults to "GigabitEthernet0/0")
        is_live: Use live data (0=Baseline, 1=Live access, default=0)
    
    Returns:
        dict: Network path information including:
            - source: Source endpoint
            - destination: Destination endpoint
            - protocol: Protocol used
            - port: Port number
            - taskID: Task ID from NetBrain API (use with GetPath API to get hop information)
            - path_info: Response from NetBrain API
            - ai_analysis: Optional AI-enhanced analysis (if LLM available)
            - error: Error message if query fails
    
    Note: The NetBrain API returns a taskID. Use the GetPath API with this taskID
    to retrieve detailed hop-by-hop path information.
    """
    # Get authentication token from netbrainauth module
    # This token is required for all NetBrain API requests
    auth_token = netbrainauth.get_auth_token()
    
    # Check if authentication token was successfully obtained
    # If not, return an error dictionary immediately
    if not auth_token:
        # Return error dictionary with error message
        # This will be sent back to the MCP client
        return {"error": "Failed to get authentication token"}
    
    # Prepare HTTP headers for the API request
    # Headers specify content type and include authentication
    # NetBrain API uses "Token" header for authentication (not Bearer Authorization)
    # Note: Using "Token" (capital T) to match the example from NetBrain API documentation
    headers = {
        "Content-Type": "application/json",  # Indicates we're sending JSON data
        "Accept": "application/json",  # Indicates we want JSON response
        "Token": auth_token  # NetBrain API uses "Token" header for authentication (capital T per example)
    }
    
    # Construct the NetBrain API endpoint URL for path calculation
    # According to NetBrain API documentation:
    # POST /ServicesAPI/API/V1/CMDB/Path/Calculation
    api_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Calculation"
    
    # Map protocol string to protocol number
    # Protocol numbers: 4=IPv4, 6=TCP, 17=UDP
    # Default to IPv4 (4) if protocol is not recognized
    protocol_map = {
        "TCP": 6,  # TCP protocol number
        "UDP": 17,  # UDP protocol number
        "IP": 4,  # IPv4 protocol number
        "IPv4": 4  # IPv4 protocol number
    }
    protocol_num = protocol_map.get(protocol.upper(), 4)  # Default to IPv4 if not found
    
    # Convert port to integer, default to 0 if not provided or invalid
    try:
        source_port = int(port) if port else 0
        dest_port = int(port) if port else 0
    except ValueError:
        # If port conversion fails, default to 0
        source_port = 0
        dest_port = 0
    
    # Build the request payload (body) for the API call
    # According to NetBrain API documentation, required fields are:
    # - sourceIP* (required): Source IP address
    # - sourceGwIP* (required): Gateway IP address (using source IP as default if not provided)
    # - sourceGwDev* (required): Gateway device hostname (using source as default if not provided)
    # - sourceGwIntf* (required): Gateway interface name (using default interface if not provided)
    # - destIP* (required): Destination IP address
    # - destPort* (required): Destination port (can be 0)
    # - pathAnalysisSet* (required): 1=L3 Path, 2=L2 Path, 3=L3 Active Path
    # - protocol* (required): Protocol number (4=IPv4, 6=TCP, 17=UDP)
    # Optional fields:
    # - sourcePort: Source port (default to 0)
    # - isLive: 0=Baseline, 1=Live access (default to 0)
    # Note: Field order matches the example from NetBrain API documentation
    # Prepare gateway values - ensure they are not empty strings
    # The API requires all gateway fields to be valid (not empty)
    source_gw_ip_value = source_gw_ip if source_gw_ip and source_gw_ip.strip() else source
    source_gw_dev_value = source_gw_dev if source_gw_dev and source_gw_dev.strip() else source
    source_gw_intf_value = source_gw_intf if source_gw_intf and source_gw_intf.strip() else "GigabitEthernet0/0"
    
    # Validate that gateway fields are not empty
    # The API error suggests SourceGateway is required, which likely means all gateway fields must be valid
    if not source_gw_ip_value or not source_gw_dev_value or not source_gw_intf_value:
        return {
            "error": "Missing required gateway information",
            "details": {
                "statusCode": 791009,
                "statusDescription": "The parameter 'SourceGateway' is invalid value. The SourceGateway field is required.",
                "missing_fields": {
                    "sourceGwIP": not bool(source_gw_ip_value),
                    "sourceGwDev": not bool(source_gw_dev_value),
                    "sourceGwIntf": not bool(source_gw_intf_value)
                }
            },
            "payload_sent": {
                "sourceIP": source,
                "sourceGwIP": source_gw_ip_value,
                "sourceGwDev": source_gw_dev_value,
                "sourceGwIntf": source_gw_intf_value
            }
        }
    
    # Use provided gateway values or fall back to defaults
    # Note: Field names match the example from NetBrain API documentation
    # The API expects these exact field names (camelCase)
    # The order matches the example: sourceIP, sourcePort, sourceGwIP, sourceGwDev, sourceGwIntf, destIP, destPort, pathAnalysisSet, protocol, isLive
    payload = {
        "sourceIP": source,  # Source IP address (required)
        "sourcePort": source_port,  # Source port (optional, default 0)
        "sourceGwIP": source_gw_ip_value,  # Gateway IP address (required)
        "sourceGwDev": source_gw_dev_value,  # Gateway device hostname (required)
        "sourceGwIntf": source_gw_intf_value,  # Gateway interface name (required)
        "destIP": destination,  # Destination IP address (required)
        "destPort": dest_port,  # Destination port (required, can be 0)
        "pathAnalysisSet": 1,  # Path type: 1=L3 Path, 2=L2 Path, 3=L3 Active Path
        "protocol": protocol_num,  # Protocol number (4=IPv4, 6=TCP, 17=UDP)
        "isLive": is_live  # Use live data (0=Baseline, 1=Live access) - must be integer, not boolean
    }
    
    # Debug: Print payload for troubleshooting (can be removed in production)
    # This helps verify the payload structure matches the API requirements
    print(f"DEBUG: Sending payload to {api_url}: {json.dumps(payload, indent=2)}")
    
    # Wrap API call in try-except to handle various error conditions
    try:
        # Create SSL context for HTTPS connections
        # ssl.create_default_context() creates a default SSL context
        ssl_context = ssl.create_default_context()
        
        # Disable hostname verification (allows self-signed certificates)
        # This is necessary for development/testing with self-signed certs
        ssl_context.check_hostname = False
        
        # Disable certificate verification (allows self-signed certificates)
        # CERT_NONE means no certificate verification will be performed
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create an async HTTP client session using aiohttp with timeout
        # Timeout prevents requests from hanging indefinitely
        # total timeout: 60 seconds (30s connect + 30s read)
        timeout = aiohttp.ClientTimeout(total=60, connect=30, sock_read=30)
        # async with ensures the session is properly closed after use
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Make an async POST request to the NetBrain API
            # async with ensures the response is properly closed after use
            async with session.post(api_url, headers=headers, json=payload, ssl=ssl_context) as response:
                # Check if the HTTP response status is not 200 (OK)
                if response.status != 200:
                    # Read the error response text
                    # await is needed because response.text() is async
                    # Try to parse as JSON first, then get text if that fails
                    try:
                        # Try to parse as JSON to get more detailed error information
                        error_json = await response.json()
                        # Print error details to console for debugging
                        print(f"ERROR: HTTP {response.status} from {api_url}")
                        print(f"ERROR DETAILS: {json.dumps(error_json, indent=2)}")
                        print(f"PAYLOAD SENT: {json.dumps(payload, indent=2)}")
                        # Return error dictionary with status code, endpoint, and error details
                        # Include parsed JSON for better debugging
                        return {
                            "error": f"HTTP error {response.status}",
                            "endpoint": api_url,  # Include the endpoint that was called
                            "details": error_json,  # Parsed error JSON
                            "error_message": str(error_json),  # String representation of error
                            "payload_sent": payload  # Include the payload that was sent for debugging
                        }
                    except Exception as e:
                        # If JSON parsing fails, read as text
                        error_text = await response.text()
                        # Print error details to console for debugging
                        print(f"ERROR: HTTP {response.status} from {api_url}")
                        print(f"ERROR DETAILS (text): {error_text}")
                        print(f"JSON PARSE ERROR: {str(e)}")
                        print(f"PAYLOAD SENT: {json.dumps(payload, indent=2)}")
                        return {
                            "error": f"HTTP error {response.status}",
                            "endpoint": api_url,  # Include the endpoint that was called
                            "details": error_text,  # Raw error text
                            "payload_sent": payload  # Include the payload that was sent for debugging
                        }
                
                # Parse the JSON response body into a Python dictionary
                # await is needed because response.json() is async
                data = await response.json()
        
        # Check if the API call was successful
        # NetBrain API returns statusCode 790200 for success
        if data.get("statusCode") != 790200:
            # If status code is not success, return error
            status_code = data.get("statusCode", "Unknown")
            status_description = data.get("statusDescription", "No description")
            return {
                "error": f"NetBrain API error: statusCode={status_code}",
                "statusDescription": status_description,
                "response": data
            }
        
        # Format the response into a structured result dictionary
        # This organizes the data for easier consumption by clients
        # The API returns a taskID which can be used with GetPath API to get hop information
        result = {
            "source": source,  # Include source from parameters
            "destination": destination,  # Include destination from parameters
            "protocol": protocol,  # Include protocol from parameters
            "port": port,  # Include port from parameters
            "taskID": data.get("taskID"),  # Task ID from NetBrain API (use with GetPath API)
            "statusCode": data.get("statusCode"),  # Status code from API response
            "statusDescription": data.get("statusDescription"),  # Status description from API response
            "path_info": data  # Full API response for reference
        }
        
        # Try to enhance with LLM analysis if available
        # Check if MCP server has an LLM instance and it's not None
        if hasattr(mcp, 'llm') and mcp.llm is not None:
            # Wrap LLM analysis in try-except to handle LLM errors gracefully
            # If LLM analysis fails, we still return the basic result
            try:
                # Create a system prompt for the LLM
                # This defines the role and expected output format
                analysis_prompt = {
                    "role": "system",  # System message sets the context
                    "content": """You are a network analysis assistant. Analyze the network path information and provide:
                    1. A summary of the path
                    2. Connectivity status
                    3. Key devices in the path
                    4. Any potential issues or recommendations
                    
                    Format your response as a JSON object with these fields:
                    {
                        "summary": "string",
                        "connectivity": "string",
                        "key_devices": ["string"],
                        "recommendations": ["string"]
                    }"""
                }
                
                # Create a user message with the network path data
                # json.dumps() converts the result dictionary to a formatted JSON string
                # indent=2 makes it human-readable
                user_message = {
                    "role": "user",  # User message contains the actual query
                    "content": f"Analyze this network path:\n{json.dumps(result, indent=2)}"
                }
                
                # Combine system and user messages into a messages list
                # This is the format expected by the LLM chat interface
                messages = [analysis_prompt, user_message]
                
                # Call the LLM's async chat method to get AI analysis
                # await is needed because achat() is an async method
                llm_response = await mcp.llm.achat(messages=messages)
                
                # Parse the LLM response
                # Check if response is a string (needs JSON parsing)
                if isinstance(llm_response, str):
                    # Parse the JSON string into a dictionary
                    analysis = json.loads(llm_response)
                else:
                    # If already a dictionary, use it directly
                    analysis = llm_response
                
                # Add the AI analysis to the result dictionary
                # This enhances the result with intelligent insights
                result["ai_analysis"] = analysis
            except Exception:
                # If LLM analysis fails, silently continue without AI analysis
                # The basic result will still be returned
                pass
        
        # Return the complete result dictionary
        # This includes path info and optionally AI analysis
        return result
        
    # Catch aiohttp-specific client errors (network issues, connection problems)
    except aiohttp.ClientError as e:
        # Return error dictionary with network error message
        # str(e) converts the exception to a readable string
        return {"error": f"Network error: {str(e)}"}
    
    # Catch timeout errors (request took too long)
    except asyncio.TimeoutError:
        # Return error dictionary with timeout message
        return {"error": "Request timed out. Please try again later."}
    
    # Catch any other unexpected exceptions
    except Exception as e:
        # Return error dictionary with generic error message
        # This is a catch-all for any unhandled exceptions
        return {"error": f"An unexpected error occurred: {str(e)}"}

# Check if this script is being run directly (not imported as a module)
# __name__ will be "__main__" when the script is executed directly
# This allows the script to be both runnable and importable
if __name__ == "__main__":
    # Start the MCP server with stdio transport
    # transport="stdio" means the server communicates via standard input/output
    # This allows the server to be spawned as a subprocess by MCP clients
    mcp.run(transport="stdio")
