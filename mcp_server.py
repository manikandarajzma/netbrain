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
async def query_network_path(source: str, destination: str, protocol: str, port: str):
    """
    Query network path between source and destination.
    
    This function:
    1. Authenticates with NetBrain API
    2. Sends a network path query request
    3. Processes the response
    4. Optionally enhances results with AI analysis
    
    Args:
        source: Source IP address or hostname (e.g., "192.168.1.1" or "server1")
        destination: Destination IP address or hostname (e.g., "192.168.1.100" or "server2")
        protocol: Network protocol to query (e.g., "TCP" or "UDP")
        port: Port number to query (e.g., "80", "443", "22")
    
    Returns:
        dict: Network path information including:
            - source: Source endpoint
            - destination: Destination endpoint
            - protocol: Protocol used
            - port: Port number
            - path_info: Detailed path information from NetBrain API
            - ai_analysis: Optional AI-enhanced analysis (if LLM available)
            - error: Error message if query fails
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
    # NetBrain API uses "token" header for authentication (not Bearer Authorization)
    headers = {
        "Content-Type": "application/json",  # Indicates we're sending JSON data
        "Accept": "application/json",  # Indicates we want JSON response
        "token": auth_token  # NetBrain API uses "token" header for authentication
    }
    
    # Construct the NetBrain API endpoint URL for path queries
    # f-string formats the URL with the base NETBRAIN_URL
    api_url = f"{NETBRAIN_URL}/api/network/path"
    
    # Build the request payload (body) for the API call
    # This dictionary contains the query parameters
    payload = {
        "source": source,  # Source IP/hostname from function parameter
        "destination": destination,  # Destination IP/hostname from function parameter
        "protocol": protocol,  # Protocol from function parameter
        "port": port  # Port number from function parameter
    }
    
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
        
        # Create an async HTTP client session using aiohttp
        # async with ensures the session is properly closed after use
        async with aiohttp.ClientSession() as session:
            # Make an async POST request to the NetBrain API
            # async with ensures the response is properly closed after use
            async with session.post(api_url, headers=headers, json=payload, ssl=ssl_context) as response:
                # Check if the HTTP response status is not 200 (OK)
                if response.status != 200:
                    # Read the error response text
                    # await is needed because response.text() is async
                    error_text = await response.text()
                    # Return error dictionary with status code and error details
                    return {"error": f"HTTP error {response.status}", "details": error_text}
                
                # Parse the JSON response body into a Python dictionary
                # await is needed because response.json() is async
                data = await response.json()
        
        # Format the response into a structured result dictionary
        # This organizes the data for easier consumption by clients
        result = {
            "source": source,  # Include source from parameters
            "destination": destination,  # Include destination from parameters
            "protocol": protocol,  # Include protocol from parameters
            "port": port,  # Include port from parameters
            # Extract path information from API response
            # Try "data" key first, then "path" key, otherwise use entire response
            "path_info": data.get("data", data.get("path", data))
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
