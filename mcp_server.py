"""
MCP Server for NetBrain Network Query
FastMCP-based server that provides network path querying capabilities.

This module:
- Exposes a query_network_path tool via MCP protocol
- Connects to NetBrain API for network path information
- Follows the three-step Path Calculation API process:
  1. Resolve device gateway
  2. Calculate path
  3. Get path details
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

# Import type hints for better code documentation and IDE support
# Optional: indicates a value can be None
# Dict, Any, List: type hints for dictionaries, any type, and lists
from typing import Optional, Dict, Any, List

# Import the local netbrainauth module for authentication
# This module handles getting access tokens from NetBrain API
import netbrainauth

# Import FastMCP for creating MCP servers easily
# FastMCP provides decorators and utilities for MCP server development
from fastmcp import FastMCP

# Import ChatOllama for LLM integration with Ollama
# This allows AI-enhanced analysis of network path data
from langchain_ollama import ChatOllama

# Import LangChain prompt templates for structured prompt management
# ChatPromptTemplate provides better maintainability, reusability, and variable substitution
from langchain_core.prompts import ChatPromptTemplate

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
    is_live: int = 0
):
    """
    Query network path between source and destination using NetBrain Path Calculation API.
    
    This function follows the three-step process from NetBrain API documentation:
    1. Resolve device gateway (GET /V1/CMDB/Path/Gateways)
    2. Calculate path (POST /V1/CMDB/Path/Calculation)
    3. Get path details (GET /V1/CMDB/Path/Calculation/{taskID}/OverView)
    
    Args:
        source: Source IP address or hostname (e.g., "192.168.1.1")
        destination: Destination IP address (e.g., "192.168.1.100")
        protocol: Network protocol to query (e.g., "TCP" or "UDP")
        port: Port number to query (e.g., "80", "443", "22")
        is_live: Use live data (0=Baseline, 1=Live access, default=0)
    
    Returns:
        dict: Network path information including:
            - source: Source endpoint
            - destination: Destination endpoint
            - protocol: Protocol used
            - port: Port number
            - taskID: Task ID from NetBrain API
            - path_details: Detailed hop-by-hop path information
            - ai_analysis: Optional AI-enhanced analysis (if LLM available)
            - error: Error message if query fails
    """
    # Debug: Print function entry and parameters
    print(f"DEBUG: query_network_path called with source={source}, destination={destination}, protocol={protocol}, port={port}, is_live={is_live}")
    
    # Get authentication token from netbrainauth module
    # This token is required for all NetBrain API requests
    auth_token = netbrainauth.get_auth_token()
    
    # Check if authentication token was successfully obtained
    # If not, return an error dictionary immediately
    if not auth_token:
        print("DEBUG: Failed to get authentication token")
        return {"error": "Failed to get authentication token"}
    
    print(f"DEBUG: Authentication token obtained: {auth_token[:20]}...")
    
    # Prepare HTTP headers for all API requests
    # NetBrain API uses "token" header (lowercase) based on cURL example in documentation
    headers = {
        "Content-Type": "application/json",  # Indicates we're sending JSON data
        "Accept": "application/json",  # Indicates we want JSON response
        "token": auth_token  # NetBrain API uses "token" header for authentication
    }
    
    # Map protocol string to protocol number
    # Protocol numbers: 4=IPv4, 6=TCP, 17=UDP
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
        source_port = 0
        dest_port = 0
    
    # Trim source and destination to remove whitespace
    source_trimmed = source.strip() if source else ""
    destination_trimmed = destination.strip() if destination else ""
    
    # Create SSL context for HTTPS connections
    # Disable SSL verification for self-signed certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Create an async HTTP client session with timeout
    timeout = aiohttp.ClientTimeout(total=60, connect=30, sock_read=30)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # ====================================================================
            # STEP 1: Resolve Device Gateway
            # GET /V1/CMDB/Path/Gateways?ipOrHost=<source>
            # ====================================================================
            print(f"DEBUG: Step 1 - Resolving gateway for source: {source_trimmed}")
            gateway_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Gateways"
            gateway_params = {"ipOrHost": source_trimmed}
            
            async with session.get(gateway_url, headers=headers, params=gateway_params, ssl=ssl_context) as gateway_response:
                if gateway_response.status != 200:
                    error_text = await gateway_response.text()
                    print(f"ERROR: Step 1 failed - HTTP {gateway_response.status}: {error_text}")
                    return {
                        "error": f"Failed to resolve gateway: HTTP {gateway_response.status}",
                        "step": 1,
                        "details": error_text
                    }
                
                gateway_data = await gateway_response.json()
                
                # Check if gateway resolution was successful
                if gateway_data.get("statusCode") != 790200:
                    status_code = gateway_data.get("statusCode", "Unknown")
                    status_desc = gateway_data.get("statusDescription", "No description")
                    print(f"ERROR: Step 1 failed - statusCode={status_code}: {status_desc}")
                    return {
                        "error": f"Gateway resolution failed: statusCode={status_code}",
                        "step": 1,
                        "statusDescription": status_desc,
                        "response": gateway_data
                    }
                
                # Get the gateway list from the response
                gateway_list = gateway_data.get("gatewayList", [])
                if not gateway_list:
                    print("ERROR: Step 1 - No gateways found")
                    return {
                        "error": "No gateways found for source device",
                        "step": 1,
                        "response": gateway_data
                    }
                
                # Use the first gateway from the list
                # The gateway object contains: gatewayName, type, payload
                source_gateway = gateway_list[0]
                print(f"DEBUG: Step 1 - Gateway resolved: {source_gateway.get('gatewayName')}")
            
            # ====================================================================
            # STEP 2: Calculate Path
            # POST /V1/CMDB/Path/Calculation
            # ====================================================================
            print(f"DEBUG: Step 2 - Calculating path from {source_trimmed} to {destination_trimmed}")
            calc_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Calculation"
            
            # Build payload for path calculation
            # sourceGateway must be the object from Step 1, not separate fields
            payload = {
                "sourceIP": source_trimmed,  # IP address of the source device
                "sourcePort": source_port,  # Source port (0 if not provided)
                "sourceGateway": source_gateway,  # Gateway object from Step 1 (required as object)
                "destIP": destination_trimmed,  # IP address of the destination device
                "destPort": dest_port,  # Destination port (0 if not provided)
                "pathAnalysisSet": 1,  # 1=L3 Path; 2=L2 Path; 3=L3 Active Path
                "protocol": protocol_num,  # Protocol number (4=IPv4, 6=TCP, 17=UDP)
                "isLive": 1 if is_live else 0  # 0=Current Baseline; 1=Live access
            }
            
            print(f"DEBUG: Step 2 - Payload: {json.dumps(payload, indent=2)}")
            
            async with session.post(calc_url, headers=headers, json=payload, ssl=ssl_context) as calc_response:
                if calc_response.status != 200:
                    error_text = await calc_response.text()
                    print(f"ERROR: Step 2 failed - HTTP {calc_response.status}: {error_text}")
                    return {
                        "error": f"Path calculation failed: HTTP {calc_response.status}",
                        "step": 2,
                        "details": error_text,
                        "payload_sent": payload,
                        "auth_token": auth_token
                    }
                
                calc_data = await calc_response.json()
                
                # Check if path calculation was successful
                if calc_data.get("statusCode") != 790200:
                    status_code = calc_data.get("statusCode", "Unknown")
                    status_desc = calc_data.get("statusDescription", "No description")
                    print(f"ERROR: Step 2 failed - statusCode={status_code}: {status_desc}")
                    return {
                        "error": f"Path calculation failed: statusCode={status_code}",
                        "step": 2,
                        "statusDescription": status_desc,
                        "response": calc_data,
                        "payload_sent": payload,
                        "auth_token": auth_token
                    }
                
                # Get taskID from the response
                # Ensure taskID is a string (JSON may return it as different type)
                task_id = calc_data.get("taskID")
                if not task_id:
                    print("ERROR: Step 2 - No taskID in response")
                    return {
                        "error": "No taskID returned from path calculation",
                        "step": 2,
                        "response": calc_data
                    }
                
                # Convert taskID to string to ensure it's in the correct format
                task_id = str(task_id)
                
                print(f"DEBUG: Step 2 - Path calculation successful, taskID: {task_id}")
            
            # ====================================================================
            # STEP 3: Get Path Details (Optional)
            # GET /V1/CMDB/Path/Calculation/{taskID}/OverView
            # Note: This step is optional - path calculation (Step 2) is the main operation
            # Path details may not be immediately available or may require different endpoint
            # The taskID from Step 2 can be used to query path details separately if needed
            # ====================================================================
            print(f"DEBUG: Step 3 - Attempting to get path details for taskID: {task_id} (optional)")
            
            path_data = None
            step3_error = None
            
            # Try to get path details, but don't fail if this step doesn't work
            # Some NetBrain versions may have different endpoint formats or timing requirements
            try:
                # Wait a moment for path calculation to complete
                # Path calculation is asynchronous, so we need to wait for it to finish
                await asyncio.sleep(3)
                
                # Primary endpoint from documentation
                path_url = f"{NETBRAIN_URL}/ServicesAPI/API/V1/CMDB/Path/Calculation/{task_id}/OverView"
                
                print(f"DEBUG: Step 3 - Trying endpoint: {path_url}")
                async with session.get(path_url, headers=headers, ssl=ssl_context) as path_response:
                    # Read the response text first (can only read once)
                    response_text = await path_response.text()
                    
                    # Try to parse as JSON regardless of status code
                    # The API may return path data even with non-200 status codes
                    try:
                        response_json = json.loads(response_text)
                        
                        # Check if response contains path data (path_overview, path_list, etc.)
                        if "path_overview" in response_json or "path_list" in response_json or "hop_detail_list" in response_json:
                            # Response contains path data, use it even if status is not 200
                            path_data = response_json
                            print(f"DEBUG: Step 3 - Path details retrieved (status {path_response.status})")
                            
                            # If status is not 200, note it but still use the data
                            if path_response.status != 200:
                                print(f"INFO: Step 3 - Path details available but HTTP status is {path_response.status}")
                                # Store the status code in the data for reference
                                path_data["_http_status"] = path_response.status
                        elif path_response.status == 200:
                            # Standard success case
                            path_data = response_json
                            print(f"DEBUG: Step 3 - Path details retrieved successfully")
                        else:
                            # No path data in response
                            print(f"DEBUG: Step 3 - HTTP {path_response.status}: No path data in response")
                            step3_error = f"HTTP {path_response.status}: {response_text[:200]}"
                    except json.JSONDecodeError:
                        # Response is not valid JSON
                        if path_response.status == 200:
                            # Try to use it anyway
                            path_data = {"raw_response": response_text}
                        else:
                            print(f"DEBUG: Step 3 - HTTP {path_response.status}: Invalid JSON response")
                            step3_error = f"HTTP {path_response.status}: {response_text[:200]}"
            except Exception as e:
                print(f"DEBUG: Step 3 - Exception during path details retrieval: {str(e)}")
                step3_error = str(e)
            
            # Step 3 is optional - if it fails, we still return success from Step 2
            # The taskID can be used to query path details separately
            if path_data is None:
                print(f"INFO: Step 3 - Path details not available, but path calculation (Step 2) succeeded")
            
            # Build result dictionary
            result = {
                "source": source_trimmed,
                "destination": destination_trimmed,
                "protocol": protocol,
                "port": port,
                "taskID": task_id,
                "statusCode": calc_data.get("statusCode"),
                "statusDescription": calc_data.get("statusDescription"),
                "gateway_used": source_gateway.get("gatewayName"),
                "path_info": calc_data  # Original calculation response
            }
            
            # Add path details if Step 3 succeeded
            if path_data is not None:
                # Extract simplified hop information: device name, status, and reason
                simplified_hops = []
                path_status_overall = "Unknown"
                path_failure_reason = None
                
                try:
                    # Navigate through the path_overview structure
                    path_overview = path_data.get("path_overview", [])
                    for path_group in path_overview:
                        path_list = path_group.get("path_list", [])
                        for path in path_list:
                            # Get path-level status and description
                            path_status_overall = path.get("status", "Unknown")
                            path_description = path.get("description", "")
                            
                            branch_list = path.get("branch_list", [])
                            for branch in branch_list:
                                branch_status = branch.get("status", "Unknown")
                                branch_failure_reason = branch.get("failure_reason", None)
                                
                                hop_detail_list = branch.get("hop_detail_list", [])
                                for hop in hop_detail_list:
                                    from_dev = hop.get("fromDev", {})
                                    to_dev = hop.get("toDev", {})
                                    
                                    from_dev_name = from_dev.get("devName", "Unknown")
                                    to_dev_name = to_dev.get("devName") if to_dev.get("devName") else None
                                    
                                    # Only add if we have device information
                                    if from_dev_name != "Unknown" or to_dev_name:
                                        hop_info = {
                                            "hop_sequence": hop.get("sequnce", len(simplified_hops)),
                                            "from_device": from_dev_name,
                                            "to_device": to_dev_name,
                                            "status": branch_status,
                                            "failure_reason": branch_failure_reason
                                        }
                                        simplified_hops.append(hop_info)
                                
                                # If branch has failure reason, use it for path-level
                                if branch_failure_reason and not path_failure_reason:
                                    path_failure_reason = branch_failure_reason
                    
                    # If we found hops, use simplified format
                    if simplified_hops:
                        result["path_hops"] = simplified_hops
                        result["path_status"] = path_status_overall
                        result["path_status_description"] = path_data.get("statusDescription", path_failure_reason or "")
                        if path_failure_reason:
                            result["path_failure_reason"] = path_failure_reason
                    else:
                        # Fallback to full path_data if we couldn't parse it
                        result["path_details"] = path_data
                except Exception as e:
                    # If parsing fails, include full data and note the error
                    print(f"DEBUG: Error parsing path details: {str(e)}")
                    result["path_details"] = path_data
                    result["parse_error"] = str(e)
            else:
                # Add note about using taskID to query path details separately
                result["note"] = f"Path calculation succeeded. Use taskID '{task_id}' to query detailed path information separately if needed."
                if step3_error:
                    result["step3_info"] = f"Path details endpoint returned: {step3_error}. This is optional - path calculation was successful."
            
            # Try to enhance with LLM analysis if available
            if hasattr(mcp, 'llm') and mcp.llm is not None:
                try:
                    # Use LangChain ChatPromptTemplate for structured prompt management
                    # This provides better maintainability, reusability, and variable substitution
                    analysis_prompt_template = ChatPromptTemplate.from_messages([
                        ("system", """You are a network analysis assistant. Analyze the network path information and provide:
                        1. A summary of the path
                        2. Connectivity status
                        3. Key devices in the path
                        4. Any potential issues or recommendations
                        
                        Format your response as a JSON object with these fields:
                        {{
                            "summary": "string",
                            "connectivity": "string",
                            "key_devices": ["string"],
                            "recommendations": ["string"]
                        }}"""),
                        ("human", "Analyze this network path:\n{path_data}")
                    ])
                    
                    # Format the prompt with the path data
                    formatted_messages = analysis_prompt_template.format_messages(
                        path_data=json.dumps(result, indent=2)
                    )
                    
                    # Invoke the LLM with the formatted prompt
                    llm_response = await mcp.llm.ainvoke(formatted_messages)
                    
                    # Extract content from the response
                    if hasattr(llm_response, 'content'):
                        response_content = llm_response.content
                    else:
                        response_content = str(llm_response)
                    
                    # Try to parse as JSON, fallback to string if not valid JSON
                    try:
                        analysis = json.loads(response_content)
                    except json.JSONDecodeError:
                        # If response is not valid JSON, wrap it in a summary field
                        analysis = {"summary": response_content}
                    
                    result["ai_analysis"] = analysis
                except Exception as e:
                    # Log the error for debugging but don't fail the entire request
                    print(f"DEBUG: LLM analysis failed: {str(e)}")
                    pass
            
            return result
            
    except aiohttp.ClientError as e:
        return {"error": f"Network error: {str(e)}"}
    except asyncio.TimeoutError:
        return {"error": "Request timed out. Please try again later."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}


# Check if this script is being run directly (not imported as a module)
# __name__ will be "__main__" when the script is executed directly
# This allows the script to be both runnable and importable
if __name__ == "__main__":
    # Run the MCP server using stdio transport
    # stdio transport means the server communicates via standard input/output
    # This is the standard way MCP servers communicate with clients
    mcp.run(transport="stdio")
