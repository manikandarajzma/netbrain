"""
MCP Client for NetBrain Network Query.

This module provides the MCP client API used by the FastAPI chat service:
- get_mcp_session: context manager for MCP HTTP or stdio connections
- execute_network_query, execute_path_allowed_check: path and policy checks
- execute_rack_details_query, execute_racks_list_query, execute_rack_location_query: NetBox rack/device lookups
- execute_panorama_ip_object_group_query, execute_panorama_address_group_members_query: Panorama lookups
- execute_splunk_recent_denies_query: Splunk deny events

Use the FastAPI app for the chat UI: uv run python -m netbrain.app_fastapi
"""

# Import asyncio for handling asynchronous operations (needed for MCP client)
import asyncio
import sys
import warnings

# Suppress asyncio cleanup warnings on Windows (these are harmless cleanup errors)
if sys.platform == 'win32':
    # Suppress ConnectionResetError in asyncio cleanup callbacks on Windows
    def _suppress_asyncio_cleanup_warnings():
        import logging
        logging.getLogger('asyncio').setLevel(logging.ERROR)
    
    # Set up warning filter for asyncio cleanup errors
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='asyncio')

# Import ClientSession for managing MCP client connections (for stdio fallback)
from mcp import ClientSession
# Try to import FastMCP Client for HTTP transport (preferred)
try:
    from fastmcp import Client as FastMCPClient
    FASTMCP_CLIENT_AVAILABLE = True
except ImportError:
    FASTMCP_CLIENT_AVAILABLE = False
    print("DEBUG: FastMCP Client not available, will use stdio transport", file=sys.stderr, flush=True)

# Fallback to stdio if HTTP client not available
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
try:
    from mcp.shared.exceptions import McpError
except ImportError:
    # Fallback if McpError is not available
    McpError = Exception

# Import json for serialization
import json

def get_server_url():
    """
    Get the HTTP URL for the MCP server.

    Returns:
        str: Server URL for HTTP transport (streamable-http uses /mcp endpoint)
    """
    import os
    host = os.getenv("MCP_SERVER_HOST", "127.0.0.1")
    port = os.getenv("MCP_SERVER_PORT", "8765")
    return f"http://{host}:{port}/mcp"

async def get_mcp_session():
    """
    Get an MCP session using HTTP transport (preferred) or stdio (fallback).
    This is a generator that yields the session within the proper context.
    
    Yields:
        FastMCPClient or ClientSession: MCP client session
    """
    if FASTMCP_CLIENT_AVAILABLE:
        # Use FastMCP Client for HTTP transport (automatically handles streamable-http)
        try:
            server_url = get_server_url()
            print(f"DEBUG: Connecting to MCP server via HTTP at {server_url}...", file=sys.stderr, flush=True)
            # FastMCP Client automatically infers streamable-http from URL
            # Try to configure with longer timeout for long-running requests
            try:
                # Check if FastMCPClient accepts timeout parameters
                client = FastMCPClient(server_url, timeout=600)  # 10 minute timeout
            except TypeError:
                # If timeout parameter not supported, use default
                client = FastMCPClient(server_url)
            async with client:
                print(f"DEBUG: FastMCP HTTP client connected successfully", file=sys.stderr, flush=True)
                yield client
        except Exception as e:
            # HTTP connection failed - this is expected if server is running in stdio mode
            # Silently fall back to stdio (don't print verbose traceback as it's expected)
            print(f"DEBUG: HTTP connection unavailable (server may be in stdio mode), using stdio transport...", file=sys.stderr, flush=True)
            # Fall through to stdio fallback
    
    # Fallback to stdio transport
    server_params = get_server_params()
    print(f"DEBUG: Connecting to MCP server via stdio...", file=sys.stderr, flush=True)
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print(f"DEBUG: Stdio session initialized successfully", file=sys.stderr, flush=True)
            yield session

def get_server_params():
    """
    Create server parameters for stdio communication (fallback).
    
    This function creates the configuration needed to spawn the MCP server
    as a subprocess via stdio transport.
    
    Returns:
        StdioServerParameters: Server parameters for stdio communication
    """
    import os
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    server_path = os.path.join(script_dir, "mcp_server.py")
    
    # Create server parameters for stdio communication:
    # - command: The command to run (python interpreter)
    # - args: Arguments to pass (the mcp_server.py script with full path)
    # This configures the client to spawn mcp_server.py as a subprocess
    from mcp import StdioServerParameters
    return StdioServerParameters(
        command="python",  # Use Python interpreter to run the server
        args=[server_path]  # Pass mcp_server.py with full path as argument
    )

async def execute_path_allowed_check(source, destination, protocol, port, is_live):
    """
    Execute path allowed/denied check asynchronously.
    
    Args:
        source: Source IP/hostname
        destination: Destination IP/hostname
        protocol: Protocol (TCP/UDP)
        port: Port number
        is_live: Use live data (True/False)
        
    Returns:
        dict: Policy check result
    """
    import sys
    print(f"DEBUG: Starting path allowed check: {source} -> {destination}, protocol={protocol}, port={port}, is_live={is_live}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized, calling check_path_allowed tool...", file=sys.stderr, flush=True)
            
            # Use defaults if protocol or port are None (server expects strings, not None)
            protocol_str = protocol if protocol is not None else "TCP"
            port_str = port if port is not None else "0"
            
            tool_arguments = {
                "source": source,
                "destination": destination,
                "protocol": protocol_str,
                "port": port_str,
                "is_live": 1 if is_live else 0
            }
            print(f"DEBUG: Tool arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Try standard format first, then FastMCP format if needed
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    # FastMCPClient not available or not importable
                    pass
            
            try:
                # Try standard MCP format first
                result = await asyncio.wait_for(
                    client_or_session.call_tool("check_path_allowed", arguments=tool_arguments),
                    timeout=360.0  # 6 minute timeout for path checks
                )
                print(f"DEBUG: Standard format succeeded for check_path_allowed", file=sys.stderr, flush=True)
            except TypeError as e:
                if "unexpected keyword argument 'arguments'" in str(e) and is_fastmcp:
                    # FastMCP client expects **kwargs instead of arguments=dict
                    print(f"DEBUG: Standard format failed, trying FastMCP format (kwargs) for check_path_allowed", file=sys.stderr, flush=True)
                    result = await asyncio.wait_for(
                        client_or_session.call_tool("check_path_allowed", **tool_arguments),
                        timeout=360.0
                    )
                else:
                    raise
            
            # Process result based on client type
            # Check if result has content attribute (standard MCP format)
            if hasattr(result, 'content') and result.content:
                if isinstance(result.content, list) and len(result.content) > 0:
                    content_item = result.content[0]
                    result_text = content_item.text if hasattr(content_item, 'text') else str(content_item)
                else:
                    result_text = str(result.content)
            else:
                result_text = str(result)
            
            print(f"DEBUG: Result text length: {len(result_text)}", file=sys.stderr, flush=True)
            print(f"DEBUG: Result text (first 500 chars): {result_text[:500]}", file=sys.stderr, flush=True)
            
            try:
                result_dict = json.loads(result_text)
                return result_dict
            except json.JSONDecodeError:
                return {"result": result_text}
    
    except asyncio.TimeoutError:
        return {"error": "Path allowed check timed out. Please try again."}
    except Exception as e:
        import traceback
        print(f"DEBUG: Error in execute_path_allowed_check: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": f"Error executing path allowed check: {str(e)}"}


async def execute_network_query(source, destination, protocol, port, is_live):
    """
    Execute network path query asynchronously.
    
    Args:
        source: Source IP/hostname
        destination: Destination IP/hostname
        protocol: Protocol (TCP/UDP)
        port: Port number
        is_live: Use live data (True/False)
        
    Returns:
        dict: Query result
    """
    import sys
    print(f"DEBUG: Starting network query: {source} -> {destination}, protocol={protocol}, port={port}, is_live={is_live}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized, calling tool...", file=sys.stderr, flush=True)
            
            # Use defaults if protocol or port are None (server expects strings, not None)
            protocol_str = protocol if protocol is not None else "TCP"
            port_str = port if port is not None else "0"
            
            tool_arguments = {
                "source": source,
                "destination": destination,
                "protocol": protocol_str,
                "port": port_str,
                "is_live": 1 if is_live else 0,
                "continue_on_policy_denial": True  # Always continue even if denied by policy
            }
            print(f"DEBUG: Tool arguments: {tool_arguments}", file=sys.stderr, flush=True)
            print(f"DEBUG: Calling tool with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            print(f"DEBUG: Client type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}, FASTMCP_CLIENT_AVAILABLE: {FASTMCP_CLIENT_AVAILABLE}", file=sys.stderr, flush=True)
            try:
                # Try to detect FastMCP Client - be conservative, default to standard format
                is_fastmcp = False
                if FASTMCP_CLIENT_AVAILABLE:
                    try:
                        # Check if it's actually a FastMCPClient instance
                        is_fastmcp = isinstance(client_or_session, FastMCPClient)
                        print(f"DEBUG: isinstance check result: {is_fastmcp}", file=sys.stderr, flush=True)
                    except (NameError, TypeError) as e:
                        print(f"DEBUG: isinstance check failed: {e}, checking by module name", file=sys.stderr, flush=True)
                        # Fallback: check by module name (more reliable)
                        module_name = type(client_or_session).__module__
                        is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
                        print(f"DEBUG: Module-based check result: {is_fastmcp} (module: {module_name})", file=sys.stderr, flush=True)
                
                print(f"DEBUG: Final detection - is_fastmcp: {is_fastmcp}, type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}", file=sys.stderr, flush=True)
                
                # Try standard format first (safest), then FastMCP format if needed
                try:
                    # Standard MCP ClientSession format: pass arguments as a dictionary
                    print(f"DEBUG: Trying standard format (arguments=dict) for query_network_path...", file=sys.stderr, flush=True)
                    tool_result = await asyncio.wait_for(
                        client_or_session.call_tool("query_network_path", arguments=tool_arguments),
                        timeout=360.0  # 6 minute timeout for network path queries (server polls up to 120 times)
                    )
                    print(f"DEBUG: Standard format succeeded for query_network_path, result type: {type(tool_result)}", file=sys.stderr, flush=True)
                except TypeError as e:
                    error_str = str(e)
                    # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                    if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                        print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for query_network_path...", file=sys.stderr, flush=True)
                        tool_result_list = await asyncio.wait_for(
                            client_or_session.call_tool("query_network_path", **tool_arguments),
                            timeout=360.0  # 6 minute timeout for network path queries (server polls up to 120 times)
                        )
                        print(f"DEBUG: FastMCP call completed, result type: {type(tool_result_list)}, length: {len(tool_result_list) if isinstance(tool_result_list, list) else 'N/A'}", file=sys.stderr, flush=True)
                        # FastMCP returns a list of results, each with .text attribute
                        # Convert to standard format for processing
                        if tool_result_list and len(tool_result_list) > 0:
                            # Create a mock tool_result object with content attribute
                            class FastMCPToolResult:
                                def __init__(self, results):
                                    self.content = []
                                    for r in results:
                                        text = r.text if hasattr(r, 'text') else str(r)
                                        self.content.append(type('Content', (), {'text': text})())
                            tool_result = FastMCPToolResult(tool_result_list)
                            print(f"DEBUG: FastMCP result wrapped, content items: {len(tool_result.content)}", file=sys.stderr, flush=True)
                        else:
                            tool_result = None
                            print(f"DEBUG: FastMCP returned empty result", file=sys.stderr, flush=True)
                        print(f"DEBUG: FastMCP format succeeded for query_network_path", file=sys.stderr, flush=True)
                    else:
                        # Re-raise if it's a different error
                        print(f"DEBUG: Error calling query_network_path: {e}", file=sys.stderr, flush=True)
                        raise
                print(f"DEBUG: Tool call completed, processing result...", file=sys.stderr, flush=True)
            except asyncio.TimeoutError:
                print(f"DEBUG: Tool call timed out after 360 seconds", file=sys.stderr, flush=True)
                return {"error": "Network path query timed out after 6 minutes. The query may be too complex or the server may be slow. Please try again or use baseline data instead of live data."}
            except (ConnectionResetError, ConnectionError, OSError) as conn_error:
                # Connection was closed, but server may have completed - check if we got partial result
                print(f"DEBUG: Connection error during tool call: {conn_error}", file=sys.stderr, flush=True)
                print(f"DEBUG: Connection error type: {type(conn_error).__name__}", file=sys.stderr, flush=True)
                # If we have a tool_result, try to process it anyway
                if 'tool_result' in locals() and tool_result:
                    print(f"DEBUG: Connection closed but we have a result, attempting to process...", file=sys.stderr, flush=True)
                    # Fall through to result processing below
                else:
                    import traceback
                    print(f"DEBUG: Connection error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                    return {"error": f"Connection was closed during query. The server may have completed processing, but the connection was lost. Error: {str(conn_error)}"}
            except Exception as tool_error:
                print(f"DEBUG: Tool call failed with error: {tool_error}", file=sys.stderr, flush=True)
                import traceback
                print(f"DEBUG: Tool call error traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
                # If we have a tool_result despite the error, try to process it
                if 'tool_result' in locals() and tool_result:
                    print(f"DEBUG: Error occurred but we have a result, attempting to process...", file=sys.stderr, flush=True)
                    # Fall through to result processing below
                else:
                    return {"error": f"Tool call failed: {str(tool_error)}"}
            
            if tool_result:
                print(f"DEBUG: Tool result received, type: {type(tool_result)}, dir: {[x for x in dir(tool_result) if not x.startswith('_')]}", file=sys.stderr, flush=True)
                result_text = None
                
                # Handle FastMCP Client response (list of results or FastMCPToolResult wrapper)
                if isinstance(tool_result, list):
                    if len(tool_result) > 0:
                        # FastMCP returns list of result objects with .text attribute
                        first_result = tool_result[0]
                        if hasattr(first_result, 'text'):
                            result_text = first_result.text
                        else:
                            result_text = str(first_result)
                        print(f"DEBUG: FastMCP list result (first 500 chars): {result_text[:500] if result_text else 'None'}", file=sys.stderr, flush=True)
                    else:
                        return {"error": "Tool call returned empty result"}
                # Handle FastMCPToolResult wrapper or standard MCP ClientSession response (both have .content)
                elif hasattr(tool_result, 'content') and tool_result.content:
                    print(f"DEBUG: Standard MCP result content length: {len(tool_result.content)}", file=sys.stderr, flush=True)
                    if isinstance(tool_result.content, list) and len(tool_result.content) > 0:
                        content_item = tool_result.content[0]
                        if hasattr(content_item, 'text'):
                            result_text = content_item.text
                        else:
                            result_text = str(content_item)
                    else:
                        result_text = str(tool_result.content)
                    print(f"DEBUG: Result text length: {len(result_text) if result_text else 0}", file=sys.stderr, flush=True)
                    print(f"DEBUG: Result text (first 500 chars): {result_text[:500] if result_text else 'None'}", file=sys.stderr, flush=True)
                else:
                    # Try to convert to string or check if it's already a dict
                    if isinstance(tool_result, dict):
                        print(f"DEBUG: Tool result is already a dict, returning directly", file=sys.stderr, flush=True)
                        return tool_result
                    result_text = str(tool_result)
                    print(f"DEBUG: Converted result to string (first 500 chars): {result_text[:500]}", file=sys.stderr, flush=True)
                
                if not result_text:
                    print(f"DEBUG: ERROR - result_text is None or empty", file=sys.stderr, flush=True)
                    return {"error": "Tool call returned empty or unparseable result"}
                
                # Try to parse as JSON (use module-level json)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Result parsed successfully, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}, result_text: {result_text[:200]}", file=sys.stderr, flush=True)
                    # Try to extract JSON from the text if it's embedded
                    import re
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        try:
                            result = json.loads(json_match.group())
                            print(f"DEBUG: Extracted JSON from text, keys: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}", file=sys.stderr, flush=True)
                            return result
                        except json.JSONDecodeError as e2:
                            print(f"DEBUG: Failed to parse extracted JSON: {e2}", file=sys.stderr, flush=True)
                    # If still can't parse, return the raw text wrapped
                    print(f"DEBUG: Returning raw text as result", file=sys.stderr, flush=True)
                    return {"error": f"Failed to parse JSON result: {str(e)}", "raw_result": result_text[:1000]}
            else:
                print(f"DEBUG: Tool result is None or empty", file=sys.stderr, flush=True)
                return {"error": "Tool call returned no result"}
    except asyncio.TimeoutError:
        print(f"DEBUG: Query timed out", file=sys.stderr, flush=True)
        return {"error": "Query timed out. The network path calculation is taking longer than expected."}
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        print(f"DEBUG: Exception in execute_network_query: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {error_traceback}", file=sys.stderr, flush=True)
        prefix = "Error executing query"
        if "JSON" in str(e) or "json" in str(e):
            return {"error": f"{prefix}: {e} (JSON parsing error - check server logs)"}
        return {"error": _format_tool_error(e, prefix)}
    except BaseException as e:
        print(f"DEBUG: Exception in execute_network_query (base): {e}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e, "Error executing query")}


def _unwrap_exception_message(e):
    """Recursively get the first leaf exception message from ExceptionGroup/TaskGroup."""
    sub = getattr(e, "exceptions", None)
    if sub and len(sub) > 0:
        return _unwrap_exception_message(sub[0])
    return str(e)


def _format_tool_error(e, prefix="Error executing query"):
    """Format tool execution error; recursively unwrap ExceptionGroup so user sees real cause."""
    sub = getattr(e, "exceptions", None)
    if sub and len(sub) > 0:
        try:
            leaf = _unwrap_exception_message(e)
            if leaf:
                suffix = "" if "mcp_server" in leaf or "log" in leaf else " (see mcp_server.log for details)"
                return f"{prefix}: {leaf}{suffix}"
        except Exception:
            pass
        return f"{prefix}: MCP server error (check mcp_server.log)"
    msg = str(e)
    if "TaskGroup" in msg or "unhandled errors" in msg:
        return f"{prefix}: MCP server error (check mcp_server.log)"
    return f"{prefix}: {msg}"


async def execute_rack_details_query(rack_name, format_type=None, conversation_history=None, site_name=None):
    """
    Execute rack details lookup via MCP server.

    Args:
        rack_name: Rack name to look up in NetBox
        format_type: Optional format request ("table", "json", "list")
        conversation_history: Optional conversation history for context
        site_name: Optional site name to filter racks

    Returns:
        dict: Rack details result
    """
    import sys
    print(f"DEBUG: Starting rack details query for rack: {rack_name}, site: {site_name}, format: {format_type}", file=sys.stderr, flush=True)

    try:
        print(f"DEBUG: Connecting to MCP server for rack details query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for rack details query", file=sys.stderr, flush=True)
            
            tool_arguments = {"rack_name": rack_name}
            if site_name:
                tool_arguments["site_name"] = site_name
            if format_type:
                tool_arguments["format"] = format_type
            if conversation_history:
                tool_arguments["conversation_history"] = conversation_history
            
            # Detect client type for fallback
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for rack details: FastMCP={is_fastmcp}, type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}", file=sys.stderr, flush=True)
            
            tool_result = None
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for get_rack_details...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("get_rack_details", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for get_rack_details", file=sys.stderr, flush=True)
            except TypeError as type_err:
                error_str = str(type_err)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for get_rack_details...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("get_rack_details", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for get_rack_details", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling get_rack_details: {error_str}", file=sys.stderr, flush=True)
                    raise
                
                if tool_result and tool_result.content:
                    result_text = tool_result.content[0].text
                    print(f"DEBUG: Rack details result text length: {len(result_text)}", file=sys.stderr, flush=True)
                    try:
                        result = json.loads(result_text)
                        print(f"DEBUG: Rack details result parsed successfully", file=sys.stderr, flush=True)
                        return result
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                        return {"result": result_text}
                else:
                    print(f"DEBUG: No result content returned", file=sys.stderr, flush=True)
                    return None
            # Standard format succeeded: parse and return result
            if tool_result and getattr(tool_result, "content", None):
                result_text = tool_result.content[0].text
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"result": result_text}
            elif tool_result is not None and not getattr(tool_result, "content", None):
                return None
    except asyncio.TimeoutError:
        return {"error": "Rack details lookup timed out"}
    except Exception as e:
        print(f"DEBUG: Error executing rack details query: {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}
    except BaseException as e:
        # ExceptionGroup (e.g. TaskGroup) extends BaseException, not Exception
        print(f"DEBUG: Error executing rack details query (base): {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}


async def execute_racks_list_query(site_name=None, format_type=None, conversation_history=None):
    """
    Execute racks list lookup via MCP server.
    
    Args:
        site_name: Optional site name to filter racks
        format_type: Optional format request ("table", "json", "list")
        conversation_history: Optional conversation history for context
        
    Returns:
        dict: Racks list result
    """
    import sys
    print(f"DEBUG: Starting racks list query, site: {site_name}, format: {format_type}", file=sys.stderr, flush=True)

    try:
        print(f"DEBUG: Connecting to MCP server for racks list query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for racks list query", file=sys.stderr, flush=True)
            tool_arguments = {}
            if site_name:
                tool_arguments["site_name"] = site_name
            if format_type:
                tool_arguments["format"] = format_type
            if conversation_history:
                tool_arguments["conversation_history"] = conversation_history

            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = "fastmcp" in (module_name or "").lower()
            tool_result = None
            try:
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("list_racks", arguments=tool_arguments),
                    timeout=65.0,
                )
            except TypeError as type_err:
                error_str = str(type_err)
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("list_racks", **tool_arguments),
                        timeout=65.0,
                    )
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, "text") else str(r)
                                    self.content.append(type("Content", (), {"text": text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                    else:
                        tool_result = None
                else:
                    raise

            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                print(f"DEBUG: Racks list result text length: {len(result_text)}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Racks list result parsed successfully", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                    return {"result": result_text}
            print(f"DEBUG: No result content returned", file=sys.stderr, flush=True)
            return None
    except asyncio.TimeoutError:
        return {"error": "Racks list lookup timed out"}
    except Exception as e:
        print(f"DEBUG: Error executing racks list query: {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}
    except BaseException as e:
        # ExceptionGroup (e.g. TaskGroup) extends BaseException, not Exception
        print(f"DEBUG: Error executing racks list query (base): {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}


async def execute_panorama_address_group_members_query(address_group_name, device_group=None, vsys="vsys1"):
    """
    Execute Panorama address group members query via MCP server.
    
    Args:
        address_group_name: Address group name to query
        device_group: Optional device group name
        vsys: VSYS name (default: "vsys1")
    
    Returns:
        dict: Address group members information
    """
    import sys
    print(f"DEBUG: Starting Panorama address group members query for group: {address_group_name}, device_group: {device_group}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server for Panorama address group members query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for Panorama address group members query", file=sys.stderr, flush=True)
            
            tool_arguments = {"address_group_name": address_group_name}
            if device_group:
                tool_arguments["device_group"] = device_group
            if vsys:
                tool_arguments["vsys"] = vsys
            
            print(f"DEBUG: Calling query_panorama_address_group_members with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Detect client type for fallback
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for Panorama query: FastMCP={is_fastmcp}", file=sys.stderr, flush=True)
            
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for query_panorama_address_group_members...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("query_panorama_address_group_members", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for query_panorama_address_group_members", file=sys.stderr, flush=True)
            except TypeError as type_err:
                error_str = str(type_err)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for query_panorama_address_group_members...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("query_panorama_address_group_members", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for query_panorama_address_group_members", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling query_panorama_address_group_members: {error_str}", file=sys.stderr, flush=True)
                    raise
            
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                print(f"DEBUG: Panorama address group members result text length: {len(result_text)}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Panorama address group members result parsed successfully", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        print(f"DEBUG: Panorama address group members query timed out", file=sys.stderr, flush=True)
        return {"error": "Panorama query timed out. Please try again."}
    except Exception as e:
        print(f"DEBUG: Error in Panorama address group members query: {str(e)}", file=sys.stderr, flush=True)
        import traceback
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e, "Error querying Panorama")}
    except BaseException as e:
        print(f"DEBUG: Error in Panorama address group members query (base): {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e, "Error querying Panorama")}


async def execute_panorama_ip_object_group_query(ip_address, device_group=None, vsys="vsys1"):
    """
    Execute Panorama IP object group query via MCP server.
    
    Args:
        ip_address: IP address to search for
        device_group: Optional device group name
        vsys: VSYS name (default: "vsys1")
    
    Returns:
        dict: Object group information
    """
    import sys
    print(f"DEBUG: Starting Panorama IP object group query for IP: {ip_address}, device_group: {device_group}", file=sys.stderr, flush=True)
    
    try:
        print(f"DEBUG: Connecting to MCP server for Panorama IP object group query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for Panorama IP object group query", file=sys.stderr, flush=True)
            
            tool_arguments = {"ip_address": ip_address}
            if device_group:
                tool_arguments["device_group"] = device_group
            if vsys:
                tool_arguments["vsys"] = vsys
            
            print(f"DEBUG: Calling query_panorama_ip_object_group with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Detect client type for fallback
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for Panorama query: FastMCP={is_fastmcp}", file=sys.stderr, flush=True)
            
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for query_panorama_ip_object_group...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("query_panorama_ip_object_group", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for query_panorama_ip_object_group", file=sys.stderr, flush=True)
            except TypeError as type_err:
                error_str = str(type_err)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for query_panorama_ip_object_group...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("query_panorama_ip_object_group", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for query_panorama_ip_object_group", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling query_panorama_ip_object_group: {error_str}", file=sys.stderr, flush=True)
                    raise
            
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                print(f"DEBUG: Panorama result text length: {len(result_text)}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Panorama result parsed successfully", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {e}", file=sys.stderr, flush=True)
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        return {"error": "Panorama query timed out"}
    except Exception as e:
        import sys
        import traceback
        print(f"DEBUG: Error executing Panorama query: {str(e)}", file=sys.stderr, flush=True)
        print(f"DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}
    except BaseException as e:
        print(f"DEBUG: Error executing Panorama query (base): {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}


async def execute_splunk_recent_denies_query(ip_address, limit=100, earliest_time="-24h"):
    """
    Execute Splunk recent denies query via MCP server.

    Args:
        ip_address: IP address to search for in deny events
        limit: Max number of events (default 100)
        earliest_time: Splunk time range (default "-24h")

    Returns:
        dict: ip_address, events (list), count, and optional error
    """
    import sys
    print(f"DEBUG: Starting Splunk recent denies query for IP: {ip_address}", file=sys.stderr, flush=True)
    try:
        async for client_or_session in get_mcp_session():
            tool_arguments = {"ip_address": ip_address, "limit": limit, "earliest_time": earliest_time}
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            try:
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("get_splunk_recent_denies", arguments=tool_arguments),
                    timeout=90.0
                )
            except TypeError as e:
                if "unexpected keyword argument 'arguments'" in str(e) and is_fastmcp:
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("get_splunk_recent_denies", **tool_arguments),
                        timeout=90.0
                    )
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                    else:
                        tool_result = None
                else:
                    raise
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        return {"error": "Splunk query timed out"}
    except Exception as e:
        import traceback
        print(f"DEBUG: Error executing Splunk query: {str(e)}", file=sys.stderr, flush=True)
        print(f"DEBUG: Full traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}
    except BaseException as e:
        print(f"DEBUG: Error executing Splunk query (base): {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e)}


async def execute_rack_location_query(device_name, format_type=None, conversation_history=None, intent=None, expected_rack=None):
    """
    Execute rack location lookup via MCP server.

    Args:
        device_name: Device name to look up in NetBox
        format_type: Optional format request ("table", "json", "list")
        conversation_history: Optional conversation history for context
        intent: Optional intent ("device_details", "rack_location_only", "device_type_only", etc.)
        expected_rack: Optional expected rack for yes/no questions (e.g., "A1", "B4")

    Returns:
        dict: Rack location result
    """
    import sys
    print(f"DEBUG: Starting rack location query for device: {device_name}, format: {format_type}, intent: {intent}", file=sys.stderr, flush=True)

    try:
        print(f"DEBUG: Connecting to MCP server for rack location query...", file=sys.stderr, flush=True)
        async for client_or_session in get_mcp_session():
            print(f"DEBUG: Session initialized for rack location query", file=sys.stderr, flush=True)
            
            tool_arguments = {"device_name": device_name}
            if format_type:
                tool_arguments["format"] = format_type
            if intent:
                tool_arguments["intent"] = intent
            if expected_rack:
                tool_arguments["expected_rack"] = expected_rack
            if conversation_history:
                tool_arguments["conversation_history"] = conversation_history

            print(f"DEBUG: Calling get_device_rack_location with arguments: {tool_arguments}", file=sys.stderr, flush=True)
            
            # Detect client type and call appropriately
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = 'fastmcp' in module_name.lower() if module_name else False
            
            print(f"DEBUG: Client type for rack location: FastMCP={is_fastmcp}, type: {type(client_or_session).__name__}, module: {type(client_or_session).__module__}", file=sys.stderr, flush=True)
            
            # Try standard format first (safest), then FastMCP format if needed
            try:
                # Standard MCP ClientSession format: pass arguments as a dictionary
                print(f"DEBUG: Trying standard format (arguments=dict) for get_device_rack_location...", file=sys.stderr, flush=True)
                tool_result = await asyncio.wait_for(
                    client_or_session.call_tool("get_device_rack_location", arguments=tool_arguments),
                    timeout=60.0
                )
                print(f"DEBUG: Standard format succeeded for get_device_rack_location", file=sys.stderr, flush=True)
            except TypeError as type_err:
                error_str = str(type_err)
                # If standard format fails with "unexpected keyword argument 'arguments'", try FastMCP format
                if "unexpected keyword argument 'arguments'" in error_str and is_fastmcp:
                    print(f"DEBUG: Standard format failed, trying FastMCP format (keyword args) for get_device_rack_location...", file=sys.stderr, flush=True)
                    tool_result_list = await asyncio.wait_for(
                        client_or_session.call_tool("get_device_rack_location", **tool_arguments),
                        timeout=60.0
                    )
                    # Convert FastMCP result to standard format
                    if tool_result_list and len(tool_result_list) > 0:
                        class FastMCPToolResult:
                            def __init__(self, results):
                                self.content = []
                                for r in results:
                                    text = r.text if hasattr(r, 'text') else str(r)
                                    self.content.append(type('Content', (), {'text': text})())
                        tool_result = FastMCPToolResult(tool_result_list)
                        print(f"DEBUG: FastMCP format succeeded for get_device_rack_location", file=sys.stderr, flush=True)
                    else:
                        tool_result = None
                else:
                    # Re-raise if it's a different error
                    print(f"DEBUG: Error calling get_device_rack_location: {error_str}", file=sys.stderr, flush=True)
                    raise
            
            if tool_result and tool_result.content:
                result_text = tool_result.content[0].text
                print(f"DEBUG: Raw result text length: {len(result_text)}", file=sys.stderr, flush=True)
                print(f"DEBUG: Raw result text preview: {result_text[:500]}", file=sys.stderr, flush=True)
                try:
                    result = json.loads(result_text)
                    print(f"DEBUG: Rack location result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}", file=sys.stderr, flush=True)
                    if isinstance(result, dict) and "ai_analysis" in result:
                        print(f"DEBUG: AI analysis found in result!", file=sys.stderr, flush=True)
                        print(f"DEBUG: AI analysis type: {type(result['ai_analysis'])}, content: {str(result['ai_analysis'])[:500]}", file=sys.stderr, flush=True)
                    else:
                        print(f"DEBUG: No AI analysis in result. Available keys: {list(result.keys())}", file=sys.stderr, flush=True)
                    return result
                except json.JSONDecodeError:
                    return {"result": result_text}
            return None
    except asyncio.TimeoutError:
        return {"error": "Rack location lookup timed out. Please try again."}
    except Exception as e:
        import traceback
        print(f"DEBUG: Error in execute_rack_location_query: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        print(f"DEBUG: Traceback: {traceback.format_exc()}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e, "Error executing rack location lookup")}
    except BaseException as e:
        print(f"DEBUG: Error in execute_rack_location_query (base): {str(e)}", file=sys.stderr, flush=True)
        return {"error": _format_tool_error(e, "Error executing rack location lookup")}
