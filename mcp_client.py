"""
MCP Client for Atlas network query.

This module provides the MCP client API used by the FastAPI chat service:
- get_mcp_session: context manager for MCP HTTP or stdio connections
- call_mcp_tool: call any MCP tool by name, handles both transport conventions

Use the FastAPI app for the chat UI: uv run python -m atlas.app
"""

# Import asyncio for handling asynchronous operations (needed for MCP client)
import asyncio
import logging
import sys
import warnings

logger = logging.getLogger("atlas.mcp_client")

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
    logger.debug("FastMCP Client not available, will use stdio transport")

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
            logger.debug(f"Connecting to MCP server via HTTP at {server_url}...")
            # FastMCP Client automatically infers streamable-http from URL
            # Try to configure with longer timeout for long-running requests
            try:
                # Check if FastMCPClient accepts timeout parameters
                client = FastMCPClient(server_url, timeout=600)  # 10 minute timeout
            except TypeError:
                # If timeout parameter not supported, use default
                client = FastMCPClient(server_url)
            async with client:
                logger.debug("FastMCP HTTP client connected successfully")
                yield client
        except Exception as e:
            # HTTP connection failed - this is expected if server is running in stdio mode
            # Silently fall back to stdio (don't print verbose traceback as it's expected)
            logger.debug("HTTP connection unavailable (server may be in stdio mode), using stdio transport...")
            # Fall through to stdio fallback
    
    # Fallback to stdio transport
    server_params = get_server_params()
    logger.debug("Connecting to MCP server via stdio...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            logger.debug("Stdio session initialized successfully")
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

async def call_mcp_tool(tool_name: str, tool_arguments: dict, timeout: float = 65.0):
    """
    Call any MCP tool by name and return parsed result dict.
    Handles both standard MCP (arguments=dict) and FastMCP (**kwargs) calling conventions,
    result unwrapping, and JSON parsing.
    """
    try:
        async for client_or_session in get_mcp_session():
            is_fastmcp = False
            if FASTMCP_CLIENT_AVAILABLE:
                try:
                    is_fastmcp = isinstance(client_or_session, FastMCPClient)
                except (NameError, TypeError):
                    module_name = type(client_or_session).__module__
                    is_fastmcp = "fastmcp" in (module_name or "").lower()

            try:
                result = await asyncio.wait_for(
                    client_or_session.call_tool(tool_name, arguments=tool_arguments),
                    timeout=timeout,
                )
            except TypeError as e:
                if "unexpected keyword argument 'arguments'" in str(e) and is_fastmcp:
                    result_list = await asyncio.wait_for(
                        client_or_session.call_tool(tool_name, **tool_arguments),
                        timeout=timeout,
                    )
                    if result_list:
                        class _W:
                            def __init__(self, items):
                                self.content = [
                                    type("_C", (), {"text": i.text if hasattr(i, "text") else str(i)})()
                                    for i in items
                                ]
                        result = _W(result_list)
                    else:
                        result = None
                else:
                    raise

            if result is None:
                return None

            # Unwrap MCP result into text
            if isinstance(result, list):
                first = result[0] if result else None
                result_text = first.text if first and hasattr(first, "text") else (str(first) if first else None)
            elif hasattr(result, "content") and result.content:
                item = result.content[0]
                result_text = item.text if hasattr(item, "text") else str(item)
            elif isinstance(result, dict):
                return result
            else:
                result_text = str(result)

            if not result_text:
                return None

            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                import re as _re
                m = _re.search(r"\{.*\}", result_text, _re.DOTALL)
                if m:
                    try:
                        return json.loads(m.group())
                    except json.JSONDecodeError:
                        pass
                return {"result": result_text}
    except asyncio.TimeoutError:
        return {"error": f"{tool_name} timed out. Please try again."}
    except Exception as e:
        return {"error": _format_tool_error(e, f"Error calling {tool_name}")}
    except BaseException as e:
        return {"error": _format_tool_error(e, f"Error calling {tool_name}")}




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
