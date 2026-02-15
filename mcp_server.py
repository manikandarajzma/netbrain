"""
MCP Server for NetBrain Network Query – Entry Point.

This module imports domain tool modules to register all MCP tools on the
shared FastMCP instance, then starts the server.

Domain modules:
  - tools.splunk_tools   : get_splunk_recent_denies
  - tools.netbox_tools   : get_rack_details, list_racks, get_device_rack_location
  - tools.panorama_tools : query_panorama_ip_object_group, query_panorama_address_group_members
  - tools.netbrain_tools : query_network_path, check_path_allowed
"""

import logging
import os
import sys

# Ensure the directory containing this file is on sys.path so that
# ``import tools.shared`` and sibling imports work regardless of cwd.
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# Import the shared FastMCP instance and config
from tools.shared import mcp, MCP_SERVER_HOST, MCP_SERVER_PORT

# Import domain modules – the act of importing triggers @mcp.tool() registration
import tools.splunk_tools      # noqa: F401
import tools.netbox_tools      # noqa: F401
import tools.panorama_tools    # noqa: F401
import tools.netbrain_tools    # noqa: F401

logger = logging.getLogger("netbrain.server")

# ---------------------------------------------------------------------------
# Health check endpoint (available at GET /health on the MCP HTTP server)
# ---------------------------------------------------------------------------
from starlette.requests import Request
from starlette.responses import JSONResponse


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Return server status and registered tool count."""
    tools = await mcp.get_tools()
    return JSONResponse({
        "status": "ok",
        "server": mcp.name,
        "tools_registered": len(tools),
    })


if __name__ == "__main__":
    # Add a file handler so all log output also goes to mcp_server.log
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, "mcp_server.log")
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(file_handler)
    logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Server logs will be written to: %s", log_file_path)

    # Run the MCP server using streamable-http transport
    mcp.run(transport="streamable-http", port=MCP_SERVER_PORT, host=MCP_SERVER_HOST)
