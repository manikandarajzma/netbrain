"""
MCP Server for Atlas network query – Entry Point.

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

logger = logging.getLogger("atlas.server")

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
    # Send all logs to file only (no console output)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, "mcp_server.log")
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root = logging.getLogger()
    # Remove all existing handlers (console etc.) so output goes only to the file
    for h in root.handlers[:]:
        root.removeHandler(h)
    root.addHandler(file_handler)
    root.setLevel(logging.DEBUG)
    # Remove StreamHandlers from any logger that got one (e.g. from tools.shared.setup_logging)
    for name, logr in logging.Logger.manager.loggerDict.items():
        if isinstance(logr, logging.Logger):
            for h in logr.handlers[:]:
                if isinstance(h, logging.StreamHandler):
                    logr.removeHandler(h)

    logger.info("MCP server starting; logs written to %s", log_file_path)

    # Run the MCP server using streamable-http transport
    mcp.run(transport="streamable-http", port=MCP_SERVER_PORT, host=MCP_SERVER_HOST)
