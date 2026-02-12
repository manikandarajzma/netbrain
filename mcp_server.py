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


if __name__ == "__main__":
    # Redirect stderr to a log file for easier debugging
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, "mcp_server.log")
    log_file = open(log_file_path, "a", encoding="utf-8")

    class TeeStderr:
        def __init__(self, file, stderr):
            self.file = file
            self.stderr = stderr
        def write(self, text):
            self.stderr.write(text)
            self.file.write(text)
            self.file.flush()
        def flush(self):
            self.stderr.flush()
            self.file.flush()

    sys.stderr = TeeStderr(log_file, sys.__stderr__)
    print(f"DEBUG: Server logs will be written to: {log_file_path}", file=sys.__stderr__, flush=True)

    # Run the MCP server using streamable-http transport
    mcp.run(transport="streamable-http", port=MCP_SERVER_PORT, host=MCP_SERVER_HOST)
