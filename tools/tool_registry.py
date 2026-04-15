"""Owned tool registry for Atlas agents."""
from __future__ import annotations

try:
    from atlas.tools import all_tools as _all_tools
except ImportError:
    from tools import all_tools as _all_tools  # type: ignore


class ToolRegistry:
    """Owns the shared tool collections exposed to Atlas agents."""

    def __init__(self) -> None:
        self._all_tools = _all_tools.ALL_TOOLS
        self._connectivity_tools = _all_tools.CONNECTIVITY_TOOLS
        self._network_ops_tools = _all_tools.NETWORK_OPS_TOOLS

    def get_all_tools(self):
        return self._all_tools

    def get_connectivity_tools(self):
        return self._connectivity_tools

    def get_network_ops_tools(self):
        return self._network_ops_tools


tool_registry = ToolRegistry()
