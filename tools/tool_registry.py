"""Capability-based tool registry for Atlas agents."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

try:
    from atlas.tools import connectivity_agent_tools as _connectivity_tools
    from atlas.tools import device_agent_tools as _device_tools
    from atlas.tools import knowledge_agent_tools as _knowledge_tools
    from atlas.tools import memory_agent_tools as _memory_tools
    from atlas.tools import path_agent_tools as _path_tools
    from atlas.tools import routing_agent_tools as _routing_tools
    from atlas.tools import servicenow_agent_tools as _servicenow_tools
    from atlas.tools import servicenow_workflow_tools as _servicenow_workflow_tools
except ImportError:
    from tools import connectivity_agent_tools as _connectivity_tools  # type: ignore
    from tools import device_agent_tools as _device_tools  # type: ignore
    from tools import knowledge_agent_tools as _knowledge_tools  # type: ignore
    from tools import memory_agent_tools as _memory_tools  # type: ignore
    from tools import path_agent_tools as _path_tools  # type: ignore
    from tools import routing_agent_tools as _routing_tools  # type: ignore
    from tools import servicenow_agent_tools as _servicenow_tools  # type: ignore
    from tools import servicenow_workflow_tools as _servicenow_workflow_tools  # type: ignore


Capability = str
ProfileName = str


@dataclass(frozen=True)
class RegisteredTool:
    """Single agent-facing tool plus the capabilities it satisfies."""

    tool: Any
    capabilities: frozenset[Capability] = field(default_factory=frozenset)


class ToolRegistry:
    """
    Owns tool registration and profile-to-capability resolution.

    Agent builders should ask for profile tools, not hard-code tool lists.
    New tools are added by registering capabilities here.
    New agents are added by declaring a capability profile here.
    """

    def __init__(self) -> None:
        self._registry: list[RegisteredTool] = []
        self._profiles: dict[ProfileName, tuple[Capability, ...]] = {}
        self._profile_cache: dict[ProfileName, tuple[Any, ...]] = {}
        self._register_default_tools()
        self._register_default_profiles()

    def register_tool(self, tool: Any, *capabilities: Capability) -> None:
        self._registry.append(RegisteredTool(tool=tool, capabilities=frozenset(capabilities)))
        self._profile_cache.clear()

    def register_profile(self, profile_name: ProfileName, capabilities: Iterable[Capability]) -> None:
        self._profiles[profile_name] = tuple(capabilities)
        self._profile_cache.pop(profile_name, None)

    def get_tools_for_capabilities(self, capabilities: Iterable[Capability]) -> tuple[Any, ...]:
        requested = set(capabilities)
        resolved: list[Any] = []
        seen: set[int] = set()
        for entry in self._registry:
            if not (entry.capabilities & requested):
                continue
            tool_id = id(entry.tool)
            if tool_id in seen:
                continue
            seen.add(tool_id)
            resolved.append(entry.tool)
        return tuple(resolved)

    def get_profile_tools(self, profile_name: ProfileName) -> tuple[Any, ...]:
        if profile_name in self._profile_cache:
            return self._profile_cache[profile_name]
        capabilities = self._profiles.get(profile_name)
        if capabilities is None:
            raise KeyError(f"Unknown tool profile: {profile_name}")
        tools = self.get_tools_for_capabilities(capabilities)
        self._profile_cache[profile_name] = tools
        return tools

    def describe_profiles(self) -> dict[ProfileName, list[Capability]]:
        return {
            profile_name: list(capabilities)
            for profile_name, capabilities in sorted(self._profiles.items())
        }

    def describe_registered_tools(self) -> list[dict[str, Any]]:
        described: list[dict[str, Any]] = []
        for entry in self._registry:
            callable_obj = getattr(entry.tool, "func", None) or getattr(entry.tool, "coroutine", None)
            module_name = getattr(callable_obj, "__module__", getattr(entry.tool, "__module__", ""))
            tool_name = getattr(entry.tool, "name", getattr(entry.tool, "__name__", type(entry.tool).__name__))
            described.append(
                {
                    "name": tool_name,
                    "module": module_name,
                    "capabilities": sorted(entry.capabilities),
                }
            )
        return sorted(described, key=lambda item: (item["module"], item["name"]))

    def get_all_tools(self):
        return self.get_profile_tools("troubleshoot.general")

    def get_connectivity_tools(self):
        return self.get_profile_tools("troubleshoot.connectivity")

    def get_network_ops_tools(self):
        return self.get_profile_tools("network_ops")

    def _register_default_tools(self) -> None:
        for tool, capabilities in _path_tools.PATH_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)
        for tool, capabilities in _device_tools.DEVICE_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)
        for tool, capabilities in _routing_tools.ROUTING_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)
        for tool, capabilities in _connectivity_tools.CONNECTIVITY_WORKFLOW_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)
        for tool, capabilities in _servicenow_workflow_tools.SERVICENOW_WORKFLOW_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)
        for tool, capabilities in _knowledge_tools.KNOWLEDGE_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)
        for tool, capabilities in _memory_tools.MEMORY_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)
        for tool, capabilities in _servicenow_tools.SERVICENOW_TOOL_CAPABILITIES:
            self.register_tool(tool, *capabilities)

    def _register_default_profiles(self) -> None:
        self.register_profile(
            "troubleshoot.general",
            (
                "workflow.path.trace",
                "workflow.path.reverse_trace",
                "workflow.connectivity.ping",
                "workflow.connectivity.tcp_test",
                "workflow.routing.check",
                "workflow.interfaces.counters",
                "workflow.interfaces.detail",
                "workflow.interfaces.inventory",
                "workflow.device.syslog",
                "workflow.ospf.peering.inspect",
                "workflow.connectivity.snapshot",
                "workflow.ospf.neighbors",
                "workflow.ospf.interfaces",
                "workflow.ospf.history",
                "workflow.routing.history",
                "servicenow.search",
                "servicenow.incident.read",
                "memory.recall",
                "knowledge.vendor.lookup",
            ),
        )
        self.register_profile(
            "troubleshoot.connectivity",
            (
                "workflow.path.trace",
                "workflow.path.reverse_trace",
                "workflow.connectivity.ping",
                "workflow.routing.check",
                "workflow.connectivity.snapshot",
                "workflow.routing.history",
                "servicenow.search",
                "servicenow.incident.read",
                "knowledge.vendor.lookup",
            ),
        )
        self.register_profile(
            "network_ops",
            (
                "workflow.path.trace",
                "servicenow.search",
                "servicenow.incident.read",
                "servicenow.change.read",
                "servicenow.incident.create",
                "servicenow.change.create",
                "servicenow.change.update",
            ),
        )


tool_registry = ToolRegistry()
