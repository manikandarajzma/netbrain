"""Declarative registry for Atlas agent specifications."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.tools.tool_registry import tool_registry
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from tools.tool_registry import tool_registry  # type: ignore


_SKILLS_DIR = Path(__file__).parent.parent / "skills"
_EXPLICIT_CI_PATTERN = r"\b(?:ci\s*name|configuration\s*item|cmdb[_\s-]*ci)\s*:\s*\S+"


@dataclass(frozen=True)
class ToolProfileRule:
    """Declarative override for which tool profile an agent should use."""

    profile_name: str
    scenarios: frozenset[str] = field(default_factory=frozenset)
    prompt_regex: str | None = None

    def matches(self, prompt: str, scenario: str) -> bool:
        scenario_name = str(scenario or "").strip().lower()
        if self.scenarios and scenario_name not in self.scenarios:
            return False
        if self.prompt_regex and not re.search(self.prompt_regex, prompt or "", re.IGNORECASE):
            return False
        return True


@dataclass(frozen=True)
class AgentSpec:
    """Declarative description of one Atlas runtime agent."""

    name: str
    route_key: str
    description: str
    llm_role: str
    workflow_type: str
    workflow_owner: str
    default_tool_profile: str
    core_prompt_path: Path
    default_scenario: str = "general"
    scenario_dir: Path | None = None
    scenario_files: dict[str, str] = field(default_factory=dict)
    scenario_descriptions: dict[str, str] = field(default_factory=dict)
    scenario_guidance: tuple[str, ...] = ()
    scenario_selector_owner: str | None = None
    tool_profile_rules: tuple[ToolProfileRule, ...] = ()

    def resolve_scenario_path(self, scenario: str) -> Path | None:
        fname = self.scenario_files.get(str(scenario or "").strip())
        if not fname or self.scenario_dir is None:
            return None
        path = self.scenario_dir / fname
        return path if path.exists() else None

    def load_system_prompt(self, scenario: str | None = None) -> str:
        scenario_name = str(scenario or self.default_scenario).strip() or self.default_scenario
        core = self.core_prompt_path.read_text(encoding="utf-8").strip() if self.core_prompt_path.exists() else ""
        scenario_path = self.resolve_scenario_path(scenario_name)
        if scenario_path:
            scenario_text = scenario_path.read_text(encoding="utf-8").strip()
            return core + "\n\n---\n\n" + scenario_text
        return core

    def resolve_tool_profile(self, prompt: str = "", scenario: str | None = None) -> str:
        scenario_name = str(scenario or self.default_scenario).strip() or self.default_scenario
        for rule in self.tool_profile_rules:
            if rule.matches(prompt, scenario_name):
                return rule.profile_name
        return self.default_tool_profile

    def resolve_tools(self, prompt: str = "", scenario: str | None = None) -> tuple[Any, ...]:
        return tool_registry.get_profile_tools(self.resolve_tool_profile(prompt, scenario))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "route_key": self.route_key,
            "description": self.description,
            "llm_role": self.llm_role,
            "workflow_type": self.workflow_type,
            "workflow_owner": self.workflow_owner,
            "scenario_selector_owner": self.scenario_selector_owner,
            "default_scenario": self.default_scenario,
            "default_tool_profile": self.default_tool_profile,
            "core_prompt_path": str(self.core_prompt_path),
            "scenarios": sorted(self.scenario_files.keys()),
            "scenario_descriptions": dict(sorted(self.scenario_descriptions.items())),
            "scenario_guidance": list(self.scenario_guidance),
            "tool_profile_rules": [
                {
                    "profile_name": rule.profile_name,
                    "scenarios": sorted(rule.scenarios),
                    "prompt_regex": rule.prompt_regex,
                }
                for rule in self.tool_profile_rules
            ],
        }


class AgentRegistry:
    """Owns declarative runtime agent specifications."""

    def __init__(self) -> None:
        troubleshoot_spec = AgentSpec(
            name="troubleshoot",
            route_key="troubleshoot",
            description="Connectivity, protocol, and device-health investigation agent.",
            llm_role="troubleshoot",
            workflow_type="troubleshoot",
            workflow_owner="TroubleshootWorkflowService",
            scenario_selector_owner="TroubleshootScenarioService",
            default_scenario="general",
            default_tool_profile="troubleshoot.general",
            core_prompt_path=_SKILLS_DIR / "troubleshooter.md",
            scenario_dir=_SKILLS_DIR / "troubleshooting_scenarios",
            scenario_files={
                "connectivity": "connectivity.md",
                "performance": "performance.md",
                "intermittent": "intermittent.md",
            },
            scenario_descriptions={
                "connectivity": "blocked, denied, unreachable, no route, connection refused, endpoint-to-endpoint path problems",
                "performance": "slow, degraded, high latency, throughput issues",
                "intermittent": "flapping, sporadic, unstable, works sometimes and fails sometimes",
                "general": "everything else",
            },
            scenario_guidance=(
                "If the request is clearly endpoint-to-endpoint connectivity or port reachability, choose connectivity.",
                "If the request is clearly about slowness, latency, or throughput degradation, choose performance.",
                "If the request is clearly unstable or intermittent over time, choose intermittent.",
                "If it is a device, protocol, or network issue that does not clearly fit the above, choose general.",
            ),
            tool_profile_rules=(
                ToolProfileRule(
                    profile_name="troubleshoot.connectivity",
                    scenarios=frozenset({"connectivity"}),
                ),
            ),
        )
        network_ops_spec = AgentSpec(
            name="network_ops",
            route_key="network_ops",
            description="ServiceNow incident/change and operational request agent.",
            llm_role="network_ops",
            workflow_type="network_ops",
            workflow_owner="NetworkOpsWorkflowService",
            scenario_selector_owner="NetworkOpsScenarioService",
            default_scenario="general",
            default_tool_profile="network_ops",
            core_prompt_path=_SKILLS_DIR / "network_ops.md",
            scenario_dir=_SKILLS_DIR / "network_ops_scenarios",
            scenario_files={
                "incident_record": "incident_record.md",
                "record_lookup": "record_lookup.md",
                "change_record": "change_record.md",
                "change_update": "change_update.md",
                "access_change": "access_change.md",
            },
            scenario_descriptions={
                "incident_record": "create/open/raise a new incident or ticket",
                "record_lookup": "fetch/show/get the details or status of an existing incident or change record",
                "change_record": "create/open/submit a generic ServiceNow change request",
                "change_update": "update/close/reassign/add work notes to an existing change request",
                "access_change": "open a port, allow/deny traffic, whitelist access, or document a firewall/security rule change",
                "general": "another operational request that does not clearly fit the above",
            },
            scenario_guidance=(
                "If the user wants a new incident or ticket created, choose incident_record.",
                "If the user references an existing INC or CHG and wants status/details, choose record_lookup.",
                "If the user wants to create a generic change request and is not describing a firewall, rule, or port access change, choose change_record.",
                "If the user wants to modify or close an existing change request, choose change_update.",
                "If the user is asking for access, whitelisting, opening ports, or rule/policy changes, choose access_change.",
            ),
            tool_profile_rules=(
                ToolProfileRule(
                    profile_name="network_ops.no_path",
                    scenarios=frozenset({"record_lookup", "change_update"}),
                ),
                ToolProfileRule(
                    profile_name="network_ops.no_path",
                    scenarios=frozenset({"incident_record", "change_record"}),
                    prompt_regex=_EXPLICIT_CI_PATTERN,
                ),
            ),
        )
        self._specs: dict[str, AgentSpec] = {
            troubleshoot_spec.name: troubleshoot_spec,
            network_ops_spec.name: network_ops_spec,
        }

    def get(self, agent_name: str) -> AgentSpec:
        try:
            return self._specs[agent_name]
        except KeyError as exc:
            raise KeyError(f"Unknown agent spec: {agent_name}") from exc

    def describe(self) -> dict[str, dict[str, Any]]:
        return {
            name: spec.describe()
            for name, spec in sorted(self._specs.items())
        }

    def route_descriptions(self) -> dict[str, str]:
        return {
            spec.route_key: spec.description
            for spec in self._specs.values()
        }

    def valid_route_keys(self) -> set[str]:
        return {spec.route_key for spec in self._specs.values()}


def _llm_for_role(spec: AgentSpec, *, factory=agent_factory):
    builders = {
        "troubleshoot": factory.build_troubleshoot_llm,
        "network_ops": factory.build_network_ops_llm,
    }
    try:
        return builders[spec.llm_role]()
    except KeyError as exc:
        raise KeyError(f"Unknown llm_role for agent spec '{spec.name}': {spec.llm_role}") from exc


def build_agent_from_spec(
    spec: AgentSpec,
    *,
    prompt: str = "",
    scenario: str | None = None,
    llm=None,
    factory=agent_factory,
):
    """Instantiate a request-scoped runtime agent from a declarative spec."""
    runtime_llm = llm or _llm_for_role(spec, factory=factory)
    scenario_name = str(scenario or spec.default_scenario).strip() or spec.default_scenario
    tools = spec.resolve_tools(prompt, scenario_name)
    system_prompt = spec.load_system_prompt(scenario_name)
    return factory.create_specialized_agent(runtime_llm, tools, system_prompt, spec.route_key)


agent_registry = AgentRegistry()
