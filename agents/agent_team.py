"""
Agent team for Atlas troubleshooting.

A supervisor makes a single planning call to decide which specialist agents
to invoke (and which can run in parallel), then executes them and returns
the combined outputs.

Agents
------
path_agent     — network path tracing (DB / NetBrain)
evidence_agent — ServiceNow incidents/changes + Splunk logs
device_agent   — interface error counters, device inventory
security_agent — Panorama firewall policy checks

The supervisor only activates the agents relevant to the query; for example
a pure ServiceNow lookup skips path_agent and device_agent entirely.
"""

import asyncio
import json
import logging
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

logger = logging.getLogger("atlas.agent_team")


def _parse_evidence_hint(evidence_output: str) -> str:
    """
    Extract structured signals from evidence_agent output to guide device_agent.
    Returns a JSON string hint, or "" if nothing useful found.
    """
    incidents = list(dict.fromkeys(re.findall(r'\bINC\d{6,}\b', evidence_output, re.IGNORECASE)))
    devices = list(dict.fromkeys(re.findall(r'\b([a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)*\d+)\b', evidence_output)))
    devices = [d for d in devices if len(d) >= 4 and not d.isdigit()][:6]
    interfaces = list(dict.fromkeys(re.findall(
        r'\b(?:Ethernet|GigabitEthernet|Management)\d+(?:/\d+)*\b',
        evidence_output, re.IGNORECASE
    )))[:6]

    if not devices and not incidents:
        return ""

    hint: dict = {}
    if incidents:
        hint["open_incidents"] = incidents[:4]
    if devices:
        hint["focus_devices"] = devices
    if interfaces:
        hint["flagged_interfaces"] = interfaces
    return json.dumps(hint)

# ---------------------------------------------------------------------------
# Agent registry
# ---------------------------------------------------------------------------

AGENT_REGISTRY: dict[str, dict] = {
    "path_agent": {
        "role": "Network Path Agent",
        "description": "Traces forward path between IPs, identifies hops, interfaces, and routing",
        "skill": (
            "You are a network path tracing specialist. "
            "Trace the hop-by-hop network path between the source and destination. "
            "Use the path tracing tools to discover all devices and interfaces in the path."
        ),
    },
    "evidence_agent": {
        "role": "Operations Evidence Agent",
        "description": "Searches ServiceNow for incidents/changes related to a device or incident number",
        "skill": (
            "You are an operations evidence specialist. "
            "If the query contains an incident number (e.g. INC0010035), look it up directly using get_servicenow_incident — do NOT pass the INC number as a device name. "
            "Otherwise search for incidents and changes by device name using hours_back=720 (30 days). "
            "Do not call Splunk — it requires IP addresses."
        ),
    },
    "device_agent": {
        "role": "Device Diagnostics Agent",
        "description": "Checks interface error counters and device inventory",
        "skill": (
            "You are a device diagnostics specialist. "
            "Check interface error counters on path devices to detect CRC errors, "
            "input/output drops, or other signal-layer problems."
        ),
    },
    "security_agent": {
        "role": "Security Policy Agent",
        "description": "Checks Panorama firewall policies between source and destination",
        "skill": (
            "You are a firewall policy specialist. "
            "Check Panorama to identify security policies affecting traffic between "
            "the source and destination IPs. Look for permit or deny rules."
        ),
    },
}

AgentName = Literal["path_agent", "evidence_agent", "device_agent", "security_agent"]

# ---------------------------------------------------------------------------
# Supervisor plan
# ---------------------------------------------------------------------------

_SUPERVISOR_PROMPT = """\
You are a network troubleshooting coordinator. Given the user's query, decide
which specialist agents to run.

Available agents:
{agent_descriptions}

Return parallel_groups: a list of agent groups to execute in order.
Agents within the same group run in parallel; groups run sequentially.

Rules:
- path_agent: ONLY include when both a source AND destination IP address are explicitly mentioned
- evidence_agent: almost always useful for troubleshooting
- device_agent: include when checking interface errors, device health, or a specific device is named
- security_agent: only when firewall/policy/Panorama is relevant

Example for connectivity troubleshoot (has src+dst IPs):
  parallel_groups: [["path_agent", "evidence_agent"], ["device_agent"]]

Example for device health query ("what's going on with arista1?", "is arista1 healthy?"):
  parallel_groups: [["evidence_agent", "device_agent"]]

Example for policy lookup:
  parallel_groups: [["security_agent"]]
"""


class TeamPlan(BaseModel):
    parallel_groups: list[list[AgentName]]
    reasoning: str


async def _plan(query: str, llm, available_agents: list[str]) -> TeamPlan:
    """Ask the supervisor LLM to plan which agents to run."""
    descs = "\n".join(
        f"  {name}: {AGENT_REGISTRY[name]['description']}"
        for name in available_agents
    )
    try:
        plan: TeamPlan = await llm.with_structured_output(TeamPlan).ainvoke([
            SystemMessage(content=_SUPERVISOR_PROMPT.format(agent_descriptions=descs)),
            HumanMessage(content=query),
        ])
        logger.info("Team plan: %s | reasoning: %s", plan.parallel_groups, plan.reasoning)
        return plan
    except Exception as exc:
        logger.warning("Supervisor planning failed (%s) — running default plan", exc)
        # Sensible default: path + evidence in parallel
        return TeamPlan(
            parallel_groups=[["path_agent", "evidence_agent"], ["device_agent"]],
            reasoning="fallback",
        )


# ---------------------------------------------------------------------------
# Sub-agent execution
# ---------------------------------------------------------------------------

async def _run_sub_agent(
    agent_name: str,
    query: str,
    context: str,
    tools: list,
    llm,
) -> str:
    """Run one specialist agent via the shared agent loop."""
    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    defn = AGENT_REGISTRY[agent_name]
    system = defn["skill"]
    if context.strip():
        system += f"\n\nFindings from earlier agents:\n{context}"

    task = query
    logger.info("Running %s", agent_name)
    try:
        result = await run_agent_loop(task=task, system_prompt=system, tools=tools)
        return f"## {defn['role']}\n\n{result}"
    except Exception as exc:
        logger.warning("%s failed: %s", agent_name, exc)
        return ""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_agent_team(
    query: str,
    llm,
    tools_by_agent: dict[str, list],
    session_id: str = "default",
) -> list[str]:
    """
    Run the agent team for a query.

    Parameters
    ----------
    query          : user query
    llm            : the LLM instance (already configured)
    tools_by_agent : dict mapping agent_name → list of LangChain tools
    session_id     : for status bus push

    Returns
    -------
    List of markdown text outputs from each agent that ran.
    """
    try:
        try:
            import atlas.status_bus as status_bus
        except ImportError:
            import status_bus  # type: ignore
        await status_bus.push(session_id, "Planning agent team...")
    except Exception:
        pass

    available = [name for name in AGENT_REGISTRY if name in tools_by_agent and tools_by_agent[name]]
    plan = await _plan(query, llm, available)

    all_outputs: list[str] = []
    context = ""
    evidence_output = ""  # track evidence_agent output for structured handoff

    for group in plan.parallel_groups:
        valid_group = [a for a in group if a in tools_by_agent and tools_by_agent[a]]
        if not valid_group:
            continue

        # Push status for this group
        labels = " | ".join(AGENT_REGISTRY[a]["role"] for a in valid_group)
        try:
            await status_bus.push(session_id, labels + "...")
        except Exception:
            pass

        def _context_for(agent_name: str) -> str:
            """Give device_agent a structured hint from evidence findings."""
            if agent_name == "device_agent" and evidence_output:
                hint = _parse_evidence_hint(evidence_output)
                if hint:
                    return context + f"\n\nStructured evidence hint: {hint}"
            return context

        group_results = await asyncio.gather(*[
            _run_sub_agent(a, query, _context_for(a), tools_by_agent[a], llm)
            for a in valid_group
        ])

        for agent_name, result in zip(valid_group, group_results):
            if result.strip():
                all_outputs.append(result)
                context += result + "\n\n"
                if agent_name == "evidence_agent":
                    evidence_output = result

    return all_outputs
