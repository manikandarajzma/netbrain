"""
Troubleshoot Orchestrator — diagnoses connectivity problems between two endpoints.

Uses a LangGraph ReAct agent where the LLM reasons at each step before deciding
which specialist agent to call next based on what the path and prior findings reveal.
"""
import logging
import pathlib
import uuid

import httpx
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

logger = logging.getLogger("atlas.agents.troubleshoot")

NETBRAIN_AGENT_URL  = "http://localhost:8004"
PANORAMA_AGENT_URL  = "http://localhost:8003"
SPLUNK_AGENT_URL    = "http://localhost:8002"

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "troubleshoot_orchestrator.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent caller helper
# ---------------------------------------------------------------------------

async def _call_agent(url: str, task: str, timeout: float = 180.0) -> str:
    payload = {
        "id": str(uuid.uuid4()),
        "message": {"role": "user", "parts": [{"type": "text", "text": task}]},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            artifacts = data.get("artifacts", [])
            if artifacts:
                text = next(
                    (p.get("text") for p in artifacts[0].get("parts", []) if p.get("type") == "text"),
                    None,
                )
                if text:
                    return text
        return "Agent returned no data."
    except Exception as e:
        logger.warning("Agent call to %s failed: %s", url, e)
        return f"Agent unavailable: {e}"


# ---------------------------------------------------------------------------
# Tools — each specialist agent exposed as a tool to the ReAct orchestrator
# ---------------------------------------------------------------------------

_session_id: str | None = None


@tool
async def call_netbrain_agent(task: str) -> str:
    """
    Trace the hop-by-hop network path between two IP addresses and check if traffic is allowed.
    Use this first for any path troubleshooting query.
    Pass a natural language task describing the source and destination IPs.
    Example: "Trace the path from 10.0.0.1 to 10.0.1.1 and check if traffic is allowed."
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Tracing network path with NetBrain...")
    except Exception:
        pass
    return await _call_agent(NETBRAIN_AGENT_URL, task)


@tool
async def call_panorama_agent(task: str) -> str:
    """
    Check whether Panorama security policies permit or deny traffic between two IPs.
    Pass the source and destination IPs. The agent will return the verdict (allowed/denied/unknown) and the exact policy name that matched.
    Example: "Check if traffic from 10.0.0.1 to 11.0.0.1 is allowed or denied in Panorama security policies."
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Checking Panorama security policies...")
    except Exception:
        pass
    return await _call_agent(PANORAMA_AGENT_URL, task)


@tool
async def call_splunk_agent(task: str) -> str:
    """
    Check Splunk firewall logs for recent deny events and traffic patterns for an IP address.
    Use when you need to correlate path findings with actual observed traffic — deny counts, blocked ports, destination spread.
    Pass a natural language task describing the IP and what to look for.
    Example: "Check Splunk for recent deny events and traffic summary for 10.0.0.1 in the last 24 hours."
    """
    try:
        import atlas.status_bus as status_bus
        await status_bus.push(_session_id or "default", "Querying Splunk for deny events...")
    except Exception:
        pass
    return await _call_agent(SPLUNK_AGENT_URL, task)


@tool
async def call_cisco_agent(task: str) -> str:
    """
    Check interface errors, drops, and hardware faults on a Cisco device.
    Use when a Cisco switch or router in the path has suspected interface issues.
    Pass a natural language task describing the device and interfaces to check.
    Example: "Check interface errors on SW-EDGE-02 Gi0/1 and Gi0/2."
    NOTE: This agent is not yet available. Return a note that Cisco agent is pending implementation.
    """
    logger.info("Cisco agent called (not yet implemented): %s", task)
    return "Cisco agent is not yet available. Interface-level diagnostics for Cisco devices are pending implementation."


TROUBLESHOOT_TOOLS = [
    call_netbrain_agent,
    call_panorama_agent,
    call_splunk_agent,
    call_cisco_agent,
]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def orchestrate_troubleshoot(
    prompt: str,
    username: str | None = None,
    session_id: str | None = None,
) -> dict:
    global _session_id
    _session_id = session_id
    """
    Run the troubleshoot ReAct agent.
    The LLM reasons at each step before deciding which specialist agent to call next.
    Returns a structured troubleshooting report.
    """
    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    logger.info("Troubleshoot ReAct agent: %s (user=%s)", prompt, username)

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )

    agent = create_react_agent(
        model=llm,
        tools=TROUBLESHOOT_TOOLS,
        prompt=SystemMessage(content=_load_skill()),
    )

    result = await agent.ainvoke(
        {"messages": [("user", prompt)]},
        config={"recursion_limit": 25},
    )

    # Extract the final assistant message
    messages = result.get("messages", [])
    final = next(
        (m.content for m in reversed(messages) if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)),
        "Investigation complete — no summary generated.",
    )

    return {
        "role": "assistant",
        "content": {"direct_answer": final},
    }
