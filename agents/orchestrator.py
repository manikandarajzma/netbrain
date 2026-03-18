"""
Risk Orchestrator — fans out to Panorama + Splunk A2A agents in parallel,
then uses Ollama to synthesize a risk assessment from their responses.
"""
import asyncio
import logging
import pathlib
import re
import uuid
from typing import Any

import httpx

logger = logging.getLogger("atlas.agents.orchestrator")

PANORAMA_AGENT_URL = "http://localhost:8003"
SPLUNK_AGENT_URL   = "http://localhost:8002"
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b")


# ---------------------------------------------------------------------------
# A2A client helper
# ---------------------------------------------------------------------------

async def _call_agent(url: str, task_text: str, timeout: float = 120.0) -> str | None:
    """Send a natural language task to a specialist agent and return its text response."""
    task = {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": task_text}],
        },
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, json=task)
            response.raise_for_status()
            data = response.json()
            artifacts = data.get("artifacts", [])
            if not artifacts:
                return None
            return next(
                (p.get("text") for p in artifacts[0].get("parts", []) if p.get("type") == "text"),
                None,
            )
    except Exception as e:
        logger.warning("Agent call to %s failed: %s", url, e)
    return None


# ---------------------------------------------------------------------------
# Synthesis via Ollama
# ---------------------------------------------------------------------------

async def _synthesize(ip: str, panorama: str | None, splunk: str | None, prompt: str) -> str:
    """Call Ollama to produce a risk assessment from the agent summaries."""
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    skill_path = pathlib.Path(__file__).parent.parent / "skills" / "risk_synthesis.md"
    skill = skill_path.read_text(encoding="utf-8").strip() if skill_path.exists() else ""

    panorama_text = panorama or "No Panorama data available."
    splunk_text   = splunk   or "No Splunk data available."

    user_content = (
        f"User query: {prompt}\n\n"
        f"IP address: {ip}\n\n"
        f"--- Panorama Agent Summary ---\n{panorama_text}\n\n"
        f"--- Splunk Agent Summary ---\n{splunk_text}"
    )

    llm = ChatOpenAI(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.0, api_key="docker")
    messages = []
    if skill:
        messages.append(SystemMessage(content=skill))
    messages.append(HumanMessage(content=user_content))

    response = await llm.ainvoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

async def orchestrate_ip_risk(
    prompt: str,
    username: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """
    Extract IP from prompt, fan out to Panorama + Splunk agents in parallel
    with descriptive task strings, then synthesize a risk assessment.
    """
    ip_match = _IP_RE.search(prompt)
    if not ip_match:
        return {"role": "assistant", "content": "No IP address found in your query."}

    ip = ip_match.group(0)
    logger.info("Orchestrator: risk assessment for %s", ip)

    panorama_task = (
        f"Assess the Panorama security posture for IP {ip}. "
        f"Find which address group it belongs to, list the group members, "
        f"and show all referencing security policies."
    )
    splunk_task = (
        f"Analyze Splunk firewall data for IP {ip}. "
        f"Get recent deny events, a traffic summary broken down by action, "
        f"and destination spread (unique destination IPs and ports)."
    )

    try:
        import atlas.status_bus as status_bus
        await status_bus.push(session_id or "default", "Checking Panorama address groups and policies...")
        await status_bus.push(session_id or "default", "Querying Splunk traffic data...")
    except Exception:
        pass

    panorama_result, splunk_result = await asyncio.gather(
        _call_agent(PANORAMA_AGENT_URL, panorama_task, timeout=120.0),
        _call_agent(SPLUNK_AGENT_URL,   splunk_task,   timeout=120.0),
        return_exceptions=True,
    )

    if isinstance(panorama_result, Exception):
        logger.warning("Panorama agent exception: %s", panorama_result)
        panorama_result = None
    if isinstance(splunk_result, Exception):
        logger.warning("Splunk agent exception: %s", splunk_result)
        splunk_result = None

    try:
        import atlas.status_bus as status_bus
        await status_bus.push(session_id or "default", "Synthesizing risk assessment...")
    except Exception:
        pass

    synthesis = await _synthesize(ip, panorama_result, splunk_result, prompt)

    return {
        "role": "assistant",
        "content": {
            "direct_answer": synthesis,
        },
    }
