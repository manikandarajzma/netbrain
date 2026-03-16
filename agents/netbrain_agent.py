"""
NetBrain A2A Agent — port 8004.

Accepts natural language path queries. Uses an LLM reasoning loop to run
path queries via NetBrain. If a Palo Alto firewall is found in the path,
calls the Panorama agent (port 8003) to get security zones and device groups.
"""
import json
import logging
import pathlib
import sys
import uuid
from pathlib import Path
from typing import Optional

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.tools import tool

logging.getLogger("atlas").setLevel(logging.INFO)

logger = logging.getLogger("atlas.agents.netbrain")

app = FastAPI(title="Atlas NetBrain Agent")

PANORAMA_AGENT_URL = "http://localhost:8003"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def netbrain_query_path(
    source: str,
    destination: str,
    protocol: str = "IP",
    port: str = "0",
) -> dict:
    """
    Trace the hop-by-hop network path between two IP addresses using NetBrain.
    Returns path hops with device names, interfaces, and firewall info.
    Use for: "find path from X to Y", "show hops between X and Y", "trace route from X to Y".
    """
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool(
        "query_network_path",
        {"source": source, "destination": destination, "protocol": protocol, "port": port},
        timeout=90.0,
    )


@tool
async def netbrain_check_allowed(
    source: str,
    destination: str,
    protocol: str,
    port: str,
) -> dict:
    """
    Check if traffic between two IP addresses is allowed or denied by firewall policy.
    Use for: "is traffic from X to Y allowed?", "can X reach Y on TCP 443?", "is path allowed?".
    """
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool(
        "check_path_allowed",
        {"source": source, "destination": destination, "protocol": protocol, "port": port},
        timeout=90.0,
    )


@tool
async def ask_panorama_agent(task: str) -> str:
    """
    Ask the Panorama agent a question about firewall zones or device groups.
    Use this when a path hop is a Palo Alto firewall and you need to know:
    - Which security zones its interfaces belong to
    - Which Panorama device group it belongs to
    Pass a clear natural language task describing the firewall name and its interfaces.
    Example: "Get security zones and device group for firewall PA-FW-01 with interfaces Ethernet1/1 and Ethernet1/2"
    """
    payload = {
        "id": str(uuid.uuid4()),
        "message": {
            "role": "user",
            "parts": [{"type": "text", "text": task}],
        },
    }
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(PANORAMA_AGENT_URL, json=payload)
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
        return "Panorama agent returned no data."
    except Exception as e:
        logger.warning("Panorama agent call failed: %s", e)
        return f"Panorama agent unavailable: {e}"


NETBRAIN_TOOLS = [netbrain_query_path, netbrain_check_allowed, ask_panorama_agent]

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "netbrain_agent.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Atlas NetBrain Agent",
    "description": (
        "Traces network paths between IP addresses using NetBrain. "
        "When a Palo Alto firewall is found in the path, queries the Panorama agent "
        "for security zones and device groups."
    ),
    "url": "http://localhost:8004",
    "version": "1.0.0",
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "network_path_query",
            "name": "Network Path Query",
            "description": "Trace the hop-by-hop path between two IPs, with Panorama enrichment for firewall hops.",
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Find path from 10.0.0.1 to 10.0.1.1",
                "Is traffic from 10.0.0.1 to 10.0.1.1 on TCP 443 allowed?",
            ],
        }
    ],
}


@app.get("/.well-known/agent.json")
async def agent_card():
    return JSONResponse(AGENT_CARD)


# ---------------------------------------------------------------------------
# A2A Task endpoint
# ---------------------------------------------------------------------------

@app.post("/")
async def handle_task(request: Request) -> JSONResponse:
    body = await request.json()
    task_id = body.get("id") or str(uuid.uuid4())
    message = body.get("message", {})
    parts = message.get("parts", [])
    text = next((p.get("text", "") for p in parts if p.get("type") == "text"), "")

    if not text:
        return _error_response(task_id, "No task text provided.")

    logger.info("NetBrain agent task: %s", text)

    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    try:
        result = await run_agent_loop(
            task=text,
            system_prompt=_load_skill(),
            tools=NETBRAIN_TOOLS,
        )
    except Exception as e:
        logger.exception("NetBrain agent loop error")
        return _error_response(task_id, f"Agent error: {e}")

    return _success_response(task_id, result)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _success_response(task_id: str, text: str) -> JSONResponse:
    return JSONResponse({
        "id": task_id,
        "status": {"state": "completed"},
        "artifacts": [{"parts": [{"type": "text", "text": text}]}],
    })


def _error_response(task_id: str, message: str) -> JSONResponse:
    return JSONResponse({
        "id": task_id,
        "status": {"state": "failed", "message": message},
        "artifacts": [],
    }, status_code=200)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    uvicorn.run("netbrain_agent:app", host="0.0.0.0", port=8004, reload=False)
