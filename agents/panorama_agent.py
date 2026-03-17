"""
Panorama A2A Agent — port 8003.

Accepts natural language A2A tasks. Uses an LLM reasoning loop to decide
which Panorama tools to call, then returns a natural language summary.
"""
import logging
import pathlib
import sys
import uuid
from pathlib import Path
from typing import Optional

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.tools import tool

logging.getLogger("atlas").setLevel(logging.INFO)

logger = logging.getLogger("atlas.agents.panorama")

app = FastAPI(title="Atlas Panorama Agent")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def panorama_ip_object_group(ip_address: str, device_group: Optional[str] = None) -> dict:
    """Find which Panorama address groups contain a given IP address. Returns group names and device groups."""
    from atlas.mcp_client import call_mcp_tool
    params = {"ip_address": ip_address}
    if device_group:
        params["device_group"] = device_group
    return await call_mcp_tool("query_panorama_ip_object_group", params, timeout=65.0)


@tool
async def panorama_address_group_members(address_group_name: str, device_group: Optional[str] = None) -> dict:
    """Get members and referencing security policies for a Panorama address group."""
    from atlas.mcp_client import call_mcp_tool
    params = {"address_group_name": address_group_name}
    if device_group:
        params["device_group"] = device_group
    return await call_mcp_tool("query_panorama_address_group_members", params, timeout=65.0)


@tool
async def panorama_unused_objects() -> dict:
    """Find unused or orphaned Panorama address objects and address groups."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("find_unused_panorama_objects", {}, timeout=65.0)


@tool
async def panorama_firewall_zones(firewall_name: str, interfaces: list) -> dict:
    """
    Get the security zones assigned to specific interfaces on a Palo Alto firewall.
    Use when a path query reveals a firewall hop and you need to know which zones its interfaces belong to.
    Returns a mapping of interface name to zone name (e.g. {"Ethernet1/1": "trust", "Ethernet1/2": "untrust"}).
    """
    from atlas.tools.panorama_tools import get_zones_for_firewall_interfaces
    # Normalise: LLM may pass [{"name": "Ethernet1/1"}] instead of ["Ethernet1/1"]
    normalised = []
    for iface in interfaces:
        if isinstance(iface, dict):
            normalised.append(iface.get("name") or iface.get("interface") or str(iface))
        else:
            normalised.append(str(iface))
    return await get_zones_for_firewall_interfaces(
        firewall_name=firewall_name,
        interfaces=normalised,
        template="Global",
    )


@tool
async def panorama_firewall_device_group(firewall_names: list) -> dict:
    """
    Get the Panorama device group for one or more Palo Alto firewalls.
    Use when a path query reveals firewall hops and you need to know which device group they belong to.
    Returns a mapping of firewall name to device group name.
    """
    from atlas.tools.panorama_tools import get_device_groups_for_firewalls
    return await get_device_groups_for_firewalls(firewall_names=firewall_names)


@tool
async def panorama_check_policy(source_ip: str, dest_ip: str) -> dict:
    """
    Check Panorama security policies to determine if traffic from source_ip to dest_ip
    is permitted or denied. Use for troubleshooting connectivity problems.
    Returns matching policies with their action (allow/deny) and a verdict.
    Example: source_ip="10.0.0.1", dest_ip="11.0.0.1"
    """
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("check_panorama_policy", {"source_ip": source_ip, "dest_ip": dest_ip}, timeout=90.0)


PANORAMA_TOOLS = [
    panorama_ip_object_group,
    panorama_address_group_members,
    panorama_unused_objects,
    panorama_firewall_zones,
    panorama_firewall_device_group,
    panorama_check_policy,
]

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "panorama_agent.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Atlas Panorama Agent",
    "description": (
        "Analyzes Panorama address groups, members, and policies for a given IP address. "
        "Uses LLM reasoning to determine which Panorama data to fetch."
    ),
    "url": "http://localhost:8003",
    "version": "2.0.0",
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "panorama_ip_lookup",
            "name": "Panorama IP Lookup",
            "description": "Find the address group, members, and policies for an IP address.",
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "What group is 10.0.0.1 in?",
                "Assess Panorama security posture for 10.0.0.1",
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

    logger.info("Panorama agent task: %s", text)

    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    try:
        result = await run_agent_loop(
            task=text,
            system_prompt=_load_skill(),
            tools=PANORAMA_TOOLS,
            max_iterations=10,
        )
    except Exception as e:
        logger.exception("Panorama agent loop error")
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
    uvicorn.run("panorama_agent:app", host="0.0.0.0", port=8003, reload=False)
