"""
NetBox Agent — port 8007.

Wraps NetBox IPAM lookups as an A2A agent:
  - get_gateway_for_prefix  — VIP/gateway for a prefix
  - get_prefix_for_ip       — containing prefix for a host IP
  - get_ip_info             — metadata for a specific IP address
"""
import logging
import pathlib
import sys
import uuid
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.getLogger("atlas").setLevel(logging.INFO)
logger = logging.getLogger("atlas.agents.netbox")

app = FastAPI(title="Atlas NetBox Agent")


def _get_tools():
    try:
        from atlas.tools.netbox_tools import get_gateway_for_prefix, get_prefix_for_ip, get_ip_info
    except ImportError:
        from tools.netbox_tools import get_gateway_for_prefix, get_prefix_for_ip, get_ip_info
    return [get_gateway_for_prefix, get_prefix_for_ip, get_ip_info]


_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "netbox_agent.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Atlas NetBox Agent",
    "description": (
        "Queries NetBox IPAM for gateway (VIP) lookups, prefix containment, "
        "and IP address metadata."
    ),
    "url": "http://localhost:8007",
    "version": "1.0.0",
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "netbox_ipam",
            "name": "NetBox IPAM Lookup",
            "description": "Look up gateways, prefixes, and IP metadata from NetBox.",
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "What is the gateway for 10.0.100.55?",
                "What prefix contains 10.0.200.10?",
                "Show NetBox info for 10.0.100.1",
            ],
        }
    ],
}


@app.get("/.well-known/agent.json")
async def agent_card():
    return JSONResponse(AGENT_CARD)


@app.post("/")
async def handle_task(request: Request) -> JSONResponse:
    body = await request.json()
    task_id = body.get("id") or str(uuid.uuid4())
    message = body.get("message", {})
    parts = message.get("parts", [])
    text = next((p.get("text", "") for p in parts if p.get("type") == "text"), "")

    if not text:
        return _error_response(task_id, "No task text provided.")

    logger.info("NetBox agent task: %s", text)

    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    try:
        result = await run_agent_loop(
            task=text,
            system_prompt=_load_skill(),
            tools=_get_tools(),
        )
    except Exception as e:
        logger.exception("NetBox agent loop error")
        return _error_response(task_id, f"Agent error: {e}")

    return _success_response(task_id, result)


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


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    uvicorn.run("netbox_agent:app", host="0.0.0.0", port=8007, reload=False)
