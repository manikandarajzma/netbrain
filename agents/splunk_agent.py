"""
Splunk A2A Agent — port 8002.

Accepts natural language A2A tasks. Uses an LLM reasoning loop to decide
which Splunk tools to call, then returns a natural language summary.
"""
import logging
import pathlib
import sys
import uuid
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_core.tools import tool

logging.getLogger("atlas").setLevel(logging.INFO)

logger = logging.getLogger("atlas.agents.splunk")

app = FastAPI(title="Atlas Splunk Agent")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
async def splunk_recent_denies(ip_address: str, limit: int = 100, earliest_time: str = "-24h") -> dict:
    """Get recent firewall deny events for an IP address from Splunk."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool(
        "get_splunk_recent_denies",
        {"ip_address": ip_address, "limit": limit, "earliest_time": earliest_time},
        timeout=95.0,
    )


@tool
async def splunk_traffic_summary(ip_address: str) -> dict:
    """Get a summary of all firewall traffic for an IP address, broken down by action (allow/deny)."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("get_splunk_traffic_summary", {"ip_address": ip_address}, timeout=95.0)


@tool
async def splunk_unique_destinations(ip_address: str) -> dict:
    """Get the count of unique destination IPs and ports that an IP address has communicated with."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("get_splunk_unique_destinations", {"ip_address": ip_address}, timeout=95.0)


SPLUNK_TOOLS = [splunk_recent_denies, splunk_traffic_summary, splunk_unique_destinations]

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "splunk_agent.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Atlas Splunk Agent",
    "description": "Retrieves and analyzes Splunk firewall events for a given IP address.",
    "url": "http://localhost:8002",
    "version": "2.0.0",
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "splunk_ip_analysis",
            "name": "Splunk IP Analysis",
            "description": "Analyze recent firewall events, traffic patterns, and destination spread for an IP.",
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Are there recent deny events for 10.0.0.1?",
                "Analyze Splunk traffic for 10.0.0.1",
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

    logger.info("Splunk agent task: %s", text)

    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    try:
        result = await run_agent_loop(
            task=text,
            system_prompt=_load_skill(),
            tools=SPLUNK_TOOLS,
        )
    except Exception as e:
        logger.exception("Splunk agent loop error")
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
    uvicorn.run("splunk_agent:app", host="0.0.0.0", port=8002, reload=False)
