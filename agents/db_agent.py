"""
DB Agent — port 8008.

Queries pre-collected network device data stored in PostgreSQL:
  - routing_table, arp_table, interface_ips, mac_table  (current state)
  - arp_history, routing_history, mac_history           (historical snapshots)

Used for hop-by-hop path tracing and temporal queries like
"what was the ARP table yesterday?"
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
from langchain_core.tools import tool

logging.getLogger("atlas").setLevel(logging.INFO)
logger = logging.getLogger("atlas.agents.db")

app = FastAPI(title="Atlas DB Agent")


# ---------------------------------------------------------------------------
# Current-state tools
# ---------------------------------------------------------------------------

@tool
async def lookup_route(device: str, destination_ip: str) -> dict:
    """
    Longest-prefix-match lookup in the collected routing table for a destination IP on a device.
    Returns next_hop, egress_interface, protocol, and matched prefix.
    Use this to find where a device will forward traffic toward the destination.
    """
    try:
        try:
            from atlas.db import fetchrow
        except ImportError:
            from db import fetchrow
        row = await fetchrow(
            """
            SELECT prefix, next_hop::text, egress_interface, protocol, vrf
            FROM routing_table
            WHERE device = $1 AND $2::inet << prefix
            ORDER BY masklen(prefix) DESC LIMIT 1
            """,
            device, destination_ip,
        )
        if not row:
            return {"found": False, "device": device, "destination": destination_ip}
        return {
            "found": True,
            "device": device,
            "destination": destination_ip,
            "prefix": str(row["prefix"]),
            "next_hop": row["next_hop"],
            "egress_interface": row["egress_interface"],
            "protocol": row["protocol"],
            "vrf": row["vrf"],
        }
    except Exception as e:
        return {"found": False, "error": str(e), "device": device}


@tool
async def lookup_interface_owner(ip: str) -> dict:
    """
    Find which device owns a given IP address by querying the collected interface_ips table.
    Use this to resolve a next-hop IP to a device hostname.
    Returns device name and interface, or found=False if no device owns that IP.
    """
    try:
        try:
            from atlas.db import fetchrow
        except ImportError:
            from db import fetchrow
        row = await fetchrow(
            "SELECT device, interface, prefix_len, vrf FROM interface_ips WHERE ip = $1::inet LIMIT 1",
            ip,
        )
        if not row:
            return {"found": False, "ip": ip}
        return {
            "found": True,
            "ip": ip,
            "device": row["device"],
            "interface": row["interface"],
            "prefix_len": row["prefix_len"],
            "vrf": row["vrf"],
        }
    except Exception as e:
        return {"found": False, "error": str(e), "ip": ip}


@tool
async def lookup_arp(device: str, ip: str) -> dict:
    """
    Look up an IP in the collected ARP table for a device.
    Returns MAC address and interface. Use for the final hop when the destination is directly connected.
    """
    try:
        try:
            from atlas.db import fetchrow
        except ImportError:
            from db import fetchrow
        row = await fetchrow(
            "SELECT mac, interface, vrf FROM arp_table WHERE device = $1 AND ip = $2::inet LIMIT 1",
            device, ip,
        )
        if not row:
            return {"found": False, "device": device, "ip": ip}
        return {"found": True, "device": device, "ip": ip, "mac": row["mac"],
                "interface": row["interface"], "vrf": row["vrf"]}
    except Exception as e:
        return {"found": False, "error": str(e), "device": device}


@tool
async def lookup_mac(device: str, mac: str) -> dict:
    """
    Look up a MAC address in the collected MAC address table for a device.
    Use for L2 last-hop resolution after finding a MAC via ARP.
    """
    try:
        try:
            from atlas.db import fetchrow
        except ImportError:
            from db import fetchrow
        row = await fetchrow(
            "SELECT interface, vlan, entry_type FROM mac_table WHERE device = $1 AND mac = $2 LIMIT 1",
            device, mac,
        )
        if not row:
            return {"found": False, "device": device, "mac": mac}
        return {"found": True, "device": device, "mac": mac,
                "interface": row["interface"], "vlan": row["vlan"], "entry_type": row["entry_type"]}
    except Exception as e:
        return {"found": False, "error": str(e), "device": device}


# ---------------------------------------------------------------------------
# Historical tools
# ---------------------------------------------------------------------------

@tool
async def lookup_arp_history(device: str, at_time: str) -> dict:
    """
    Query the ARP history table for a device at a specific point in time.
    at_time must be an ISO 8601 timestamp or a plain date like '2026-03-28'.
    Returns the most recent ARP snapshot collected before or at that time.
    Use this to answer questions like "what was the ARP table yesterday?".
    """
    try:
        try:
            from atlas.db import fetch
        except ImportError:
            from db import fetch
        rows = await fetch(
            """
            SELECT DISTINCT ON (ip)
                ip::text, mac, interface, vrf, collected_at
            FROM arp_history
            WHERE device = $1 AND collected_at <= $2::timestamptz
            ORDER BY ip, collected_at DESC
            LIMIT 500
            """,
            device, at_time,
        )
        if not rows:
            return {"found": False, "device": device, "at_time": at_time,
                    "message": "No ARP history found for that device/time."}
        return {
            "found": True,
            "device": device,
            "at_time": at_time,
            "count": len(rows),
            "entries": [
                {"ip": r["ip"], "mac": r["mac"], "interface": r["interface"],
                 "vrf": r["vrf"], "collected_at": str(r["collected_at"])}
                for r in rows
            ],
        }
    except Exception as e:
        return {"found": False, "error": str(e), "device": device}


@tool
async def lookup_routing_history(device: str, at_time: str) -> dict:
    """
    Query the routing table history for a device at a specific point in time.
    at_time must be an ISO 8601 timestamp or a plain date like '2026-03-28'.
    Returns routes from the most recent snapshot before or at that time.
    Use this to answer questions like "what routes did arista1 have yesterday?".
    """
    try:
        try:
            from atlas.db import fetch
        except ImportError:
            from db import fetch
        rows = await fetch(
            """
            SELECT DISTINCT ON (vrf, prefix)
                vrf, prefix::text, next_hop::text, egress_interface, protocol,
                admin_distance, metric, collected_at
            FROM routing_history
            WHERE device = $1 AND collected_at <= $2::timestamptz
            ORDER BY vrf, prefix, collected_at DESC
            LIMIT 1000
            """,
            device, at_time,
        )
        if not rows:
            return {"found": False, "device": device, "at_time": at_time,
                    "message": "No routing history found for that device/time."}
        return {
            "found": True,
            "device": device,
            "at_time": at_time,
            "count": len(rows),
            "routes": [
                {"vrf": r["vrf"], "prefix": r["prefix"], "next_hop": r["next_hop"],
                 "egress_interface": r["egress_interface"], "protocol": r["protocol"],
                 "collected_at": str(r["collected_at"])}
                for r in rows
            ],
        }
    except Exception as e:
        return {"found": False, "error": str(e), "device": device}


@tool
async def lookup_mac_history(device: str, at_time: str) -> dict:
    """
    Query the MAC address table history for a device at a specific point in time.
    at_time must be an ISO 8601 timestamp or a plain date like '2026-03-28'.
    Use this to answer questions like "what MACs were on arista1 yesterday?".
    """
    try:
        try:
            from atlas.db import fetch
        except ImportError:
            from db import fetch
        rows = await fetch(
            """
            SELECT DISTINCT ON (mac, vlan)
                mac, vlan, interface, entry_type, collected_at
            FROM mac_history
            WHERE device = $1 AND collected_at <= $2::timestamptz
            ORDER BY mac, vlan, collected_at DESC
            LIMIT 500
            """,
            device, at_time,
        )
        if not rows:
            return {"found": False, "device": device, "at_time": at_time,
                    "message": "No MAC history found for that device/time."}
        return {
            "found": True,
            "device": device,
            "at_time": at_time,
            "count": len(rows),
            "entries": [
                {"mac": r["mac"], "vlan": r["vlan"], "interface": r["interface"],
                 "entry_type": r["entry_type"], "collected_at": str(r["collected_at"])}
                for r in rows
            ],
        }
    except Exception as e:
        return {"found": False, "error": str(e), "device": device}


DB_TOOLS = [
    lookup_route,
    lookup_interface_owner,
    lookup_arp,
    lookup_mac,
    lookup_arp_history,
    lookup_routing_history,
    lookup_mac_history,
]

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "db_agent.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Atlas DB Agent",
    "description": (
        "Queries pre-collected network device data (routing, ARP, MAC, interface IPs) "
        "from PostgreSQL. Supports current-state path tracing and historical queries "
        "(e.g. 'what was the ARP table yesterday?')."
    ),
    "url": "http://localhost:8008",
    "version": "1.0.0",
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "db_path_trace",
            "name": "Database Path Trace",
            "description": "Trace network path using collected routing/ARP/MAC tables.",
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Trace path from 10.0.100.100 to 10.0.200.200 using the database",
                "What was the ARP table on arista1 yesterday?",
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

    logger.info("DB agent task: %s", text)

    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    try:
        result = await run_agent_loop(
            task=text,
            system_prompt=_load_skill(),
            tools=DB_TOOLS,
        )
    except Exception as e:
        logger.exception("DB agent loop error")
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
    uvicorn.run("db_agent:app", host="0.0.0.0", port=8008, reload=False)
