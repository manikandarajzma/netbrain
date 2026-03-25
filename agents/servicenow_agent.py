"""
ServiceNow A2A Agent — port 8005.

Accepts natural language tasks about ServiceNow incidents, change requests,
problems, CMDB CIs, users, and knowledge articles.
"""
import hashlib
import json
import logging
import os
import pathlib
import re
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
logger = logging.getLogger("atlas.agents.servicenow")

app = FastAPI(title="Atlas ServiceNow Agent")

# ---------------------------------------------------------------------------
# Response-level Redis cache (caches full LLM + tool output for read queries)
# ---------------------------------------------------------------------------

_RESPONSE_CACHE_TTL = 120  # seconds

# Write-intent keywords — these tasks must never be cached
_WRITE_PATTERN = re.compile(
    r'\b(create|update|close|resolve|assign|add note|work note|open|submit|delete|remove)\b',
    re.IGNORECASE,
)


def _response_cache_key(text: str) -> str:
    digest = hashlib.sha256(text.strip().lower().encode()).hexdigest()[:20]
    return f"atlas:snow:response:{digest}"


def _response_cache_get(text: str) -> str | None:
    if _WRITE_PATTERN.search(text):
        return None
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        return r.get(_response_cache_key(text))
    except Exception:
        return None


def _response_cache_set(text: str, result: str) -> None:
    if _WRITE_PATTERN.search(text):
        return
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.setex(_response_cache_key(text), _RESPONSE_CACHE_TTL, result)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Tools (thin wrappers over MCP tools)
# ---------------------------------------------------------------------------

@tool
async def snow_get_incident(number: str) -> dict:
    """Get a ServiceNow incident by number (e.g. INC0010001)."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("get_servicenow_incident", {"number": number}, timeout=30.0)


@tool
async def snow_list_incidents(state: str = "", priority: str = "", assigned_to: str = "", limit: int = 20) -> dict:
    """List ServiceNow incidents. Filter by state (new/in_progress/on_hold/resolved/closed), priority (1-4), or assigned_to username."""
    from atlas.mcp_client import call_mcp_tool
    args = {"limit": limit}
    if state:
        args["state"] = state
    if priority:
        args["priority"] = priority
    if assigned_to:
        args["assigned_to"] = assigned_to
    return await call_mcp_tool("list_servicenow_incidents", args, timeout=30.0)


@tool
async def snow_create_incident(short_description: str, description: str = "", urgency: str = "3", impact: str = "3", category: str = "network", assignment_group: str = "") -> dict:
    """Create a new ServiceNow incident. Urgency/impact: 1=high, 2=medium, 3=low."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("create_servicenow_incident", {
        "short_description": short_description,
        "description": description,
        "urgency": urgency,
        "impact": impact,
        "category": category,
        "assignment_group": assignment_group,
    }, timeout=30.0)


@tool
async def snow_update_incident(number: str, state: str = "", work_notes: str = "", assigned_to: str = "", close_notes: str = "") -> dict:
    """Update a ServiceNow incident — change state, add work notes, or assign."""
    from atlas.mcp_client import call_mcp_tool
    args = {"number": number}
    if state:
        args["state"] = state
    if work_notes:
        args["work_notes"] = work_notes
    if assigned_to:
        args["assigned_to"] = assigned_to
    if close_notes:
        args["close_notes"] = close_notes
    return await call_mcp_tool("update_servicenow_incident", args, timeout=30.0)


@tool
async def snow_get_change_request(number: str) -> dict:
    """Get a ServiceNow change request by number (e.g. CHG0010001)."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("get_servicenow_change_request", {"number": number}, timeout=30.0)


@tool
async def snow_list_change_requests(query: str = "", state: str = "", risk: str = "", limit: int = 20) -> dict:
    """List ServiceNow change requests. Use query to search by device name, IP, or keyword (e.g. query='EDGE-RTR-01'). Filter by state or risk (high/moderate/low)."""
    from atlas.mcp_client import call_mcp_tool
    args = {"limit": limit}
    if query:
        args["query"] = query
    if state:
        args["state"] = state
    if risk:
        args["risk"] = risk
    return await call_mcp_tool("list_servicenow_change_requests", args, timeout=30.0)


@tool
async def snow_update_change_request(number: str, state: str = "", work_notes: str = "", assigned_to: str = "", close_notes: str = "") -> dict:
    """Update or close a ServiceNow change request (CHG...). Use state='closed' with close_notes to close it — the full approval workflow runs automatically."""
    from atlas.mcp_client import call_mcp_tool
    args = {"number": number}
    if state:
        args["state"] = state
    if work_notes:
        args["work_notes"] = work_notes
    if assigned_to:
        args["assigned_to"] = assigned_to
    if close_notes:
        args["close_notes"] = close_notes
    return await call_mcp_tool("update_servicenow_change_request", args, timeout=120.0)


@tool
async def snow_create_change_request(short_description: str, description: str = "", risk: str = "3", assignment_group: str = "", justification: str = "", implementation_plan: str = "", ci_name: str = "") -> dict:
    """Create a new ServiceNow change request. Always pass ci_name with the affected device hostname (e.g. 'PA-FW-01')."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("create_servicenow_change_request", {
        "short_description": short_description,
        "description": description,
        "risk": risk,
        "assignment_group": assignment_group,
        "justification": justification,
        "implementation_plan": implementation_plan,
        "ci_name": ci_name,
    }, timeout=30.0)


@tool
async def snow_list_problems(state: str = "", limit: int = 20) -> dict:
    """List ServiceNow problem records. Filter by state (open/known_error/closed)."""
    from atlas.mcp_client import call_mcp_tool
    args = {"limit": limit}
    if state:
        args["state"] = state
    return await call_mcp_tool("list_servicenow_problems", args, timeout=30.0)


@tool
async def snow_create_problem(short_description: str, description: str = "", assignment_group: str = "") -> dict:
    """Create a new ServiceNow problem record."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("create_servicenow_problem", {
        "short_description": short_description,
        "description": description,
        "assignment_group": assignment_group,
    }, timeout=30.0)


@tool
async def snow_get_ci(name: str = "", ip_address: str = "", limit: int = 10) -> dict:
    """Search CMDB for a configuration item by name or IP address."""
    from atlas.mcp_client import call_mcp_tool
    args = {"limit": limit}
    if name:
        args["name"] = name
    if ip_address:
        args["ip_address"] = ip_address
    return await call_mcp_tool("get_servicenow_ci", args, timeout=30.0)


@tool
async def snow_search_incidents(query: str, limit: int = 10) -> dict:
    """Search across all incident fields (description, work notes) for a keyword, device name, or IP."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("search_servicenow_incidents", {"query": query, "limit": limit}, timeout=30.0)


@tool
async def snow_get_user(username: str = "", email: str = "", name: str = "") -> dict:
    """Look up a ServiceNow user by username, email, or full name."""
    from atlas.mcp_client import call_mcp_tool
    args = {}
    if username:
        args["username"] = username
    if email:
        args["email"] = email
    if name:
        args["name"] = name
    return await call_mcp_tool("get_servicenow_user", args, timeout=30.0)


@tool
async def snow_search_knowledge(query: str, limit: int = 5) -> dict:
    """Search the ServiceNow knowledge base for articles matching a query."""
    from atlas.mcp_client import call_mcp_tool
    return await call_mcp_tool("search_servicenow_knowledge", {"query": query, "limit": limit}, timeout=30.0)


def _extract_field(text: str, keywords: list[str]) -> str:
    """Extract a field value from labeled user text (e.g. 'Justification — value').
    Only matches at the START of a line to avoid false keyword matches inside other fields.
    """
    for line in text.splitlines():
        low = line.lower().lstrip(" -•*\t")
        for kw in keywords:
            if low.startswith(kw):
                rest = line[line.lower().find(kw) + len(kw):]
                value = rest.lstrip(" :—–-•\t").strip()
                if value:
                    return value
    return ""


async def _create_record_directly(is_chg: bool, user_provided: str) -> str:
    """Parse user_provided text and call the servicenow_tools functions directly."""
    try:
        from atlas.tools.servicenow_tools import (
            _do_create_change_request,
            _do_create_incident,
        )
    except ImportError:
        from tools.servicenow_tools import (
            _do_create_change_request,
            _do_create_incident,
        )

    def _get(keywords):
        return _extract_field(user_provided, keywords)

    if is_chg:
        short_desc    = _get(["short description", "short desc"])
        ci_name       = _get(["ci / device affected", "ci/device affected", "ci/device", "device affected"])
        if ci_name and ci_name.lower() not in short_desc.lower():
            short_desc = f"{short_desc} on {ci_name}"
        justification = _get(["justification", "reason"])
        impl_plan     = _get(["implementation plan"])
        assignment    = _get(["assignment group"])
        risk_raw      = _get(["risk level", "risk"])
        risk_map      = {"high": "1", "moderate": "2", "medium": "2", "low": "3"}
        risk          = risk_map.get(risk_raw.lower().strip(), "3") if risk_raw else "3"

        result = await _do_create_change_request(
            short_description=short_desc,
            description=short_desc,
            risk=risk,
            assignment_group=assignment,
            justification=justification,
            implementation_plan=impl_plan,
            ci_name=ci_name,
        )
        if isinstance(result, dict) and "error" in result:
            return f"Failed to create change request: {result['error']}"
        record = result.get("result", result) if isinstance(result, dict) else {}
        num = record.get("number", "")
        return (f"Change request created successfully.\n\n"
                f"**Number**: {num}\n"
                f"**Short description**: {short_desc}\n"
                f"**CI / Device**: {ci_name}\n"
                f"**Justification**: {justification}\n"
                f"**Implementation plan**: {impl_plan}\n"
                f"**Assignment group**: {assignment}\n"
                f"**Risk**: {risk_raw or 'low'}")
    else:
        short_desc  = _get(["short description", "short desc"]) or _get(["description"])
        ci_name     = _get(["ci / device affected", "ci/device affected", "ci/device", "device affected", "ci"])
        if ci_name and ci_name.lower() not in short_desc.lower():
            short_desc = f"{short_desc} on {ci_name}"
        description = _get(["description"]) or short_desc
        urgency_raw = _get(["urgency"])
        impact_raw  = _get(["impact"])
        assignment  = _get(["assignment group", "group", "team"])
        lvl_map     = {"high": "1", "medium": "2", "low": "3"}
        urgency     = lvl_map.get((urgency_raw or "").lower().strip(), "3")
        impact      = lvl_map.get((impact_raw or "").lower().strip(), "3")

        result = await _do_create_incident(
            short_description=short_desc,
            description=description,
            urgency=urgency,
            impact=impact,
            category="network",
            assignment_group=assignment,
            ci_name=ci_name,
        )
        if isinstance(result, dict) and "error" in result:
            return f"Failed to create incident: {result['error']}"
        record = result.get("result", result) if isinstance(result, dict) else {}
        num = record.get("number", "")
        return (f"Incident created successfully.\n\n"
                f"**Number**: {num}\n"
                f"**Short description**: {short_desc}\n"
                f"**CI / Device**: {ci_name}\n"
                f"**Urgency**: {urgency_raw or 'low'}\n"
                f"**Impact**: {impact_raw or 'low'}\n"
                f"**Assignment group**: {assignment}")


SERVICENOW_TOOLS = [
    snow_get_incident,
    snow_list_incidents,
    snow_search_incidents,
    snow_create_incident,
    snow_update_incident,
    snow_get_change_request,
    snow_list_change_requests,
    snow_update_change_request,
    snow_create_change_request,
    snow_list_problems,
    snow_create_problem,
    snow_get_ci,
    snow_get_user,
    snow_search_knowledge,
]

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "servicenow_agent.md"


def _load_skill() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "Atlas ServiceNow Agent",
    "description": "Manages ServiceNow incidents, change requests, problems, CMDB, users, and knowledge articles.",
    "url": "http://localhost:8005",
    "version": "1.0.0",
    "capabilities": {"streaming": False},
    "skills": [
        {
            "id": "servicenow_itsm",
            "name": "ServiceNow ITSM",
            "description": "Create, read, and update incidents, change requests, problems, and CMDB records.",
            "inputModes": ["text"],
            "outputModes": ["text"],
            "examples": [
                "Show me open critical incidents",
                "Create an incident for network outage in building A",
                "What change requests are scheduled this week?",
                "Find the CI for IP 10.0.0.1",
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

    logger.info("ServiceNow agent task: %s", text)

    # --- Intercept bare create requests before the LLM runs ---
    _lower = text.lower().strip()
    _is_create_chg = re.search(r'\b(create|open|raise|submit|log)\b.{0,30}\b(change request|change|cr)\b', _lower)
    _is_create_inc = re.search(r'\b(create|open|raise|submit|log)\b.{0,30}\b(incident|inc)\b', _lower)
    _has_details = len(text.split()) > 10  # bare "create a change request" has no details

    _CHG_FIELDS = [
        ("short_description", ["short description", "short desc", "description"]),
        ("ci_device",         ["ci", "device", "ci/device", "affected"]),
        ("justification",     ["justification", "reason", "why"]),
        ("implementation_plan", ["implementation plan", "implementation", "how"]),
        ("assignment_group",  ["assignment group", "group", "team", "assigned to"]),
        ("risk",              ["risk level", "risk"]),
    ]
    _INC_FIELDS = [
        ("short_description", ["short description", "short desc"]),
        ("ci_device",         ["ci", "device", "ci/device", "affected"]),
        ("description",       ["description", "details", "problem"]),
        ("urgency",           ["urgency", "urgent"]),
        ("impact",            ["impact"]),
        ("assignment_group",  ["assignment group", "group", "team"]),
    ]
    _CHG_LABELS = {
        "short_description":   "**Short description** — what is changing?",
        "ci_device":           "**CI / Device affected** — which device or system?",
        "justification":       "**Justification** — why is this change needed?",
        "implementation_plan": "**Implementation plan** — how will it be done?",
        "assignment_group":    "**Assignment group**",
        "risk":                "**Risk level** — low, moderate, or high?",
    }
    _INC_LABELS = {
        "short_description": "**Short description** — what is the issue?",
        "ci_device":         "**CI / Device affected** — which device or system?",
        "description":       "**Description** — full details of the problem",
        "urgency":           "**Urgency** — high, medium, or low?",
        "impact":            "**Impact** — high, medium, or low?",
        "assignment_group":  "**Assignment group**",
    }

    def _missing_fields(user_text: str, fields: list) -> list:
        """Return field keys not yet mentioned in user_text."""
        low = user_text.lower()
        missing = []
        for key, keywords in fields:
            if not any(kw in low for kw in keywords):
                missing.append(key)
        return missing

    if _is_create_chg and not _has_details:
        return _success_response(task_id, (
            "I need a few details to create the change request. Please provide:\n\n"
            + "\n".join(f"- {v}" for v in _CHG_LABELS.values())
        ))

    if _is_create_inc and not _has_details:
        return _success_response(task_id, (
            "I need a few details to create the incident. Please provide:\n\n"
            + "\n".join(f"- {v}" for v in _INC_LABELS.values())
        ))

    # Handle accumulated follow-up replies from create form
    if text.startswith("[CREATE FORM]"):
        form_line = text.split("\n")[0]
        is_chg_form = "change" in form_line.lower()  # matches "change request", "create_change_request"
        fields = _CHG_FIELDS if is_chg_form else _INC_FIELDS
        labels = _CHG_LABELS if is_chg_form else _INC_LABELS
        user_provided = text.split("[USER PROVIDED]", 1)[-1].strip() if "[USER PROVIDED]" in text else text
        missing = _missing_fields(user_provided, fields)
        # Short description is the only hard requirement
        if "short_description" in missing:
            still_needed = "\n".join(f"- {labels[k]}" for k in missing)
            return _success_response(task_id,
                f"Still need the following before I can create the record:\n\n{still_needed}")
        if missing:
            still_needed = "\n".join(f"- {labels[k]}" for k in missing)
            return _success_response(task_id,
                f"Almost there — just need a few more details:\n\n{still_needed}")
        # All required fields present — call the create tool directly (no LLM)
        result = await _create_record_directly(is_chg_form, user_provided)
        return _success_response(task_id, result)

    cached = _response_cache_get(text)
    if cached:
        logger.info("ServiceNow response cache hit")
        return _success_response(task_id, cached)

    try:
        from atlas.agents.agent_loop import run_agent_loop
    except ImportError:
        from agent_loop import run_agent_loop

    try:
        result = await run_agent_loop(
            task=text,
            system_prompt=_load_skill(),
            tools=SERVICENOW_TOOLS,
        )
    except Exception as e:
        logger.exception("ServiceNow agent loop error")
        return _error_response(task_id, f"Agent error: {e}")

    _response_cache_set(text, result)
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
    uvicorn.run("servicenow_agent:app", host="0.0.0.0", port=8005, reload=False)
