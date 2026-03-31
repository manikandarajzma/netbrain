"""
ServiceNow MCP tools — incidents, change requests, problems, CMDB, users, knowledge.

Exposes the most common ServiceNow Table API operations as MCP tools.
"""

import aiohttp
import asyncio
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

from tools.shared import mcp, setup_logging
import servicenowauth

logger = setup_logging(__name__)


# ---------------------------------------------------------------------------
# Redis cache helpers (read-only tools only; writes are never cached)
# ---------------------------------------------------------------------------

# TTLs per tool (seconds)
_SNOW_CACHE_TTL = {
    "get_incident":         60,   # single record — short TTL so updates are visible quickly
    "list_incidents":      120,
    "search_incidents":    120,
    "get_change_request":   60,
    "list_change_requests": 120,
    "list_problems":       120,
    "get_ci":              300,   # CMDB records change infrequently
    "get_user":            300,
    "search_knowledge":    600,
}


def _snow_cache_key(tool: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:16]
    return f"atlas:snow:{tool}:{digest}"


def _snow_cache_get(key: str) -> Any:
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        val = r.get(key)
        return json.loads(val) if val else None
    except Exception:
        return None


def _snow_cache_set(key: str, value: Any, ttl: int) -> None:
    try:
        import redis as _redis
        r = _redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        r.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


async def _cached_get(tool: str, cache_parts: list, path: str, params: dict = None) -> Dict[str, Any]:
    """GET with Redis read-through cache. Skips cache on error responses."""
    key = _snow_cache_key(tool, *cache_parts)
    cached = _snow_cache_get(key)
    if cached is not None:
        logger.info("snow cache hit [%s]: %s", tool, key)
        return cached
    result = await _get(path, params)
    if "error" not in result:
        _snow_cache_set(key, result, _SNOW_CACHE_TTL[tool])
    return result


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return servicenowauth.SERVICENOW_INSTANCE_URL.rstrip("/")


def _auth() -> aiohttp.BasicAuth:
    return aiohttp.BasicAuth(servicenowauth.SERVICENOW_USER, servicenowauth.SERVICENOW_PASSWORD)


async def _get(path: str, params: dict = None) -> Dict[str, Any]:
    """GET request to ServiceNow REST API. Returns parsed JSON or {'error': ...}."""
    url = f"{_base_url()}{path}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    # sysparm_display_value=true returns human-readable labels instead of sys_ids
    if params is None:
        params = {}
    params.setdefault("sysparm_display_value", "true")
    try:
        async with aiohttp.ClientSession(auth=_auth(), headers=headers) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 200:
                    return await resp.json()
                text = await resp.text()
                return {"error": f"HTTP {resp.status}: {text[:300]}"}
    except Exception as exc:
        return {"error": str(exc)}


async def _post(path: str, payload: dict, params: dict = None) -> Dict[str, Any]:
    """POST request to ServiceNow REST API."""
    url = f"{_base_url()}{path}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession(auth=_auth(), headers=headers) as session:
            async with session.post(url, json=payload, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                body = await resp.json()
                if resp.status in (200, 201):
                    return body
                return {"error": f"HTTP {resp.status}: {body}"}
    except Exception as exc:
        return {"error": str(exc)}


async def _patch(path: str, payload: dict) -> Dict[str, Any]:
    """PATCH request to ServiceNow REST API."""
    url = f"{_base_url()}{path}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession(auth=_auth(), headers=headers) as session:
            async with session.patch(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                body = await resp.json()
                if resp.status == 200:
                    return body
                return {"error": f"HTTP {resp.status}: {body}"}
    except Exception as exc:
        return {"error": str(exc)}


def _records(result: Dict[str, Any]) -> List[Dict]:
    return result.get("result", []) if isinstance(result.get("result"), list) else []


def _record(result: Dict[str, Any]) -> Dict:
    return result.get("result", {}) if isinstance(result.get("result"), dict) else {}


# ---------------------------------------------------------------------------
# Incidents
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_servicenow_incident(number: str) -> Dict[str, Any]:
    """
    Get a ServiceNow incident by incident number (e.g. INC0010001).

    Args:
        number: Incident number (e.g. "INC0010001")

    Returns:
        Incident record with number, short_description, state, priority, assigned_to, etc.
    """
    logger.info("get_servicenow_incident: %s", number)
    result = await _cached_get("get_incident", [number], "/api/now/table/incident", params={
        "sysparm_query": f"number={number}",
        "sysparm_limit": 1,
        "sysparm_fields": "number,sys_id,short_description,description,state,priority,urgency,impact,category,assignment_group,assigned_to,caller_id,opened_at,resolved_at,closed_at,close_notes,work_notes,comments",
    })
    records = _records(result)
    if not records:
        return {"error": f"Incident {number} not found."}
    return {"result": records[0]}


@mcp.tool()
async def list_servicenow_incidents(
    state: Optional[str] = None,
    priority: Optional[str] = None,
    assigned_to: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List ServiceNow incidents with optional filters.

    Args:
        state:       Filter by state: "new", "in_progress", "on_hold", "resolved", "closed"
        priority:    Filter by priority: "1" (critical), "2" (high), "3" (moderate), "4" (low)
        assigned_to: Filter by assigned user name or username
        limit:       Max number of results (default 20, max 100)

    Returns:
        List of incidents with number, short_description, state, priority, assigned_to.
    """
    STATE_MAP = {"new": "1", "in_progress": "2", "on_hold": "3", "resolved": "6", "closed": "7"}
    query_parts = []
    if state:
        query_parts.append(f"state={STATE_MAP.get(state.lower(), state)}")
    if priority:
        query_parts.append(f"priority={priority}")
    if assigned_to:
        query_parts.append(f"assigned_to.user_nameSTARTSWITH{assigned_to}^ORequested_bySTARTSWITH{assigned_to}")

    params = {
        "sysparm_query": "^".join(query_parts) + "^ORDERBYDESCopened_at" if query_parts else "ORDERBYDESCopened_at",
        "sysparm_limit": min(limit, 100),
        "sysparm_fields": "number,short_description,state,priority,assigned_to,assignment_group,opened_at,caller_id",
    }
    logger.info("list_servicenow_incidents: query=%s", params["sysparm_query"])
    return await _cached_get("list_incidents", [params["sysparm_query"], str(params["sysparm_limit"])],
                             "/api/now/table/incident", params=params)


async def _do_create_incident(
    short_description: str,
    description: str = "",
    urgency: str = "3",
    impact: str = "3",
    category: str = "network",
    assignment_group: str = "",
    ci_name: str = "",
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "short_description": short_description,
        "description": description,
        "urgency": urgency,
        "impact": impact,
        "category": category,
    }
    if assignment_group:
        payload["assignment_group"] = assignment_group
    if ci_name:
        ci_records = []
        for query in (f"name={ci_name}", f"nameLIKE{ci_name}"):
            ci_result = await _get("/api/now/table/cmdb_ci", params={
                "sysparm_query": query,
                "sysparm_fields": "sys_id,name",
                "sysparm_limit": 1,
            })
            ci_records = _records(ci_result)
            if ci_records:
                break
        if ci_records:
            raw = ci_records[0].get("sys_id", "")
            sys_id = raw.get("value", raw) if isinstance(raw, dict) else raw
            payload["cmdb_ci"] = sys_id
            logger.info("create_servicenow_incident: resolved ci=%s → sys_id=%s", ci_name, sys_id)
        else:
            logger.info("create_servicenow_incident: CI '%s' not in CMDB — creating it", ci_name)
            ci_create = await _post("/api/now/table/cmdb_ci", {"name": ci_name})
            created = ci_create.get("result", {})
            sys_id = created.get("sys_id", "")
            if sys_id:
                payload["cmdb_ci"] = sys_id
                logger.info("create_servicenow_incident: created CI '%s' → sys_id=%s", ci_name, sys_id)
            else:
                logger.warning("create_servicenow_incident: failed to create CI '%s': %s", ci_name, ci_create)
    logger.info("create_servicenow_incident: %s ci=%s", short_description, ci_name)
    return await _post("/api/now/table/incident", payload)


@mcp.tool()
async def create_servicenow_incident(
    short_description: str,
    description: str = "",
    urgency: str = "3",
    impact: str = "3",
    category: str = "network",
    assignment_group: str = "",
    ci_name: str = "",
) -> Dict[str, Any]:
    """
    Create a new ServiceNow incident.

    Args:
        short_description: Brief summary of the issue (required)
        description:       Detailed description of the issue
        urgency:           "1" (high), "2" (medium), "3" (low) — default "3"
        impact:            "1" (high), "2" (medium), "3" (low) — default "3"
        category:          Category (e.g. "network", "hardware", "software") — default "network"
        assignment_group:  Name of the assignment group
        ci_name:           Hostname or name of the affected CI/device (e.g. "PA-FW-01")

    Returns:
        Created incident record with sys_id and number.
    """
    return await _do_create_incident(
        short_description=short_description,
        description=description,
        urgency=urgency,
        impact=impact,
        category=category,
        assignment_group=assignment_group,
        ci_name=ci_name,
    )


@mcp.tool()
async def update_servicenow_incident(
    number: str,
    state: Optional[str] = None,
    work_notes: Optional[str] = None,
    assigned_to: Optional[str] = None,
    close_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing ServiceNow incident.

    Args:
        number:      Incident number (e.g. "INC0010001")
        state:       New state: "new", "in_progress", "on_hold", "resolved", "closed"
        work_notes:  Internal notes to add
        assigned_to: Username to assign the incident to
        close_notes: Resolution notes (required when closing)

    Returns:
        Updated incident record.
    """
    STATE_MAP = {"new": "1", "in_progress": "2", "on_hold": "3", "resolved": "6", "closed": "7"}

    # Get sys_id first
    lookup = await _get("/api/now/table/incident", params={
        "sysparm_query": f"number={number}",
        "sysparm_limit": 1,
        "sysparm_fields": "sys_id",
    })
    records = _records(lookup)
    if not records:
        return {"error": f"Incident {number} not found."}
    sys_id = records[0]["sys_id"]

    payload: Dict[str, Any] = {}
    if state:
        mapped = STATE_MAP.get(state.lower(), state)
        payload["state"] = mapped
        if mapped == "6":  # resolved — ServiceNow requires these fields
            payload["resolved_by"] = assigned_to or "admin"
            payload["caller_id"] = "admin"
            payload["close_code"] = "Solved (Permanently)"
    if work_notes:
        payload["work_notes"] = work_notes
    if assigned_to:
        payload["assigned_to"] = assigned_to
    if close_notes:
        payload["close_notes"] = close_notes

    logger.info("update_servicenow_incident: %s sys_id=%s", number, sys_id)
    return await _patch(f"/api/now/table/incident/{sys_id}", payload)


# ---------------------------------------------------------------------------
# Change Requests
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_servicenow_change_request(number: str) -> Dict[str, Any]:
    """
    Get a ServiceNow change request by number (e.g. CHG0010001).

    Args:
        number: Change request number (e.g. "CHG0010001")

    Returns:
        Change request record with number, short_description, state, risk, start/end dates.
    """
    logger.info("get_servicenow_change_request: %s", number)
    result = await _cached_get("get_change_request", [number], "/api/now/table/change_request", params={
        "sysparm_query": f"number={number}",
        "sysparm_limit": 1,
        "sysparm_fields": "number,sys_id,short_description,description,state,risk,impact,priority,assignment_group,assigned_to,start_date,end_date,close_notes,work_notes,justification,implementation_plan",
    })
    records = _records(result)
    if not records:
        return {"error": f"Change request {number} not found."}
    return {"result": records[0]}


@mcp.tool()
async def list_servicenow_change_requests(
    query: Optional[str] = None,
    state: Optional[str] = None,
    risk: Optional[str] = None,
    updated_within_hours: Optional[int] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List ServiceNow change requests with optional filters.

    Args:
        query: Keyword search — supports "A OR B OR C" to search multiple device names/terms
               across short_description, description, and work_notes (e.g. "PA-FW-01 OR CORE-SW-01")
        state: Filter by state: "new", "assess", "authorize", "scheduled", "implement", "review", "closed"
        risk:  Filter by risk: "high", "moderate", "low"
        updated_within_hours: Only return records updated within the last N hours (e.g. 1 for last hour)
        limit: Max number of results (default 20)

    Returns:
        List of change requests.
    """
    import re as _re
    from datetime import datetime, timezone, timedelta
    RISK_MAP = {"high": "1", "moderate": "2", "low": "3"}
    STATE_MAP = {"new": "-5", "assess": "-4", "authorize": "-3", "scheduled": "-2", "implement": "-1", "review": "0", "closed": "3"}
    query_parts = []
    if state:
        query_parts.append(f"state={STATE_MAP.get(state.lower(), state)}")
    if risk:
        query_parts.append(f"risk={RISK_MAP.get(risk.lower(), risk)}")
    if updated_within_hours:
        since = datetime.now(timezone.utc) - timedelta(hours=updated_within_hours)
        query_parts.append(f"sys_updated_on>={since.strftime('%Y-%m-%d %H:%M:%S')}")
    q = (query or "").strip()
    if q:
        terms = [t.strip() for t in _re.split(r'\bOR\b', q, flags=_re.IGNORECASE) if t.strip()]
        if not terms:
            logger.info("list_servicenow_change_requests: no valid keywords in query=%r", q)
            return {"result": []}
        or_clauses = "^OR".join(
            f"short_descriptionLIKE{t}^ORdescriptionLIKE{t}^ORwork_notesLIKE{t}^ORcmdb_ci.nameLIKE{t}"
            for t in terms
        )
        query_parts.append(or_clauses)

    base = "^".join(query_parts) if query_parts else ""
    params = {
        "sysparm_query": f"{base}^ORDERBYDESCsys_updated_on" if base else "ORDERBYDESCsys_updated_on",
        "sysparm_limit": min(limit, 100),
        "sysparm_fields": "number,short_description,description,state,risk,assigned_to,start_date,end_date,close_notes,work_notes,cmdb_ci",
    }
    logger.info("list_servicenow_change_requests: query=%s", params["sysparm_query"])
    # Skip cache for time-scoped queries (updated_within_hours) — results change by the minute
    if updated_within_hours:
        return await _get("/api/now/table/change_request", params=params)
    return await _cached_get("list_change_requests", [params["sysparm_query"], str(params["sysparm_limit"])],
                             "/api/now/table/change_request", params=params)


async def _do_create_change_request(
    short_description: str,
    description: str = "",
    risk: str = "3",
    assignment_group: str = "",
    justification: str = "",
    implementation_plan: str = "",
    ci_name: str = "",
) -> Dict[str, Any]:
    if ci_name and ci_name.lower() not in short_description.lower():
        short_description = f"{short_description} on {ci_name}"
    payload: Dict[str, Any] = {
        "short_description": short_description,
        "description": description,
        "risk": risk,
        "type": "normal",
    }
    if assignment_group:
        payload["assignment_group"] = assignment_group
    if justification:
        payload["justification"] = justification
    if implementation_plan:
        payload["implementation_plan"] = implementation_plan
    if ci_name:
        # Look up the CI sys_id by name so we can link it directly.
        # Try exact match first, then case-insensitive LIKE match.
        ci_records = []
        for query in (f"name={ci_name}", f"nameLIKE{ci_name}"):
            ci_result = await _get("/api/now/table/cmdb_ci", params={
                "sysparm_query": query,
                "sysparm_fields": "sys_id,name",
                "sysparm_display_value": "true",
                "sysparm_limit": 1,
            })
            ci_records = _records(ci_result)
            if ci_records:
                break
        if ci_records:
            # sys_id may come back as a dict when display_value=true
            raw = ci_records[0].get("sys_id", "")
            sys_id = raw.get("value", raw) if isinstance(raw, dict) else raw
            payload["cmdb_ci"] = sys_id
            logger.info("create_servicenow_change_request: resolved ci=%s → sys_id=%s", ci_name, sys_id)
        else:
            # CI doesn't exist — create a minimal record so future lookups and searches work
            logger.info("create_servicenow_change_request: CI '%s' not in CMDB — creating it", ci_name)
            ci_create = await _post("/api/now/table/cmdb_ci", {"name": ci_name})
            created = ci_create.get("result", {})
            sys_id = created.get("sys_id", "")
            if sys_id:
                payload["cmdb_ci"] = sys_id
                logger.info("create_servicenow_change_request: created CI '%s' → sys_id=%s", ci_name, sys_id)
            else:
                logger.warning("create_servicenow_change_request: failed to create CI '%s': %s", ci_name, ci_create)
    logger.info("create_servicenow_change_request: %s ci=%s", short_description, ci_name)
    return await _post("/api/now/table/change_request", payload)


@mcp.tool()
async def create_servicenow_change_request(
    short_description: str,
    description: str = "",
    risk: str = "3",
    assignment_group: str = "",
    justification: str = "",
    implementation_plan: str = "",
    ci_name: str = "",
) -> Dict[str, Any]:
    """
    Create a new ServiceNow change request.

    Args:
        short_description:   Brief summary of the change (required)
        description:         Detailed description
        risk:                "1" (high), "2" (moderate), "3" (low) — default "3"
        assignment_group:    Name of the assignment group
        justification:       Business justification for the change
        implementation_plan: Step-by-step implementation plan
        ci_name:             Hostname or name of the affected CI/device (e.g. "PA-FW-01")

    Returns:
        Created change request with sys_id and number.
    """
    return await _do_create_change_request(
        short_description=short_description,
        description=description,
        risk=risk,
        assignment_group=assignment_group,
        justification=justification,
        implementation_plan=implementation_plan,
        ci_name=ci_name,
    )


async def _approve_pending(sys_id: str) -> int:
    """Approve all pending sysapproval_approver records for a change request. Returns count approved."""
    import json as _json
    result = await _get("/api/now/table/sysapproval_approver", params={
        "sysparm_query": f"document_id={sys_id}^stateNOT INapproved,rejected,cancelled",
        "sysparm_fields": "sys_id",
        "sysparm_limit": "50",
        "sysparm_display_value": "false",
    })
    approvals = _records(result)
    count = 0
    for a in approvals:
        r = await _patch(f"/api/now/table/sysapproval_approver/{a['sys_id']}",
                         {"state": "approved", "comments": "Approved"})
        if not r.get("error"):
            count += 1
    return count


async def _chg_rest_patch(sys_id: str, payload: dict) -> Dict[str, Any]:
    """PATCH via the sn_chg_rest API (enforces change model state machine)."""
    import json as _json
    url = f"{_base_url()}/api/sn_chg_rest/change/{sys_id}"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    try:
        async with aiohttp.ClientSession(auth=_auth(), headers=headers) as session:
            async with session.patch(url, json=payload, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                body = await resp.json()
                if resp.status == 200:
                    return body
                # Parse the structured error detail from the change model
                err_detail = body.get("error", {}).get("detail", "")
                try:
                    return {"error": _json.loads(err_detail)}
                except Exception:
                    return {"error": body.get("error", body)}
    except Exception as exc:
        return {"error": str(exc)}


async def _get_chg_state(sys_id: str) -> str:
    """Return the current numeric state value of a change request."""
    r = await _get("/api/now/table/change_request", params={
        "sysparm_query": f"sys_id={sys_id}",
        "sysparm_fields": "state",
        "sysparm_limit": "1",
        "sysparm_display_value": "false",
    })
    recs = _records(r)
    return str(recs[0].get("state", "")) if recs else ""


async def _walk_to_closed(sys_id: str, close_notes: str, assignment_group: str = "") -> Dict[str, Any]:
    """
    Walk a change request through the full Normal workflow to Closed state,
    automatically approving any pending approval records at each blocked transition.
    State path: New(-5) → Assess(-4) → Authorize(-3) → Implement(-1) → Review(0) → Closed(3)
    """
    STATE_ORDER = {"-5": 0, "-4": 1, "-3": 2, "-2": 3, "-1": 4, "0": 5, "3": 6}
    path = ["-4", "-3", "-1", "0", "3"]

    # Ensure assignment_group is set (required for New→Assess transition)
    if assignment_group:
        await _patch(f"/api/now/table/change_request/{sys_id}", {"assignment_group": assignment_group})
    else:
        grp = await _get("/api/now/table/sys_user_group", params={
            "sysparm_query": "nameLIKENetwork", "sysparm_fields": "sys_id",
            "sysparm_limit": "1", "sysparm_display_value": "false",
        })
        grp_records = _records(grp)
        if grp_records:
            await _patch(f"/api/now/table/change_request/{sys_id}",
                         {"assignment_group": grp_records[0]["sys_id"]})

    for target_state in path:
        # Re-read current state each iteration to catch auto-transitions from approvals
        for _retry in range(4):  # up to 4 approve-and-retry cycles per state
            current = await _get_chg_state(sys_id)
            if current == "3":
                break  # closed — done

            # Already at or past this target (auto-transitioned)
            if STATE_ORDER.get(current, -1) >= STATE_ORDER.get(target_state, -1):
                break

            payload: Dict[str, Any] = {"state": target_state}
            if target_state == "3":
                payload["close_notes"] = close_notes or "Successfully implemented"

            r = await _chg_rest_patch(sys_id, payload)
            err = r.get("error")

            if not err:
                break  # transition succeeded — move to next path entry

            # Approve pending records (handles both explicit and silent approval blocks)
            conditions = err.get("conditions", []) if isinstance(err, dict) else []
            has_approval_condition = any(
                "approv" in c.get("condition", {}).get("name", "").lower()
                for c in conditions if not c.get("passed")
            )
            transition_unavailable = isinstance(err, dict) and err.get("transition_available") is False

            if has_approval_condition or transition_unavailable:
                approved = await _approve_pending(sys_id)
                logger.info("_walk_to_closed: approved %d pending records targeting state %s",
                            approved, target_state)
                if approved > 0:
                    continue  # retry — approvals may have triggered auto-transition
            # No approvals to grant or non-approval block — move on to next state
            logger.info("_walk_to_closed: could not reach state %s: %s",
                        target_state, err.get("display_value") or str(err)[:80])
            break  # outer for-loop moves to next target_state

        current = await _get_chg_state(sys_id)
        if current == "3":
            break  # closed

    # Return final record
    final = await _get("/api/now/table/change_request", params={
        "sysparm_query": f"sys_id={sys_id}",
        "sysparm_fields": "number,state",
        "sysparm_limit": "1",
        "sysparm_display_value": "true",
    })
    final_rec = _records(final)
    if final_rec:
        return {"result": final_rec[0]}
    return {"result": {"sys_id": sys_id}}


@mcp.tool()
async def update_servicenow_change_request(
    number: str,
    state: Optional[str] = None,
    work_notes: Optional[str] = None,
    assigned_to: Optional[str] = None,
    close_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update an existing ServiceNow change request.
    When state="closed", automatically walks through the full approval workflow
    (Assess → Authorize → Implement → Review → Closed), approving any pending
    approval records along the way.

    Args:
        number:      Change request number (e.g. "CHG0030001")
        state:       New state: "new", "assess", "authorize", "scheduled", "implement", "review", "closed"
        work_notes:  Internal notes to add
        assigned_to: Username to assign the change to
        close_notes: Closure notes (required when closing)

    Returns:
        Updated change request record.
    """
    STATE_MAP = {
        "new": "-5", "assess": "-4", "authorize": "-3", "scheduled": "-2",
        "implement": "-1", "review": "0", "closed": "3",
    }

    lookup = await _get("/api/now/table/change_request", params={
        "sysparm_query": f"number={number}",
        "sysparm_limit": 1,
        "sysparm_fields": "sys_id",
        "sysparm_display_value": "false",
    })
    records = _records(lookup)
    if not records:
        return {"error": f"Change request {number} not found."}
    sys_id = records[0]["sys_id"]

    logger.info("update_servicenow_change_request: %s sys_id=%s state=%s", number, sys_id, state)

    # Handle closing with full workflow walk
    if state and STATE_MAP.get(state.lower()) == "3":
        result = await _walk_to_closed(sys_id, close_notes or "", assigned_to or "")
        # Also apply work_notes if provided
        if work_notes:
            await _patch(f"/api/now/table/change_request/{sys_id}", {"work_notes": work_notes})
        return result

    # Simple field update (no state or non-closing state change)
    payload: Dict[str, Any] = {}
    if state:
        payload["state"] = STATE_MAP.get(state.lower(), state)
    if work_notes:
        payload["work_notes"] = work_notes
    if assigned_to:
        payload["assigned_to"] = assigned_to
    if close_notes:
        payload["close_notes"] = close_notes

    return await _patch(f"/api/now/table/change_request/{sys_id}", payload)


# ---------------------------------------------------------------------------
# Problems
# ---------------------------------------------------------------------------

@mcp.tool()
async def list_servicenow_problems(
    state: Optional[str] = None,
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List ServiceNow problem records.

    Args:
        state: Filter by state: "open", "known_error", "closed"
        limit: Max number of results (default 20)

    Returns:
        List of problem records with number, short_description, state, and assigned_to.
    """
    STATE_MAP = {"open": "1", "known_error": "2", "closed": "3"}
    query_parts = []
    if state:
        query_parts.append(f"state={STATE_MAP.get(state.lower(), state)}")

    params = {
        "sysparm_query": "^".join(query_parts) + "^ORDERBYDESCsys_created_on" if query_parts else "ORDERBYDESCsys_created_on",
        "sysparm_limit": min(limit, 100),
        "sysparm_fields": "number,short_description,state,priority,assigned_to,assignment_group,opened_at",
    }
    return await _cached_get("list_problems", [params["sysparm_query"], str(params["sysparm_limit"])],
                             "/api/now/table/problem", params=params)


@mcp.tool()
async def create_servicenow_problem(
    short_description: str,
    description: str = "",
    assignment_group: str = "",
) -> Dict[str, Any]:
    """
    Create a new ServiceNow problem record.

    Args:
        short_description: Brief summary (required)
        description:       Detailed description
        assignment_group:  Name of the assignment group

    Returns:
        Created problem record with sys_id and number.
    """
    payload: Dict[str, Any] = {
        "short_description": short_description,
        "description": description,
    }
    if assignment_group:
        payload["assignment_group"] = assignment_group
    logger.info("create_servicenow_problem: %s", short_description)
    return await _post("/api/now/table/problem", payload)


# ---------------------------------------------------------------------------
# CMDB — Configuration Items
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_servicenow_ci(
    name: str = "",
    ip_address: str = "",
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search for a ServiceNow CMDB configuration item by name or IP address.

    Args:
        name:       CI name or partial name to search for
        ip_address: IP address of the CI
        limit:      Max number of results (default 10)

    Returns:
        List of matching CIs with name, ip_address, sys_class_name, operational_status.
    """
    query_parts = []
    if name:
        query_parts.append(f"nameLIKE{name}")
    if ip_address:
        query_parts.append(f"ip_address={ip_address}")

    if not query_parts:
        return {"error": "Provide at least one of: name, ip_address"}

    params = {
        "sysparm_query": "^OR".join(query_parts),
        "sysparm_limit": min(limit, 50),
        "sysparm_fields": "name,sys_id,ip_address,sys_class_name,operational_status,serial_number,manufacturer,model_id,location,assigned_to",
    }
    logger.info("get_servicenow_ci: name=%s ip=%s", name, ip_address)
    return await _cached_get("get_ci", [name, ip_address, str(params["sysparm_limit"])],
                             "/api/now/table/cmdb_ci", params=params)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

@mcp.tool()
async def get_servicenow_user(
    username: str = "",
    email: str = "",
    name: str = "",
) -> Dict[str, Any]:
    """
    Look up a ServiceNow user by username, email, or full name.

    Args:
        username: ServiceNow user_name field
        email:    User email address
        name:     User full name (partial match)

    Returns:
        Matching user records with name, email, department, title, active status.
    """
    query_parts = []
    if username:
        query_parts.append(f"user_name={username}")
    if email:
        query_parts.append(f"email={email}")
    if name:
        query_parts.append(f"nameLIKE{name}")

    if not query_parts:
        return {"error": "Provide at least one of: username, email, name"}

    params = {
        "sysparm_query": "^OR".join(query_parts),
        "sysparm_limit": 5,
        "sysparm_fields": "user_name,name,email,department,title,active,phone,manager",
    }
    logger.info("get_servicenow_user: %s", query_parts)
    return await _cached_get("get_user", [username, email, name], "/api/now/table/sys_user", params=params)


# ---------------------------------------------------------------------------
# Full-text search across incidents
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_servicenow_incidents(
    query: str,
    limit: int = 10,
    updated_within_hours: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Full-text search across ServiceNow incidents — searches short_description,
    description, and work_notes for the given query string.

    Use this when the user asks for tickets related to a device, IP, or keyword
    (e.g. "any tickets related to PA-FW-01", "incidents mentioning 10.0.0.1").

    Args:
        query:                Search term (e.g. "PA-FW-01", "10.0.0.1", "network outage")
        limit:                Max number of results (default 10)
        updated_within_hours: Only return incidents updated within the last N hours (e.g. 24)

    Returns:
        List of matching incidents with number, short_description, state, priority, work_notes.
    """
    # Support "A OR B OR C" — split into separate LIKE conditions per term
    import re as _re
    terms = [t.strip() for t in _re.split(r'\bOR\b', query, flags=_re.IGNORECASE) if t.strip()]
    if not terms:
        q = (query or "").strip()
        terms = [q] if q else []
    # Empty LIKE terms match broadly in some instances — never query unscoped
    if not terms:
        logger.info("search_servicenow_incidents: empty query, returning no rows")
        return {"result": []}
    or_clauses = "^OR".join(
        f"short_descriptionLIKE{t}^ORdescriptionLIKE{t}^ORwork_notesLIKE{t}^ORcmdb_ciLIKE{t}"
        for t in terms
    )
    time_filter = ""
    if updated_within_hours:
        from datetime import datetime, timezone, timedelta
        since = datetime.now(timezone.utc) - timedelta(hours=updated_within_hours)
        time_filter = f"^sys_updated_on>={since.strftime('%Y-%m-%d %H:%M:%S')}"
    params = {
        "sysparm_query": f"{or_clauses}{time_filter}^ORDERBYDESCsys_updated_on",
        "sysparm_limit": min(limit, 50),
        "sysparm_fields": "number,short_description,description,state,priority,urgency,impact,assigned_to,assignment_group,cmdb_ci,opened_at,resolved_at,sys_updated_on,close_notes,work_notes",
    }
    logger.info("search_servicenow_incidents: %s (terms=%s, hours=%s)", query, terms, updated_within_hours)
    if updated_within_hours:
        return await _get("/api/now/table/incident", params=params)
    return await _cached_get("search_incidents", [query, str(limit)], "/api/now/table/incident", params=params)


# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

@mcp.tool()
async def search_servicenow_knowledge(
    query: str,
    limit: int = 5,
) -> Dict[str, Any]:
    """
    Search the ServiceNow knowledge base for articles matching a query.

    Args:
        query: Search terms (e.g. "VPN setup", "password reset")
        limit: Max number of articles to return (default 5)

    Returns:
        List of knowledge articles with number, short_description, and text.
    """
    params = {
        "sysparm_query": f"short_descriptionLIKE{query}^ORtextLIKE{query}^workflow_state=published",
        "sysparm_limit": min(limit, 20),
        "sysparm_fields": "number,short_description,text,kb_category,sys_updated_on",
    }
    logger.info("search_servicenow_knowledge: %s", query)
    return await _cached_get("search_knowledge", [query, str(limit)], "/api/now/table/kb_knowledge", params=params)
