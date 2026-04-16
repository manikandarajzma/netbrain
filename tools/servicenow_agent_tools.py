"""Agent-facing product tools for ServiceNow records and change/incident CRUD."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.mcp_client import call_mcp_tool
    from atlas.services.backend_contracts import (
        lookup_error,
        not_found,
        operation_failed,
        unexpected_response,
        verification_failed,
    )
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from mcp_client import call_mcp_tool  # type: ignore
    from services.backend_contracts import (  # type: ignore
        lookup_error,
        not_found,
        operation_failed,
        unexpected_response,
        verification_failed,
    )
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


logger = logging.getLogger("atlas.tools.servicenow")

@tool
async def get_incident_details(incident_number: str, config: RunnableConfig) -> str:
    """
    Fetch full details for a specific ServiceNow incident by number (e.g. INC0010035).
    Use when the user references a specific incident and wants its description or to troubleshoot it.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Fetching {incident_number}...")

    try:
        data = await call_mcp_tool(
            "get_servicenow_incident",
            {"number": incident_number.upper().strip()},
            timeout=20.0,
        )
    except Exception as exc:
        return lookup_error("Incident", exc)

    if "error" in data:
        return not_found("Incident", data["error"])
    r = data.get("result", {})
    lines = [
        f"**{r.get('number')}** — {r.get('short_description', '')}",
        f"State: {r.get('state')} | Priority: {r.get('priority')} | Opened: {r.get('opened_at')}",
        f"Assigned to: {(r.get('assigned_to') or {}).get('display_value', 'Unassigned')}",
    ]
    if desc := r.get("description") or r.get("short_description"):
        lines.append(f"Description: {desc}")
    if notes := r.get("close_notes"):
        lines.append(f"Resolution: {notes}")
    return "\n".join(lines)


@tool
async def get_change_request_details(change_number: str, config: RunnableConfig) -> str:
    """
    Fetch full details for a specific ServiceNow change request by number (e.g. CHG0010001).
    Use when the user references a specific change request and wants its details or current state.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Fetching {change_number}...")

    try:
        data = await call_mcp_tool(
            "get_servicenow_change_request",
            {"number": change_number.upper().strip()},
            timeout=20.0,
        )
    except Exception as exc:
        return lookup_error("Change request", exc)

    if "error" in data:
        return not_found("Change request", data["error"])
    r = data.get("result", {})
    lines = [
        f"**{r.get('number')}** — {r.get('short_description', '')}",
        f"State: {r.get('state')} | Risk: {r.get('risk', 'Unknown')}",
        f"Assignment group: {(r.get('assignment_group') or {}).get('display_value', 'Unassigned')}",
        f"Configuration Item: {(r.get('cmdb_ci') or {}).get('display_value', r.get('cmdb_ci', 'Unknown'))}",
    ]
    if desc := r.get("description") or r.get("short_description"):
        lines.append(f"Description: {desc}")
    if just := r.get("justification"):
        lines.append(f"Justification: {just}")
    if plan := r.get("implementation_plan"):
        lines.append(f"Implementation plan: {plan}")
    if notes := r.get("close_notes"):
        lines.append(f"Close notes: {notes}")
    return "\n".join(lines)


@tool
async def create_servicenow_incident(
    short_description: str,
    config: RunnableConfig,
    description: str = "",
    urgency: str = "2",
    impact: str = "2",
    ci_name: str = "",
) -> str:
    """
    Create a ServiceNow incident.
    Use for explicit requests to create/open/raise an incident or ticket.
    Returns the created incident number and key details.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, "Creating ServiceNow incident...")

    try:
        result = await call_mcp_tool(
            "create_servicenow_incident",
            {
                "short_description": short_description,
                "description": description,
                "urgency": urgency,
                "impact": impact,
                "category": "network",
                "ci_name": ci_name,
            },
            timeout=20.0,
        )
    except Exception as exc:
        return operation_failed("Incident creation", exc)

    if not isinstance(result, dict):
        return unexpected_response("Incident creation", result)
    if "error" in result:
        return operation_failed("Incident creation", result["error"])

    r = result.get("result", {}) if isinstance(result.get("result"), dict) else result
    number = r.get("number") or r.get("display_value") or "unknown"
    sys_id = r.get("sys_id") or "unknown"
    if number != "unknown":
        verify = await call_mcp_tool(
            "get_servicenow_incident",
            {"number": str(number).upper().strip()},
            timeout=20.0,
        )
        if isinstance(verify, dict) and "error" in verify:
            return verification_failed("Incident creation", str(number), verify["error"])
    return (
        f"Created ServiceNow incident {number}.\n"
        f"sys_id: {sys_id}\n"
        f"short_description: {short_description}"
    )


@tool
async def create_servicenow_change_request(
    short_description: str,
    config: RunnableConfig,
    description: str = "",
    risk: str = "3",
    assignment_group: str = "",
    justification: str = "",
    implementation_plan: str = "",
    ci_name: str = "",
) -> str:
    """
    Create a ServiceNow change request.
    Use for explicit requests to create/open/submit a change request.
    Returns the created change number and key details.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, "Creating ServiceNow change request...")

    try:
        result = await call_mcp_tool(
            "create_servicenow_change_request",
            {
                "short_description": short_description,
                "description": description,
                "risk": risk,
                "assignment_group": assignment_group,
                "justification": justification,
                "implementation_plan": implementation_plan,
                "ci_name": ci_name,
            },
            timeout=20.0,
        )
    except Exception as exc:
        return operation_failed("Change request creation", exc)

    if not isinstance(result, dict):
        return unexpected_response("Change request creation", result)
    if "error" in result:
        return operation_failed("Change request creation", result["error"])

    r = result.get("result", {}) if isinstance(result.get("result"), dict) else result
    number = r.get("number") or r.get("display_value") or "unknown"
    sys_id = r.get("sys_id") or "unknown"
    if number != "unknown":
        verify = await call_mcp_tool(
            "get_servicenow_change_request",
            {"number": str(number).upper().strip()},
            timeout=20.0,
        )
        if isinstance(verify, dict) and "error" in verify:
            return verification_failed("Change request creation", str(number), verify["error"])
    return (
        f"Created ServiceNow change request {number}.\n"
        f"sys_id: {sys_id}\n"
        f"short_description: {short_description}"
    )


@tool
async def update_servicenow_change_request(
    number: str,
    config: RunnableConfig,
    state: str = "",
    work_notes: str = "",
    assigned_to: str = "",
    close_notes: str = "",
) -> str:
    """
    Update an existing ServiceNow change request.
    Use for explicit requests to close/update a change request such as CHG0030001.
    """
    session_id = sid_from_config(config)
    await push_status(session_id, f"Updating ServiceNow change request {number}...")

    args: dict[str, Any] = {"number": number}
    if state:
        args["state"] = state
    if work_notes:
        args["work_notes"] = work_notes
    if assigned_to:
        args["assigned_to"] = assigned_to
    if close_notes:
        args["close_notes"] = close_notes

    try:
        result = await call_mcp_tool("update_servicenow_change_request", args, timeout=20.0)
    except Exception as exc:
        return operation_failed("Change request update", exc)

    if not isinstance(result, dict):
        return unexpected_response("Change request update", result)
    if "error" in result:
        return operation_failed("Change request update", result["error"])

    r = result.get("result", {}) if isinstance(result.get("result"), dict) else result
    out_number = r.get("number") or number
    out_state = r.get("state") or state or "updated"
    return f"Updated ServiceNow change request {out_number}.\nstate: {out_state}"


SERVICENOW_TOOL_CAPABILITIES = (
    (get_incident_details, ("servicenow.incident.read",)),
    (get_change_request_details, ("servicenow.change.read",)),
    (create_servicenow_incident, ("servicenow.incident.create",)),
    (create_servicenow_change_request, ("servicenow.change.create",)),
    (update_servicenow_change_request, ("servicenow.change.update",)),
)
