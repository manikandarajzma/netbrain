"""Agent-facing product tools for ServiceNow records and change/incident CRUD."""
from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

try:
    from atlas.integrations.mcp_client import call_mcp_tool
    from atlas.services.backend_contracts import (
        lookup_error,
        not_found,
        operation_failed,
        unexpected_response,
        verification_failed,
    )
    from atlas.services.pending_approval import pending_approval_store
    from atlas.tools.tool_runtime import push_status, sid_from_config
except ImportError:
    from integrations.mcp_client import call_mcp_tool  # type: ignore
    from services.backend_contracts import (  # type: ignore
        lookup_error,
        not_found,
        operation_failed,
        unexpected_response,
        verification_failed,
    )
    from services.pending_approval import pending_approval_store  # type: ignore
    from tools.tool_runtime import push_status, sid_from_config  # type: ignore


logger = logging.getLogger("atlas.tools.servicenow")


def _config_value(config: RunnableConfig, key: str, default: str = "") -> str:
    return str((config or {}).get("configurable", {}).get(key, default) or default)


def _approval_mode(config: RunnableConfig) -> str:
    return _config_value(config, "approval_mode", "draft").strip().lower()


def _request_prompt(config: RunnableConfig) -> str:
    return _config_value(config, "request_prompt", "")


def _format_fields(fields: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in fields.items():
        if value in ("", None):
            continue
        label = str(key).replace("_", " ").title()
        lines.append(f"- {label}: {value}")
    return "\n".join(lines) if lines else "- No fields captured."


def _build_pending_approval_payload(
    tool_name: str,
    action_label: str,
    arguments: dict[str, Any],
    config: RunnableConfig,
) -> dict[str, Any]:
    return {
        "approval_id": uuid4().hex,
        "tool_name": tool_name,
        "action_label": action_label,
        "fields": {k: v for k, v in arguments.items() if v not in ("", None)},
        "original_prompt": _request_prompt(config),
    }


def _build_pending_approval_text(action_label: str, fields: dict[str, Any]) -> str:
    return (
        f"Proposed action: {action_label}\n\n"
        "This write action has not been executed yet.\n\n"
        "Fields:\n"
        f"{_format_fields(fields)}\n\n"
        "Reply with `confirm` to execute it, `cancel` to discard it, or tell me what to change."
    )


def _stage_pending_approval(
    session_id: str,
    tool_name: str,
    action_label: str,
    arguments: dict[str, Any],
    config: RunnableConfig,
) -> str:
    payload = _build_pending_approval_payload(tool_name, action_label, arguments, config)
    pending_approval_store.set(session_id, payload)
    return _build_pending_approval_text(action_label, payload["fields"])


async def _execute_create_servicenow_incident(arguments: dict[str, Any], session_id: str) -> str:
    await push_status(session_id, "Creating ServiceNow incident...")

    try:
        result = await call_mcp_tool(
            "create_servicenow_incident",
            {
                "short_description": arguments.get("short_description", ""),
                "description": arguments.get("description", ""),
                "urgency": arguments.get("urgency", "2"),
                "impact": arguments.get("impact", "2"),
                "category": "network",
                "ci_name": arguments.get("ci_name", ""),
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
        f"short_description: {arguments.get('short_description', '')}"
    )


async def _execute_create_servicenow_change_request(arguments: dict[str, Any], session_id: str) -> str:
    await push_status(session_id, "Creating ServiceNow change request...")

    try:
        result = await call_mcp_tool(
            "create_servicenow_change_request",
            {
                "short_description": arguments.get("short_description", ""),
                "description": arguments.get("description", ""),
                "risk": arguments.get("risk", "3"),
                "assignment_group": arguments.get("assignment_group", ""),
                "justification": arguments.get("justification", ""),
                "implementation_plan": arguments.get("implementation_plan", ""),
                "ci_name": arguments.get("ci_name", ""),
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
        f"short_description: {arguments.get('short_description', '')}"
    )


async def _execute_update_servicenow_change_request(arguments: dict[str, Any], session_id: str) -> str:
    number = str(arguments.get("number") or "")
    await push_status(session_id, f"Updating ServiceNow change request {number}...")

    try:
        result = await call_mcp_tool("update_servicenow_change_request", arguments, timeout=20.0)
    except Exception as exc:
        return operation_failed("Change request update", exc)

    if not isinstance(result, dict):
        return unexpected_response("Change request update", result)
    if "error" in result:
        return operation_failed("Change request update", result["error"])

    r = result.get("result", {}) if isinstance(result.get("result"), dict) else result
    out_number = r.get("number") or number
    out_state = r.get("state") or arguments.get("state") or "updated"
    return f"Updated ServiceNow change request {out_number}.\nstate: {out_state}"


async def execute_pending_write_action(approval: dict[str, Any], session_id: str) -> str:
    tool_name = str(approval.get("tool_name") or "").strip()
    arguments = approval.get("fields") or approval.get("arguments") or {}
    if not isinstance(arguments, dict):
        return "Pending approval execution failed: invalid stored arguments"

    if tool_name == "create_servicenow_incident":
        return await _execute_create_servicenow_incident(arguments, session_id)
    if tool_name == "create_servicenow_change_request":
        return await _execute_create_servicenow_change_request(arguments, session_id)
    if tool_name == "update_servicenow_change_request":
        return await _execute_update_servicenow_change_request(arguments, session_id)
    return f"Pending approval execution failed: unsupported tool '{tool_name}'"

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
    Stage a ServiceNow incident for approval.
    Use for explicit requests to create/open/raise an incident or ticket.
    Atlas executes the write only after the user confirms.
    """
    session_id = sid_from_config(config)
    args = {
        "short_description": short_description,
        "description": description,
        "urgency": urgency,
        "impact": impact,
        "ci_name": ci_name,
    }
    if _approval_mode(config) != "execute":
        await push_status(session_id, "Staging ServiceNow incident for approval...")
        return _stage_pending_approval(
            session_id,
            "create_servicenow_incident",
            "Create ServiceNow incident",
            args,
            config,
        )
    return await _execute_create_servicenow_incident(args, session_id)


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
    Stage a ServiceNow change request for approval.
    Use for explicit requests to create/open/submit a change request.
    Atlas executes the write only after the user confirms.
    """
    session_id = sid_from_config(config)
    args = {
        "short_description": short_description,
        "description": description,
        "risk": risk,
        "assignment_group": assignment_group,
        "justification": justification,
        "implementation_plan": implementation_plan,
        "ci_name": ci_name,
    }
    if _approval_mode(config) != "execute":
        await push_status(session_id, "Staging ServiceNow change request for approval...")
        return _stage_pending_approval(
            session_id,
            "create_servicenow_change_request",
            "Create ServiceNow change request",
            args,
            config,
        )
    return await _execute_create_servicenow_change_request(args, session_id)


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
    Stage an update to an existing ServiceNow change request for approval.
    Use for explicit requests to close/update a change request such as CHG0030001.
    Atlas executes the write only after the user confirms.
    """
    session_id = sid_from_config(config)
    args: dict[str, Any] = {"number": number}
    if state:
        args["state"] = state
    if work_notes:
        args["work_notes"] = work_notes
    if assigned_to:
        args["assigned_to"] = assigned_to
    if close_notes:
        args["close_notes"] = close_notes
    if _approval_mode(config) != "execute":
        await push_status(session_id, f"Staging change request update for {number}...")
        return _stage_pending_approval(
            session_id,
            "update_servicenow_change_request",
            f"Update ServiceNow change request {number}",
            args,
            config,
        )
    return await _execute_update_servicenow_change_request(args, session_id)


SERVICENOW_TOOL_CAPABILITIES = (
    (get_incident_details, ("servicenow.incident.read",)),
    (get_change_request_details, ("servicenow.change.read",)),
    (create_servicenow_incident, ("servicenow.incident.create",)),
    (create_servicenow_change_request, ("servicenow.change.create",)),
    (update_servicenow_change_request, ("servicenow.change.update",)),
)
