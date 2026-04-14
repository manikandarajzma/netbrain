"""Prompt and message preprocessing helpers used by graph orchestration."""
from __future__ import annotations

import logging
import re

logger = logging.getLogger("atlas.request_preprocessor")

_INC_RE = re.compile(r"\bINC\d+\b", re.IGNORECASE)
_IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


def extract_ips(text: str) -> tuple[str, str]:
    ips = _IP_RE.findall(text or "")
    return (ips[0] if ips else ""), (ips[1] if len(ips) > 1 else "")


def extract_port(text: str) -> str:
    m = re.search(r"\bport\s+(\d{1,5})\b", text or "", re.IGNORECASE)
    return m.group(1) if m else ""


def extract_final_text(messages: list) -> str:
    text = next(
        (
            m.content
            for m in reversed(messages)
            if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)
        ),
        "",
    )
    text = re.sub(r"<plan>.*?</plan>", "", text, flags=re.DOTALL)
    text = re.sub(r"<reflection>.*?</reflection>", "", text, flags=re.DOTALL)
    return text.strip()


def looks_like_clarification_request(text: str) -> bool:
    lowered = (text or "").lower()
    cues = (
        "please provide the following",
        "i need more details",
        "to proceed, please provide",
        "once i have these details",
        "please provide the following information",
        "i need the following information",
    )
    return any(cue in lowered for cue in cues)


async def resolve_incident_prompt(prompt: str) -> tuple[str, dict | None]:
    """Expand INC→connectivity prompt when the prompt has an INC number but no IPs."""
    match = _INC_RE.search(prompt or "")
    if not match or _IP_RE.search(prompt or ""):
        return prompt, None
    inc_num = match.group(0).upper()
    try:
        try:
            from atlas.mcp_client import call_mcp_tool
        except ImportError:
            from mcp_client import call_mcp_tool  # type: ignore

        data = await call_mcp_tool(
            "get_servicenow_incident",
            {"number": inc_num},
            timeout=20.0,
        )
        if not isinstance(data, dict) or "error" in data:
            return prompt, None

        result = data.get("result", {})
        short_desc = str(result.get("short_description") or "")
        description = str(result.get("description") or "")
        work_notes = str(result.get("work_notes") or "")
        comments = str(result.get("comments") or "")
        combined = " ".join(
            part for part in (short_desc, description, work_notes, comments) if part
        )
        ips = _IP_RE.findall(combined)
        if len(ips) < 2:
            return prompt, None

        port_hit = re.search(
            r"\b(?:tcp\s+port|udp\s+port|port)\s+(\d+)\b",
            combined,
            re.IGNORECASE,
        )
        port_str = f" port {port_hit.group(1)}" if port_hit else ""
        new_prompt = (
            f"help me troubleshoot connectivity from {ips[0]} to {ips[1]}{port_str} "
            f"based on incident {inc_num}"
        )
        logger.info("INC→IP resolved: %s → %s", inc_num, new_prompt[-80:])
        summary = {
            "number": result.get("number", inc_num),
            "short_description": short_desc,
            "state": result.get("state", ""),
            "priority": result.get("priority", ""),
            "opened_at": result.get("opened_at", ""),
            "assigned_to": (result.get("assigned_to") or {}).get("display_value") or "Unassigned",
        }
        return new_prompt, summary
    except Exception as exc:
        logger.warning("INC→IP resolution failed: %s", exc)
        return prompt, None

