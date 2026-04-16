"""Agent-facing vendor knowledge tools."""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

try:
    from atlas.tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL
except ImportError:
    from tools.shared import OLLAMA_BASE_URL, OLLAMA_MODEL  # type: ignore


logger = logging.getLogger("atlas.tools.knowledge")

_PLATFORM_TO_VENDOR = {
    "arista": "arista_eos",
    "eos": "arista_eos",
    "cisco": "cisco_ios",
    "ios": "cisco_ios",
    "nxos": "cisco_nxos",
    "nx-os": "cisco_nxos",
    "junos": "junos",
    "panos": "panos",
}

_VENDOR_SYSTEM_PROMPTS = {
    "arista_eos": """\
You are an expert Arista EOS network engineer writing concise vendor knowledge base entries.
Given a description of a diagnosed network problem, produce 2-3 KB entries.

Each entry must follow this exact format:
**[Short descriptive title]**
[2-4 sentences: explain the EOS-specific cause and include exact CLI commands to verify and fix.]
Reference: [EOS documentation section or relevant `show` command]

Rules:
- Use exact Arista EOS CLI syntax (e.g. `router ospf 1`, `network 10.0.0.0/8 area 0`, `ip ospf area 0`)
- Do not invent bug IDs or URLs
- Be specific and actionable""",
    "cisco_ios": """\
You are an expert Cisco IOS network engineer writing concise vendor knowledge base entries.
Given a description of a diagnosed network problem, produce 2-3 KB entries.

Each entry must follow this exact format:
**[Short descriptive title]**
[2-4 sentences: explain the IOS-specific cause and include exact CLI commands to verify and fix.]
Reference: [IOS documentation section or relevant `show` command]

Rules:
- Use exact Cisco IOS CLI syntax
- Do not invent bug IDs or URLs
- Be specific and actionable""",
}


async def _detect_vendor(devices: list[str]) -> str:
    """Return the primary vendor key for the given device list using the DB."""
    if not devices:
        return "unknown"
    try:
        try:
            from atlas.persistence.db import fetch
        except ImportError:
            from persistence.db import fetch  # type: ignore
        rows = await fetch(
            "SELECT platform FROM devices WHERE hostname = ANY($1::text[])",
            devices,
        )
        counts: dict[str, int] = {}
        for row in rows:
            platform = (row["platform"] or "").lower()
            vendor = next((v for k, v in _PLATFORM_TO_VENDOR.items() if k in platform), None)
            if vendor:
                counts[vendor] = counts.get(vendor, 0) + 1
        return max(counts, key=counts.get) if counts else "unknown"
    except Exception as exc:
        logger.warning("_detect_vendor: %s", exc)
        return "unknown"


@tool
async def lookup_vendor_kb(
    symptoms: str,
    devices: list[str],
    context: str | None = None,
) -> str:
    """
    Look up vendor-specific knowledge base entries for the given symptoms and devices.

    Use this when you need vendor-specific CLI commands, configuration examples, or
    troubleshooting tips — for example, exact Arista EOS or Cisco IOS syntax to fix
    a diagnosed problem. Call it as soon as you have enough context to describe the
    symptoms; you do not need to wait until the end of the investigation.

    The vendor is auto-detected from the device list via the device database.

    Args:
        symptoms: Natural language description of the issue
                  (e.g. "OSPF process running but ospf_interface_count=0 on arista2").
        devices:  List of device hostnames in the path (used to detect vendor).
        context:  Optional additional context from previous tool results
                  (e.g. "TCP port 443 connection refused from last hop device").
    """
    vendor = await _detect_vendor(devices)
    system_prompt = _VENDOR_SYSTEM_PROMPTS.get(vendor)
    if not system_prompt:
        return f"No vendor KB handler for vendor={vendor!r} (devices: {devices})."

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
        max_tokens=800,
    )
    user_content = f"Symptoms:\n{symptoms}"
    if context:
        user_content += f"\n\nAdditional context:\n{context}"
    user_content += "\n\nWrite 2-3 knowledge base entries."

    resp = await llm.ainvoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ])
    result = (resp.content or "").strip()
    logger.info("lookup_vendor_kb: vendor=%s devices=%s, %d chars returned", vendor, devices, len(result))
    return result or "No KB entries generated."


KNOWLEDGE_TOOL_CAPABILITIES = (
    (lookup_vendor_kb, ("knowledge.vendor.lookup",)),
)
