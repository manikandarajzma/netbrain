"""
Vendor Knowledge Base lookup agent.

After the runbook collects diagnostic evidence, this agent:
  1. Extracts structured symptoms from rb_ctx (vendor-agnostic)
  2. Detects the vendor for each device in the path (from DB)
  3. Dispatches to a vendor-specific search function
  4. Returns KB article titles + snippets to enrich LLM synthesis

Adding a new vendor: implement an async _search_<vendor>(queries) function
and map its key in VENDOR_DISPATCH.
"""

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger("atlas.vendor_lookup")


# ---------------------------------------------------------------------------
# Symptom extraction (vendor-agnostic)
# ---------------------------------------------------------------------------

# Syslog patterns → symptom type
_SYSLOG_SYMPTOM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"OSPF.*(down|dropped|dead.timer|adjacency)", re.I),  "ospf_adjacency_event"),
    (re.compile(r"LINEPROTO.*down|Interface.*down",              re.I),  "interface_link_down_event"),
    (re.compile(r"BGP.*(down|notification|reset)",               re.I),  "bgp_session_reset"),
    (re.compile(r"err.?disabl",                                  re.I),  "interface_errdisable"),
    (re.compile(r"duplex.mismatch|half.duplex",                  re.I),  "duplex_mismatch"),
]


def extract_symptoms(rb_ctx: dict) -> list[dict]:
    """
    Distil structured symptoms from the runbook context dict.
    Each symptom is {"type": str, "device"?: str, "interface"?: str, "sample"?: list[str]}.
    """
    symptoms: list[dict] = []

    # ── OSPF symptoms ──────────────────────────────────────────────────────
    ospf_ifaces  = (rb_ctx.get("ospf_interfaces") or {}).get("ospf_interfaces", {})
    ospf_history = (rb_ctx.get("ospf_history")    or {}).get("ospf_history",    {})
    ospf_nbrs    = (rb_ctx.get("ospf_neighbors")  or {}).get("ospf_neighbors",  {})

    for device, intf_data in ospf_ifaces.items():
        count = intf_data.get("ospf_interface_count", -1)
        hist_snaps = ospf_history.get(device, {}).get("history", [])
        had_neighbors = any(s.get("neighbor_count", 0) > 0 for s in hist_snaps)

        if count == 0 and had_neighbors:
            symptoms.append({"type": "ospf_not_configured", "device": device})
        elif count == 0:
            symptoms.append({"type": "ospf_process_no_interfaces", "device": device})
        elif count > 0 and ospf_nbrs.get(device, {}).get("count", 0) == 0:
            symptoms.append({"type": "ospf_adjacency_lost", "device": device})

    # ── Interface symptoms ─────────────────────────────────────────────────
    inv_device = rb_ctx.get("investigation_device", "")
    detail     = rb_ctx.get("investigation_intf_detail") or {}
    if detail and "error" not in detail:
        oper = detail.get("oper_status", "")
        lp   = detail.get("line_protocol", "")
        intf = detail.get("interface", "")
        if oper in ("disabled", "adminDown"):
            symptoms.append({"type": "interface_admin_shutdown", "device": inv_device, "interface": intf})
        elif lp == "down":
            symptoms.append({"type": "interface_link_down", "device": inv_device, "interface": intf})

    # ── Connectivity symptoms ──────────────────────────────────────────────
    if rb_ctx.get("ping_failed"):
        symptoms.append({"type": "ping_failure"})

    # ── Syslog symptom patterns ────────────────────────────────────────────
    syslog_lines: list[str] = list(rb_ctx.get("investigation_syslog") or [])
    syslog_lines += (rb_ctx.get("syslog") or {}).get("logs", [])
    seen_types: set[str] = set()
    for line in syslog_lines:
        for pattern, sym_type in _SYSLOG_SYMPTOM_PATTERNS:
            if sym_type not in seen_types and pattern.search(line):
                symptoms.append({"type": sym_type, "sample": [line]})
                seen_types.add(sym_type)

    # Deduplicate by type+device
    seen: set[str] = set()
    deduped: list[dict] = []
    for s in symptoms:
        key = f"{s['type']}:{s.get('device', '')}"
        if key not in seen:
            seen.add(key)
            deduped.append(s)

    return deduped


# ---------------------------------------------------------------------------
# Vendor detection
# ---------------------------------------------------------------------------

_PLATFORM_TO_VENDOR = {
    "arista":    "arista_eos",
    "eos":       "arista_eos",
    "cisco":     "cisco",
    "ios":       "cisco",
    "nxos":      "cisco",
    "nx-os":     "cisco",
    "panos":     "panos",
    "panorama":  "panos",
    "junos":     "junos",
    "citrix":    "citrix_adc",
    "netscaler": "citrix_adc",
}


async def detect_vendors(path_devices: list[str]) -> dict[str, str]:
    """Returns {hostname: vendor_key} for each device in path, using DB platform field."""
    if not path_devices:
        return {}
    try:
        from db import fetch
        rows = await fetch(
            "SELECT hostname, platform FROM devices WHERE hostname = ANY($1::text[])",
            path_devices,
        )
        result: dict[str, str] = {}
        for row in rows:
            platform = (row["platform"] or "").lower()
            vendor = next((v for k, v in _PLATFORM_TO_VENDOR.items() if k in platform), "unknown")
            result[row["hostname"]] = vendor
        return result
    except Exception as exc:
        logger.warning("detect_vendors: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Vendor-specific LLM guidance functions
# ---------------------------------------------------------------------------
# Each function takes a list of symptom dicts and returns KB-style entries:
#   [{"title": str, "snippet": str, "url": str}, ...]
#
# The local LLM is used instead of external search to avoid rate-limiting /
# bot-detection issues and to return precise CLI commands from training data.
# Adding a new vendor: implement async _guide_<vendor>(symptoms) and register
# it in VENDOR_DISPATCH.
# ---------------------------------------------------------------------------

_ARISTA_SYMPTOM_CONTEXT: dict[str, str] = {
    "ospf_not_configured":        "OSPF process is running but zero interfaces are participating (no `network` command or `ip ospf area` on any interface).",
    "ospf_process_no_interfaces": "OSPF process configured but no interfaces have OSPF enabled.",
    "ospf_adjacency_lost":        "OSPF adjacency was lost — interfaces are configured but no neighbor formed.",
    "interface_admin_shutdown":   "An interface is administratively shut down (`shutdown` applied).",
    "interface_link_down":        "An interface has lost its physical link (line-protocol down).",
    "ping_failure":               "End-to-end ICMP ping failed across the path.",
    "ospf_adjacency_event":       "Syslog shows OSPF adjacency state change events (neighbor went down).",
    "interface_link_down_event":  "Syslog shows interface link-down / LINEPROTO-down events.",
    "bgp_session_reset":          "BGP session reset or notification received.",
    "interface_errdisable":       "Interface placed into err-disable state.",
    "duplex_mismatch":            "Duplex mismatch detected (half-duplex on one side).",
}

_ARISTA_SYSTEM_PROMPT = """\
You are an expert Arista EOS network engineer writing vendor knowledge base entries.
Given a list of detected network symptoms, produce exactly 2-3 concise KB entries.

Each entry must follow this exact format:
**[Short article-style title]**
[2-4 sentences: explain the EOS-specific root cause, include exact `show` commands to verify, and the exact config commands to fix it. Be precise about EOS syntax.]
Reference: [EOS documentation section name or relevant `show` command]

Rules:
- Use exact Arista EOS CLI syntax (e.g. `router ospf 1`, `network 10.0.0.0/8 area 0`, `ip ospf area 0`)
- Do not invent bug IDs or article URLs
- Focus on the most actionable fix for each symptom
- Do not repeat the same content across entries
"""


async def _guide_arista(symptoms: list[dict]) -> list[dict]:
    """Use the local LLM to generate Arista EOS-specific KB entries for detected symptoms."""
    try:
        from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
    except ImportError:
        from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL

    symptom_lines = []
    for s in symptoms:
        ctx = _ARISTA_SYMPTOM_CONTEXT.get(s["type"], s["type"])
        device_part = f" (device: {s['device']})" if s.get("device") else ""
        symptom_lines.append(f"- {ctx}{device_part}")

    if not symptom_lines:
        return []

    user_msg = "Detected symptoms:\n" + "\n".join(symptom_lines) + "\n\nWrite 2-3 Arista EOS knowledge base entries for these symptoms."

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
        max_tokens=800,
    )
    resp = await llm.ainvoke([
        SystemMessage(content=_ARISTA_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ])
    raw = (resp.content or "").strip()
    if not raw:
        return []

    # Parse entries: split on bold headings **...**
    entries = []
    blocks = re.split(r'\n(?=\*\*)', raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        title_m = re.match(r'\*\*([^*]+)\*\*\s*\n?(.*)', block, re.DOTALL)
        if title_m:
            title   = title_m.group(1).strip()
            body    = title_m.group(2).strip()
            # Extract Reference line if present
            ref_m   = re.search(r'Reference:\s*(.+)', body)
            snippet = re.sub(r'\nReference:.*', '', body, flags=re.DOTALL).strip()
            ref     = ref_m.group(1).strip() if ref_m else ""
            url     = f"https://www.arista.com/en/support/product-documentation/eos — {ref}" if ref else ""
            entries.append({"title": title, "snippet": snippet, "url": url})

    logger.info("_guide_arista: parsed %d entries from LLM response", len(entries))
    return entries[:3]


# Dispatch table — add new vendors here
VENDOR_DISPATCH: dict[str, Any] = {
    "arista_eos": _guide_arista,
    # "cisco":     _guide_cisco,
    # "panos":     _guide_panos,
}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def lookup(rb_ctx: dict, path_devices: list[str]) -> dict:
    """
    Extract symptoms, detect vendor, run KB search.

    Returns:
        {
          "vendor":   str,
          "symptoms": [{"type": str, ...}, ...],
          "kb_results": [{"title": str, "snippet": str, "url": str}, ...],
        }
    """
    symptoms = extract_symptoms(rb_ctx)
    logger.info("vendor_lookup: %d symptoms extracted: %s", len(symptoms), [s["type"] for s in symptoms])

    if not symptoms:
        return {"vendor": "unknown", "symptoms": [], "kb_results": []}

    vendor_map   = await detect_vendors(path_devices)
    vendor_counts: dict[str, int] = {}
    for v in vendor_map.values():
        vendor_counts[v] = vendor_counts.get(v, 0) + 1
    primary_vendor = max(vendor_counts, key=vendor_counts.get) if vendor_counts else "unknown"

    guide_fn = VENDOR_DISPATCH.get(primary_vendor)
    if not guide_fn:
        logger.info("vendor_lookup: no guide handler for vendor=%r", primary_vendor)
        return {"vendor": primary_vendor, "symptoms": symptoms, "kb_results": []}

    kb_results = await guide_fn(symptoms)
    logger.info("vendor_lookup: %d KB entries generated for vendor=%r", len(kb_results), primary_vendor)
    return {"vendor": primary_vendor, "symptoms": symptoms, "kb_results": kb_results}
