"""
Atlas troubleshooting agent.

Single entry point: orchestrate_troubleshoot().

Architecture:
  create_react_agent(llm, ALL_TOOLS, prompt=system_prompt)
  ↓
  LLM drives all tool calls (parallel where sensible).
  Tools write structured side-effect data (path_hops, interface_counters)
  to a per-session-id store keyed on RunnableConfig["configurable"]["session_id"].
  ↓
  After the ReAct loop finishes, we read that store and attach structured
  data to the API response alongside the LLM's text report.
"""
from __future__ import annotations

import logging
import pathlib
import re

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

try:
    from atlas.tools.all_tools import ALL_TOOLS, pop_session_data
    from atlas.tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL
except ImportError:
    from tools.all_tools import ALL_TOOLS, pop_session_data  # type: ignore
    from tools.shared import OLLAMA_MODEL, OLLAMA_BASE_URL    # type: ignore

logger = logging.getLogger("atlas.troubleshoot_agent")

_SKILLS_DIR    = pathlib.Path(__file__).parent.parent / "skills"
_CORE_PROMPT   = _SKILLS_DIR / "troubleshooter.md"
_SCENARIOS_DIR = _SKILLS_DIR / "troubleshooting_scenarios"

# Maps issue_type (from classify_intent) or keyword patterns → scenario file name
_SCENARIO_FILES = {
    "blocked":      "connectivity.md",
    "connectivity": "connectivity.md",
    "slow":         "performance.md",
    "performance":  "performance.md",
    "intermittent": "intermittent.md",
    "flapping":     "intermittent.md",
}

_SCENARIO_KEYWORDS = [
    (re.compile(r"\b(block|deny|denied|reject|can.?t connect|port|tcp|udp|refused|unreachable)\b", re.IGNORECASE), "connectivity.md"),
    (re.compile(r"\b(slow|latency|lag|delay|degraded|throughput|performance|high rtt)\b", re.IGNORECASE),           "performance.md"),
    (re.compile(r"\b(intermittent|flap|unstable|sporadic|random|drops in and out)\b", re.IGNORECASE),               "intermittent.md"),
]


def _pick_scenario(prompt: str, issue_type: str) -> str | None:
    """Return the path of the most relevant scenario file, or None."""
    # Prefer the explicit issue_type from the classifier
    fname = _SCENARIO_FILES.get(issue_type)
    if not fname:
        # Fall back to keyword scan on the prompt
        for pattern, candidate in _SCENARIO_KEYWORDS:
            if pattern.search(prompt):
                fname = candidate
                break
    if not fname:
        return None
    path = _SCENARIOS_DIR / fname
    return str(path) if path.exists() else None


def _load_system_prompt(prompt: str = "", issue_type: str = "general") -> str:
    core = _CORE_PROMPT.read_text(encoding="utf-8").strip() if _CORE_PROMPT.exists() else ""
    scenario_path = _pick_scenario(prompt, issue_type)
    if scenario_path:
        scenario_text = pathlib.Path(scenario_path).read_text(encoding="utf-8").strip()
        logger.info("Loaded scenario: %s", scenario_path)
        return core + "\n\n---\n\n" + scenario_text
    return core


def _extract_ips(text: str) -> tuple[str, str]:
    ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
    return (ips[0] if len(ips) > 0 else ""), (ips[1] if len(ips) > 1 else "")


def _extract_port(text: str) -> str:
    m = re.search(r'\bport\s+(\d+)\b|/(\d+)\b|\b(\d+)/(tcp|udp)\b', text, re.IGNORECASE)
    if m:
        return next(g for g in m.groups() if g and g.isdigit())
    return ""


async def _resolve_inc(prompt: str) -> tuple[str, dict | None]:
    """
    If the prompt references an INC number but has no IPs, look it up and
    append the IPs from the incident description to the prompt.
    Returns (updated_prompt, incident_summary_dict_or_None).
    """
    _INC_RE = re.compile(r'\bINC\d+\b', re.IGNORECASE)
    m = _INC_RE.search(prompt)
    if not m or _extract_ips(prompt)[0]:
        return prompt, None

    inc_num = m.group(0).upper()
    try:
        try:
            from atlas.tools.servicenow_tools import get_servicenow_incident as _t
        except ImportError:
            from tools.servicenow_tools import get_servicenow_incident as _t  # type: ignore
        fn   = getattr(_t, "fn", None) or _t
        data = await fn(inc_num)
        if "error" in data:
            return prompt, None
        r     = data.get("result", {})
        desc  = r.get("description") or r.get("short_description") or ""
        ips   = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', desc)
        if len(ips) < 2:
            return prompt, None

        port_hit = re.search(r'\bport\s+(\d+)\b', desc, re.IGNORECASE)
        port_str = f" port {port_hit.group(1)}" if port_hit else ""
        new_prompt = f"{prompt} (source: {ips[0]}, destination: {ips[1]}{port_str})"
        logger.info("INC→IP resolved: %s → %s", inc_num, new_prompt[-60:])

        inc_summary = {
            "number":           r.get("number", inc_num),
            "short_description": r.get("short_description", ""),
            "state":            r.get("state", ""),
            "priority":         r.get("priority", ""),
            "opened_at":        r.get("opened_at", ""),
            "assigned_to":      (r.get("assigned_to") or {}).get("display_value") or "Unassigned",
        }
        return new_prompt, inc_summary
    except Exception as exc:
        logger.warning("INC→IP resolution failed: %s", exc)
        return prompt, None


async def _push_status(session_id: str, msg: str) -> None:
    try:
        try:
            import atlas.status_bus as sb
        except ImportError:
            import status_bus as sb  # type: ignore
        await sb.push(session_id, msg)
    except Exception:
        pass


async def orchestrate_troubleshoot(
    prompt: str,
    username: str | None = None,
    session_id: str | None = None,
    issue_type: str = "general",
) -> dict:
    """
    Run the troubleshooting ReAct agent and return a structured response dict.

    The LLM (via create_react_agent) drives all tool calls.  Tools write
    structured side-effect data to the per-session store.  After the agent
    finishes, we attach that data to the response for the frontend.
    """
    session_id = session_id or "default"
    logger.info("troubleshoot_agent: prompt=%r user=%s session=%s", prompt[:80], username, session_id)

    await _push_status(session_id, "Investigating...")

    # Resolve INC number → IPs when the prompt has no IPs
    prompt, inc_summary = await _resolve_inc(prompt)

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )

    system_prompt = _load_system_prompt(prompt, issue_type)
    agent = create_react_agent(llm, ALL_TOOLS, prompt=SystemMessage(content=system_prompt))

    config = {
        "configurable": {
            "session_id": session_id,
            # LangGraph checkpointer thread — enables multi-turn memory if checkpointer attached
            "thread_id":  session_id,
        }
    }

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=prompt)]},
            config=config,
        )
    except Exception as exc:
        import traceback
        logger.error("ReAct agent failed: %s\n%s", exc, traceback.format_exc())
        return {
            "role":    "assistant",
            "content": {"direct_answer": f"Troubleshooting failed: {exc}"},
        }

    # Extract the last non-tool-call AIMessage as the final text report
    messages  = result.get("messages", [])
    final_text = next(
        (m.content for m in reversed(messages)
         if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)),
        "",
    )

    # Strip internal reasoning tags that some models leak
    final_text = re.sub(r"<plan>.*?</plan>",             "", final_text, flags=re.DOTALL).strip()
    final_text = re.sub(r"<reflection>.*?</reflection>", "", final_text, flags=re.DOTALL).strip()

    # Gather structured side-effect data from session store
    session_data = pop_session_data(session_id)
    path_hops         = session_data.get("path_hops", [])
    reverse_path_hops = session_data.get("reverse_path_hops", [])
    interface_counters = session_data.get("interface_counters", [])

    src_ip, dst_ip = _extract_ips(prompt)

    # Build response dict — frontend consumes path_hops for PathVisualization,
    # interface_counters for the InterfaceCounters widget.
    content: dict = {}
    if final_text:
        content["direct_answer"] = final_text
    if path_hops:
        content["path_hops"]   = path_hops
        content["source"]      = src_ip
        content["destination"] = dst_ip
    if reverse_path_hops:
        content["reverse_path_hops"] = reverse_path_hops
    if interface_counters:
        content["interface_counters"] = interface_counters
    if inc_summary:
        content["incident_summary"] = inc_summary

    logger.info(
        "troubleshoot_agent done: keys=%s path_hops=%d counters=%d",
        list(content.keys()), len(path_hops), len(interface_counters),
    )
    return {"role": "assistant", "content": content}
