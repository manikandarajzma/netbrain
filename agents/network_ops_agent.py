"""
Atlas network operations agent.

Handles structured output workflows that reuse path and Panorama tools
but are NOT layered troubleshooting:
  - Firewall change request / spreadsheet generation
  - Policy review (what rule currently matches this flow?)
  - Access request documentation

Shares ALL_TOOLS with troubleshoot_agent — no duplicate tools.
Gets a different system prompt: concise, document-oriented, no deep diagnosis.
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

logger = logging.getLogger("atlas.network_ops_agent")

_SKILL_PATH = pathlib.Path(__file__).parent.parent / "skills" / "network_ops.md"


def _load_system_prompt() -> str:
    return _SKILL_PATH.read_text(encoding="utf-8").strip() if _SKILL_PATH.exists() else ""


def _extract_ips(text: str) -> tuple[str, str]:
    ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', text)
    return (ips[0] if ips else ""), (ips[1] if len(ips) > 1 else "")


async def _push_status(session_id: str, msg: str) -> None:
    try:
        try:
            import atlas.status_bus as sb
        except ImportError:
            import status_bus as sb  # type: ignore
        await sb.push(session_id, msg)
    except Exception:
        pass


async def handle(
    prompt: str,
    username: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Run the network-ops ReAct agent and return a structured response dict.

    The agent traces the path, conditionally checks Panorama, then produces
    structured output (spreadsheet rows, ticket description, recommended rules).
    """
    session_id = session_id or "default"
    logger.info("network_ops_agent: prompt=%r user=%s", prompt[:80], username)

    await _push_status(session_id, "Processing network ops request...")

    system_prompt = _load_system_prompt()
    if not system_prompt:
        return {
            "role":    "assistant",
            "content": "Network ops agent is not configured yet (skills/network_ops.md missing).",
        }

    llm = ChatOpenAI(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        api_key="docker",
    )

    agent = create_react_agent(llm, ALL_TOOLS, prompt=SystemMessage(content=system_prompt))

    config = {
        "configurable": {
            "session_id": session_id,
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
        logger.error("network_ops_agent failed: %s\n%s", exc, traceback.format_exc())
        return {"role": "assistant", "content": {"direct_answer": f"Network ops agent failed: {exc}"}}

    messages = result.get("messages", [])
    final_text = next(
        (m.content for m in reversed(messages)
         if hasattr(m, "content") and m.content and not getattr(m, "tool_calls", None)),
        "",
    )

    # Attach path_hops for visualization (same as troubleshoot_agent)
    session_data      = pop_session_data(session_id)
    path_hops         = session_data.get("path_hops", [])
    reverse_path_hops = session_data.get("reverse_path_hops", [])

    src_ip, dst_ip = _extract_ips(prompt)

    content: dict = {}
    if final_text:
        content["direct_answer"] = final_text
    if path_hops:
        content["path_hops"]   = path_hops
        content["source"]      = src_ip
        content["destination"] = dst_ip
    if reverse_path_hops:
        content["reverse_path_hops"] = reverse_path_hops

    return {"role": "assistant", "content": content}
