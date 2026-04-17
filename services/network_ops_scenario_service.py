"""Owned LLM-based network-ops scenario selection."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from atlas.agents.agent_factory import agent_factory
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore

logger = logging.getLogger("atlas.network_ops_scenario")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_SCENARIO_SYSTEM_PROMPT = """You are Atlas's network-operations scenario selector.

Choose exactly one scenario for the user's network-operations request:
- incident_record: create/open/raise a new incident or ticket
- record_lookup: fetch/show/get the details or status of an existing incident or change record
- change_record: create/open/submit a generic ServiceNow change request
- change_update: update/close/reassign/add work notes to an existing change request
- access_change: open a port, allow/deny traffic, whitelist access, or document a firewall/security rule change
- general: another operational request that does not clearly fit the above

Guidelines:
- If the user wants a new incident or ticket created, choose incident_record.
- If the user references an existing INC or CHG and wants status/details, choose record_lookup.
- If the user wants to create a generic change request and is not describing a firewall/rule/port access change, choose change_record.
- If the user wants to modify or close an existing change request, choose change_update.
- If the user is asking for access, whitelisting, opening ports, or rule/policy changes, choose access_change.

Return ONLY valid JSON with this shape:
{"scenario":"incident_record|record_lookup|change_record|change_update|access_change|general","reason":"short explanation"}
"""


class NetworkOpsScenarioService:
    """Owns LLM-based network-ops scenario selection."""

    @staticmethod
    def _extract_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        return str(content or "").strip()

    @staticmethod
    def _parse_decision(text: str) -> dict[str, Any] | None:
        candidate = text.strip()
        if not candidate:
            return None
        if not candidate.startswith("{"):
            match = _JSON_BLOCK_RE.search(candidate)
            if not match:
                return None
            candidate = match.group(0)
        try:
            data = json.loads(candidate)
        except Exception:
            return None
        scenario = str(data.get("scenario") or "").strip()
        if scenario not in {
            "incident_record",
            "record_lookup",
            "change_record",
            "change_update",
            "access_change",
            "general",
        }:
            return None
        return {"scenario": scenario, "reason": str(data.get("reason") or "").strip()}

    async def select_scenario(self, prompt: str) -> str:
        try:
            llm = agent_factory.build_default_llm()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=_SCENARIO_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            logger.debug("network-ops scenario selection failed: %s", exc)
            return "general"

        text = self._extract_text(response)
        decision = self._parse_decision(text)
        if not decision:
            logger.debug("network-ops scenario selector returned unparsable content: %r", text)
            return "general"
        return str(decision["scenario"])


network_ops_scenario_service = NetworkOpsScenarioService()
