"""Owned LLM-based troubleshoot scenario selection."""
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

logger = logging.getLogger("atlas.troubleshoot_scenario")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_SCENARIO_SYSTEM_PROMPT = """You are Atlas's troubleshooting scenario selector.

Choose exactly one scenario for the user's troubleshooting request:
- connectivity: blocked, denied, unreachable, no route, connection refused, endpoint-to-endpoint path problems
- performance: slow, degraded, high latency, throughput issues
- intermittent: flapping, sporadic, unstable, works sometimes and fails sometimes
- general: everything else

Guidelines:
- If the request is clearly endpoint-to-endpoint connectivity or port reachability, choose connectivity.
- If the request is clearly about slowness, latency, or throughput degradation, choose performance.
- If the request is clearly unstable or intermittent over time, choose intermittent.
- If it is a device, protocol, or network issue that does not clearly fit the above, choose general.

Return ONLY valid JSON with this shape:
{"scenario":"connectivity|performance|intermittent|general","reason":"short explanation"}
"""


class TroubleshootScenarioService:
    """Owns LLM-based troubleshoot scenario selection."""

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
        if scenario not in {"connectivity", "performance", "intermittent", "general"}:
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
            logger.debug("troubleshoot scenario selection failed: %s", exc)
            return "general"

        text = self._extract_text(response)
        decision = self._parse_decision(text)
        if not decision:
            logger.debug("troubleshoot scenario selector returned unparsable content: %r", text)
            return "general"
        return str(decision["scenario"])


troubleshoot_scenario_service = TroubleshootScenarioService()
