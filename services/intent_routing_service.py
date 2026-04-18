"""Owned LLM-assisted coarse intent routing with deterministic fallback support."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from atlas.agents.agent_registry import agent_registry
    from atlas.agents.agent_factory import agent_factory
except ImportError:
    from agents.agent_registry import agent_registry  # type: ignore
    from agents.agent_factory import agent_factory  # type: ignore

logger = logging.getLogger("atlas.intent_routing")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class IntentRoutingService:
    """Owns LLM-based coarse routing decisions for Atlas."""

    @staticmethod
    def _valid_intents() -> set[str]:
        return agent_registry.valid_route_keys() | {"dismiss"}

    def _router_system_prompt(self) -> str:
        route_lines = [
            f"- {route_key}: {description}"
            for route_key, description in sorted(agent_registry.route_descriptions().items())
        ]
        intent_list = "|".join(sorted(agent_registry.valid_route_keys() | {"dismiss"}))
        return (
            "You are Atlas's coarse request router.\n\n"
            "Classify the user's request into exactly one intent:\n"
            f"{chr(10).join(route_lines)}\n"
            "- dismiss: unsupported request, casual chat, or a request Atlas should not handle\n\n"
            "Important routing rules:\n"
            "- If the user is diagnosing why an existing rule, request, policy, or change is not working, choose troubleshoot.\n"
            "- If the user wants to create/open/raise/update/close an incident or change, choose network_ops.\n"
            "- If the user wants Atlas to write poetry or do unrelated general tasks, choose dismiss.\n"
            "- Do not infer nonexistent urgency or hidden intent.\n\n"
            "Return ONLY valid JSON, no markdown fences, with this shape:\n"
            f'{{"intent":"{intent_list}","confidence":0.0,"reason":"short explanation"}}'
        )

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

    @classmethod
    def _parse_decision(cls, text: str) -> dict[str, Any] | None:
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
        intent = str(data.get("intent") or "").strip()
        if intent not in cls._valid_intents():
            return None
        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = str(data.get("reason") or "").strip()
        return {"intent": intent, "confidence": confidence, "reason": reason}

    async def route_prompt(self, prompt: str) -> dict[str, Any] | None:
        try:
            llm = agent_factory.build_router_llm()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=self._router_system_prompt()),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            logger.debug("intent router LLM call failed: %s", exc)
            return None

        text = self._extract_text(response)
        decision = self._parse_decision(text)
        if not decision:
            logger.debug("intent router returned unparsable content: %r", text)
            return None
        return decision


intent_routing_service = IntentRoutingService()
