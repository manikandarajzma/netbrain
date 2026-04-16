"""Owned LLM-assisted coarse intent routing with deterministic fallback support."""
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

logger = logging.getLogger("atlas.intent_routing")

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

_ROUTER_SYSTEM_PROMPT = """You are Atlas's coarse request router.

Classify the user's request into exactly one intent:
- troubleshoot: investigate, diagnose, debug, or explain an existing network, device, routing, protocol, or connectivity problem
- network_ops: create/update/show incidents or changes, request access, ask for an operational action, or retrieve operational records
- dismiss: unsupported request, casual chat, or a request Atlas should not handle

Important routing rules:
- If the user is diagnosing why an existing rule, request, policy, or change is not working, choose troubleshoot.
- If the user wants to create/open/raise/update/close an incident or change, choose network_ops.
- If the user wants Atlas to write poetry or do unrelated general tasks, choose dismiss.
- Do not infer nonexistent urgency or hidden intent.

Return ONLY valid JSON, no markdown fences, with this shape:
{"intent":"troubleshoot|network_ops|dismiss","confidence":0.0,"reason":"short explanation"}
"""


class IntentRoutingService:
    """Owns LLM-based coarse routing decisions for Atlas."""

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
        intent = str(data.get("intent") or "").strip()
        if intent not in {"troubleshoot", "network_ops", "dismiss"}:
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
            llm = agent_factory.build_default_llm()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
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
