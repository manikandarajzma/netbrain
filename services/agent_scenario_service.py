"""Generic LLM-based scenario selection driven by AgentSpec metadata."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

try:
    from atlas.agents.agent_factory import agent_factory
    from atlas.agents.agent_registry import AgentSpec
except ImportError:
    from agents.agent_factory import agent_factory  # type: ignore
    from agents.agent_registry import AgentSpec  # type: ignore


logger = logging.getLogger("atlas.agent_scenario")
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


class AgentScenarioService:
    """Owns generic LLM-based scenario selection for spec-driven agents."""

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
    def _valid_scenarios(spec: AgentSpec) -> set[str]:
        return set(spec.scenario_descriptions.keys()) | {spec.default_scenario}

    @classmethod
    def _parse_decision(cls, spec: AgentSpec, text: str) -> dict[str, Any] | None:
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
        if scenario not in cls._valid_scenarios(spec):
            return None
        return {"scenario": scenario, "reason": str(data.get("reason") or "").strip()}

    def _system_prompt(self, spec: AgentSpec) -> str:
        route_label = spec.route_key.replace("_", " ")
        scenario_lines = [
            f"- {name}: {description}"
            for name, description in sorted(spec.scenario_descriptions.items())
        ]
        guidance_lines = [f"- {line}" for line in spec.scenario_guidance]
        scenario_list = "|".join(sorted(self._valid_scenarios(spec)))
        blocks = [
            f"You are Atlas's scenario selector for the {route_label} agent.",
            "",
            f"Choose exactly one scenario for the user's {route_label} request:",
            *scenario_lines,
        ]
        if guidance_lines:
            blocks.extend(["", "Guidelines:", *guidance_lines])
        blocks.extend(
            [
                "",
                "Return ONLY valid JSON with this shape:",
                f'{{"scenario":"{scenario_list}","reason":"short explanation"}}',
            ]
        )
        return "\n".join(blocks)

    async def select_scenario(self, spec: AgentSpec, prompt: str) -> str:
        try:
            llm = agent_factory.build_selector_llm()
            response = await llm.ainvoke(
                [
                    SystemMessage(content=self._system_prompt(spec)),
                    HumanMessage(content=prompt),
                ]
            )
        except Exception as exc:
            logger.debug("%s scenario selection failed: %s", spec.route_key, exc)
            return spec.default_scenario

        text = self._extract_text(response)
        decision = self._parse_decision(spec, text)
        if not decision:
            logger.debug("%s scenario selector returned unparsable content: %r", spec.route_key, text)
            return spec.default_scenario
        return str(decision["scenario"])


agent_scenario_service = AgentScenarioService()
