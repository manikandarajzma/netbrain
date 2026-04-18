import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agents.agent_registry import agent_registry
from services.agent_scenario_service import agent_scenario_service


class AgentScenarioServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_system_prompt_uses_spec_defined_scenarios_and_guidance(self):
        spec = agent_registry.get("network_ops")

        prompt = agent_scenario_service._system_prompt(spec)

        self.assertIn("incident_record", prompt)
        self.assertIn("access_change", prompt)
        self.assertIn("If the user wants a new incident or ticket created", prompt)

    def test_parse_decision_validates_against_spec_scenarios(self):
        spec = agent_registry.get("troubleshoot")

        self.assertIsNone(agent_scenario_service._parse_decision(spec, "not json"))
        self.assertIsNone(
            agent_scenario_service._parse_decision(spec, '{"scenario":"other","reason":"bad"}')
        )
        self.assertEqual(
            agent_scenario_service._parse_decision(
                spec,
                '{"scenario":"performance","reason":"latency complaint"}',
            ),
            {"scenario": "performance", "reason": "latency complaint"},
        )

    @patch("services.agent_scenario_service.agent_factory.build_selector_llm")
    async def test_select_scenario_uses_default_when_model_fails(self, mock_build_selector_llm):
        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("model unavailable")
        mock_build_selector_llm.return_value = llm

        scenario = await agent_scenario_service.select_scenario(
            agent_registry.get("network_ops"),
            "show me the latest ticket status",
        )

        self.assertEqual(scenario, "general")

    @patch("services.agent_scenario_service.agent_factory.build_selector_llm")
    async def test_select_scenario_returns_model_choice(self, mock_build_selector_llm):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content='{"scenario":"connectivity","reason":"endpoint-to-endpoint failure"}'
        )
        mock_build_selector_llm.return_value = llm

        scenario = await agent_scenario_service.select_scenario(
            agent_registry.get("troubleshoot"),
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
        )

        self.assertEqual(scenario, "connectivity")


if __name__ == "__main__":
    unittest.main()
