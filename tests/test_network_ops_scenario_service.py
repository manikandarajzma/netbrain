import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.network_ops_scenario_service import network_ops_scenario_service


class NetworkOpsScenarioServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_decision_rejects_invalid_payload(self):
        self.assertIsNone(network_ops_scenario_service._parse_decision("not json"))
        self.assertIsNone(
            network_ops_scenario_service._parse_decision('{"scenario":"other","reason":"bad"}')
        )

    def test_parse_decision_accepts_valid_payload(self):
        decision = network_ops_scenario_service._parse_decision(
            '{"scenario":"incident_record","reason":"create a ticket"}'
        )
        self.assertEqual(decision, {"scenario": "incident_record", "reason": "create a ticket"})

    @patch("services.network_ops_scenario_service.agent_factory.build_default_llm")
    async def test_select_scenario_returns_model_choice(self, mock_build_default_llm):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content='{"scenario":"access_change","reason":"open port request"}'
        )
        mock_build_default_llm.return_value = llm

        scenario = await network_ops_scenario_service.select_scenario(
            "open port 443 from 10.0.100.100 to 10.0.200.200"
        )

        self.assertEqual(scenario, "access_change")

    @patch("services.network_ops_scenario_service.agent_factory.build_default_llm")
    async def test_select_scenario_defaults_to_general_on_failure(self, mock_build_default_llm):
        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("model unavailable")
        mock_build_default_llm.return_value = llm

        scenario = await network_ops_scenario_service.select_scenario("show me the latest ticket status")

        self.assertEqual(scenario, "general")


if __name__ == "__main__":
    unittest.main()
