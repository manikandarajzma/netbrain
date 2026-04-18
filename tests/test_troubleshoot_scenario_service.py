import unittest
from unittest.mock import AsyncMock, patch

from services.troubleshoot_scenario_service import troubleshoot_scenario_service


class TroubleshootScenarioServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_decision_rejects_invalid_payload(self):
        self.assertIsNone(troubleshoot_scenario_service._parse_decision("not json"))
        self.assertIsNone(
            troubleshoot_scenario_service._parse_decision('{"scenario":"other","reason":"bad"}')
        )

    def test_parse_decision_accepts_valid_payload(self):
        decision = troubleshoot_scenario_service._parse_decision(
            '{"scenario":"performance","reason":"latency complaint"}'
        )
        self.assertEqual(decision, {"scenario": "performance", "reason": "latency complaint"})

    @patch("services.troubleshoot_scenario_service.agent_scenario_service.select_scenario", new_callable=AsyncMock)
    async def test_select_scenario_returns_model_choice(self, mock_select_scenario):
        mock_select_scenario.return_value = "connectivity"

        scenario = await troubleshoot_scenario_service.select_scenario(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200"
        )

        self.assertEqual(scenario, "connectivity")
        mock_select_scenario.assert_awaited_once()

    @patch("services.troubleshoot_scenario_service.agent_scenario_service.select_scenario", new_callable=AsyncMock)
    async def test_select_scenario_defaults_to_general_on_failure(self, mock_select_scenario):
        mock_select_scenario.return_value = "general"

        scenario = await troubleshoot_scenario_service.select_scenario("why is bgp down on core-rtr-01")

        self.assertEqual(scenario, "general")
        mock_select_scenario.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
