import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from services.intent_routing_service import intent_routing_service


class IntentRoutingServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_decision_rejects_invalid_payload(self):
        self.assertIsNone(intent_routing_service._parse_decision("not json"))
        self.assertIsNone(intent_routing_service._parse_decision('{"intent":"other","confidence":1.0}'))

    def test_parse_decision_accepts_valid_payload(self):
        decision = intent_routing_service._parse_decision(
            '{"intent":"troubleshoot","confidence":0.88,"reason":"existing failure"}'
        )
        self.assertEqual(
            decision,
            {"intent": "troubleshoot", "confidence": 0.88, "reason": "existing failure"},
        )

    @patch("services.intent_routing_service.agent_factory.build_router_llm")
    async def test_route_prompt_returns_none_on_model_failure(self, mock_build_router_llm):
        llm = AsyncMock()
        llm.ainvoke.side_effect = RuntimeError("router unavailable")
        mock_build_router_llm.return_value = llm

        decision = await intent_routing_service.route_prompt("help")

        self.assertIsNone(decision)

    @patch("services.intent_routing_service.agent_factory.build_router_llm")
    async def test_route_prompt_extracts_json_decision(self, mock_build_router_llm):
        llm = AsyncMock()
        llm.ainvoke.return_value = SimpleNamespace(
            content='{"intent":"network_ops","confidence":0.91,"reason":"record request"}'
        )
        mock_build_router_llm.return_value = llm

        decision = await intent_routing_service.route_prompt("show me INC0010043")

        self.assertEqual(
            decision,
            {"intent": "network_ops", "confidence": 0.91, "reason": "record request"},
        )


if __name__ == "__main__":
    unittest.main()
