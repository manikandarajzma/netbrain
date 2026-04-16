import unittest
from unittest.mock import AsyncMock, patch

from graph.graph_nodes import classify_intent


class GraphNodeRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def test_create_incident_routes_to_network_ops(self):
        with patch(
            "graph.graph_nodes.intent_routing_service.route_prompt",
            new_callable=AsyncMock,
            return_value={"intent": "network_ops", "confidence": 0.92, "reason": "incident creation"},
        ):
            state = {"prompt": 'Create an incident for "Connectivity issues between 10.0.100.100 and 10.0.200.200 on tcp port 443"', "session_id": "s1"}
            result = await classify_intent(state)
        self.assertEqual(result["intent"], "network_ops")

    async def test_connectivity_prompt_routes_to_troubleshoot(self):
        with patch(
            "graph.graph_nodes.intent_routing_service.route_prompt",
            new_callable=AsyncMock,
            return_value={"intent": "troubleshoot", "confidence": 0.91, "reason": "diagnostic request"},
        ):
            state = {"prompt": "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443", "session_id": "s2"}
            result = await classify_intent(state)
        self.assertEqual(result["intent"], "troubleshoot")

    async def test_diagnostic_framing_beats_network_ops_keywords(self):
        with patch(
            "graph.graph_nodes.intent_routing_service.route_prompt",
            new_callable=AsyncMock,
            return_value={"intent": "troubleshoot", "confidence": 0.88, "reason": "existing failure investigation"},
        ):
            state = {"prompt": "why is the firewall rule not matching traffic from 10.0.100.100 to 10.0.200.200", "session_id": "s2b"}
            result = await classify_intent(state)
        self.assertEqual(result["intent"], "troubleshoot")

    async def test_explicit_change_request_routes_to_network_ops(self):
        with patch(
            "graph.graph_nodes.intent_routing_service.route_prompt",
            new_callable=AsyncMock,
            return_value={"intent": "network_ops", "confidence": 0.90, "reason": "change creation"},
        ):
            state = {"prompt": "create a change request for arista-ai1 route map update", "session_id": "s2c"}
            result = await classify_intent(state)
        self.assertEqual(result["intent"], "network_ops")

    async def test_unknown_prompt_can_dismiss_when_router_says_dismiss(self):
        with patch(
            "graph.graph_nodes.intent_routing_service.route_prompt",
            new_callable=AsyncMock,
            return_value={"intent": "dismiss", "confidence": 0.94, "reason": "unsupported creative task"},
        ):
            state = {"prompt": "write me a poem about routers", "session_id": "s3"}
            result = await classify_intent(state)
        self.assertEqual(result["intent"], "dismiss")
        self.assertEqual(result["final_response"]["content"], "Atlas is not equipped to help with it.")

    async def test_plain_acknowledgement_without_pending_context_dismisses(self):
        state = {"prompt": "okay", "session_id": "s3b"}
        result = await classify_intent(state)
        self.assertEqual(result["intent"], "dismiss")

    @patch("graph.graph_nodes.memory_manager.get_pending_context", return_value=("create a change request", "network_ops"))
    @patch("graph.graph_nodes.memory_manager.has_pending_context", return_value=True)
    async def test_pending_network_ops_follow_up_stays_network_ops(self, _has_pending_context, _get_pending_context):
        state = {"prompt": "1. Short Description: route map update", "session_id": "s4"}
        result = await classify_intent(state)
        self.assertEqual(result["intent"], "network_ops")

    async def test_llm_router_can_route_ambiguous_prompt(self):
        with patch(
            "graph.graph_nodes.intent_routing_service.route_prompt",
            new_callable=AsyncMock,
            return_value={"intent": "troubleshoot", "confidence": 0.52, "reason": "User is reporting an existing failure."},
        ):
            state = {"prompt": "something broke between 10.0.100.100 and 10.0.200.200", "session_id": "s5"}
            result = await classify_intent(state)
        self.assertEqual(result["intent"], "troubleshoot")

    async def test_invalid_router_result_requests_clarification(self):
        with patch(
            "graph.graph_nodes.intent_routing_service.route_prompt",
            new_callable=AsyncMock,
            return_value=None,
        ):
            state = {"prompt": "something broke", "session_id": "s6"}
            result = await classify_intent(state)
        self.assertEqual(result["intent"], "dismiss")
        self.assertIn("could not classify", result["final_response"]["content"].lower())


if __name__ == "__main__":
    unittest.main()
