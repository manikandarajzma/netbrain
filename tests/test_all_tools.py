import unittest
from unittest.mock import AsyncMock, patch

from tools.all_tools import check_routing, get_all_interfaces, pop_session_data, search_servicenow


class ServiceNowToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        pop_session_data("tool-test")

    async def asyncTearDown(self):
        pop_session_data("tool-test")

    @patch("tools.all_tools._push_status", new_callable=AsyncMock)
    @patch("tools.all_tools.call_mcp_tool", new_callable=AsyncMock)
    async def test_search_servicenow_stores_summary_on_success(self, mock_call_mcp_tool, _mock_push_status):
        mock_call_mcp_tool.side_effect = [
            {"result": [{"number": "INC0010043", "short_description": "Connectivity issues", "state": "New", "priority": "5"}]},
            {"result": [{"number": "CHG0030042", "short_description": "route map update on arista-ai1", "state": "Closed", "risk": "Moderate"}]},
        ]

        result = await search_servicenow.coroutine(
            device_names=["arista-ai1"],
            source_ip="10.0.100.100",
            dest_ip="10.0.200.200",
            port="443",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertIn("Incidents found: 1", result)
        self.assertIn("Change requests found: 1", result)
        self.assertIn("CHG0030042", result)

        session = pop_session_data("tool-test")
        self.assertIn("CHG0030042", session["servicenow_summary"])

    @patch("tools.all_tools._push_status", new_callable=AsyncMock)
    @patch("tools.all_tools.call_mcp_tool", new_callable=AsyncMock)
    async def test_search_servicenow_stores_unavailable_summary_on_partial_failure(self, mock_call_mcp_tool, _mock_push_status):
        mock_call_mcp_tool.side_effect = [
            {"result": []},
            {"error": "User is not authenticated"},
        ]

        result = await search_servicenow.coroutine(
            device_names=["arista-ai1"],
            source_ip="10.0.100.100",
            dest_ip="10.0.200.200",
            port="443",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(
            result,
            "ServiceNow unavailable: change search failed: User is not authenticated",
        )

        session = pop_session_data("tool-test")
        self.assertEqual(
            session["servicenow_summary"],
            "ServiceNow unavailable: change search failed: User is not authenticated",
        )

    @patch("tools.all_tools._push_status", new_callable=AsyncMock)
    @patch("tools.all_tools.retry_async", new_callable=AsyncMock)
    async def test_check_routing_uses_standardized_nornir_failure_contract(self, mock_retry_async, _mock_push_status):
        mock_retry_async.side_effect = RuntimeError("route lookup timed out")

        result = await check_routing.coroutine(
            devices=["arista-ai1"],
            destination="10.0.200.200",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(
            result,
            "Nornir unavailable during routing check: route lookup timed out",
        )

    @patch("tools.all_tools._push_status", new_callable=AsyncMock)
    @patch("tools.all_tools._nornir_post", new_callable=AsyncMock)
    async def test_get_all_interfaces_uses_standardized_nornir_failure_contract(self, mock_nornir_post, _mock_push_status):
        mock_nornir_post.side_effect = RuntimeError("device access failed")

        result = await get_all_interfaces.coroutine(
            device="arista-ai1",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(
            result,
            "Nornir unavailable during interface inventory lookup for arista-ai1: device access failed",
        )


if __name__ == "__main__":
    unittest.main()
