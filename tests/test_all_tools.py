import unittest
from unittest.mock import AsyncMock, patch

from tools.all_tools import (
    _store,
    check_routing,
    get_all_interfaces,
    pop_session_data,
    recall_similar_cases,
    search_servicenow,
)


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

    @patch("tools.all_tools._push_status", new_callable=AsyncMock)
    @patch("tools.all_tools.recall_memory", new_callable=AsyncMock, create=True)
    @patch("tools.all_tools.recall_incidents_by_devices", new_callable=AsyncMock, create=True)
    @patch("tools.all_tools.format_memory_context", create=True)
    async def test_recall_similar_cases_defers_without_live_evidence(
        self,
        mock_format_memory_context,
        mock_recall_incidents,
        mock_recall_memory,
        _mock_push_status,
    ):
        result = await recall_similar_cases.coroutine(
            query="same issue again",
            devices=["arista-ai1"],
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertIn("Memory recall deferred", result)
        mock_recall_memory.assert_not_awaited()
        mock_recall_incidents.assert_not_awaited()
        mock_format_memory_context.assert_not_called()

    @patch("tools.all_tools._push_status", new_callable=AsyncMock)
    @patch("agent_memory.format_memory_context", return_value="Matched prior case")
    @patch("agent_memory.recall_incidents_by_devices", new_callable=AsyncMock)
    @patch("agent_memory.recall_memory", new_callable=AsyncMock)
    async def test_recall_similar_cases_runs_after_live_evidence(
        self,
        mock_recall_memory,
        mock_recall_incidents,
        mock_format_memory_context,
        _mock_push_status,
    ):
        _store("tool-test")["interface_details"]["arista-ai1:Ethernet1"] = {
            "device": "arista-ai1",
            "interface": "Ethernet1",
            "oper_status": "down",
            "line_protocol": "down",
        }
        mock_recall_memory.return_value = [{"summary": "Prior route map issue"}]
        mock_recall_incidents.return_value = []

        result = await recall_similar_cases.coroutine(
            query="same issue again",
            devices=["arista-ai1"],
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertIn("Memory recall triggered by live signals", result)
        self.assertIn("interface state anomaly", result)
        self.assertIn("Matched prior case", result)
        mock_recall_memory.assert_awaited_once()
        mock_recall_incidents.assert_awaited_once_with(["arista-ai1"], query="same issue again", top_k=5)
        mock_format_memory_context.assert_called_once()


if __name__ == "__main__":
    unittest.main()
