import importlib
import unittest
from unittest.mock import AsyncMock, patch

from tools.all_tools import (
    check_routing,
    get_all_interfaces,
    search_servicenow,
)
from tools.memory_agent_tools import recall_similar_cases
from services.session_store import session_store


class ServiceNowToolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        session_store.pop("tool-test")

    async def asyncTearDown(self):
        session_store.pop("tool-test")

    @patch("tools.servicenow_workflow_tools.push_status", new_callable=AsyncMock)
    @patch("tools.servicenow_workflow_tools.servicenow_search_service.search_summary", new_callable=AsyncMock)
    @patch("tools.servicenow_workflow_tools.servicenow_search_service.resolve_devices", new_callable=AsyncMock)
    async def test_search_servicenow_stores_summary_on_success(
        self,
        mock_resolve_devices,
        mock_search_summary,
        _mock_push_status,
    ):
        mock_resolve_devices.return_value = ["arista-ai1"]
        mock_search_summary.return_value = (
            "Incidents found: 1\n"
            "Change requests found: 1\n\n"
            "### Incidents\n"
            "- **INC0010043**\n\n"
            "### Change Requests\n"
            "- **CHG0030042**"
        )

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

        session = session_store.pop("tool-test")
        self.assertIn("CHG0030042", session["servicenow_summary"])

    @patch("tools.servicenow_workflow_tools.push_status", new_callable=AsyncMock)
    @patch("tools.servicenow_workflow_tools.servicenow_search_service.search_summary", new_callable=AsyncMock)
    @patch("tools.servicenow_workflow_tools.servicenow_search_service.resolve_devices", new_callable=AsyncMock)
    async def test_search_servicenow_stores_unavailable_summary_on_partial_failure(
        self,
        mock_resolve_devices,
        mock_search_summary,
        _mock_push_status,
    ):
        mock_resolve_devices.return_value = ["arista-ai1"]
        mock_search_summary.return_value = "ServiceNow unavailable: change search failed: User is not authenticated"

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

        session = session_store.pop("tool-test")
        self.assertEqual(
            session["servicenow_summary"],
            "ServiceNow unavailable: change search failed: User is not authenticated",
        )

    @patch("tools.device_agent_tools.push_status", new_callable=AsyncMock)
    @patch("tools.device_agent_tools.device_diagnostics_service.routing_check_summary", new_callable=AsyncMock)
    async def test_check_routing_uses_standardized_nornir_failure_contract(
        self,
        mock_routing_check_summary,
        _mock_push_status,
    ):
        mock_routing_check_summary.return_value = "Nornir unavailable during routing check: route lookup timed out"

        result = await check_routing.coroutine(
            devices=["arista-ai1"],
            destination="10.0.200.200",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(
            result,
            "Nornir unavailable during routing check: route lookup timed out",
        )

    @patch("tools.device_agent_tools.push_status", new_callable=AsyncMock)
    @patch("tools.device_agent_tools.device_diagnostics_service.all_interfaces_summary", new_callable=AsyncMock)
    async def test_get_all_interfaces_uses_standardized_nornir_failure_contract(
        self,
        mock_all_interfaces_summary,
        _mock_push_status,
    ):
        mock_all_interfaces_summary.return_value = (
            "Nornir unavailable during interface inventory lookup for arista-ai1: device access failed"
        )

        result = await get_all_interfaces.coroutine(
            device="arista-ai1",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(
            result,
            "Nornir unavailable during interface inventory lookup for arista-ai1: device access failed",
        )

    @patch("tools.memory_agent_tools.push_status", new_callable=AsyncMock)
    @patch("tools.memory_agent_tools.recall_memory", new_callable=AsyncMock, create=True)
    @patch("tools.memory_agent_tools.recall_incidents_by_devices", new_callable=AsyncMock, create=True)
    @patch("tools.memory_agent_tools.format_memory_context", create=True)
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

    @patch("tools.memory_agent_tools.push_status", new_callable=AsyncMock)
    async def test_recall_similar_cases_runs_after_live_evidence(self, _mock_push_status):
        session_store.get("tool-test")["interface_details"]["arista-ai1:Ethernet1"] = {
            "device": "arista-ai1",
            "interface": "Ethernet1",
            "oper_status": "down",
            "line_protocol": "down",
        }
        try:
            memory_module = importlib.import_module("atlas.memory.agent_memory")
        except ImportError:
            memory_module = importlib.import_module("memory.agent_memory")

        with (
            patch.object(memory_module, "recall_memory", new_callable=AsyncMock) as mock_recall_memory,
            patch.object(memory_module, "recall_incidents_by_devices", new_callable=AsyncMock) as mock_recall_incidents,
            patch.object(memory_module, "format_memory_context", return_value="Matched prior case") as mock_format_memory_context,
        ):
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
