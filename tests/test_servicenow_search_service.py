import unittest
from unittest.mock import AsyncMock, patch

from services.servicenow_search_service import ServiceNowSearchService


class ServiceNowSearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_resolve_devices_includes_path_and_history_context(self):
        service = ServiceNowSearchService()
        store = {
            "path_hops": [
                {"from_device": "10.0.100.100", "to_device": "arista-ai1"},
                {"from_device": "arista-ai1", "to_device": "arista-ai2"},
            ],
            "reverse_path_hops": [
                {"from_device": "10.0.200.200", "to_device": "arista-ai4"},
            ],
            "routing_history": {
                "historical_devices": ["arista-ai3"],
                "peer_hint": {"from_device": "arista-ai2", "to_device": "arista-ai3"},
            },
        }

        devices = await service.resolve_devices(
            store=store,
            device_names=["arista-ai1"],
            dest_ip="10.0.200.200",
        )

        self.assertEqual(devices, ["arista-ai1", "arista-ai2", "arista-ai4", "arista-ai3"])

    @patch("services.servicenow_search_service.call_mcp_tool", new_callable=AsyncMock)
    async def test_search_summary_formats_incidents_and_changes(self, mock_call_mcp_tool):
        service = ServiceNowSearchService()
        mock_call_mcp_tool.side_effect = [
            {
                "result": [
                    {
                        "number": "INC0010043",
                        "short_description": "Connectivity issues",
                        "state": "New",
                        "priority": "5",
                    }
                ]
            },
            {
                "result": [
                    {
                        "number": "CHG0030042",
                        "short_description": "route map update on arista-ai1",
                        "state": "Closed",
                        "risk": "Moderate",
                    }
                ]
            },
        ]

        summary = await service.search_summary(
            session_id="svc-test",
            devices=["arista-ai1"],
            source_ip="10.0.100.100",
            dest_ip="10.0.200.200",
            port="443",
        )

        self.assertIn("Incidents found: 1", summary)
        self.assertIn("Change requests found: 1", summary)
        self.assertIn("INC0010043", summary)
        self.assertIn("CHG0030042", summary)

    @patch("services.servicenow_search_service.call_mcp_tool", new_callable=AsyncMock)
    async def test_search_summary_surfaces_partial_failure(self, mock_call_mcp_tool):
        service = ServiceNowSearchService()
        mock_call_mcp_tool.side_effect = [
            {"result": []},
            {"error": "User is not authenticated"},
        ]

        summary = await service.search_summary(
            session_id="svc-test",
            devices=["arista-ai1"],
            source_ip="10.0.100.100",
            dest_ip="10.0.200.200",
            port="443",
        )

        self.assertEqual(
            summary,
            "ServiceNow unavailable: change search failed: User is not authenticated",
        )


if __name__ == "__main__":
    unittest.main()
