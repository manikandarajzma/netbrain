import unittest
from unittest.mock import AsyncMock, patch

from tools.servicenow_agent_tools import (
    create_servicenow_incident,
    get_change_request_details,
    get_incident_details,
    update_servicenow_change_request,
)


class ServiceNowAgentToolContractTests(unittest.IsolatedAsyncioTestCase):
    @patch("tools.servicenow_agent_tools.push_status", new_callable=AsyncMock)
    @patch("tools.servicenow_agent_tools.call_mcp_tool", new_callable=AsyncMock)
    async def test_get_incident_details_uses_standard_lookup_error_contract(self, mock_call_mcp_tool, _mock_push_status):
        mock_call_mcp_tool.side_effect = RuntimeError("service unavailable")

        result = await get_incident_details.coroutine(
            incident_number="INC0010043",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(result, "Incident lookup error: service unavailable")

    @patch("tools.servicenow_agent_tools.push_status", new_callable=AsyncMock)
    @patch("tools.servicenow_agent_tools.call_mcp_tool", new_callable=AsyncMock)
    async def test_get_change_request_details_uses_standard_not_found_contract(self, mock_call_mcp_tool, _mock_push_status):
        mock_call_mcp_tool.return_value = {"error": "Change request CHG0030042 not found."}

        result = await get_change_request_details.coroutine(
            change_number="CHG0030042",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(result, "Change request not found: Change request CHG0030042 not found.")

    @patch("tools.servicenow_agent_tools.push_status", new_callable=AsyncMock)
    @patch("tools.servicenow_agent_tools.call_mcp_tool", new_callable=AsyncMock)
    async def test_create_servicenow_incident_uses_standard_unexpected_response_contract(self, mock_call_mcp_tool, _mock_push_status):
        mock_call_mcp_tool.return_value = "bad-result"

        result = await create_servicenow_incident.coroutine(
            short_description="Connectivity issue",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(result, "Incident creation failed: unexpected response 'bad-result'")

    @patch("tools.servicenow_agent_tools.push_status", new_callable=AsyncMock)
    @patch("tools.servicenow_agent_tools.call_mcp_tool", new_callable=AsyncMock)
    async def test_update_change_request_uses_standard_failed_contract(self, mock_call_mcp_tool, _mock_push_status):
        mock_call_mcp_tool.return_value = {"error": "Update rejected"}

        result = await update_servicenow_change_request.coroutine(
            number="CHG0030042",
            state="Closed",
            config={"configurable": {"session_id": "tool-test"}},
        )

        self.assertEqual(result, "Change request update failed: Update rejected")


if __name__ == "__main__":
    unittest.main()
