import unittest
from unittest.mock import AsyncMock, patch

from graph_nodes import call_network_ops_agent, call_troubleshoot_agent


class GraphWorkflowDelegationTests(unittest.IsolatedAsyncioTestCase):
    @patch("graph_nodes.troubleshoot_workflow_service.run", new_callable=AsyncMock)
    async def test_call_troubleshoot_agent_delegates_to_workflow_service(self, mock_run):
        mock_run.return_value = {"final_response": {"role": "assistant", "content": {"direct_answer": "ok"}}}
        state = {"prompt": "help me troubleshoot connectivity", "session_id": "s1"}

        result = await call_troubleshoot_agent(state)

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "ok")
        mock_run.assert_awaited_once_with(state)

    @patch("graph_nodes.network_ops_workflow_service.run", new_callable=AsyncMock)
    async def test_call_network_ops_agent_delegates_to_workflow_service(self, mock_run):
        mock_run.return_value = {"final_response": {"role": "assistant", "content": {"direct_answer": "ok"}}}
        state = {"prompt": "create an incident", "session_id": "s2"}

        result = await call_network_ops_agent(state)

        self.assertEqual(result["final_response"]["content"]["direct_answer"], "ok")
        mock_run.assert_awaited_once_with(state)


if __name__ == "__main__":
    unittest.main()
