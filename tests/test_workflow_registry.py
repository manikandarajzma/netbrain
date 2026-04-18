import unittest
from unittest.mock import AsyncMock, patch

from agents.agent_registry import agent_registry
from services.workflow_registry import workflow_registry


class WorkflowRegistryTests(unittest.IsolatedAsyncioTestCase):
    def test_describe_exposes_registered_workflow_runners(self):
        described = workflow_registry.describe()

        self.assertEqual(described["troubleshoot"], "TroubleshootWorkflowService")
        self.assertEqual(described["network_ops"], "NetworkOpsWorkflowService")

    @patch("services.workflow_registry.troubleshoot_workflow_service.run", new_callable=AsyncMock)
    async def test_run_uses_spec_workflow_type(self, mock_run):
        mock_run.return_value = {"final_response": {"role": "assistant", "content": "ok"}}
        spec = agent_registry.get("troubleshoot")
        state = {"intent": "troubleshoot", "prompt": "help me troubleshoot"}

        result = await workflow_registry.run(spec, state)

        self.assertEqual(result["final_response"]["content"], "ok")
        mock_run.assert_awaited_once_with(state)


if __name__ == "__main__":
    unittest.main()
