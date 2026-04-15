import unittest
from unittest.mock import AsyncMock, patch

from atlas_application import AtlasApplication
from services.memory_manager import MemoryManager
from tools.tool_registry import ToolRegistry


class AtlasApplicationTests(unittest.IsolatedAsyncioTestCase):
    @patch("atlas_application.atlas_runtime.extract_final_response")
    @patch("atlas_application.atlas_runtime.invoke_atlas_graph", new_callable=AsyncMock)
    async def test_process_query_delegates_to_runtime(self, mock_invoke, mock_extract):
        mock_invoke.return_value = {"final_response": {"role": "assistant", "content": "ok"}}
        mock_extract.return_value = {"role": "assistant", "content": "ok"}

        app = AtlasApplication()
        result = await app.process_query(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
            [{"role": "user", "content": "previous"}],
            username="alice",
            session_id="session-1",
        )

        self.assertEqual(result, {"role": "assistant", "content": "ok"})
        mock_invoke.assert_awaited_once_with(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
            [{"role": "user", "content": "previous"}],
            username="alice",
            session_id="session-1",
        )
        mock_extract.assert_called_once_with({"final_response": {"role": "assistant", "content": "ok"}})


class MemoryManagerTests(unittest.TestCase):
    @patch("services.memory_manager._set_pending_context")
    def test_memory_manager_delegates_pending_context_set(self, mock_set_pending_context):
        MemoryManager().set_pending_context("session-1", "prompt", "network_ops")
        mock_set_pending_context.assert_called_once_with("session-1", "prompt", "network_ops")

    def test_memory_manager_extracts_recall_signals_from_live_evidence(self):
        manager = MemoryManager()
        signals = manager.get_recall_signals({
            "interface_details": {
                "arista-ai1:Ethernet1": {
                    "oper_status": "down",
                    "line_protocol": "down",
                }
            },
            "ping_results": [{"success": False}],
        })

        self.assertIn("interface state anomaly", signals)
        self.assertIn("failed reachability test", signals)


class ToolRegistryTests(unittest.TestCase):
    def test_tool_registry_returns_stable_tool_collections(self):
        registry = ToolRegistry()
        self.assertIs(registry.get_all_tools(), registry.get_all_tools())
        self.assertIs(registry.get_connectivity_tools(), registry.get_connectivity_tools())
        self.assertIs(registry.get_network_ops_tools(), registry.get_network_ops_tools())


if __name__ == "__main__":
    unittest.main()
