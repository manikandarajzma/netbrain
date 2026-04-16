import unittest
from unittest.mock import ANY, AsyncMock, patch

from atlas_application import AtlasApplication
from services.memory_manager import MemoryManager
from services.session_store import session_store
from tools import all_tools, knowledge_agent_tools, memory_agent_tools, servicenow_agent_tools
from tools.tool_registry import ToolRegistry


class AtlasApplicationTests(unittest.IsolatedAsyncioTestCase):
    @patch("atlas_application.atlas_runtime.extract_final_response")
    @patch("atlas_application.atlas_runtime.invoke_atlas_graph", new_callable=AsyncMock)
    @patch("atlas_application.metrics_recorder.observe_ms")
    @patch("atlas_application.metrics_recorder.increment")
    async def test_process_query_delegates_to_runtime(self, mock_increment, mock_observe_ms, mock_invoke, mock_extract):
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
            request_id=ANY,
        )
        mock_extract.assert_called_once_with({"final_response": {"role": "assistant", "content": "ok"}})
        mock_increment.assert_any_call("atlas.query.started")
        mock_increment.assert_any_call("atlas.query.completed", content_type="text")
        mock_observe_ms.assert_called_once()

    def test_application_owns_routing_diagnostics_service(self):
        app = AtlasApplication()
        self.assertEqual(type(app.routing_diagnostics_service).__name__, "RoutingDiagnosticsService")

    def test_application_owns_device_diagnostics_service(self):
        app = AtlasApplication()
        self.assertEqual(type(app.device_diagnostics_service).__name__, "DeviceDiagnosticsService")

    def test_application_owns_workflow_and_status_services(self):
        app = AtlasApplication()
        self.assertEqual(type(app.troubleshoot_workflow_service).__name__, "TroubleshootWorkflowService")
        self.assertEqual(type(app.network_ops_workflow_service).__name__, "NetworkOpsWorkflowService")
        self.assertEqual(type(app.workflow_state_service).__name__, "WorkflowStateService")
        self.assertEqual(type(app.status_service).__name__, "StatusService")

    @patch("atlas_application.diagnostics_service.build_snapshot", new_callable=AsyncMock)
    async def test_get_diagnostics_snapshot_delegates_to_service(self, mock_build_snapshot):
        mock_build_snapshot.return_value = {"metrics": {}, "tools": {}}

        app = AtlasApplication()
        result = await app.get_diagnostics_snapshot()

        self.assertEqual(result, {"metrics": {}, "tools": {}})
        mock_build_snapshot.assert_awaited_once_with()


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


class SessionStoreTests(unittest.TestCase):
    def tearDown(self):
        session_store.pop("session-test")

    def test_session_store_owns_default_side_effect_shape(self):
        data = session_store.get("session-test")

        self.assertIn("path_hops", data)
        self.assertIn("reverse_path_hops", data)
        self.assertIn("interface_counters", data)
        self.assertIn("servicenow_summary", data)


class ToolRegistryTests(unittest.TestCase):
    def test_tool_capability_manifests_cover_workflow_and_product_tools(self):
        workflow_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool, _caps in all_tools.WORKFLOW_TOOL_CAPABILITIES}
        knowledge_names = {
            getattr(tool, "name", getattr(tool, "__name__", ""))
            for tool, _caps in knowledge_agent_tools.KNOWLEDGE_TOOL_CAPABILITIES
        }
        memory_names = {
            getattr(tool, "name", getattr(tool, "__name__", ""))
            for tool, _caps in memory_agent_tools.MEMORY_TOOL_CAPABILITIES
        }
        servicenow_names = {
            getattr(tool, "name", getattr(tool, "__name__", ""))
            for tool, _caps in servicenow_agent_tools.SERVICENOW_TOOL_CAPABILITIES
        }

        self.assertIn("trace_path", workflow_names)
        self.assertIn("search_servicenow", workflow_names)
        self.assertIn("lookup_vendor_kb", knowledge_names)
        self.assertIn("recall_similar_cases", memory_names)
        self.assertIn("get_incident_details", servicenow_names)
        self.assertIn("update_servicenow_change_request", servicenow_names)

    def test_tool_registry_returns_stable_tool_collections(self):
        registry = ToolRegistry()
        self.assertIs(registry.get_all_tools(), registry.get_all_tools())
        self.assertIs(registry.get_connectivity_tools(), registry.get_connectivity_tools())
        self.assertIs(registry.get_network_ops_tools(), registry.get_network_ops_tools())

    def test_tool_registry_resolves_profile_from_capabilities(self):
        registry = ToolRegistry()

        network_ops_tools = registry.get_profile_tools("network_ops")
        direct_tools = registry.get_tools_for_capabilities(
            (
                "workflow.path.trace",
                "servicenow.search",
                "servicenow.incident.read",
                "servicenow.change.read",
                "servicenow.incident.create",
                "servicenow.change.create",
                "servicenow.change.update",
            )
        )

        self.assertEqual(network_ops_tools, direct_tools)

    def test_tool_registry_exposes_product_and_workflow_capabilities(self):
        registry = ToolRegistry()
        tools = registry.get_tools_for_capabilities(
            (
                "servicenow.incident.read",
                "workflow.connectivity.snapshot",
            )
        )
        tool_names = {getattr(tool, "name", getattr(tool, "__name__", "")) for tool in tools}

        self.assertIn("get_incident_details", tool_names)
        self.assertIn("collect_connectivity_snapshot", tool_names)

    def test_tool_registry_keeps_uniform_interface_over_mixed_tool_modules(self):
        registry = ToolRegistry()
        tools = registry.get_tools_for_capabilities(
            (
                "servicenow.incident.read",
                "workflow.connectivity.snapshot",
            )
        )
        modules = {
            getattr(getattr(tool, "func", None) or getattr(tool, "coroutine", None), "__module__", "")
            for tool in tools
        }

        self.assertTrue(any(module.endswith("servicenow_agent_tools") for module in modules))
        self.assertTrue(
            any(
                module.endswith(suffix)
                for module in modules
                for suffix in (
                    "path_agent_tools",
                    "device_agent_tools",
                    "routing_agent_tools",
                    "connectivity_agent_tools",
                    "servicenow_workflow_tools",
                )
            )
        )

    def test_tool_registry_exposes_memory_tools_from_dedicated_module(self):
        registry = ToolRegistry()
        tools = registry.get_tools_for_capabilities(("memory.recall",))
        modules = {
            getattr(getattr(tool, "func", None) or getattr(tool, "coroutine", None), "__module__", "")
            for tool in tools
        }

        self.assertTrue(any(module.endswith("memory_agent_tools") for module in modules))

    def test_tool_registry_exposes_knowledge_tools_from_dedicated_module(self):
        registry = ToolRegistry()
        tools = registry.get_tools_for_capabilities(("knowledge.vendor.lookup",))
        modules = {
            getattr(getattr(tool, "func", None) or getattr(tool, "coroutine", None), "__module__", "")
            for tool in tools
        }

        self.assertTrue(any(module.endswith("knowledge_agent_tools") for module in modules))


if __name__ == "__main__":
    unittest.main()
