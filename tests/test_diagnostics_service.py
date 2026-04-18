import unittest
from unittest.mock import AsyncMock, patch

from services.diagnostics_service import DiagnosticsService


class DiagnosticsServiceTests(unittest.IsolatedAsyncioTestCase):
    @patch("services.diagnostics_service.session_store.active_session_count")
    @patch("services.diagnostics_service.checkpointer_runtime.get_status")
    @patch("services.diagnostics_service.metrics_recorder.snapshot")
    @patch("services.diagnostics_service.tool_registry.describe_registered_tools")
    @patch("services.diagnostics_service.tool_registry.describe_profiles")
    @patch("services.diagnostics_service.workflow_registry.describe")
    @patch("services.diagnostics_service.agent_registry.describe")
    @patch("services.diagnostics_service.health_service.build_snapshot", new_callable=AsyncMock)
    async def test_build_snapshot_includes_runtime_tools_metrics_and_health(
        self,
        mock_health_snapshot,
        mock_agent_registry,
        mock_workflow_registry,
        mock_profiles,
        mock_registered_tools,
        mock_metrics_snapshot,
        mock_checkpointer_status,
        mock_active_session_count,
    ):
        mock_agent_registry.return_value = {
            "troubleshoot": {
                "route_key": "troubleshoot",
                "default_tool_profile": "troubleshoot.general",
            }
        }
        mock_workflow_registry.return_value = {"troubleshoot": "TroubleshootWorkflowService"}
        mock_profiles.return_value = {"network_ops": ["servicenow.incident.read"]}
        mock_registered_tools.return_value = [
            {
                "name": "get_incident_details",
                "module": "atlas.tools.servicenow_agent_tools",
                "capabilities": ["servicenow.incident.read"],
            }
        ]
        mock_metrics_snapshot.return_value = {"counters": [], "timings": []}
        mock_checkpointer_status.return_value = {
            "ready": True,
            "state": "enabled",
            "label": "Enabled",
            "error": None,
        }
        mock_active_session_count.return_value = 2
        mock_health_snapshot.return_value = {"overall": {"status": "healthy", "label": "All systems OK"}}

        snapshot = await DiagnosticsService().build_snapshot()

        self.assertEqual(
            snapshot["runtime"]["checkpointer"],
            {"ready": True, "state": "enabled", "label": "Enabled", "error": None},
        )
        self.assertEqual(snapshot["runtime"]["active_tool_sessions"], 2)
        self.assertEqual(snapshot["agents"]["count"], 1)
        self.assertEqual(snapshot["agents"]["registered"]["troubleshoot"]["route_key"], "troubleshoot")
        self.assertEqual(snapshot["workflows"]["troubleshoot"], "TroubleshootWorkflowService")
        self.assertEqual(snapshot["tools"]["profile_count"], 1)
        self.assertEqual(snapshot["tools"]["registered_tool_count"], 1)
        self.assertEqual(snapshot["tools"]["profiles"]["network_ops"], ["servicenow.incident.read"])
        self.assertEqual(snapshot["tools"]["registered"][0]["name"], "get_incident_details")
        self.assertEqual(snapshot["metrics"], {"counters": [], "timings": []})
        self.assertEqual(snapshot["health"]["overall"]["label"], "All systems OK")
        self.assertEqual(snapshot["owners"]["device_diagnostics_service"], "DeviceDiagnosticsService")
        self.assertEqual(snapshot["owners"]["agent_registry"], "AgentRegistry")
        self.assertEqual(snapshot["owners"]["workflow_registry"], "WorkflowRegistry")
        self.assertEqual(snapshot["owners"]["troubleshoot_workflow_service"], "TroubleshootWorkflowService")
        self.assertEqual(snapshot["owners"]["network_ops_workflow_service"], "NetworkOpsWorkflowService")
        self.assertEqual(snapshot["owners"]["routing_diagnostics_service"], "RoutingDiagnosticsService")
        self.assertEqual(snapshot["owners"]["status_service"], "StatusService")
        self.assertEqual(snapshot["owners"]["session_store"], "SessionStore")
        self.assertEqual(snapshot["owners"]["workflow_state_service"], "WorkflowStateService")
        self.assertEqual(snapshot["owners"]["diagnostics_service"], "DiagnosticsService")


if __name__ == "__main__":
    unittest.main()
