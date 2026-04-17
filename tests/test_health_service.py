import unittest
from unittest.mock import AsyncMock, patch

from services.health_service import HealthService


class HealthServiceTests(unittest.IsolatedAsyncioTestCase):
    @patch.object(HealthService, "_check_nornir", new_callable=AsyncMock)
    @patch.object(HealthService, "_check_ollama", new_callable=AsyncMock)
    @patch.object(HealthService, "_check_mcp", new_callable=AsyncMock)
    async def test_build_snapshot_reports_overall_health(
        self,
        mock_check_mcp,
        mock_check_ollama,
        mock_check_nornir,
    ):
        mock_check_mcp.return_value = {"status": "ok", "tools_registered": 12}
        mock_check_ollama.return_value = {
            "status": "ok",
            "models": {
                "router": "gemma4:latest",
                "selector": "gemma4:latest",
                "network_ops": "gemma4:latest",
                "troubleshoot": "gemma4:latest",
            },
            "missing_models": [],
            "all_models_available": True,
        }
        mock_check_nornir.return_value = {"status": "ok", "device_count": 4}

        snapshot = await HealthService().build_snapshot()

        self.assertEqual(snapshot["overall"]["status"], "healthy")
        self.assertEqual(snapshot["overall"]["label"], "All systems OK")
        self.assertEqual(snapshot["services"]["nornir"]["device_count"], 4)

    @patch.object(HealthService, "_check_nornir", new_callable=AsyncMock)
    @patch.object(HealthService, "_check_ollama", new_callable=AsyncMock)
    @patch.object(HealthService, "_check_mcp", new_callable=AsyncMock)
    async def test_build_snapshot_reports_degraded_when_dependency_offline(
        self,
        mock_check_mcp,
        mock_check_ollama,
        mock_check_nornir,
    ):
        mock_check_mcp.return_value = {"status": "ok", "tools_registered": 12}
        mock_check_ollama.return_value = {
            "status": "ok",
            "models": {
                "router": "gemma4:latest",
                "selector": "gemma4:latest",
                "network_ops": "gemma4:latest",
                "troubleshoot": "gemma4:latest",
            },
            "missing_models": [],
            "all_models_available": True,
        }
        mock_check_nornir.return_value = {"status": "unreachable", "device_count": None}

        snapshot = await HealthService().build_snapshot()

        self.assertEqual(snapshot["overall"]["status"], "degraded")
        self.assertEqual(snapshot["overall"]["label"], "Dependency offline")


if __name__ == "__main__":
    unittest.main()
