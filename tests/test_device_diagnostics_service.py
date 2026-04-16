import unittest
from unittest.mock import AsyncMock, patch

from services.device_diagnostics_service import DeviceDiagnosticsService
from services.session_store import session_store


class DeviceDiagnosticsServiceTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        session_store.pop("device-diag-test")

    @patch("services.device_diagnostics_service.nornir_client.post", new_callable=AsyncMock)
    async def test_ping_summary_resolves_source_interface_from_path_meta_and_stores_result(self, mock_post):
        service = DeviceDiagnosticsService()
        session_store.get("device-diag-test")["path_meta"] = {
            "first_hop_device": "arista-ai1",
            "first_hop_lan_interface": "Ethernet1",
            "last_hop_device": "arista-ai4",
            "last_hop_egress_interface": "Ethernet1",
        }
        mock_post.return_value = {"success": True, "rtt_avg_ms": 2.5}

        result = await service.ping_summary(
            session_id="device-diag-test",
            device="arista-ai1",
            destination="10.0.200.200",
        )

        self.assertIn("source Ethernet1", result)
        self.assertIn("SUCCESS", result)
        stored = session_store.get("device-diag-test")["ping_results"][0]
        self.assertEqual(stored["source_interface"], "Ethernet1")

    @patch("services.device_diagnostics_service.nornir_client.post", new_callable=AsyncMock)
    async def test_interface_counters_summary_stores_structured_entries(self, mock_post):
        service = DeviceDiagnosticsService()
        mock_post.return_value = {
            "active_errors": [],
            "clean_interfaces": ["Ethernet1", "Ethernet2"],
            "poll_interval_s": 3,
            "iterations": 3,
        }

        result = await service.interface_counters_summary(
            session_id="device-diag-test",
            devices_and_interfaces=[{"device": "arista-ai1", "interfaces": ["Ethernet1", "Ethernet2"]}],
        )

        self.assertIn("Interface counters:", result)
        self.assertIn("all interfaces clean over 6s", result)
        stored = session_store.get("device-diag-test")["interface_counters"][0]
        self.assertEqual(stored["device"], "arista-ai1")
        self.assertEqual(stored["window_s"], 6)
        self.assertEqual(stored["clean"], ["Ethernet1", "Ethernet2"])

    @patch("services.device_diagnostics_service.nornir_client.post", new_callable=AsyncMock)
    async def test_all_interfaces_summary_formats_inventory_and_stores_it(self, mock_post):
        service = DeviceDiagnosticsService()
        mock_post.return_value = {
            "device": "arista-ai1",
            "interfaces": [
                {"interface": "Ethernet1", "up": True, "oper_status": "up", "primary_ip": "10.0.0.1", "prefix_len": 24},
                {"interface": "Ethernet2", "up": False, "oper_status": "down", "description": "uplink"},
            ],
        }

        result = await service.all_interfaces_summary(
            session_id="device-diag-test",
            device="arista-ai1",
        )

        self.assertIn("arista-ai1: 2 interfaces, 1 DOWN", result)
        self.assertIn("✓ Ethernet1 — up ip 10.0.0.1/24", result)
        self.assertIn("✗ Ethernet2 — link-down (down) (uplink)", result)
        self.assertEqual(session_store.get("device-diag-test")["all_interfaces"]["arista-ai1"]["device"], "arista-ai1")


if __name__ == "__main__":
    unittest.main()
