import types
import unittest
from unittest.mock import AsyncMock, patch

from services.routing_diagnostics_service import RoutingDiagnosticsService
from services.session_store import session_store


class RoutingDiagnosticsServiceTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self):
        session_store.pop("routing-test")

    @patch("services.routing_diagnostics_service.nornir_client.post", new_callable=AsyncMock)
    async def test_ospf_neighbors_summary_formats_and_stores_neighbors(self, mock_post):
        service = RoutingDiagnosticsService()
        payload = {
            "ospf_neighbors": {
                "arista-ai1": {
                    "count": 1,
                    "neighbors": [
                        {"router_id": "1.1.1.1", "interface": "Ethernet2", "state": "FULL"}
                    ],
                }
            }
        }
        mock_post.return_value = payload

        result = await service.ospf_neighbors_summary(
            session_id="routing-test",
            devices=["arista-ai1"],
        )

        self.assertIn("OSPF neighbors:", result)
        self.assertIn("arista-ai1: 1 neighbor(s) — 1.1.1.1 via Ethernet2 (FULL)", result)
        self.assertEqual(session_store.get("routing-test")["ospf_neighbors"], payload)

    @patch("services.routing_diagnostics_service.nornir_client.post", new_callable=AsyncMock)
    async def test_device_syslog_summary_correlates_local_ospf_ip_to_interface(self, mock_post):
        service = RoutingDiagnosticsService()
        session_store.get("routing-test")["all_interfaces"]["arista-ai1"] = {
            "interfaces": [
                {
                    "interface": "Ethernet2",
                    "primary_ip": "169.254.0.1",
                    "up": False,
                    "oper_status": "down",
                }
            ]
        }
        mock_post.return_value = {
            "logs": [
                "%OSPF-5-ADJCHG: OSPF adjacency lost to 169.254.0.1 on Ethernet2",
            ]
        }

        result = await service.device_syslog_summary(
            session_id="routing-test",
            device="arista-ai1",
            interface="Ethernet2",
        )

        self.assertIn("arista-ai1 syslog:", result)
        self.assertIn("OSPF interface correlation:", result)
        self.assertIn("Correlated OSPF syslog IP 169.254.0.1 -> Ethernet2 (down (down))", result)
        correlations = session_store.get("routing-test")["syslog"]["arista-ai1"]["correlations"]
        self.assertEqual(correlations[0]["interface"], "Ethernet2")
        self.assertEqual(correlations[0]["ip"], "169.254.0.1")

    @patch("services.routing_diagnostics_service.nornir_client.post", new_callable=AsyncMock)
    async def test_routing_history_summary_stores_peer_hint_and_history(self, mock_post):
        service = RoutingDiagnosticsService()
        fake_db = types.SimpleNamespace(
            fetch=AsyncMock(
                return_value=[
                    {"device": "arista-ai1"},
                    {"device": "arista-ai2"},
                ]
            ),
            fetchrow=AsyncMock(
                side_effect=[
                    {
                        "device": "arista-ai4",
                        "egress_interface": "Ethernet2",
                        "next_hop": "169.254.0.5",
                        "protocol": "ospf",
                        "prefix": "10.0.200.0/24",
                        "collected_at": "2026-04-15T12:00:00+00:00",
                    },
                    {
                        "device": "arista-ai3",
                        "egress_interface": "Ethernet2",
                        "next_hop": "169.254.0.5",
                        "protocol": "ospf",
                        "prefix": "10.0.200.0/24",
                        "collected_at": "2026-04-15T12:00:00+00:00",
                    },
                ]
            ),
        )
        mock_post.return_value = {
            "found": True,
            "device": "arista-ai4",
            "interface": "Ethernet2",
        }

        with patch.dict(
            "sys.modules",
            {
                "atlas.persistence.db": fake_db,
                "persistence.db": fake_db,
            },
        ):
            result = await service.routing_history_summary(
                session_id="routing-test",
                destination_ip="10.0.200.200",
            )

        self.assertIn("Historically known path devices: arista-ai1, arista-ai2", result)
        self.assertIn("Primary OSPF peering to troubleshoot: arista-ai3 Ethernet2 <-> arista-ai4 Ethernet2", result)
        stored = session_store.get("routing-test")["routing_history"]
        self.assertEqual(stored["historical_devices"], ["arista-ai1", "arista-ai2"])
        self.assertEqual(stored["peer_hint"]["to_device"], "arista-ai4")


if __name__ == "__main__":
    unittest.main()
