import unittest
from unittest.mock import AsyncMock, patch

from services.connectivity_snapshot_service import ConnectivitySnapshotService


class ConnectivitySnapshotServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_build_snapshot_returns_unavailable_without_devices(self):
        service = ConnectivitySnapshotService()

        result = await service.build_snapshot_summary(
            session_id="snapshot-test",
            store={},
            source_ip="10.0.100.100",
            dest_ip="10.0.200.200",
            push_status=AsyncMock(),
        )

        self.assertEqual(
            result,
            "Connectivity snapshot unavailable: no in-scope devices were discovered from live path or history.",
        )

    @patch("services.connectivity_snapshot_service.nornir_client.post", new_callable=AsyncMock)
    async def test_build_snapshot_collects_and_stores_structured_evidence(self, mock_nornir_post):
        service = ConnectivitySnapshotService()
        mock_nornir_post.return_value = {
            "protocol_discovery": {
                "device": "arista-ai1",
                "routing_protocols": ["ospf"],
                "configured_routing_protocols": ["ospf"],
                "observed_route_types": ["ospf", "connected"],
            },
            "all_interfaces": {
                "device": "arista-ai1",
                "interfaces": [
                    {
                        "interface": "Ethernet2",
                        "up": True,
                        "oper_status": "up",
                        "primary_ip": "169.254.0.1",
                        "prefix_len": 31,
                    }
                ],
            },
            "syslog": {"device": "arista-ai1", "logs": [], "relevant": []},
            "route_to_destination": {
                "found": True,
                "protocol": "ospf",
                "interface": "Ethernet2",
                "next_hop": "169.254.0.2",
            },
            "route_to_source": {
                "found": True,
                "protocol": "connected",
                "interface": "Ethernet1",
                "next_hop": None,
            },
            "ospf_neighbors": {"device": "arista-ai1", "count": 1, "neighbors": ["1.1.1.1"]},
            "ospf_interfaces": {"device": "arista-ai1", "ospf_interface_count": 1},
            "interface_details": {
                "Ethernet2": {
                    "interface": "Ethernet2",
                    "up": True,
                    "oper_status": "up",
                    "primary_ip": "169.254.0.1",
                    "prefix_len": 31,
                    "input_errors": 0,
                    "output_errors": 0,
                    "input_discards": 0,
                    "output_discards": 0,
                }
            },
        }
        store = {
            "path_hops": [
                {"from_device": "10.0.100.100", "to_device": "arista-ai1", "in_interface": "Ethernet1"},
                {"from_device": "arista-ai1", "to_device": "10.0.200.200", "out_interface": "Ethernet2"},
            ],
            "reverse_path_hops": [],
            "routing_history": {},
            "path_meta": {"first_hop_device": "arista-ai1", "last_hop_device": "arista-ai1"},
            "reverse_path_meta": {},
            "protocol_discovery": {},
            "all_interfaces": {},
            "syslog": {},
            "interface_details": {},
            "interface_counters": [],
            "peering_inspections": [],
        }

        result = await service.build_snapshot_summary(
            session_id="snapshot-test",
            store=store,
            source_ip="10.0.100.100",
            dest_ip="10.0.200.200",
            push_status=AsyncMock(),
        )

        self.assertIn("Connectivity incident snapshot:", result)
        self.assertIn("forward_path_devices: arista-ai1", result)
        self.assertIn("route_to_destination=ospf via Ethernet2 next-hop 169.254.0.2", result)
        self.assertIn("connectivity_snapshot", store)
        self.assertEqual(store["connectivity_snapshot"]["devices"], ["arista-ai1"])
        self.assertTrue(store["connectivity_snapshot"]["live_evidence_available"])


if __name__ == "__main__":
    unittest.main()
