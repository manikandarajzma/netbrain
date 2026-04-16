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

    @patch("services.connectivity_snapshot_service.path_trace_service.infer_vrf", new_callable=AsyncMock)
    @patch("services.connectivity_snapshot_service.nornir_client.post", new_callable=AsyncMock)
    async def test_build_snapshot_uses_trace_metadata_for_forward_and_reverse_pings(
        self,
        mock_nornir_post,
        mock_infer_vrf,
    ):
        service = ConnectivitySnapshotService()
        mock_infer_vrf.return_value = "dest-vrf"
        ping_payloads: list[dict[str, object]] = []

        async def _side_effect(endpoint, payload, *args, **kwargs):
            if endpoint == "/device-snapshot":
                device = payload["device"]
                return {
                    "protocol_discovery": {
                        "device": device,
                        "routing_protocols": ["ospf"],
                        "configured_routing_protocols": ["ospf"],
                        "observed_route_types": ["ospf", "connected"],
                    },
                    "all_interfaces": {
                        "device": device,
                        "interfaces": [
                            {
                                "interface": "Ethernet1",
                                "up": True,
                                "oper_status": "up",
                                "primary_ip": "10.0.100.1" if device == "arista-ai1" else "10.0.200.1",
                                "prefix_len": 24,
                            },
                            {
                                "interface": "Ethernet2",
                                "up": True,
                                "oper_status": "up",
                                "primary_ip": "169.254.0.1" if device == "arista-ai1" else "169.254.0.2",
                                "prefix_len": 31,
                            },
                        ],
                    },
                    "syslog": {"device": device, "logs": [], "relevant": []},
                    "route_to_destination": {
                        "found": True,
                        "protocol": "ospf" if device == "arista-ai1" else "connected",
                        "interface": "Ethernet2" if device == "arista-ai1" else "Ethernet1",
                        "next_hop": "169.254.0.2" if device == "arista-ai1" else None,
                    },
                    "route_to_source": {
                        "found": True,
                        "protocol": "connected" if device == "arista-ai1" else "ospf",
                        "interface": "Ethernet1" if device == "arista-ai1" else "Ethernet2",
                        "next_hop": None if device == "arista-ai1" else "169.254.0.1",
                    },
                    "ospf_neighbors": {"device": device, "count": 1, "neighbors": ["1.1.1.1"]},
                    "ospf_interfaces": {"device": device, "ospf_interface_count": 1},
                    "interface_details": {
                        "Ethernet1": {
                            "interface": "Ethernet1",
                            "up": True,
                            "oper_status": "up",
                            "primary_ip": "10.0.100.1" if device == "arista-ai1" else "10.0.200.1",
                            "prefix_len": 24,
                            "input_errors": 0,
                            "output_errors": 0,
                            "input_discards": 0,
                            "output_discards": 0,
                        },
                        "Ethernet2": {
                            "interface": "Ethernet2",
                            "up": True,
                            "oper_status": "up",
                            "primary_ip": "169.254.0.1" if device == "arista-ai1" else "169.254.0.2",
                            "prefix_len": 31,
                            "input_errors": 0,
                            "output_errors": 0,
                            "input_discards": 0,
                            "output_discards": 0,
                        },
                    },
                }
            if endpoint == "/ping":
                ping_payloads.append(dict(payload))
                return {"success": True, "loss_pct": 0, "rtt_avg_ms": 4.2}
            raise AssertionError(f"Unexpected endpoint: {endpoint}")

        mock_nornir_post.side_effect = _side_effect
        store = {
            "path_hops": [
                {"from_device": "10.0.100.100", "to_device": "arista-ai1", "in_interface": "Ethernet1"},
                {"from_device": "arista-ai1", "to_device": "arista-ai2", "out_interface": "Ethernet2", "in_interface": "Ethernet1"},
            ],
            "reverse_path_hops": [
                {"from_device": "10.0.200.200", "to_device": "arista-ai4", "in_interface": "Ethernet1"},
                {"from_device": "arista-ai4", "to_device": "arista-ai3", "out_interface": "Ethernet2", "in_interface": "Ethernet2"},
            ],
            "routing_history": {},
            "path_meta": {
                "first_hop_device": "arista-ai1",
                "first_hop_lan_interface": "Ethernet1",
                "src_vrf": "source-vrf",
            },
            "reverse_path_meta": {
                "reverse_first_hop_device": "arista-ai4",
                "reverse_first_hop_lan_interface": "Ethernet1",
            },
            "protocol_discovery": {},
            "all_interfaces": {},
            "syslog": {},
            "interface_details": {},
            "interface_counters": [],
            "ping_results": [],
            "peering_inspections": [],
        }

        result = await service.build_snapshot_summary(
            session_id="snapshot-test",
            store=store,
            source_ip="10.0.100.100",
            dest_ip="10.0.200.200",
            push_status=AsyncMock(),
        )

        self.assertEqual(
            ping_payloads,
            [
                {
                    "device": "arista-ai1",
                    "destination": "10.0.200.200",
                    "source_interface": "Ethernet1",
                    "vrf": "source-vrf",
                },
                {
                    "device": "arista-ai4",
                    "destination": "10.0.100.100",
                    "source_interface": "Ethernet1",
                    "vrf": "dest-vrf",
                },
            ],
        )
        self.assertEqual(store["connectivity_snapshot"]["forward_ping"]["device"], "arista-ai1")
        self.assertEqual(store["connectivity_snapshot"]["reverse_ping"]["device"], "arista-ai4")
        self.assertIn("forward_ping: arista-ai1 -> 10.0.200.200", result)
        self.assertIn("reverse_ping: arista-ai4 -> 10.0.100.100", result)


if __name__ == "__main__":
    unittest.main()
