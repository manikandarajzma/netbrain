import unittest

from services.path_trace_service import PathTraceService


class PathTraceServiceTests(unittest.IsolatedAsyncioTestCase):
    def test_extract_path_metadata_from_structured_hops(self):
        hops = [
            {
                "from_device": "10.0.100.100",
                "to_device": "arista-ai1",
                "in_interface": "Ethernet1",
            },
            {
                "from_device": "arista-ai1",
                "to_device": "arista-ai2",
                "out_interface": "Ethernet2",
                "in_interface": "Ethernet1",
            },
            {
                "from_device": "arista-ai2",
                "to_device": "10.0.200.200",
                "out_interface": "Ethernet2",
            },
        ]

        meta = PathTraceService.extract_path_metadata(hops)

        self.assertEqual(meta["first_hop_device"], "arista-ai1")
        self.assertEqual(meta["first_hop_lan_interface"], "Ethernet1")
        self.assertEqual(meta["first_hop_egress_interface"], "Ethernet2")
        self.assertEqual(meta["last_hop_device"], "arista-ai2")
        self.assertEqual(meta["last_hop_egress_interface"], "Ethernet2")
        self.assertEqual(meta["path_devices"], ["arista-ai1", "arista-ai2"])

    def test_extract_reverse_path_metadata_from_structured_hops(self):
        hops = [
            {
                "from_device": "10.0.200.200",
                "to_device": "arista-ai4",
                "in_interface": "Ethernet1",
            },
            {
                "from_device": "arista-ai4",
                "to_device": "arista-ai3",
                "out_interface": "Ethernet2",
                "in_interface": "Ethernet2",
            },
            {
                "from_device": "arista-ai3",
                "to_device": "10.0.100.100",
                "out_interface": "Ethernet1",
            },
        ]

        meta = PathTraceService.extract_reverse_path_metadata(hops)

        self.assertEqual(meta["reverse_first_hop_device"], "arista-ai4")
        self.assertEqual(meta["reverse_first_hop_lan_interface"], "Ethernet1")
        self.assertEqual(meta["reverse_first_hop_egress_interface"], "Ethernet2")
        self.assertEqual(meta["reverse_last_hop_device"], "arista-ai3")
        self.assertEqual(meta["reverse_last_hop_egress_interface"], "Ethernet1")

    async def test_infer_vrf_defaults_without_device(self):
        vrf = await PathTraceService.infer_vrf("10.0.100.100", "")
        self.assertEqual(vrf, "default")


if __name__ == "__main__":
    unittest.main()
