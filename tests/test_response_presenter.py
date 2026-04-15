import unittest

from services.response_presenter import (
    ResponsePresenter,
    response_presenter,
)


class ResponsePresenterTests(unittest.TestCase):
    def test_response_presenter_class_groups_duplicate_devices(self):
        grouped = ResponsePresenter().group_interface_counters(
            [
                {"device": "arista-ai1", "window_s": 6, "clean": ["Ethernet2"], "active": []},
                {"device": "arista-ai1", "window_s": 6, "clean": ["Ethernet1"], "active": []},
            ]
        )

        self.assertEqual(
            grouped,
            [
                {
                    "device": "arista-ai1",
                    "window_s": 6,
                    "ssh_error": "",
                    "active": [],
                    "clean": ["Ethernet1", "Ethernet2"],
                }
            ],
        )

    def test_group_interface_counters_merges_duplicate_devices(self):
        grouped = response_presenter.group_interface_counters(
            [
                {
                    "device": "arista-ai1",
                    "window_s": 6,
                    "clean": ["Ethernet2"],
                    "active": [],
                },
                {
                    "device": "arista-ai1",
                    "window_s": 6,
                    "clean": ["Ethernet1", "Ethernet2"],
                    "active": [],
                },
            ]
        )

        self.assertEqual(
            grouped,
            [
                {
                    "device": "arista-ai1",
                    "window_s": 6,
                    "ssh_error": "",
                    "active": [],
                    "clean": ["Ethernet1", "Ethernet2"],
                }
            ],
        )

    def test_build_network_ops_content_hides_path_for_incident_creation(self):
        content = response_presenter.build_network_ops_content(
            "Created incident INC0010044",
            {
                "path_hops": [{"node_type": "host"}, {"node_type": "device"}],
                "reverse_path_hops": [{"node_type": "host"}, {"node_type": "device"}],
            },
            'Create an incident for "Connectivity issues between 10.0.100.100 and 10.0.200.200 on tcp port 443"',
        )

        self.assertEqual(content, {"direct_answer": "Created incident INC0010044"})

    def test_build_network_ops_content_keeps_path_for_firewall_style_request(self):
        content = response_presenter.build_network_ops_content(
            "Reviewed firewall path",
            {
                "path_hops": [{"node_type": "host"}, {"node_type": "device"}],
                "reverse_path_hops": [{"node_type": "host"}, {"node_type": "device"}],
            },
            "show me the firewall path from 10.0.100.100 to 10.0.200.200",
        )

        self.assertEqual(content["direct_answer"], "Reviewed firewall path")
        self.assertEqual(content["source"], "10.0.100.100")
        self.assertEqual(content["destination"], "10.0.200.200")
        self.assertIn("path_hops", content)
        self.assertIn("reverse_path_hops", content)

    def test_build_troubleshoot_content_replaces_servicenow_section_and_groups_counters(self):
        result = response_presenter.build_troubleshoot_content(
            "## Summary\nIssue found\n\n## ServiceNow\nold data",
            {
                "path_hops": [{"node_type": "host"}, {"node_type": "device"}],
                "interface_counters": [
                    {"device": "arista-ai1", "window_s": 6, "clean": ["Ethernet2"], "active": []},
                    {"device": "arista-ai1", "window_s": 6, "clean": ["Ethernet1"], "active": []},
                ],
                "servicenow_summary": "Incidents found: 2\n\n### Change Requests\n- CHG0030042",
            },
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443",
            None,
        )

        self.assertIn("direct_answer", result)
        self.assertIn("CHG0030042", result["direct_answer"])
        self.assertEqual(
            result["interface_counters"],
            [
                {
                    "device": "arista-ai1",
                    "window_s": 6,
                    "ssh_error": "",
                    "active": [],
                    "clean": ["Ethernet1", "Ethernet2"],
                }
            ],
        )

    def test_build_troubleshoot_content_fails_closed_when_live_evidence_is_unavailable(self):
        result = response_presenter.build_troubleshoot_content(
            "## Root Cause\nIncorrect OSPF guess",
            {
                "servicenow_summary": "Incidents found: 2\n\n### Change Requests\n- CHG0030042",
                "connectivity_snapshot": {
                    "live_evidence_available": False,
                    "errors": {
                        "arista-ai1": "TCP connection to device failed",
                        "arista-ai2": "TCP connection to device failed",
                    },
                },
            },
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443",
            None,
        )

        self.assertIn("Unable to determine the current root cause from live evidence", result["direct_answer"])
        self.assertIn("arista-ai1: TCP connection to device failed", result["direct_answer"])
        self.assertIn("CHG0030042", result["direct_answer"])
        self.assertNotIn("Incorrect OSPF guess", result["direct_answer"])


if __name__ == "__main__":
    unittest.main()
