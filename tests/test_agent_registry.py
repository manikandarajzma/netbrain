import unittest

from agents.agent_registry import agent_registry
from tools.tool_registry import tool_registry


class AgentRegistryTests(unittest.TestCase):
    def test_registry_exposes_existing_agent_specs(self):
        description = agent_registry.describe()

        self.assertIn("troubleshoot", description)
        self.assertIn("network_ops", description)
        self.assertEqual(description["troubleshoot"]["workflow_owner"], "TroubleshootWorkflowService")
        self.assertEqual(description["network_ops"]["workflow_owner"], "NetworkOpsWorkflowService")
        self.assertEqual(description["troubleshoot"]["workflow_type"], "troubleshoot")
        self.assertEqual(description["network_ops"]["workflow_type"], "network_ops")
        self.assertIn("connectivity", description["troubleshoot"]["scenario_descriptions"])
        self.assertIn("incident_record", description["network_ops"]["scenario_descriptions"])

    def test_troubleshoot_connectivity_spec_resolves_connectivity_profile(self):
        spec = agent_registry.get("troubleshoot")

        self.assertEqual(spec.resolve_tool_profile("help troubleshoot tcp 443", "connectivity"), "troubleshoot.connectivity")
        self.assertIs(
            spec.resolve_tools("help troubleshoot tcp 443", "connectivity"),
            tool_registry.get_profile_tools("troubleshoot.connectivity"),
        )

    def test_network_ops_spec_resolves_no_path_profile_for_explicit_ci_flows(self):
        spec = agent_registry.get("network_ops")

        self.assertEqual(spec.resolve_tool_profile("show me CHG0030042", "record_lookup"), "network_ops.no_path")
        self.assertEqual(
            spec.resolve_tool_profile(
                "create an incident short description: issue ci name: arista-ai1",
                "incident_record",
            ),
            "network_ops.no_path",
        )
        self.assertEqual(
            spec.resolve_tool_profile(
                "create a change request short description: route-map update",
                "change_record",
            ),
            "network_ops",
        )

    def test_registry_exposes_valid_route_keys_and_route_descriptions(self):
        self.assertEqual(agent_registry.valid_route_keys(), {"troubleshoot", "network_ops"})
        descriptions = agent_registry.route_descriptions()
        self.assertIn("Connectivity", descriptions["troubleshoot"])
        self.assertIn("ServiceNow", descriptions["network_ops"])


if __name__ == "__main__":
    unittest.main()
