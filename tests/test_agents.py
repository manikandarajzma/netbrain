import unittest
from unittest.mock import patch

from agents import network_ops_agent, troubleshoot_agent


class TroubleshootAgentBuilderTests(unittest.TestCase):
    @patch("agents.troubleshoot_agent.agent_factory.create_specialized_agent")
    @patch("agents.troubleshoot_agent.agent_factory.build_troubleshoot_llm")
    def test_connectivity_prompt_uses_connectivity_toolset(self, mock_build_troubleshoot_llm, mock_create_specialized_agent):
        mock_build_troubleshoot_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        result = troubleshoot_agent.build_agent(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200 on tcp port 443",
            "connectivity",
        )

        self.assertEqual(result, "agent")
        mock_create_specialized_agent.assert_called_once()
        args, kwargs = mock_create_specialized_agent.call_args
        self.assertEqual(args[0], "llm")
        self.assertIs(args[1], troubleshoot_agent.CONNECTIVITY_TOOLS)
        self.assertEqual(args[3], "troubleshoot")
        self.assertEqual(kwargs, {})

    @patch("agents.troubleshoot_agent.agent_factory.create_specialized_agent")
    @patch("agents.troubleshoot_agent.agent_factory.build_troubleshoot_llm")
    def test_general_troubleshoot_prompt_keeps_full_toolset_available(self, mock_build_troubleshoot_llm, mock_create_specialized_agent):
        mock_build_troubleshoot_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        troubleshoot_agent.build_agent("help me troubleshoot bgp on core-rtr-01", "general")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], troubleshoot_agent.ALL_TOOLS)

    @patch("agents.troubleshoot_agent.agent_factory.create_specialized_agent")
    @patch("agents.troubleshoot_agent.agent_factory.build_troubleshoot_llm")
    def test_performance_scenario_loads_performance_prompt(self, mock_build_troubleshoot_llm, mock_create_specialized_agent):
        mock_build_troubleshoot_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        troubleshoot_agent.build_agent(
            "traffic is slow from 10.0.100.100 to 10.0.200.200",
            "performance",
        )

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], troubleshoot_agent.ALL_TOOLS)
        self.assertIn("Scenario: Performance", args[2])

    @patch("agents.troubleshoot_agent.agent_factory.create_specialized_agent")
    @patch("agents.troubleshoot_agent.agent_factory.build_troubleshoot_llm")
    def test_intermittent_scenario_loads_intermittent_prompt(self, mock_build_troubleshoot_llm, mock_create_specialized_agent):
        mock_build_troubleshoot_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        troubleshoot_agent.build_agent(
            "the link keeps flapping between arista-ai3 and arista-ai4",
            "intermittent",
        )

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], troubleshoot_agent.ALL_TOOLS)
        self.assertIn("Scenario: Intermittent", args[2])


class NetworkOpsAgentBuilderTests(unittest.TestCase):
    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_network_ops_builder_uses_network_ops_toolset(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        result = network_ops_agent.build_agent("show me CHG0030042", "general")

        self.assertEqual(result, "agent")
        mock_create_specialized_agent.assert_called_once()
        args, kwargs = mock_create_specialized_agent.call_args
        self.assertEqual(args[0], "llm")
        self.assertIs(args[1], network_ops_agent.NETWORK_OPS_TOOLS)
        self.assertEqual(args[3], "network_ops")
        self.assertEqual(kwargs, {})

    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_incident_request_with_explicit_ci_uses_no_path_toolset(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        network_ops_agent.build_agent(
            "create an incident short description: connectivity failure ci name: arista-ai1",
            "incident_record",
        )

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], network_ops_agent.NETWORK_OPS_TOOLS_NO_PATH)

    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_change_request_with_explicit_ci_uses_no_path_toolset(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        network_ops_agent.build_agent(
            "create a change request short description: route map update ci name: arista-ai1",
            "change_record",
        )

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], network_ops_agent.NETWORK_OPS_TOOLS_NO_PATH)

    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_generic_change_request_uses_change_record_scenario(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        network_ops_agent.build_agent("create a change request for arista-ai1 route map update", "change_record")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIn("Generic ServiceNow Change Request", args[2])
        self.assertNotIn("Access / Rule / Port Change Request", args[2])

    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_access_request_uses_network_change_template_scenario(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        network_ops_agent.build_agent("open port 443 from 10.0.100.100 to 10.0.200.200", "access_change")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIn("## Network Change Request", args[2])
        self.assertIn("Access / Rule / Port Change Request", args[2])

    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_incident_request_uses_incident_record_scenario(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        network_ops_agent.build_agent("create a sev2 incident for arista-ai1 routing failure", "incident_record")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIn("Scenario: Incident Record", args[2])

    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_record_lookup_uses_record_lookup_scenario(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        network_ops_agent.build_agent("show me INC0010035", "record_lookup")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], network_ops_agent.NETWORK_OPS_TOOLS_NO_PATH)
        self.assertIn("Scenario: Record Lookup", args[2])

    @patch("agents.network_ops_agent.agent_factory.create_specialized_agent")
    @patch("agents.network_ops_agent.agent_factory.build_network_ops_llm")
    def test_change_update_uses_change_update_scenario(self, mock_build_network_ops_llm, mock_create_specialized_agent):
        mock_build_network_ops_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        network_ops_agent.build_agent("close CHG0030042 with success notes", "change_update")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], network_ops_agent.NETWORK_OPS_TOOLS_NO_PATH)
        self.assertIn("Scenario: Change Update", args[2])


if __name__ == "__main__":
    unittest.main()
