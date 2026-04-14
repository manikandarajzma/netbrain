import unittest
from unittest.mock import patch

from agents import network_ops_agent, troubleshoot_agent


class TroubleshootAgentBuilderTests(unittest.TestCase):
    @patch("agents.troubleshoot_agent.create_specialized_agent")
    @patch("agents.troubleshoot_agent.build_default_llm")
    def test_connectivity_prompt_uses_connectivity_toolset(self, mock_build_default_llm, mock_create_specialized_agent):
        mock_build_default_llm.return_value = "llm"
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

    @patch("agents.troubleshoot_agent.create_specialized_agent")
    @patch("agents.troubleshoot_agent.build_default_llm")
    def test_general_troubleshoot_prompt_does_not_auto_enable_memory(self, mock_build_default_llm, mock_create_specialized_agent):
        mock_build_default_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        troubleshoot_agent.build_agent("help me troubleshoot bgp on core-rtr-01", "general")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], troubleshoot_agent.TROUBLESHOOT_BASE_TOOLS)

    @patch("agents.troubleshoot_agent.create_specialized_agent")
    @patch("agents.troubleshoot_agent.build_default_llm")
    def test_history_hint_enables_memory_augmented_tools(self, mock_build_default_llm, mock_create_specialized_agent):
        mock_build_default_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        troubleshoot_agent.build_agent("same issue again on arista-ai1 after last time", "general")

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], troubleshoot_agent.MEMORY_AUGMENTED_TOOLS)

    @patch("agents.troubleshoot_agent.create_specialized_agent")
    @patch("agents.troubleshoot_agent.build_default_llm")
    def test_performance_scenario_keeps_memory_available(self, mock_build_default_llm, mock_create_specialized_agent):
        mock_build_default_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        troubleshoot_agent.build_agent(
            "traffic is slow from 10.0.100.100 to 10.0.200.200",
            "performance",
        )

        args, _kwargs = mock_create_specialized_agent.call_args
        self.assertIs(args[1], troubleshoot_agent.MEMORY_AUGMENTED_TOOLS)


class NetworkOpsAgentBuilderTests(unittest.TestCase):
    @patch("agents.network_ops_agent.create_specialized_agent")
    @patch("agents.network_ops_agent.build_default_llm")
    def test_network_ops_builder_uses_network_ops_toolset(self, mock_build_default_llm, mock_create_specialized_agent):
        mock_build_default_llm.return_value = "llm"
        mock_create_specialized_agent.return_value = "agent"

        result = network_ops_agent.build_agent()

        self.assertEqual(result, "agent")
        mock_create_specialized_agent.assert_called_once()
        args, kwargs = mock_create_specialized_agent.call_args
        self.assertEqual(args[0], "llm")
        self.assertIs(args[1], network_ops_agent.NETWORK_OPS_TOOLS)
        self.assertEqual(args[3], "network_ops")
        self.assertEqual(kwargs, {})


if __name__ == "__main__":
    unittest.main()
