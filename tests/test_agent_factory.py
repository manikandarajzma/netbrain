import unittest
from unittest.mock import patch

from langchain_core.messages import SystemMessage

from agents.agent_factory import AgentFactory, create_specialized_agent


class AgentFactoryTests(unittest.TestCase):
    @patch("agents.agent_factory.create_react_agent")
    def test_agent_factory_class_builds_minimal_react_agent(self, mock_create_react_agent):
        mock_create_react_agent.return_value = "agent"

        agent = AgentFactory().create_specialized_agent("llm", ["tool-a"], "system prompt", "troubleshoot")

        self.assertEqual(agent, "agent")
        args, kwargs = mock_create_react_agent.call_args
        self.assertEqual(args[0], "llm")
        self.assertEqual(args[1], ["tool-a"])
        self.assertEqual(kwargs["name"], "troubleshoot")
        self.assertIsInstance(kwargs["prompt"], SystemMessage)
        self.assertEqual(kwargs["prompt"].content, "system prompt")

    @patch("agents.agent_factory.create_react_agent")
    def test_create_specialized_agent_builds_minimal_react_agent(self, mock_create_react_agent):
        mock_create_react_agent.return_value = "agent"

        agent = create_specialized_agent("llm", ["tool-a"], "system prompt", "troubleshoot")

        self.assertEqual(agent, "agent")
        mock_create_react_agent.assert_called_once()
        args, kwargs = mock_create_react_agent.call_args
        self.assertEqual(args[0], "llm")
        self.assertEqual(args[1], ["tool-a"])
        self.assertEqual(kwargs["name"], "troubleshoot")
        self.assertIsInstance(kwargs["prompt"], SystemMessage)
        self.assertEqual(kwargs["prompt"].content, "system prompt")


if __name__ == "__main__":
    unittest.main()
