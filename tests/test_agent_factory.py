import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import SystemMessage

from agents.agent_factory import create_specialized_agent


class AgentFactoryTests(unittest.TestCase):
    @patch("agents.agent_factory.create_react_agent")
    def test_create_specialized_agent_forwards_common_options(self, mock_create_react_agent):
        fake_agent = SimpleNamespace()
        mock_create_react_agent.return_value = fake_agent

        agent = create_specialized_agent(
            "llm",
            ["tool-a"],
            "system prompt",
            "troubleshoot",
            checkpointer="cp",
            stream_mode="updates",
            response_format={"type": "json_schema"},
            debug=True,
            version="v1",
        )

        self.assertIs(agent, fake_agent)
        mock_create_react_agent.assert_called_once()
        _, kwargs = mock_create_react_agent.call_args
        self.assertEqual(kwargs["checkpointer"], "cp")
        self.assertEqual(kwargs["response_format"], {"type": "json_schema"})
        self.assertTrue(kwargs["debug"])
        self.assertEqual(kwargs["version"], "v1")
        self.assertEqual(kwargs["name"], "troubleshoot")
        self.assertIsInstance(kwargs["prompt"], SystemMessage)
        self.assertEqual(kwargs["prompt"].content, "system prompt")
        self.assertEqual(getattr(fake_agent, "atlas_stream_mode"), "updates")


if __name__ == "__main__":
    unittest.main()
