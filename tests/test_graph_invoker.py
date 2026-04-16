import types
import unittest
from unittest.mock import ANY, AsyncMock, patch

from services.graph_runtime import atlas_runtime


class GraphInvokerTests(unittest.IsolatedAsyncioTestCase):
    @patch("services.graph_runtime.ensure_checkpointer", new_callable=AsyncMock)
    async def test_invoke_atlas_graph_builds_state_and_config_and_calls_graph(self, mock_ensure):
        fake_graph = types.SimpleNamespace(ainvoke=AsyncMock(return_value={"final_response": {"content": "ok"}}))
        fake_module = types.SimpleNamespace(atlas_graph=fake_graph)

        with patch.dict("sys.modules", {"atlas.graph_builder": fake_module}):
            result = await atlas_runtime.invoke_atlas_graph(
                "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
                [{"role": "user", "content": "previous"}],
                username="alice",
                session_id="session-1",
            )

        self.assertEqual(result, {"final_response": {"content": "ok"}})
        mock_ensure.assert_awaited_once()
        fake_graph.ainvoke.assert_awaited_once_with(
            {
                "prompt": "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
                "conversation_history": [{"role": "user", "content": "previous"}],
                "username": "alice",
                "session_id": "session-1",
                "request_id": ANY,
                "intent": None,
                "rbac_error": None,
                "final_response": None,
            },
            config={"recursion_limit": 50, "configurable": {"thread_id": "session-1"}},
        )


if __name__ == "__main__":
    unittest.main()
