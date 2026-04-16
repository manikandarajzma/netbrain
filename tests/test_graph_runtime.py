import unittest
from unittest.mock import AsyncMock, patch

from services.graph_runtime import AtlasRuntime, atlas_runtime


class GraphRuntimeHelperTests(unittest.IsolatedAsyncioTestCase):
    def test_build_initial_state_sets_expected_defaults(self):
        state = atlas_runtime.build_initial_state(
            "hello",
            [{"role": "user", "content": "prev"}],
            "alice",
            "session-1",
            None,
        )
        self.assertEqual(state["prompt"], "hello")
        self.assertEqual(state["conversation_history"], [{"role": "user", "content": "prev"}])
        self.assertEqual(state["username"], "alice")
        self.assertEqual(state["session_id"], "session-1")
        self.assertTrue(state["request_id"])
        self.assertIsNone(state["intent"])
        self.assertIsNone(state["rbac_error"])
        self.assertIsNone(state["final_response"])

    def test_build_initial_state_uses_supplied_request_id(self):
        state = atlas_runtime.build_initial_state(
            "hello",
            [],
            "alice",
            "session-1",
            "req-123",
        )
        self.assertEqual(state["request_id"], "req-123")

    def test_build_graph_config_includes_thread_id_only_when_present(self):
        runtime = AtlasRuntime()
        self.assertEqual(runtime.build_graph_config(None), {"recursion_limit": 50})
        self.assertEqual(
            runtime.build_graph_config("session-1"),
            {"recursion_limit": 50, "configurable": {"thread_id": "session-1"}},
        )

    @patch("services.graph_runtime.checkpointer_runtime.ensure_ready", new_callable=AsyncMock)
    async def test_invoke_atlas_graph_propagates_request_id(self, _mock_checkpointer):
        import types

        fake_graph = types.SimpleNamespace(ainvoke=AsyncMock(return_value={"final_response": {"role": "assistant", "content": "ok"}}))
        fake_builder = types.SimpleNamespace(get_graph=lambda: fake_graph)
        fake_module = types.SimpleNamespace(graph_builder=fake_builder)

        with patch.dict("sys.modules", {"atlas.graph_builder": fake_module}):
            await atlas_runtime.invoke_atlas_graph(
                "hello",
                [],
                username="alice",
                session_id="session-1",
                request_id="req-123",
            )

        args, kwargs = fake_graph.ainvoke.call_args
        self.assertEqual(args[0]["request_id"], "req-123")
        self.assertEqual(kwargs["config"], {"recursion_limit": 50, "configurable": {"thread_id": "session-1"}})

    def test_extract_final_response_returns_fallback_message(self):
        self.assertEqual(
            atlas_runtime.extract_final_response({}),
            {"role": "assistant", "content": "Something went wrong — please try again."},
        )


if __name__ == "__main__":
    unittest.main()
