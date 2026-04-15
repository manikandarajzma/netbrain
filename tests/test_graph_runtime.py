import unittest

from services.graph_runtime import AtlasRuntime, atlas_runtime


class GraphRuntimeHelperTests(unittest.TestCase):
    def test_build_initial_state_sets_expected_defaults(self):
        state = atlas_runtime.build_initial_state(
            "hello",
            [{"role": "user", "content": "prev"}],
            "alice",
            "session-1",
        )
        self.assertEqual(state["prompt"], "hello")
        self.assertEqual(state["conversation_history"], [{"role": "user", "content": "prev"}])
        self.assertEqual(state["username"], "alice")
        self.assertEqual(state["session_id"], "session-1")
        self.assertIsNone(state["intent"])
        self.assertIsNone(state["rbac_error"])
        self.assertIsNone(state["final_response"])

    def test_build_graph_config_includes_thread_id_only_when_present(self):
        runtime = AtlasRuntime()
        self.assertEqual(runtime.build_graph_config(None), {"recursion_limit": 50})
        self.assertEqual(
            runtime.build_graph_config("session-1"),
            {"recursion_limit": 50, "configurable": {"thread_id": "session-1"}},
        )

    def test_extract_final_response_returns_fallback_message(self):
        self.assertEqual(
            atlas_runtime.extract_final_response({}),
            {"role": "assistant", "content": "Something went wrong — please try again."},
        )


if __name__ == "__main__":
    unittest.main()
