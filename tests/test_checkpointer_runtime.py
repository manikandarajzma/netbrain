import unittest
import importlib

import services.checkpointer_runtime as checkpointer_runtime


class CheckpointerRuntimeTests(unittest.TestCase):
    def test_default_status_is_pending_first_graph_run(self):
        module = importlib.reload(checkpointer_runtime)

        status = module.checkpointer_runtime.get_status()

        self.assertEqual(status["state"], "pending")
        self.assertEqual(status["label"], "Pending first graph run")
        self.assertFalse(status["ready"])
        self.assertIsNone(status["error"])


if __name__ == "__main__":
    unittest.main()
