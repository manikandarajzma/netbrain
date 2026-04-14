import unittest
from unittest.mock import AsyncMock, patch

from chat_service import process_message


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    @patch("chat_service.atlas_runtime.extract_final_response")
    @patch("chat_service.atlas_runtime.invoke_atlas_graph", new_callable=AsyncMock)
    async def test_process_message_delegates_to_graph_execution_runtime(self, mock_invoke, mock_extract):
        mock_invoke.return_value = {"final_response": {"role": "assistant", "content": "done"}}
        mock_extract.return_value = {"role": "assistant", "content": "done"}

        result = await process_message(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
            [{"role": "user", "content": "previous"}],
            username="alice",
            session_id="session-1",
            ignored_flag=True,
        )

        self.assertEqual(result, {"role": "assistant", "content": "done"})
        mock_invoke.assert_awaited_once_with(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
            [{"role": "user", "content": "previous"}],
            username="alice",
            session_id="session-1",
        )
        mock_extract.assert_called_once_with({"final_response": {"role": "assistant", "content": "done"}})


if __name__ == "__main__":
    unittest.main()
