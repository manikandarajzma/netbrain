import unittest
from unittest.mock import AsyncMock, patch

from chat_service import process_message


class ChatServiceTests(unittest.IsolatedAsyncioTestCase):
    @patch("chat_service.atlas_application.process_query", new_callable=AsyncMock)
    async def test_process_message_delegates_to_application(self, mock_process_query):
        mock_process_query.return_value = {"role": "assistant", "content": "done"}

        result = await process_message(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
            [{"role": "user", "content": "previous"}],
            username="alice",
            session_id="session-1",
            ignored_flag=True,
        )

        self.assertEqual(result, {"role": "assistant", "content": "done"})
        mock_process_query.assert_awaited_once_with(
            "help me troubleshoot connectivity from 10.0.100.100 to 10.0.200.200",
            [{"role": "user", "content": "previous"}],
            username="alice",
            session_id="session-1",
        )

if __name__ == "__main__":
    unittest.main()
