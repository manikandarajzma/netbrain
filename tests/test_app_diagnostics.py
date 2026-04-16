import unittest
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from atlas.app import api_internal_diagnostics


class AppDiagnosticsRouteTests(unittest.IsolatedAsyncioTestCase):
    @patch("atlas.atlas_application.atlas_application.get_diagnostics_snapshot", new_callable=AsyncMock)
    async def test_api_internal_diagnostics_returns_application_snapshot(self, mock_get_snapshot):
        mock_get_snapshot.return_value = {"owners": {}, "metrics": {}, "tools": {}}

        result = await api_internal_diagnostics(username="alice")

        self.assertEqual(result, {"owners": {}, "metrics": {}, "tools": {}})
        mock_get_snapshot.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
