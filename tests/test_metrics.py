import unittest

from services.metrics import MetricsRecorder


class MetricsRecorderTests(unittest.TestCase):
    def test_increment_records_counter_with_tags(self):
        recorder = MetricsRecorder()
        recorder.increment("atlas.query.started", agent_type="troubleshoot")
        recorder.increment("atlas.query.started", agent_type="troubleshoot")

        snapshot = recorder.snapshot()
        self.assertEqual(
            snapshot["counters"],
            [
                {
                    "name": "atlas.query.started",
                    "tags": {"agent_type": "troubleshoot"},
                    "value": 2,
                }
            ],
        )

    def test_observe_ms_records_aggregate_timing(self):
        recorder = MetricsRecorder()
        recorder.observe_ms("atlas.query.duration_ms", 10, content_type="dict")
        recorder.observe_ms("atlas.query.duration_ms", 20, content_type="dict")

        snapshot = recorder.snapshot()
        self.assertEqual(
            snapshot["timings"],
            [
                {
                    "name": "atlas.query.duration_ms",
                    "tags": {"content_type": "dict"},
                    "count": 2,
                    "total_ms": 30,
                    "max_ms": 20,
                    "avg_ms": 15,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
