"""Lightweight in-process metrics owner for Atlas."""
from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import Any


def _normalize_tags(tags: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    normalized: list[tuple[str, str]] = []
    for key, value in sorted(tags.items()):
        if value is None:
            continue
        normalized.append((str(key), str(value)))
    return tuple(normalized)


class MetricsRecorder:
    """Owns lightweight counters and timing aggregates for Atlas."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = defaultdict(int)
        self._timings: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, int]] = defaultdict(
            lambda: {"count": 0, "total_ms": 0, "max_ms": 0}
        )

    def increment(self, name: str, value: int = 1, **tags: Any) -> None:
        key = (name, _normalize_tags(tags))
        with self._lock:
            self._counters[key] += value

    def observe_ms(self, name: str, value_ms: int, **tags: Any) -> None:
        key = (name, _normalize_tags(tags))
        with self._lock:
            bucket = self._timings[key]
            bucket["count"] += 1
            bucket["total_ms"] += int(value_ms)
            bucket["max_ms"] = max(bucket["max_ms"], int(value_ms))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = [
                {"name": name, "tags": dict(tags), "value": value}
                for (name, tags), value in self._counters.items()
            ]
            timings = []
            for (name, tags), bucket in self._timings.items():
                count = bucket["count"]
                total_ms = bucket["total_ms"]
                timings.append(
                    {
                        "name": name,
                        "tags": dict(tags),
                        "count": count,
                        "total_ms": total_ms,
                        "max_ms": bucket["max_ms"],
                        "avg_ms": int(total_ms / count) if count else 0,
                    }
                )
            return {"counters": counters, "timings": timings}

    def clear(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()


metrics_recorder = MetricsRecorder()
