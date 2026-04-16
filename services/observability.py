"""Shared structured observability helpers for Atlas."""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4


def new_request_id() -> str:
    return uuid4().hex


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def log_event(logger: logging.Logger, event: str, *, level: str = "info", **fields: Any) -> None:
    payload = {"event": event}
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = json_safe(value)
    message = json.dumps(payload, sort_keys=True, default=str)
    getattr(logger, level)(message)


def elapsed_ms(start: float) -> int:
    return int((perf_counter() - start) * 1000)
