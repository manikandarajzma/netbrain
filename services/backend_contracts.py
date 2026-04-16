"""Shared backend result and error contract helpers."""
from __future__ import annotations

from typing import Any


def backend_unavailable(backend: str, action: str, detail: Any, *, subject: str = "") -> str:
    reason = str(detail or "unknown error").strip() or "unknown error"
    context = f" for {subject}" if subject else ""
    return f"{backend} unavailable during {action}{context}: {reason}"


def lookup_error(entity_label: str, detail: Any) -> str:
    reason = str(detail or "unknown error").strip() or "unknown error"
    return f"{entity_label} lookup error: {reason}"


def not_found(entity_label: str, detail: Any) -> str:
    reason = str(detail or "unknown error").strip() or "unknown error"
    return f"{entity_label} not found: {reason}"


def operation_failed(operation_label: str, detail: Any) -> str:
    reason = str(detail or "unknown error").strip() or "unknown error"
    return f"{operation_label} failed: {reason}"


def unexpected_response(operation_label: str, result: Any) -> str:
    return f"{operation_label} failed: unexpected response {result!r}"


def verification_failed(entity_label: str, identifier: str, detail: Any) -> str:
    reason = str(detail or "unknown error").strip() or "unknown error"
    return f"{entity_label} could not be verified for {identifier}.\nlookup_error: {reason}"
