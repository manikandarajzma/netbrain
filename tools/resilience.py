"""
Resilience primitives for A2A and MCP HTTP calls.

Provides:
  - retry_async()   — exponential backoff with jitter for async calls
  - retry_sync()    — same for synchronous (requests) calls
  - CircuitBreaker  — per-endpoint open/half-open/closed state machine

Usage (async):
    from tools.resilience import retry_async, CircuitBreaker

    _cb = CircuitBreaker("nornir")
    result = await retry_async(_cb, my_async_fn, arg1, arg2)

Usage (sync):
    from tools.resilience import retry_sync, CircuitBreaker

    _cb = CircuitBreaker("servicenow")
    result = retry_sync(_cb, my_sync_fn, arg1, arg2)
"""

import asyncio
import logging
import random
import time
from enum import Enum, auto
from typing import Any, Callable

logger = logging.getLogger("atlas.resilience")

# ---------------------------------------------------------------------------
# Tunables (override via env vars if needed)
# ---------------------------------------------------------------------------

MAX_ATTEMPTS   = 3          # total attempts (1 original + 2 retries)
BASE_DELAY     = 0.5        # seconds before first retry
MAX_DELAY      = 8.0        # cap on backoff
JITTER         = 0.25       # ± fraction of delay added as jitter

CB_FAILURE_THRESHOLD = 3    # consecutive failures to trip breaker
CB_RECOVERY_TIMEOUT  = 30.0 # seconds before half-open probe
CB_SUCCESS_THRESHOLD = 1    # successes in half-open to close


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_retryable_http(status: int) -> bool:
    return status in (429, 500, 502, 503, 504)


def _jittered(delay: float) -> float:
    return delay * (1 + random.uniform(-JITTER, JITTER))


def _next_delay(attempt: int) -> float:
    return min(BASE_DELAY * (2 ** attempt), MAX_DELAY)


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class _CBState(Enum):
    CLOSED    = auto()   # normal
    OPEN      = auto()   # failing — reject immediately
    HALF_OPEN = auto()   # probing one request to see if service recovered


class CircuitBreaker:
    """
    Per-endpoint circuit breaker.

    States:
      CLOSED    → normal; count consecutive failures
      OPEN      → reject all calls; after recovery_timeout → HALF_OPEN
      HALF_OPEN → allow one probe; success → CLOSED, failure → OPEN
    """

    # Class-level registry so one CB instance is shared per name
    _registry: dict[str, "CircuitBreaker"] = {}

    @classmethod
    def for_endpoint(cls, name: str) -> "CircuitBreaker":
        if name not in cls._registry:
            cls._registry[name] = cls(name)
        return cls._registry[name]

    def __init__(self, name: str):
        self.name = name
        self._state      = _CBState.CLOSED
        self._failures   = 0
        self._opened_at  = 0.0
        self._half_open_successes = 0

    @property
    def state(self) -> str:
        return self._state.name

    def allow_request(self) -> bool:
        if self._state == _CBState.CLOSED:
            return True
        if self._state == _CBState.OPEN:
            if time.monotonic() - self._opened_at >= CB_RECOVERY_TIMEOUT:
                logger.info("circuit[%s]: OPEN → HALF_OPEN (probe)", self.name)
                self._state = _CBState.HALF_OPEN
                self._half_open_successes = 0
                return True
            return False
        # HALF_OPEN — allow exactly one probe at a time
        return True

    def record_success(self) -> None:
        if self._state == _CBState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= CB_SUCCESS_THRESHOLD:
                logger.info("circuit[%s]: HALF_OPEN → CLOSED", self.name)
                self._state    = _CBState.CLOSED
                self._failures = 0
        elif self._state == _CBState.CLOSED:
            self._failures = 0

    def record_failure(self) -> None:
        self._failures += 1
        if self._state == _CBState.HALF_OPEN:
            logger.warning("circuit[%s]: HALF_OPEN → OPEN (probe failed)", self.name)
            self._state     = _CBState.OPEN
            self._opened_at = time.monotonic()
        elif self._state == _CBState.CLOSED and self._failures >= CB_FAILURE_THRESHOLD:
            logger.warning(
                "circuit[%s]: CLOSED → OPEN (%d consecutive failures)",
                self.name, self._failures,
            )
            self._state     = _CBState.OPEN
            self._opened_at = time.monotonic()


# ---------------------------------------------------------------------------
# Retry wrappers
# ---------------------------------------------------------------------------

class CircuitOpenError(Exception):
    pass


async def retry_async(
    cb: CircuitBreaker,
    fn: Callable,
    *args: Any,
    retryable_exc: tuple = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Call async fn(*args, **kwargs) with retries and circuit-breaker protection.

    retryable_exc: exception types that trigger a retry (default: all Exceptions).
    Raises the last exception if all attempts fail.
    Raises CircuitOpenError immediately if the breaker is open.
    """
    if not cb.allow_request():
        raise CircuitOpenError(f"Circuit breaker open for '{cb.name}' — service unavailable")

    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = await fn(*args, **kwargs)
            cb.record_success()
            return result
        except CircuitOpenError:
            raise
        except retryable_exc as exc:
            last_exc = exc
            cb.record_failure()
            if attempt < MAX_ATTEMPTS - 1:
                delay = _jittered(_next_delay(attempt))
                logger.warning(
                    "retry_async[%s]: attempt %d/%d failed (%s), retrying in %.2fs",
                    cb.name, attempt + 1, MAX_ATTEMPTS, exc, delay,
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    "retry_async[%s]: all %d attempts failed — last error: %s",
                    cb.name, MAX_ATTEMPTS, exc,
                )
    raise last_exc  # type: ignore[misc]


def retry_sync(
    cb: CircuitBreaker,
    fn: Callable,
    *args: Any,
    retryable_exc: tuple = (Exception,),
    **kwargs: Any,
) -> Any:
    """
    Synchronous version of retry_async.
    Uses time.sleep instead of asyncio.sleep.
    """
    if not cb.allow_request():
        raise CircuitOpenError(f"Circuit breaker open for '{cb.name}' — service unavailable")

    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            result = fn(*args, **kwargs)
            cb.record_success()
            return result
        except CircuitOpenError:
            raise
        except retryable_exc as exc:
            last_exc = exc
            cb.record_failure()
            if attempt < MAX_ATTEMPTS - 1:
                delay = _jittered(_next_delay(attempt))
                logger.warning(
                    "retry_sync[%s]: attempt %d/%d failed (%s), retrying in %.2fs",
                    cb.name, attempt + 1, MAX_ATTEMPTS, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "retry_sync[%s]: all %d attempts failed — last error: %s",
                    cb.name, MAX_ATTEMPTS, exc,
                )
    raise last_exc  # type: ignore[misc]
