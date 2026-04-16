"""Owned client for the local Nornir HTTP service and its run-scoped cache."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from time import perf_counter
from typing import Any

import httpx

try:
    from atlas.services.metrics import metrics_recorder
    from atlas.services.observability import elapsed_ms, json_safe, log_event
    from atlas.tools.resilience import retry_async, CircuitBreaker
except ImportError:
    from services.metrics import metrics_recorder  # type: ignore
    from services.observability import elapsed_ms, json_safe, log_event  # type: ignore
    from tools.resilience import retry_async, CircuitBreaker  # type: ignore


logger = logging.getLogger("atlas.nornir_client")
class NornirClient:
    def __init__(self, base_url: str = "http://localhost:8006", run_cache_ttl: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.run_cache_ttl = run_cache_ttl
        self._redis_client = None
        self._redis_checked = False

    def _get_redis(self):
        if self._redis_checked:
            return self._redis_client
        self._redis_checked = True
        url = os.getenv("REDIS_URL", "").strip()
        if not url:
            return None
        try:
            import redis

            client = redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis_client = client
        except Exception:
            self._redis_client = None
        return self._redis_client

    def _run_cache_key(self, session_id: str, endpoint: str, payload: dict[str, Any]) -> str:
        blob = json.dumps({"endpoint": endpoint, "payload": payload}, sort_keys=True, default=str)
        digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return f"atlas:run_cache:{session_id}:{digest}"

    def _run_cache_index_key(self, session_id: str) -> str:
        return f"atlas:run_cache:{session_id}:keys"

    def clear_session_cache(self, session_id: str) -> None:
        r = self._get_redis()
        if not r or not session_id:
            return
        try:
            index_key = self._run_cache_index_key(session_id)
            keys = list(r.smembers(index_key) or [])
            if keys:
                r.delete(*keys)
            r.delete(index_key)
        except Exception:
            pass

    async def request_json(
        self,
        method: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float = 30.0,
        retries: bool = True,
    ) -> dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        method = method.upper()
        started_at = perf_counter()

        async def _do() -> dict:
            async with httpx.AsyncClient(timeout=timeout) as c:
                if method == "GET":
                    r = await c.get(url)
                else:
                    r = await c.post(url, json=payload or {})
                r.raise_for_status()
                return r.json()

        if not retries:
            result = await _do()
            duration_ms = elapsed_ms(started_at)
            metrics_recorder.increment("atlas.nornir.request.completed", method=method, endpoint=endpoint, retries="false")
            metrics_recorder.observe_ms("atlas.nornir.request.duration_ms", duration_ms, method=method, endpoint=endpoint)
            log_event(
                logger,
                "nornir_request_completed",
                method=method,
                endpoint=endpoint,
                timeout=timeout,
                retries=False,
                elapsed_ms=duration_ms,
            )
            return result

        cb = CircuitBreaker.for_endpoint(url)
        result = await retry_async(
            cb,
            _do,
            retryable_exc=(httpx.HTTPStatusError, httpx.TimeoutException, httpx.NetworkError),
        )
        duration_ms = elapsed_ms(started_at)
        metrics_recorder.increment("atlas.nornir.request.completed", method=method, endpoint=endpoint, retries="true")
        metrics_recorder.observe_ms("atlas.nornir.request.duration_ms", duration_ms, method=method, endpoint=endpoint)
        log_event(
            logger,
            "nornir_request_completed",
            method=method,
            endpoint=endpoint,
            timeout=timeout,
            retries=True,
            elapsed_ms=duration_ms,
        )
        return result

    async def post(self, endpoint: str, payload: dict[str, Any], *, timeout: float = 30.0, retries: bool = True) -> dict:
        return await self.request_json("POST", endpoint, payload=payload, timeout=timeout, retries=retries)

    async def get(self, endpoint: str, *, timeout: float = 30.0, retries: bool = True) -> dict:
        return await self.request_json("GET", endpoint, timeout=timeout, retries=retries)

    async def cached_post(
        self,
        session_id: str,
        endpoint: str,
        payload: dict[str, Any],
        *,
        timeout: float = 30.0,
        retries: bool = True,
    ) -> dict:
        r = self._get_redis()
        if not r or not session_id:
            return await self.post(endpoint, payload, timeout=timeout, retries=retries)

        cache_key = self._run_cache_key(session_id, endpoint, payload)
        try:
            cached = r.get(cache_key)
            if cached:
                data = json.loads(cached)
                if isinstance(data, dict):
                    metrics_recorder.increment("atlas.nornir.cache.hit", endpoint=endpoint)
                    log_event(
                        logger,
                        "nornir_cache_hit",
                        session_id=session_id,
                        endpoint=endpoint,
                    )
                    return data
        except Exception:
            pass

        data = await self.post(endpoint, payload, timeout=timeout, retries=retries)
        try:
            r.setex(cache_key, self.run_cache_ttl, json.dumps(json_safe(data)))
            r.sadd(self._run_cache_index_key(session_id), cache_key)
            r.expire(self._run_cache_index_key(session_id), self.run_cache_ttl)
            metrics_recorder.increment("atlas.nornir.cache.store", endpoint=endpoint)
            log_event(
                logger,
                "nornir_cache_store",
                session_id=session_id,
                endpoint=endpoint,
                ttl_seconds=self.run_cache_ttl,
            )
        except Exception:
            pass
        return data


nornir_client = NornirClient()
