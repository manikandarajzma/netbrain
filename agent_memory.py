"""
Agent tool result cache backed by Redis.

After each successful tool call the raw + normalized result is stored keyed by
(tool_name, tool_args) — shared across all agents. Before calling any MCP tool,
tool_executor checks this cache and skips the backend call if already known.

Redis key: atlas:toolcache:{sha256(tool+args)[:24]}
TTL: AGENT_MEMORY_TTL env var (default 86400 = 24 hours)
"""
import hashlib
import json
import logging
import os
from typing import Any

_log = logging.getLogger("atlas.agent_memory")
_MEMORY_TTL = int(os.getenv("AGENT_MEMORY_TTL", "86400"))  # 24 hours

_redis_client = None
_redis_checked = False


def _get_redis():
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    _redis_checked = True
    url = os.getenv("REDIS_URL", "").strip()
    if not url:
        return None
    try:
        import redis
        client = redis.from_url(url, decode_responses=True)
        client.ping()
        _redis_client = client
    except Exception as exc:
        _log.warning("agent_memory: Redis unavailable: %s", exc)
    return _redis_client


def _cache_key(tool_name: str, tool_args: dict) -> str:
    canonical = json.dumps({tool_name: tool_args}, sort_keys=True)
    return "atlas:toolcache:" + hashlib.sha256(canonical.encode()).hexdigest()[:24]


def store_result(tool_name: str, tool_args: dict, raw_result: Any, normalized: Any) -> None:
    """Cache raw + normalized tool result keyed by tool + args."""
    r = _get_redis()
    if not r:
        return
    try:
        payload = json.dumps({"tool": tool_name, "args": tool_args, "raw": raw_result, "result": normalized})
        r.setex(_cache_key(tool_name, tool_args), _MEMORY_TTL, payload)
        _log.debug("agent_memory: cached %s(%s)", tool_name, tool_args)
    except Exception as exc:
        _log.warning("agent_memory: failed to store result: %s", exc)


def get_cached_result(tool_name: str, tool_args: dict) -> tuple[Any, Any] | None:
    """Return (raw_result, normalized) from cache, or None if not found."""
    r = _get_redis()
    if not r:
        return None
    try:
        raw = r.get(_cache_key(tool_name, tool_args))
        if not raw:
            return None
        data = json.loads(raw)
        return data.get("raw"), data.get("result")
    except Exception:
        return None


def clear_cache() -> None:
    """Delete all cached tool results."""
    r = _get_redis()
    if not r:
        return
    try:
        keys = list(r.scan_iter("atlas:toolcache:*"))
        if keys:
            r.delete(*keys)
    except Exception as exc:
        _log.warning("agent_memory: failed to clear: %s", exc)
