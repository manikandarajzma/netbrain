"""
Semantic agent memory backed by RedisVL.

Stores past troubleshooting sessions as vector embeddings so agents can recall
semantically similar past findings as context — not as conclusions.

Redis index: atlas:memory  (vector search, cosine similarity)
Embedding:   sentence-transformers/all-MiniLM-L6-v2  (384-dim, CPU-only, ~22MB)
TTL:         30 days (configurable via AGENT_MEMORY_TTL_DAYS env var)

Also keeps the original exact-match tool result cache (atlas:toolcache:*).
"""
import hashlib
import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("atlas.agent_memory")

_MEMORY_TTL_DAYS = int(os.getenv("AGENT_MEMORY_TTL_DAYS", "30"))
_MEMORY_TTL = _MEMORY_TTL_DAYS * 86400
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_INDEX_NAME = "atlas:memory"
_DIMS = 384  # all-MiniLM-L6-v2

_index = None
_vectorizer = None


# ---------------------------------------------------------------------------
# Vectorizer (lazy-loaded — first embed call downloads the model once)
# ---------------------------------------------------------------------------

def _get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        from redisvl.utils.vectorize import HFTextVectorizer
        _vectorizer = HFTextVectorizer(model="sentence-transformers/all-MiniLM-L6-v2")
    return _vectorizer


# ---------------------------------------------------------------------------
# Index (lazy-created on first use)
# ---------------------------------------------------------------------------

async def _get_index():
    global _index
    if _index is not None:
        return _index
    try:
        from redisvl.index import AsyncSearchIndex
        from redisvl.schema import IndexSchema

        schema = IndexSchema.from_dict({
            "index": {"name": _INDEX_NAME, "prefix": "atlas:mem"},
            "fields": [
                {"name": "query",          "type": "text"},
                {"name": "result_summary", "type": "text"},
                {"name": "agent_type",     "type": "tag"},
                {"name": "user_corrected", "type": "tag"},
                {"name": "timestamp",      "type": "numeric"},
                {"name": "query_embedding", "type": "vector", "attrs": {
                    "dims": _DIMS,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                }},
            ],
        })
        idx = AsyncSearchIndex(schema=schema, redis_url=_REDIS_URL)
        await idx.create(overwrite=False)
        _index = idx
        logger.info("agent_memory: vector index ready")
    except Exception as exc:
        logger.warning("agent_memory: index init failed: %s", exc)
        return None
    return _index


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def store_memory(
    query: str,
    result_summary: str,
    agent_type: str = "troubleshoot",
) -> None:
    """
    Embed and store a query + result summary for future recall.

    Args:
        query:          The user's original query (e.g. "troubleshoot connectivity from 10.0.0.1 to 11.0.0.1")
        result_summary: The root cause + recommendation from the agent's response
        agent_type:     Agent category — "troubleshoot", "servicenow", etc.
    """
    try:
        idx = await _get_index()
        if idx is None:
            return
        import numpy as np
        vec = _get_vectorizer().embed(query)
        vec_bytes = np.array(vec, dtype=np.float32).tobytes()
        mem_key = "atlas:mem:" + hashlib.sha256(query.encode()).hexdigest()[:20]

        # If a user-corrected entry exists for this query, don't overwrite it.
        try:
            import redis.asyncio as _aioredis
            _r = _aioredis.from_url(_REDIS_URL)
            corrected_flag = await _r.hget(mem_key, "user_corrected")
            if corrected_flag and (corrected_flag == "1" if isinstance(corrected_flag, str) else corrected_flag == b"1"):
                logger.info("agent_memory: skipping overwrite — entry is user-corrected for %r", query[:60])
                await _r.aclose()
                return
            # Delete any other stale entries with auto-generated keys for this query
            existing_keys = []
            async for k in _r.scan_iter("atlas:mem:*"):
                if k == mem_key:
                    continue
                val = await _r.hget(k, "query")
                if val and (val == query if isinstance(val, str) else val.decode() == query):
                    existing_keys.append(k)
            if existing_keys:
                await _r.delete(*existing_keys)
            await _r.aclose()
        except Exception as _del_exc:
            logger.debug("agent_memory: pre-delete failed: %s", _del_exc)

        doc = {
            "query":           query,
            "result_summary":  result_summary,
            "agent_type":      agent_type,
            "timestamp":       int(time.time()),
            "query_embedding": vec_bytes,
            "user_corrected":  "0",
        }
        # Deterministic key so future corrections overwrite cleanly.
        mem_key = "atlas:mem:" + hashlib.sha256(query.encode()).hexdigest()[:20]
        await idx.load([doc], keys=[mem_key], ttl=_MEMORY_TTL)
        logger.info("agent_memory: stored %s memory for %r", agent_type, query[:80])
    except Exception as exc:
        logger.warning("agent_memory: store failed: %s", exc)


async def store_memory_correction(
    query: str,
    result_summary: str,
    agent_type: str = "troubleshoot",
) -> None:
    """Store a user-supplied correction. Marks the entry as user_corrected=1
    so automatic agent runs never overwrite it."""
    await store_memory(query, result_summary, agent_type)
    # After storing, set the user_corrected flag directly on the Redis hash
    try:
        import redis.asyncio as _aioredis
        _r = _aioredis.from_url(_REDIS_URL)
        mem_key = "atlas:mem:" + hashlib.sha256(query.encode()).hexdigest()[:20]
        await _r.hset(mem_key, "user_corrected", "1")
        await _r.aclose()
        logger.info("agent_memory: marked %r as user_corrected", query[:60])
    except Exception as exc:
        logger.warning("agent_memory: failed to set user_corrected flag: %s", exc)


async def recall_memory(
    query: str,
    agent_type: str = "troubleshoot",
    top_k: int = 3,
    min_similarity: float = 0.70,
) -> list[dict]:
    """
    Find past results semantically similar to the current query.

    Returns a list of dicts with keys: query, result_summary, timestamp.
    Only returns results above min_similarity threshold.
    """
    try:
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        idx = await _get_index()
        if idx is None:
            return []

        import numpy as np
        vec = _get_vectorizer().embed(query)
        vec_bytes = np.array(vec, dtype=np.float32).tobytes()
        tag_filter = Tag("agent_type") == agent_type
        q = VectorQuery(
            vector=vec_bytes,
            vector_field_name="query_embedding",
            return_fields=["query", "result_summary", "timestamp"],
            num_results=top_k,
            filter_expression=tag_filter,
        )
        results = await idx.query(q)

        # vector_distance is cosine distance (0=identical, 1=opposite) → similarity = 1 - distance
        filtered = []
        for r in results:
            dist = float(r.get("vector_distance", 1.0))
            similarity = 1.0 - dist
            if similarity >= min_similarity:
                r["similarity"] = round(similarity, 3)
                filtered.append(r)

        logger.info("agent_memory: recalled %d/%d results for %r", len(filtered), len(results), query[:60])
        return filtered
    except Exception as exc:
        logger.warning("agent_memory: recall failed: %s", exc)
        return []


def format_memory_context(memories: list[dict]) -> str:
    """Format recalled memories as a context block for injection into prompts."""
    if not memories:
        return ""
    lines = ["**Similar past cases (from memory — use as context only, not as conclusions):**"]
    for m in memories:
        ts = int(m.get("timestamp", 0))
        age_days = max(0, int((time.time() - ts) / 86400))
        age_str = f"{age_days}d ago" if age_days > 0 else "today"
        similarity_pct = int(float(m.get("similarity", 0)) * 100)
        lines.append(
            f"- [{age_str}, {similarity_pct}% similar] Query: {m.get('query', '')}\n"
            f"  Findings: {m.get('result_summary', '')}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool plan cache — caches the ReAct agent's tool-calling sequence so repeat
# queries skip the LLM decision loop entirely and execute tools directly.
#
# Redis index: atlas:toolplan  (vector search, cosine similarity)
# Key:         atlas:tp:{sha256(query)[:20]}
# Invalidation: tool_fingerprint tag — changes when the tool set changes
# ---------------------------------------------------------------------------

_PLAN_INDEX_NAME = "atlas:toolplan"
_plan_index = None


async def _get_plan_index():
    global _plan_index
    if _plan_index is not None:
        return _plan_index
    try:
        from redisvl.index import AsyncSearchIndex
        from redisvl.schema import IndexSchema

        schema = IndexSchema.from_dict({
            "index": {"name": _PLAN_INDEX_NAME, "prefix": "atlas:tp"},
            "fields": [
                {"name": "query",            "type": "text"},
                {"name": "tool_sequence",    "type": "text"},
                {"name": "tool_fingerprint", "type": "tag"},
                {"name": "timestamp",        "type": "numeric"},
                {"name": "query_embedding",  "type": "vector", "attrs": {
                    "dims": _DIMS,
                    "distance_metric": "cosine",
                    "algorithm": "flat",
                    "datatype": "float32",
                }},
            ],
        })
        idx = AsyncSearchIndex(schema=schema, redis_url=_REDIS_URL)
        await idx.create(overwrite=False)
        _plan_index = idx
        logger.info("agent_memory: tool plan index ready")
    except Exception as exc:
        logger.warning("agent_memory: plan index init failed: %s", exc)
        return None
    return _plan_index


async def store_tool_plan(
    query: str,
    tool_sequence: list[str],
    tool_fingerprint: str,
) -> None:
    """Store the ordered list of tools the ReAct agent called for this query."""
    try:
        idx = await _get_plan_index()
        if idx is None:
            return
        import numpy as np
        vec = _get_vectorizer().embed(query)
        vec_bytes = np.array(vec, dtype=np.float32).tobytes()
        doc = {
            "query":            query,
            "tool_sequence":    json.dumps(tool_sequence),
            "tool_fingerprint": tool_fingerprint,
            "timestamp":        int(time.time()),
            "query_embedding":  vec_bytes,
        }
        plan_key = "atlas:tp:" + hashlib.sha256(query.encode()).hexdigest()[:20]
        await idx.load([doc], keys=[plan_key])
        logger.info("agent_memory: stored tool plan %s for %r", tool_sequence, query[:60])
    except Exception as exc:
        logger.warning("agent_memory: store_tool_plan failed: %s", exc)


async def recall_tool_plan(
    query: str,
    tool_fingerprint: str,
    min_similarity: float = 0.92,
) -> list[str] | None:
    """Return the cached tool sequence for a semantically similar query,
    or None if no match above the similarity threshold or fingerprint differs."""
    try:
        from redisvl.query import VectorQuery
        from redisvl.query.filter import Tag

        idx = await _get_plan_index()
        if idx is None:
            return None

        import numpy as np
        vec = _get_vectorizer().embed(query)
        vec_bytes = np.array(vec, dtype=np.float32).tobytes()
        fp_filter = Tag("tool_fingerprint") == tool_fingerprint
        q = VectorQuery(
            vector=vec_bytes,
            vector_field_name="query_embedding",
            return_fields=["tool_sequence", "tool_fingerprint"],
            num_results=1,
            filter_expression=fp_filter,
        )
        results = await idx.query(q)
        if not results:
            return None
        r = results[0]
        dist = float(r.get("vector_distance", 1.0))
        similarity = 1.0 - dist
        if similarity < min_similarity:
            logger.info("agent_memory: plan cache miss (similarity=%.2f < %.2f)", similarity, min_similarity)
            return None
        plan = json.loads(r.get("tool_sequence", "[]"))
        logger.info("agent_memory: plan cache hit (similarity=%.2f) → %s", similarity, plan)
        return plan
    except Exception as exc:
        logger.warning("agent_memory: recall_tool_plan failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Legacy exact-match tool result cache (unchanged)
# ---------------------------------------------------------------------------

_EXACT_TTL = int(os.getenv("AGENT_MEMORY_TTL", "86400"))
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
        logger.warning("agent_memory: Redis unavailable: %s", exc)
    return _redis_client


def _cache_key(tool_name: str, tool_args: dict) -> str:
    canonical = json.dumps({tool_name: tool_args}, sort_keys=True)
    return "atlas:toolcache:" + hashlib.sha256(canonical.encode()).hexdigest()[:24]


def store_result(tool_name: str, tool_args: dict, raw_result: Any, normalized: Any) -> None:
    r = _get_redis()
    if not r:
        return
    try:
        payload = json.dumps({"tool": tool_name, "args": tool_args, "raw": raw_result, "result": normalized})
        r.setex(_cache_key(tool_name, tool_args), _EXACT_TTL, payload)
    except Exception as exc:
        logger.warning("agent_memory: failed to store result: %s", exc)


def get_cached_result(tool_name: str, tool_args: dict) -> tuple[Any, Any] | None:
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
    r = _get_redis()
    if not r:
        return
    try:
        keys = list(r.scan_iter("atlas:toolcache:*"))
        if keys:
            r.delete(*keys)
    except Exception as exc:
        logger.warning("agent_memory: failed to clear: %s", exc)
