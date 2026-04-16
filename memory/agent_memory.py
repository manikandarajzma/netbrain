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

import re as _re
# Matches device hostnames like PA-FW-01, EDGE-RTR-01, CORE-SW-01, DIST-RTR-02
_DEVICE_RE = _re.compile(r'\b([A-Z]{2,}(?:-[A-Z0-9]+)*-\d{2,})\b')

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
                {"name": "resolution",     "type": "text"},
                {"name": "agent_type",     "type": "tag"},
                {"name": "devices",        "type": "tag"},
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
        doc = {
            "query":           query,
            "result_summary":  result_summary,
            "agent_type":      agent_type,
            "timestamp":       int(time.time()),
            "query_embedding": vec_bytes,
        }
        await idx.load([doc], keys=[mem_key], ttl=_MEMORY_TTL)
        logger.info("agent_memory: stored %s memory for %r", agent_type, query[:80])
    except Exception as exc:
        logger.warning("agent_memory: store failed: %s", exc)


async def store_incident_memory(
    incident_number: str,
    short_description: str,
    close_notes: str,
    cmdb_ci: str = "",
) -> None:
    """Store a ServiceNow incident as a memory entry.
    Keyed by incident number so re-syncs are idempotent."""
    if not short_description.strip():
        return
    try:
        idx = await _get_index()
        if idx is None:
            return
        import numpy as np
        # Embed combined text so the vector captures both symptom and root cause
        combined_text = f"{short_description} {close_notes}"
        vec = _get_vectorizer().embed(combined_text)
        vec_bytes = np.array(vec, dtype=np.float32).tobytes()
        # Use cmdb_ci as primary device tag; fall back to extracting hostnames from text
        if cmdb_ci:
            found_devices = [cmdb_ci]
        else:
            _hostname_re = _re.compile(r'\b([a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)*\d+)\b')
            found_devices = list(dict.fromkeys(
                m.lower() for m in _hostname_re.findall(combined_text)
                if len(m) >= 4 and m.lower() not in ("high", "low", "state", "open", "true", "false")
            ))
        devices_tag = "|".join(found_devices) if found_devices else ""
        doc = {
            "query":           short_description,
            "result_summary":  f"[{incident_number}] {short_description}",
            "resolution":      close_notes or "",
            "agent_type":      "incident",
            "devices":         devices_tag,
            "timestamp":       int(time.time()),
            "query_embedding": vec_bytes,
        }
        mem_key = f"atlas:mem:inc:{incident_number}"
        await idx.load([doc], keys=[mem_key], ttl=_MEMORY_TTL)
        logger.debug("agent_memory: stored incident %s", incident_number)
    except Exception as exc:
        logger.warning("agent_memory: store_incident_memory failed: %s", exc)



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
            return_fields=["query", "result_summary", "resolution", "timestamp", "agent_type"],
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


async def recall_incidents_by_devices(
    devices: list[str],
    top_k: int = 8,
    query: str = "",
    min_similarity: float = 0.28,
) -> list[dict]:
    """
    Return the top_k most semantically relevant incidents for the given path devices.

    Uses a single VectorQuery with a device tag filter so Redis scores ALL incidents
    for the path devices and returns only the top_k — scales to hundreds per device.

    Query enrichment: strip IPs (no semantic value) and append device names so the
    embedding is grounded in path context ("trace path arista1 arista2" scores much
    closer to "arista1: Interface flap" than the raw query does).

    Falls back to a plain FilterQuery when no query is provided.
    Results are marked with match_type='device'.
    """
    if not devices:
        return []
    try:
        from redisvl.query.filter import Tag

        idx = await _get_index()
        if idx is None:
            return []

        type_filter = Tag("agent_type") == "incident"
        device_filter = None
        for device in devices:
            df = Tag("devices") == device
            device_filter = df if device_filter is None else (device_filter | df)
        combined_filter = type_filter & device_filter

        if query:
            import re as _re2
            import numpy as np
            from redisvl.query import VectorQuery

            clean_query = _re2.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}(?:/\d+)?\b', '', query).strip()
            enriched_query = f"{clean_query} {' '.join(devices)}".strip()

            vec = _get_vectorizer().embed(enriched_query)
            vec_bytes = np.array(vec, dtype=np.float32).tobytes()
            q = VectorQuery(
                vector=vec_bytes,
                vector_field_name="query_embedding",
                return_fields=["query", "result_summary", "resolution", "timestamp", "devices"],
                num_results=top_k,
                filter_expression=combined_filter,
            )
            results = await idx.query(q)
            filtered = []
            for r in results:
                sim = round(1.0 - float(r.get("vector_distance", 1.0)), 3)
                if sim >= min_similarity:
                    r["similarity"] = sim
                    r["match_type"] = "device"
                    filtered.append(r)
            logger.info(
                "agent_memory: device vector recall %d/%d above threshold=%.2f for devices %s, enriched_query=%r",
                len(filtered), len(results), min_similarity, devices, enriched_query[:80],
            )
            if filtered:
                return filtered
            # Vector recall found matches but all below threshold — fall back to filter query
            logger.info("agent_memory: vector recall empty — falling back to filter query for devices %s", devices)

        from redisvl.query import FilterQuery
        q = FilterQuery(
            filter_expression=combined_filter,
            return_fields=["query", "result_summary", "resolution", "timestamp", "devices"],
            num_results=top_k,
        )
        results = await idx.query(q)
        for r in results:
            r["match_type"] = "device"
        logger.info("agent_memory: device filter recall found %d incidents for devices %s", len(results), devices)
        return results
    except Exception as exc:
        logger.warning("agent_memory: recall_incidents_by_devices failed: %s", exc)
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
        entry = (
            f"- [{age_str}, {similarity_pct}% similar] Query: {m.get('query', '')}\n"
            f"  Findings: {m.get('result_summary', '')}"
        )
        resolution = (m.get("resolution") or "").strip()
        if resolution:
            entry += f"\n  Known fix: {resolution[:200]}"
        lines.append(entry)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Device health baselines — long-term counter trend tracking
# ---------------------------------------------------------------------------

def store_device_health_snapshot(snapshots: list[dict]) -> None:
    """
    Persist one counter snapshot per interface for baseline trend tracking.
    Uses synchronous Redis client (fire-and-forget from async context).

    snapshots: list of structured dicts from call_interface_counters_agent,
               shape: {"device": str, "active": [...], "clean": [...], ...}
    """
    r = _get_redis()
    if not r:
        return
    _HEALTH_TTL = 8 * 86400   # 8 days
    _MAX_SNAPSHOTS = 336       # 7d × 48 polls/day
    ts = int(time.time())
    try:
        pipe = r.pipeline(transaction=False)
        for snap in snapshots:
            dev = snap.get("device", "")
            if not dev:
                continue
            for c in snap.get("active", []):
                intf = c.get("interface")
                if not intf or "error" in c:
                    continue
                entry = json.dumps({"ts": ts, "deltas": c.get("delta_9s", {})})
                key = f"atlas:health:{dev}:{intf}"
                pipe.lpush(key, entry)
                pipe.ltrim(key, 0, _MAX_SNAPSHOTS - 1)
                pipe.expire(key, _HEALTH_TTL)
            for intf in snap.get("clean", []):
                entry = json.dumps({"ts": ts, "deltas": {}})
                key = f"atlas:health:{dev}:{intf}"
                pipe.lpush(key, entry)
                pipe.ltrim(key, 0, _MAX_SNAPSHOTS - 1)
                pipe.expire(key, _HEALTH_TTL)
        pipe.execute()
    except Exception as exc:
        logger.warning("agent_memory: store_device_health_snapshot failed: %s", exc)


def get_health_trend(device: str, interface: str, current_deltas: dict) -> str:
    """
    Compare current counter deltas against 7-day history for this interface.
    Returns a human-readable trend string, or "" if no history or no anomaly.

    current_deltas: the delta_9s dict from an active-error entry,
                    e.g. {"crc_errors": 3, "input_errors": 0, "output_drops": 0}
    """
    r = _get_redis()
    if not r:
        return ""
    try:
        key = f"atlas:health:{device}:{interface}"
        raw_list = r.lrange(key, 1, -1)   # skip index 0 = the entry just written
        if len(raw_list) < 6:             # need at least a few data points
            return ""

        # Only use history older than 24 h for the baseline
        cutoff = int(time.time()) - 86400
        historical = []
        for raw in raw_list:
            try:
                e = json.loads(raw)
            except Exception:
                continue
            if e.get("ts", 0) < cutoff:
                historical.append(e.get("deltas", {}))

        if len(historical) < 4:
            return ""

        # Compute per-counter averages from historical clean+active snapshots
        all_keys = set()
        for h in historical:
            all_keys.update(h.keys())

        trends = []
        for counter in sorted(all_keys):
            hist_vals = [h.get(counter, 0) for h in historical]
            avg = sum(hist_vals) / len(hist_vals)
            cur = current_deltas.get(counter, 0)
            if avg < 0.5 and cur > 0:
                trends.append(f"normally 0 {counter.replace('_', ' ')} — currently {cur} (new today)")
            elif avg > 0 and cur > avg * 3 and cur > 2:
                trends.append(f"{counter.replace('_', ' ')}: 7d avg {avg:.1f} → currently {cur} ({cur/avg:.1f}× spike)")

        if not trends:
            return ""
        return f"{interface} on {device}: " + "; ".join(trends)
    except Exception as exc:
        logger.warning("agent_memory: get_health_trend failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Device reputation — tracks how many times each device was a failure point
# ---------------------------------------------------------------------------

def record_device_incident(device: str, category: str, session_id: str) -> None:
    """Record that a device was involved in an incident (as failure point or path member)."""
    r = _get_redis()
    if not r:
        return
    try:
        key = f"atlas:device_rep:{device.lower()}"
        ts = int(time.time())
        member = f"{category}:{session_id}"
        r.zadd(key, {member: ts})
        r.zremrangebyrank(key, 0, -201)   # keep latest 200
        r.expire(key, 14 * 86400)
    except Exception as exc:
        logger.warning("agent_memory: record_device_incident failed: %s", exc)


def get_device_reputation(device: str, window_days: int = 7) -> dict:
    """Return incident counts for a device over the last window_days."""
    r = _get_redis()
    result = {"incident_count": 0, "failure_count": 0, "categories": {}}
    if not r:
        return result
    try:
        key = f"atlas:device_rep:{device.lower()}"
        min_ts = int(time.time()) - window_days * 86400
        members = r.zrangebyscore(key, min_ts, "+inf")
        from collections import Counter
        cats: Counter = Counter()
        failure_count = 0
        for m in members:
            cat = m.split(":")[0] if ":" in m else m
            cats[cat] += 1
            if cat not in ("was_in_path",):
                failure_count += 1
        result["incident_count"] = len(members)
        result["failure_count"] = failure_count
        result["categories"] = dict(cats)
    except Exception as exc:
        logger.warning("agent_memory: get_device_reputation failed: %s", exc)
    return result


# ---------------------------------------------------------------------------
# Confirmed resolution store — operator-verified fixes from closed SNOW tickets
# ---------------------------------------------------------------------------

def store_confirmed_resolution(
    device: str,
    incident_number: str,
    symptom: str,
    fix: str,
) -> None:
    """Store a confirmed fix for a device from a closed ServiceNow incident."""
    if not device or not fix or len(fix) < 30:
        return
    r = _get_redis()
    if not r:
        return
    try:
        key = f"atlas:resolution:{device.lower()}"
        entry = json.dumps({
            "incident_number": incident_number,
            "symptom":         symptom[:200],
            "fix":             fix[:400],
            "ts":              int(time.time()),
        })
        ts = int(time.time())
        r.zadd(key, {entry: ts})
        r.zremrangebyrank(key, 0, -51)  # keep latest 50
        r.expire(key, 90 * 86400)
        logger.debug("agent_memory: stored confirmed resolution for %s (%s)", device, incident_number)
    except Exception as exc:
        logger.warning("agent_memory: store_confirmed_resolution failed: %s", exc)


def get_confirmed_resolutions(devices: list, max_per_device: int = 3) -> list[dict]:
    """Retrieve the most recent confirmed fixes for a list of devices."""
    r = _get_redis()
    if not r or not devices:
        return []
    results = []
    try:
        for device in devices:
            key = f"atlas:resolution:{device.lower()}"
            members = r.zrevrange(key, 0, max_per_device - 1)
            for m in members:
                try:
                    entry = json.loads(m)
                    entry["device"] = device
                    results.append(entry)
                except Exception:
                    pass
    except Exception as exc:
        logger.warning("agent_memory: get_confirmed_resolutions failed: %s", exc)
    return results


def format_confirmed_resolutions(resolutions: list[dict]) -> str:
    """Format confirmed resolutions as a context block for injection into LLM prompts."""
    if not resolutions:
        return ""
    lines = ["**Confirmed fixes from closed tickets (operator-verified — high confidence):**"]
    for r in resolutions:
        age_days = max(0, int((time.time() - r.get("ts", 0)) / 86400))
        age_str = f"{age_days}d ago" if age_days > 0 else "today"
        lines.append(
            f"- [{r['device']}, {age_str}, {r['incident_number']}] "
            f"Symptom: {r['symptom']}\n"
            f"  Fix: {r['fix']}"
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
