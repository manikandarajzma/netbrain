"""
Daily sync of closed ServiceNow incidents into RedisVL semantic memory.

Runs at midnight via a background asyncio task started in app.py.
Fetches incidents closed in the last 90 days, extracts short_description + close_notes,
and stores each as a memory entry so the troubleshoot orchestrator can recall
real past root causes during investigations.
"""
import logging
import os

logger = logging.getLogger("atlas.servicenow_memory_sync")

_SYNC_DAYS = int(os.getenv("SNOW_MEMORY_SYNC_DAYS", "90"))
_BATCH_SIZE = 100  # ServiceNow page size


async def sync_closed_incidents() -> int:
    """
    Fetch closed/resolved incidents from the last SNOW_MEMORY_SYNC_DAYS days
    and store them in RedisVL memory. Returns the number of incidents stored.
    """
    logger.info("servicenow_memory_sync: starting sync (last %d days)", _SYNC_DAYS)
    try:
        from atlas.integrations import servicenowauth
        from tools.servicenow_tools import _base_url, _auth
        import aiohttp
        from atlas.memory.agent_memory import store_incident_memory, store_confirmed_resolution
    except Exception as exc:
        logger.warning("servicenow_memory_sync: import failed: %s", exc)
        return 0

    # ServiceNow query: all active + recently closed incidents updated within last N days
    snow_query = (
        f"sys_updated_on>=javascript:gs.daysAgoStart({_SYNC_DAYS})"
        f"^short_descriptionISNOTEMPTY"
    )
    params = {
        "sysparm_query":         snow_query,
        "sysparm_limit":         str(_BATCH_SIZE),
        "sysparm_fields":        "number,short_description,close_notes,description,cmdb_ci,sys_updated_on",
        "sysparm_display_value": "true",
        "sysparm_offset":        "0",
    }

    stored = 0
    offset = 0

    async with aiohttp.ClientSession() as session:
        while True:
            params["sysparm_offset"] = str(offset)
            url = _base_url() + "/api/now/table/incident"
            try:
                async with session.get(url, params=params, auth=_auth(), timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        logger.warning("servicenow_memory_sync: HTTP %s", resp.status)
                        break
                    data = await resp.json()
            except Exception as exc:
                logger.warning("servicenow_memory_sync: fetch failed at offset %d: %s", offset, exc)
                break

            records = data.get("result", [])
            if not records:
                break

            for rec in records:
                number = rec.get("number", "")
                short_desc = (rec.get("short_description") or "").strip()
                close_notes = (rec.get("close_notes") or "").strip()
                description = (rec.get("description") or "").strip()
                ci_raw = rec.get("cmdb_ci") or ""
                if isinstance(ci_raw, dict):
                    ci_raw = ci_raw.get("display_value") or ci_raw.get("value") or ""
                cmdb_ci = str(ci_raw).strip()
                notes = close_notes or description
                if not number or not short_desc:
                    continue
                # Skip generic/cleanup close notes that provide no diagnostic value
                if close_notes and any(phrase in close_notes.lower() for phrase in (
                    "test data cleanup", "closing as part of", "bulk close", "auto-closed"
                )):
                    logger.debug("servicenow_memory_sync: skipping %s — low-quality close notes", number)
                    continue
                await store_incident_memory(number, short_desc, notes, cmdb_ci=cmdb_ci)
                # Store confirmed resolution separately if cmdb_ci is a real device hostname
                if cmdb_ci and len(close_notes) >= 30:
                    store_confirmed_resolution(cmdb_ci, number, short_desc, close_notes)
                stored += 1

            if len(records) < _BATCH_SIZE:
                break  # last page
            offset += _BATCH_SIZE

    logger.info("servicenow_memory_sync: stored %d incidents", stored)
    return stored
