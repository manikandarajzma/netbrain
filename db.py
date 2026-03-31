"""
PostgreSQL connection pool — async, shared across the app.

Usage:
    from db import get_pool, fetchrow, fetch, execute

    # Single row
    row = await fetchrow(
        "SELECT next_hop, egress_interface FROM routing_table "
        "WHERE device=$1 AND vrf=$2 AND $3::inet << prefix "
        "ORDER BY masklen(prefix) DESC LIMIT 1",
        "EDGE-RTR-01", "CORP", "11.0.0.1"
    )

    # Multiple rows
    rows = await fetch("SELECT * FROM devices WHERE role=$1", "router")

    # Insert / update
    await execute(
        "INSERT INTO devices (hostname, mgmt_ip, platform) VALUES ($1, $2, $3) "
        "ON CONFLICT (hostname) DO UPDATE SET mgmt_ip=EXCLUDED.mgmt_ip",
        "EDGE-RTR-01", "10.255.0.1", "cisco_ios"
    )
"""
import logging
import os

logger = logging.getLogger("atlas.db")

_pool = None

_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/atlas")


async def get_pool():
    global _pool
    if _pool is not None:
        return _pool
    import asyncpg
    _pool = await asyncpg.create_pool(
        _DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=30,
    )
    logger.info("db: connection pool ready (%s)", _DATABASE_URL.split("@")[-1])
    return _pool


async def fetchrow(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def fetch(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def execute(query: str, *args):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def executemany(query: str, args_list: list):
    pool = await get_pool()
    async with pool.acquire() as conn:
        return await conn.executemany(query, args_list)


async def close():
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("db: connection pool closed")
