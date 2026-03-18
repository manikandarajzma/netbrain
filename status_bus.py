"""Per-session status event bus for SSE streaming."""
import asyncio

_queues: dict[str, asyncio.Queue] = {}


def register(session_id: str) -> asyncio.Queue:
    q = asyncio.Queue()
    _queues[session_id] = q
    return q


def deregister(session_id: str) -> None:
    _queues.pop(session_id, None)


async def push(session_id: str, message: str) -> None:
    q = _queues.get(session_id)
    if q:
        await q.put({"type": "status", "message": message})
