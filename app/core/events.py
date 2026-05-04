"""In-memory pub/sub for SSE event streams.

Each analysis_id has a list of asyncio.Queue subscribers. The inference runner
calls publish() to broadcast progress/log events. SSE endpoints call subscribe()
to receive them. Restart-safe: queues live for the lifetime of the process.
"""
import asyncio
from collections import defaultdict
from typing import Any

_subscribers: dict[str, list[asyncio.Queue]] = defaultdict(list)
_lock = asyncio.Lock()


async def subscribe(channel: str) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    async with _lock:
        _subscribers[channel].append(q)
    return q


async def unsubscribe(channel: str, q: asyncio.Queue) -> None:
    async with _lock:
        try:
            _subscribers[channel].remove(q)
        except ValueError:
            pass
        if not _subscribers[channel]:
            _subscribers.pop(channel, None)


async def publish(channel: str, event: dict[str, Any]) -> None:
    async with _lock:
        queues = list(_subscribers.get(channel, []))
    for q in queues:
        await q.put(event)


def publish_sync(channel: str, event: dict[str, Any]) -> None:
    """Schedule a publish from a sync context (e.g. BackgroundTasks worker)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(publish(channel, event), loop)
            return
    except RuntimeError:
        pass
    asyncio.run(publish(channel, event))
