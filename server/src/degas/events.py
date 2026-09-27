"""In-process pub/sub feeding the `/api/events` SSE stream."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

log = logging.getLogger(__name__)

Event = dict[str, Any]


class EventBus:
    def __init__(self, max_queue: int = 1000) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()
        self._max_queue = max_queue

    def publish(self, event: Event) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                # A stalled client; it refetches state when it reconnects.
                log.warning("dropping slow event subscriber")
                self._subscribers.discard(queue)

    async def subscribe(self) -> AsyncIterator[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(self._max_queue)
        self._subscribers.add(queue)
        try:
            while queue in self._subscribers or not queue.empty():
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)
