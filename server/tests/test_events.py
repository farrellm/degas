import asyncio

from degas.events import EventBus


async def test_publish_reaches_all_subscribers() -> None:
    bus = EventBus()
    a, b = bus.subscribe(), bus.subscribe()
    first_a = asyncio.ensure_future(anext(a))
    first_b = asyncio.ensure_future(anext(b))
    await asyncio.sleep(0)
    bus.publish({"type": "job", "n": 1})
    assert await first_a == {"type": "job", "n": 1}
    assert await first_b == {"type": "job", "n": 1}
    await a.aclose()
    await b.aclose()


async def test_slow_subscriber_is_dropped() -> None:
    bus = EventBus(max_queue=2)
    sub = bus.subscribe()
    pending = asyncio.ensure_future(anext(sub))
    await asyncio.sleep(0)
    for n in range(4):
        bus.publish({"n": n})
    assert await pending == {"n": 0}
    remaining = [e async for e in sub]
    assert remaining == [{"n": 1}]  # 2 overflowed the queue
