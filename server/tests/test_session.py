import asyncio
import socket
import threading
from collections.abc import Callable
from typing import Any

from degas.colab.session import BOOTSTRAP
from degas.db import now


async def until(fn: Callable[[], Any], timeout: float = 5) -> None:
    async with asyncio.timeout(timeout):
        while not fn():  # noqa: ASYNC110 - polling a plain condition
            await asyncio.sleep(0.01)


async def test_idle_shutdown(harness: Any) -> None:
    config = harness.config.model_copy(update={"idle_timeout_min": 0.002})  # 120 ms
    svc = harness.build(config)
    await svc.start()
    try:
        await svc.sessions.start("T4", high_mem=False)
        await until(lambda: svc.sessions.state == "ready")
        await until(lambda: svc.sessions.state == "stopped")
        assert harness.colab.calls[-1] == "stop"
        assert harness.worker_exits == 1
    finally:
        await svc.stop()


async def test_pending_jobs_prevent_idle_shutdown(harness: Any) -> None:
    config = harness.config.model_copy(update={"idle_timeout_min": 0.002})
    svc = harness.build(config)
    svc.sessions.has_pending_jobs = lambda: True
    await svc.sessions.start("T4", high_mem=False)
    await until(lambda: svc.sessions.state == "ready")
    await asyncio.sleep(0.3)
    assert svc.sessions.state == "ready"
    await svc.sessions.stop()
    await svc.stop()


async def test_recover_when_vm_is_gone(harness: Any) -> None:
    svc = harness.build()
    session = svc.db.create_session("L4", False)
    svc.db.update_session(session["id"], state="busy")
    job = svc.db.insert_job({"family": "sdxl"}, [1])
    svc.db.update_job(job["id"], status="running", session_id=session["id"])

    await svc.start()
    try:
        assert svc.sessions.state == "stopped"
        assert svc.db.get_job(job["id"])["status"] == "error"  # type: ignore[index]
        assert harness.tunnels == []
    finally:
        await svc.stop()


async def test_recover_reattaches_to_a_live_vm(harness: Any) -> None:
    first = harness.build()
    await first.sessions.start("L4", high_mem=True)
    await until(lambda: first.sessions.state == "ready")
    old_activity = first.sessions.session["last_activity_at"]
    await first.stop()  # server shutdown: the VM keeps running

    second = harness.build()
    second.db.add_push_subscription("https://push.example/sub/1", {"p256dh": "k", "auth": "a"})
    await second.start()
    try:
        await until(lambda: second.sessions.state == "ready")
        # The worker answered /health, so it was neither reinstalled nor restarted.
        assert harness.colab.calls.count("exec") == 1
        assert len(harness.tunnels[1].uploads) == 0
        assert second.sessions.session["last_activity_at"] > old_activity
        await second.push.drain()
        assert harness.pushed == []  # reattaching isn't a launch: no "ready" push
    finally:
        await second.stop()


async def test_touch_resets_idle_deadline(harness: Any) -> None:
    svc = harness.build()
    await svc.sessions.start("T4", high_mem=False)
    await until(lambda: svc.sessions.state == "ready")
    before = svc.sessions.idle_deadline()
    await asyncio.sleep(0.01)
    svc.sessions.touch()
    assert svc.sessions.idle_deadline() > before
    assert svc.sessions.session["last_activity_at"] <= now()
    await svc.sessions.stop()
    await svc.stop()


def test_bootstrap_waits_for_the_old_workers_port() -> None:
    """A killed worker's port can outlive it in the process list (a 14B model takes seconds
    to tear down): the bootstrap starts the new worker only once the port is free."""
    with socket.socket() as old:
        old.bind(("127.0.0.1", 0))
        old.listen()
        port = old.getsockname()[1]
        code = BOOTSTRAP.replace("{port}", str(port))
        # Only the port wait: the rest kills and starts real processes.
        wait = code[code.index("def _port_free") : code.index("\nif _port_free():")]
        scope: dict[str, Any] = {}
        exec("import socket, time\n" + wait.replace("range(900)", "range(3)"), scope)
        assert not scope["_port_free"]()
        threading.Timer(0.2, old.close).start()
        exec("import socket, time\n" + wait, scope)
        assert scope["_port_free"]()
