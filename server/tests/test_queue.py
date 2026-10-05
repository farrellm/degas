"""Phase 5: queue reordering, undoing a cancel, and Web Push."""

import base64
import threading
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from degas.notices import idle_notice, job_notice, ready_notice
from degas.push import application_server_key, load_vapid

from .conftest import FastIntervals
from .test_api import SPEC, job, session_state, wait_for
from .test_session import until

SUB = {
    "endpoint": "https://push.example/sub/1",
    "keys": {"p256dh": "BPkey", "auth": "authsecret"},
}


def queue(client: TestClient) -> list[str]:
    jobs = [j for j in client.get("/api/jobs").json() if j["status"] == "queued"]
    return [j["id"] for j in sorted(jobs, key=lambda j: j["queue_position"])]


def submit(client: TestClient, prompt: str, n: int = 1) -> str:
    spec = {**SPEC, "params": {**SPEC["params"], "prompt": prompt}}
    body = client.post("/api/jobs", json={"spec": spec, "batch_count": n}).json()
    return str(body["id"])


def test_reorder_the_queue(client: TestClient) -> None:
    a, b, c = (submit(client, p) for p in ("a", "b", "c"))
    assert queue(client) == [a, b, c]

    moved = client.patch(f"/api/jobs/{c}", json={"position": 0}).json()
    assert moved["id"] == c
    assert queue(client) == [c, a, b]
    client.patch(f"/api/jobs/{c}", json={"position": 99})  # past the end: last
    assert queue(client) == [a, b, c]

    # A new job still goes to the back of the queue.
    d = submit(client, "d")
    assert queue(client) == [a, b, c, d]

    assert client.patch("/api/jobs/nope", json={"position": 0}).status_code == 404
    assert client.patch(f"/api/jobs/{a}", json={"position": -1}).status_code == 422
    client.delete(f"/api/jobs/{a}")
    assert client.patch(f"/api/jobs/{a}", json={"position": 0}).status_code == 409


def test_undo_cancel_restores_the_queue_position(client: TestClient) -> None:
    a, b, c = (submit(client, p) for p in ("a", "b", "c"))
    client.delete(f"/api/jobs/{b}")
    assert queue(client) == [a, c]

    restored = client.post(f"/api/jobs/{b}/restore").json()
    assert restored["status"] == "queued"
    assert restored["finished_at"] is None
    assert queue(client) == [a, b, c]
    assert client.post(f"/api/jobs/{b}/restore").status_code == 409  # not cancelled
    assert client.post("/api/jobs/nope/restore").status_code == 404


def test_restored_job_runs(client: TestClient) -> None:
    a = submit(client, "a")
    client.delete(f"/api/jobs/{a}")
    client.post(f"/api/jobs/{a}/restore")
    client.post("/api/session", json={"gpu": "L4"})
    wait_for(lambda: job(client, a)["status"] == "done")
    # A job that ran can't be un-cancelled.
    assert client.post(f"/api/jobs/{a}/restore").status_code == 409


def test_push_key_and_subscriptions(client: TestClient) -> None:
    key = client.get("/api/push").json()["public_key"]
    raw = base64.urlsafe_b64decode(key + "=" * (-len(key) % 4))
    assert len(raw) == 65  # uncompressed P-256 point
    assert raw[0] == 4
    assert client.get("/api/push").json()["public_key"] == key  # stable

    assert client.post("/api/push/subscribe", json=SUB).status_code == 201
    assert client.post("/api/push/subscribe", json=SUB).status_code == 201  # idempotent
    db = client.app.state.services.db  # type: ignore[attr-defined]
    (stored,) = db.list_push_subscriptions()
    assert stored["keys"] == SUB["keys"]

    insecure = {**SUB, "endpoint": "http://push.example/x"}
    assert client.post("/api/push/subscribe", json=insecure).status_code == 422
    unsub = client.post("/api/push/unsubscribe", json={"endpoint": SUB["endpoint"]})
    assert unsub.json() == {"unsubscribed": True}
    assert db.list_push_subscriptions() == []


def test_finished_job_sends_a_notification(client: TestClient, harness: Any) -> None:
    client.post("/api/push/subscribe", json=SUB)
    first = submit(client, "a lighthouse at dusk", n=2)
    client.post("/api/session", json={"gpu": "L4"})
    wait_for(lambda: job(client, first)["status"] == "done")
    endpoint, payload = wait_for(
        lambda: next((p for p in harness.pushed if p[1]["tag"] == first), None)
    )
    assert endpoint == SUB["endpoint"]
    assert payload == {
        "title": "2 images finished",
        "body": "a lighthouse at dusk",
        "tag": first,
        "url": "/?tab=results",
    }

    # The push service says the subscription is gone: it's forgotten.
    harness.push_status = 410
    second = submit(client, "b")
    wait_for(lambda: job(client, second)["status"] == "done")
    db = client.app.state.services.db  # type: ignore[attr-defined]
    wait_for(lambda: db.list_push_subscriptions() == [])


def test_cancelled_job_sends_nothing(client: TestClient, harness: Any) -> None:
    client.post("/api/push/subscribe", json=SUB)
    a = submit(client, "a")
    client.delete(f"/api/jobs/{a}")
    assert harness.pushed == []


async def test_idle_warning_is_pushed_once(harness: Any) -> None:
    config = harness.config.model_copy(update={"idle_timeout_min": 0.01})  # 600 ms
    svc = harness.build(config)
    svc.db.add_push_subscription(SUB["endpoint"], SUB["keys"])
    await svc.start()
    try:
        await svc.sessions.start("L4", high_mem=False)
        idle = lambda: [p for _, p in harness.pushed if p["tag"] == "idle"]  # noqa: E731
        await until(idle)
        assert idle()[0]["title"] == "L4 stops in 0:10 if no job runs."
        assert idle()[0]["url"] == "/?sheet=session"
        await until(lambda: svc.sessions.state == "stopped")
        assert len(idle()) == 1
    finally:
        await svc.stop()


async def test_ready_is_pushed_on_launch_and_worker_reset(harness: Any) -> None:
    svc = harness.build()
    svc.db.add_push_subscription(SUB["endpoint"], SUB["keys"])
    await svc.start()
    ready = {"title": "L4 is ready.", "body": "", "tag": "session", "url": "/?sheet=session"}
    try:
        await svc.sessions.start("L4", high_mem=False)
        await until(lambda: harness.pushed)
        assert [p for _, p in harness.pushed] == [ready]
        await svc.sessions.reset_worker()
        await until(lambda: len(harness.pushed) == 2)
        assert harness.pushed[1][1] == ready
    finally:
        await svc.stop()


def test_notice_wording() -> None:
    assert idle_notice("L4", timedelta(seconds=115)) == "L4 stops in 2:00 if no job runs."
    assert idle_notice("A100", timedelta(seconds=61)) == "A100 stops in 1:10 if no job runs."
    assert ready_notice("A100") == "A100 is ready."
    clip = {"status": "done", "seeds": [1], "spec": {"family": "wan22", "params": {}}}
    assert job_notice(clip) == ("Clip finished", "")
    failed = {"status": "error", "error": "CUDA out of memory", "seeds": [1], "spec": {}}
    assert job_notice(failed) == ("Job failed", "CUDA out of memory")
    long = {"status": "done", "seeds": [1, 2], "spec": {"params": {"prompt": "x" * 300}}}
    title, body = job_notice(long) or ("", "")
    assert title == "2 images finished"
    assert len(body) == 140
    assert body.endswith("…")
    assert job_notice({"status": "cancelled", "seeds": [1], "spec": {}}) is None


def test_vapid_key_is_generated_once(tmp_path: Path) -> None:
    path = tmp_path / "keys" / "vapid.pem"
    key = application_server_key(load_vapid(path))
    assert path.stat().st_mode & 0o777 == 0o600
    assert application_server_key(load_vapid(path)) == key


def results(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    return list(client.get(f"/api/results?job={job_id}").json()["results"])


def test_a_dropped_tunnel_fails_no_jobs(client: TestClient, harness: Any) -> None:
    """The SSH tunnel dies mid-job while the worker carries on: the running job is followed
    again once the session reconnects, and the queued ones still run."""
    sessions = client.app.state.services.sessions  # type: ignore[attr-defined]
    harness.runner.gate = threading.Event()
    client.post("/api/session", json={"gpu": "L4"})
    a, b = submit(client, "a", n=2), submit(client, "b")
    wait_for(lambda: job(client, a)["status"] == "running")
    sessions.iv = SlowHealth()  # the stream breaks before the health check notices
    harness.tunnels[0].dropped = True
    harness.runner.gate.set()
    wait_for(lambda: job(client, b)["status"] != "queued")
    assert len(harness.tunnels) == 2  # reconnected
    wait_for(lambda: job(client, b)["status"] == "done")
    assert job(client, a)["status"] == "done"
    assert len(results(client, a)) == 2


def test_a_job_waits_for_the_tunnel_to_start(client: TestClient, harness: Any) -> None:
    sessions = client.app.state.services.sessions  # type: ignore[attr-defined]
    client.post("/api/session", json={"gpu": "L4"})
    wait_for(lambda: session_state(client) == "ready")
    sessions.iv = NoHealth()
    harness.runner.gate = threading.Event()
    harness.tunnels[0].dropped = True
    a = submit(client, "a")
    time.sleep(0.3)
    assert job(client, a)["status"] == "running"  # waiting, not failed
    harness.tunnels[0].dropped = False  # the connection came back by itself
    harness.runner.gate.set()
    wait_for(lambda: job(client, a)["status"] == "done")
    assert len(results(client, a)) == 1
    assert len(harness.tunnels) == 1


class NoHealth(FastIntervals):
    health = 3600


class SlowHealth(FastIntervals):
    health = 0.3
