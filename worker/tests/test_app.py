import hashlib
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas_worker import __version__
from degas_worker.app import create_app
from degas_worker.families.base import FamilyRunner, Output, RunContext
from degas_worker.paths import Paths


class FakeRunner:
    """Emits a few denoise steps per seed; blocks on `gate` when set (for cancel tests)."""

    gate: threading.Event | None = None
    fail: str | None = None

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        for item, seed in enumerate(seeds):
            for step in range(3):
                if self.gate is not None:
                    self.gate.wait(5)
                ctx.check_cancelled()
                ctx.progress(item, "denoise", step + 1, 3)
            if self.fail:
                raise RuntimeError(self.fail)
            yield Output(item, seed, f"img-{seed}".encode(), "image/png", "png")

    def unload(self) -> None:
        pass


@pytest.fixture
def paths(tmp_path: Path) -> Paths:
    return Paths(home=tmp_path / "degas", models=tmp_path / "models")


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def client(paths: Paths, runner: FakeRunner) -> Iterator[TestClient]:
    def factory() -> FamilyRunner:
        return runner

    with TestClient(create_app(paths, {"fake": factory})) as c:
        yield c


def events(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    with client.stream("GET", f"/jobs/{job_id}/events") as resp:
        assert resp.status_code == 200
        return [json.loads(line[5:]) for line in resp.iter_lines() if line.startswith("data:")]


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["drive_token"] is False
    assert body["loaded"] is None
    assert body["cache"]["files"] == []


def test_job_lifecycle(client: TestClient) -> None:
    resp = client.post("/jobs", json={"job_id": "j1", "spec": {"family": "fake"}, "seeds": [7, 8]})
    assert resp.status_code == 202
    evs = events(client, "j1")
    kinds = [e["t"] for e in evs]
    assert kinds.count("progress") == 6
    assert [e["seed"] for e in evs if e["t"] == "output"] == [7, 8]
    assert kinds[-1] == "done"
    assert all(e["job"] == "j1" for e in evs)

    state = client.get("/state").json()
    assert state["job"]["status"] == "done"
    assert [(o["item"], o["seed"]) for o in state["outputs"]] == [(0, 7), (1, 8)]

    resp = client.get("/outputs/j1/1")
    assert resp.content == b"img-8"
    assert resp.headers["content-type"] == "image/png"
    assert client.delete("/outputs/j1/1").status_code == 204
    assert client.get("/outputs/j1/1").status_code == 404
    assert len(client.get("/state").json()["outputs"]) == 1

    # Events can be replayed after the job has finished.
    assert events(client, "j1")[-1]["t"] == "done"


def test_job_error(client: TestClient, runner: FakeRunner) -> None:
    runner.fail = "CUDA out of memory"
    client.post("/jobs", json={"job_id": "j1", "spec": {"family": "fake"}, "seeds": [1]})
    last = events(client, "j1")[-1]
    assert last["t"] == "error"
    assert last["message"] == "CUDA out of memory"


def test_cancel_and_busy(client: TestClient, runner: FakeRunner) -> None:
    runner.gate = threading.Event()
    client.post("/jobs", json={"job_id": "j1", "spec": {"family": "fake"}, "seeds": [1]})
    resp = client.post("/jobs", json={"job_id": "j2", "spec": {"family": "fake"}, "seeds": [1]})
    assert resp.status_code == 409
    assert client.post("/jobs/j1/cancel").json() == {"cancelled": True}
    runner.gate.set()
    assert events(client, "j1")[-1]["t"] == "cancelled"
    assert client.get("/state").json()["job"]["status"] == "cancelled"


def test_unknown_family(client: TestClient) -> None:
    resp = client.post("/jobs", json={"job_id": "j1", "spec": {"family": "nope"}, "seeds": [1]})
    assert resp.status_code == 400


def test_blobs(client: TestClient) -> None:
    data = b"pixels"
    sha = hashlib.sha256(data).hexdigest()
    assert client.head(f"/blobs/{sha}").status_code == 404
    assert client.put(f"/blobs/{sha}", content=b"other").status_code == 400
    assert client.put(f"/blobs/{sha}", content=data).status_code == 204
    assert client.head(f"/blobs/{sha}").status_code == 200
    assert client.head("/blobs/not-a-sha").status_code == 400


def test_drive_token_writes_rclone_config(client: TestClient, paths: Paths) -> None:
    body = {"access_token": "ya29.x", "expiry": "2026-09-27T12:00:00Z", "root": "degas"}
    assert client.post("/drive-token", json=body).status_code == 204
    conf = paths.rclone_conf.read_text()
    assert "type = drive" in conf
    assert '"access_token": "ya29.x"' in conf
    assert paths.rclone_conf.stat().st_mode & 0o077 == 0
    assert client.get("/health").json()["drive_token"] is True


class FakePre:
    def __init__(self) -> None:
        self.calls: list[tuple[Path | None, Path, dict[str, Any]]] = []
        self.unloads = 0

    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((model, image, params))
        return {"candidates": [], "chosen": 0}

    def unload(self) -> None:
        self.unloads += 1


def test_preprocess(paths: Paths, tmp_path: Path) -> None:
    pre, edges = FakePre(), FakePre()
    rclone = tmp_path / "rclone"
    rclone.write_text('#!/bin/sh\nmkdir -p "$(dirname "$3")"\nprintf 0123456789 > "$3"\n')
    rclone.chmod(0o755)
    app = create_app(
        paths, {}, rclone=str(rclone), preprocessors={"sam": lambda: pre, "canny": lambda: edges}
    )
    with TestClient(app) as c:
        data = b"image bytes"
        sha = hashlib.sha256(data).hexdigest()
        body = {"id": "sam", "image": f"sha256:{sha}", "asset": {"path": "preprocessors/sam3"}}
        assert c.post("/preprocess", json=body).status_code == 400  # not staged
        c.put(f"/blobs/{sha}", content=data)
        assert c.post("/preprocess", json=body).status_code == 503  # no Drive token
        c.post("/drive-token", json={"access_token": "t", "expiry": "2099-01-01T00:00:00Z"})
        resp = c.post("/preprocess", json={**body, "params": {"text": "a cat"}})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"candidates": [], "chosen": 0}
        model, image, params = pre.calls[0]
        assert model == paths.models / "preprocessors/sam3"
        assert image.name == sha
        assert params == {"text": "a cat"}
        assert c.post("/preprocess", json={**body, "id": "depth"}).status_code == 400
        # Edges need no model; loading them unloads SAM, so one preprocessor is resident.
        edges_body = {"id": "canny", "image": f"sha256:{sha}", "params": {"low": 10}}
        assert c.post("/preprocess", json=edges_body).status_code == 200
        assert edges.calls[0] == (None, image, {"low": 10})
        assert pre.unloads == 1
