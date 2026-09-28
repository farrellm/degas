import time
from collections.abc import AsyncIterator, Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas import __version__
from degas.colab.worker_client import WorkerClient

from .conftest import CONFIGS, LORA, MODEL, VAE


def wait_for(fn: Callable[[], Any], timeout: float = 5) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("condition not reached")


def session_state(client: TestClient) -> str | None:
    session = client.get("/api/session").json()["session"]
    return session["state"] if session else None


def job(client: TestClient, job_id: str) -> dict[str, Any]:
    return next(j for j in client.get("/api/jobs").json() if j["id"] == job_id)


SPEC = {
    "family": "sdxl",
    "variant": "base",
    "mode": "t2i",
    "model": {"path": "models/sdxl/juggernaut.safetensors"},
    "params": {"prompt": "a lighthouse", "seed": 42},
}


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"status": "ok", "version": __version__}


def test_families_and_schema(client: TestClient) -> None:
    families = client.get("/api/families").json()
    assert [f["id"] for f in families] == ["sdxl", "wan22"]
    assert families[0]["variants"][0]["modes"] == ["t2i", "i2i", "inpaint", "outpaint"]
    wan = {v["id"]: v for v in families[1]["variants"]}
    assert wan["ti2v-5b"]["modes"] == ["t2v", "i2v"]
    assert wan["ti2v-5b"]["lora_format"] == "single"
    assert wan["i2v-a14b"]["lora_format"] == "paired_hi_lo"
    assert wan["i2v-a14b"]["model_dir"] == "models/wan22/i2v-a14b"
    schema = client.get("/api/families/sdxl/schema?variant=base&mode=t2i").json()
    assert schema["properties"]["steps"]["default"] == 30
    resp = client.get("/api/families/sdxl/schema?variant=inpaint&mode=t2i")
    assert resp.status_code == 400


def test_assets(client: TestClient) -> None:
    assets = client.get("/api/assets?family=sdxl&kind=model").json()
    assert sorted(a["path"] for a in assets) == [
        "models/sdxl/inpaint/sdxl-inpaint.safetensors",
        "models/sdxl/juggernaut.safetensors",
    ]
    (lora,) = client.get("/api/assets?kind=lora").json()
    assert lora["sidecar"]["trigger_words"] == ["filmgrain"]


def test_job_validation(client: TestClient) -> None:
    bad_model = {**SPEC, "model": {"path": "models/sdxl/missing.safetensors"}}
    assert client.post("/api/jobs", json={"spec": bad_model}).status_code == 400
    no_prompt = {**SPEC, "params": {}}
    resp = client.post("/api/jobs", json={"spec": no_prompt})
    assert resp.status_code == 400
    assert "prompt" in resp.json()["detail"]
    unknown = {**SPEC, "params": {"prompt": "x", "bogus": 1}}
    assert client.post("/api/jobs", json={"spec": unknown}).status_code == 400


def test_jobs_wait_for_a_session_and_can_be_cancelled(client: TestClient) -> None:
    body = client.post("/api/jobs", json={"spec": SPEC, "batch_count": 3}).json()
    assert body["status"] == "queued"
    assert body["seeds"] == [42, 43, 44]
    assert body["spec"]["params"]["steps"] == 30  # default filled
    assert body["spec"]["model"]["size"] == 1234  # from the Drive index
    time.sleep(0.1)
    assert job(client, body["id"])["status"] == "queued"
    assert client.delete(f"/api/jobs/{body['id']}").json() == {"cancelled": True}
    assert job(client, body["id"])["status"] == "cancelled"


def test_generate_end_to_end(client: TestClient, harness: Any) -> None:
    submitted = client.post("/api/jobs", json={"spec": SPEC, "batch_count": 2}).json()

    resp = client.post("/api/session", json={"gpu": "L4", "high_mem": False})
    assert resp.status_code == 202
    assert client.post("/api/session", json={"gpu": "L4"}).status_code == 409
    wait_for(lambda: job(client, submitted["id"])["status"] == "done")

    results = client.get(f"/api/results?job={submitted['id']}").json()["results"]
    assert sorted((r["item_index"], r["seed"]) for r in results) == [(0, 42), (1, 43)]
    assert all((r["width"], r["height"]) == (64, 48) for r in results)
    assert results[0]["spec"]["params"]["prompt"] == "a lighthouse"
    blob = client.get(f"/api/blobs/{results[0]['blob_sha']}")
    assert blob.headers["content-type"] == "image/png"
    thumb = client.get(f"/api/thumbs/{results[0]['blob_sha']}")
    assert thumb.headers["content-type"] == "image/webp"
    # Outputs were acknowledged on the worker.
    assert not list(harness.worker_paths.outputs.glob("*/*"))

    snapshot = client.get("/api/session").json()
    assert snapshot["session"]["state"] == "ready"
    assert snapshot["worker"]["status"] == "ok"
    assert snapshot["idle_deadline"] is not None
    assert "Drive is not authorized" in snapshot["drive"]["push_error"]
    assert harness.colab.calls[:2] == ["new L4 False", "exec"]
    tunnel = harness.tunnels[0]
    assert tunnel.uploads == ["/content/degas/bundle.tar"]

    client.delete("/api/session")
    assert session_state(client) == "stopped"
    assert harness.colab.calls[-1] == "stop"
    assert client.get("/api/session").json()["session"]["ended_at"] is not None


def test_job_error_is_reported(client: TestClient, harness: Any) -> None:
    harness.runner.fail = "CUDA out of memory"
    client.post("/api/session", json={"gpu": "T4"})
    submitted = client.post("/api/jobs", json={"spec": SPEC}).json()
    done = wait_for(lambda: (j := job(client, submitted["id"]))["status"] == "error" and j)
    assert done["error"] == "CUDA out of memory"
    wait_for(lambda: session_state(client) == "ready")


def test_bootstrap_failure_stops_the_vm(client: TestClient, harness: Any) -> None:
    harness.colab.exec_output = "Traceback: boom"
    client.post("/api/session", json={"gpu": "T4"})
    wait_for(lambda: session_state(client) == "error")
    snapshot = client.get("/api/session").json()
    assert "Worker did not start" in snapshot["session"]["error"]
    assert harness.colab.calls[-1] == "stop"


def test_vm_reclaimed(client: TestClient, harness: Any) -> None:
    client.post("/api/session", json={"gpu": "T4"})
    wait_for(lambda: session_state(client) == "ready")
    harness.colab.alive = False
    wait_for(lambda: session_state(client) == "error")
    assert "reclaimed" in client.get("/api/session").json()["session"]["error"]


def test_loras_are_checked_against_the_index(client: TestClient) -> None:
    missing = {**SPEC, "loras": [{"path": "loras/sdxl/gone.safetensors", "weight": 1}]}
    resp = client.post("/api/jobs", json={"spec": missing})
    assert resp.status_code == 400
    assert "LoRA loras/sdxl/gone.safetensors" in resp.json()["detail"]
    model_as_lora = {**SPEC, "loras": [{"path": MODEL["path"], "weight": 1}]}
    assert client.post("/api/jobs", json={"spec": model_as_lora}).status_code == 400

    ok = {**SPEC, "loras": [{"path": LORA["path"], "weight": 5}]}
    body = client.post("/api/jobs", json={"spec": ok}).json()
    assert body["spec"]["loras"] == [{"path": LORA["path"], "weight": 2.0, "size": 10}]


def test_sdxl_uses_the_fp16_fix_vae_unless_asked(client: TestClient) -> None:
    body = client.post("/api/jobs", json={"spec": SPEC}).json()
    assert body["spec"]["vae"] == {"path": VAE["path"], "size": 10}
    assert body["spec"]["config"] == {"path": CONFIGS[0]["path"], "size": 10}
    assert body["spec"]["params"]["vae_fp32"] is False

    fp32 = {**SPEC, "params": {**SPEC["params"], "vae_fp32": True}}
    body = client.post("/api/jobs", json={"spec": fp32}).json()
    assert "vae" not in body["spec"]
    bad = {**SPEC, "params": {**SPEC["params"], "vae_fp32": 1}}
    assert client.post("/api/jobs", json={"spec": bad}).status_code == 400

    client.app.state.services.db.replace_assets([MODEL, *CONFIGS])  # type: ignore[attr-defined]
    resp = client.post("/api/jobs", json={"spec": SPEC})
    assert resp.status_code == 400
    assert "fp16-fix VAE isn't in Drive" in resp.json()["detail"]
    assert client.post("/api/jobs", json={"spec": fp32}).status_code == 201

    client.app.state.services.db.replace_assets([MODEL, VAE])  # type: ignore[attr-defined]
    resp = client.post("/api/jobs", json={"spec": SPEC})
    assert resp.status_code == 400
    assert (
        f"configs aren't in Drive. Put them in degas/{CONFIGS[0]['path']}/" in resp.json()["detail"]
    )


def test_next_jobs_assets_are_prefetched(
    client: TestClient, harness: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[list[dict[str, Any]]] = []
    real_fetch = WorkerClient.fetch_assets

    def spy(self: WorkerClient, assets: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
        fetched.append(assets)
        return real_fetch(self, assets)

    monkeypatch.setattr(WorkerClient, "fetch_assets", spy)
    svc = client.app.state.services  # type: ignore[attr-defined]
    other = {**MODEL, "path": "models/sdxl/other.safetensors", "drive_file_id": "f2", "size": 10}
    svc.db.replace_assets([{**MODEL, "size": 10}, other, LORA, VAE, *CONFIGS])
    harness.worker_app.state.cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    harness.runner.fetch = True

    client.post("/api/jobs", json={"spec": SPEC})
    second_spec = {
        **SPEC,
        "model": {"path": other["path"]},
        "loras": [{"path": LORA["path"], "weight": 0.8}],
    }
    second = client.post("/api/jobs", json={"spec": second_spec}).json()
    client.post("/api/session", json={"gpu": "T4"})
    wait_for(lambda: job(client, second["id"])["status"] == "done")

    # Once the first job was loading, the second job's model and LoRA were requested.
    # (The VAE and configs are shared with the first job, so may already be on the VM.)
    shared = {VAE["path"], CONFIGS[0]["path"]}
    assert [a for a in fetched[0] if a["path"] not in shared] == [
        {"path": other["path"], "size": 10},
        {"path": LORA["path"], "size": 10},
    ]
    cache = wait_for(lambda: client.get("/api/session").json()["worker"]["cache"]["files"])
    assert {f["path"] for f in cache} == {MODEL["path"], other["path"], LORA["path"], *shared}
