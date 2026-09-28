import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from degas.blobs import BlobStore
from degas.db import Database
from degas.library import input_blobs, saved_config, sweep

from .conftest import LORA, png
from .test_api import SPEC, job, wait_for

LORA_SPEC = {**SPEC, "loras": [{"path": LORA["path"], "weight": 0.8}]}


def generate(client: TestClient, spec: dict[str, Any] = LORA_SPEC, n: int = 2) -> list[Any]:
    submitted = client.post("/api/jobs", json={"spec": spec, "batch_count": n}).json()
    client.post("/api/session", json={"gpu": "L4"})
    wait_for(lambda: job(client, submitted["id"])["status"] == "done")
    results: list[Any] = client.get(f"/api/results?job={submitted['id']}").json()["results"]
    return sorted(results, key=lambda r: r["item_index"])


def test_keep_saves_a_replayable_config(client: TestClient) -> None:
    first, second = generate(client)
    assert first["library_id"] is None

    item = client.post(f"/api/results/{second['id']}/save").json()
    config = item["config"]
    assert config["degas_version"] == 1
    assert config["model"] == {"path": SPEC["model"]["path"], "size": 1234}
    assert config["loras"] == [{"path": LORA["path"], "weight": 0.8, "size": 10}]
    assert config["params"]["prompt"] == "a lighthouse"
    assert config["params"]["seed"] == 43  # the seed of this item, not the batch's first
    assert config["runtime"]["gpu"] == "L4"
    assert config["runtime"]["duration_s"] >= 0
    assert (item["kind"], item["width"], item["height"]) == ("image", 64, 48)

    # Keeping twice returns the same item; the result now points at it.
    assert client.post(f"/api/results/{second['id']}/save").json()["id"] == item["id"]
    (kept,) = [r for r in client.get("/api/results").json()["results"] if r["library_id"]]
    assert kept["id"] == second["id"]
    assert client.post("/api/results/nope/save").status_code == 404

    # Replaying the config submits the same job.
    replay = client.post("/api/jobs", json={"spec": config}).json()
    assert replay["seeds"] == [43]
    assert replay["spec"]["loras"] == config["loras"]


def test_library_search_edit_and_delete(client: TestClient, harness: Any) -> None:
    (lighthouse,) = generate(client, SPEC, 1)
    (harbour,) = generate(client, {**SPEC, "params": {"prompt": "a 100% harbour"}}, 1)
    a = client.post(f"/api/results/{lighthouse['id']}/save").json()
    b = client.post(f"/api/results/{harbour['id']}/save").json()

    items = client.get("/api/library").json()["items"]
    assert [i["id"] for i in items] == [b["id"], a["id"]]
    assert [i["id"] for i in client.get("/api/library?q=LIGHT").json()["items"]] == [a["id"]]
    assert [i["id"] for i in client.get("/api/library", params={"q": "100%"}).json()["items"]] == [
        b["id"]
    ]
    assert client.get("/api/library?q=a%25b").json()["items"] == []

    edited = client.patch(f"/api/library/{a['id']}", json={"tags": [" sea ", "Sea", "dusk"]})
    assert edited.json()["tags"] == ["sea", "dusk"]
    assert [i["id"] for i in client.get("/api/library?q=dusk").json()["items"]] == [a["id"]]

    page = client.get("/api/library?limit=1").json()
    assert page["cursor"] is not None
    rest = client.get(f"/api/library?limit=1&cursor={page['cursor']}").json()
    assert [i["id"] for i in rest["items"]] == [a["id"]]

    assert client.delete(f"/api/library/{a['id']}").json() == {"deleted": True}
    assert client.get(f"/api/library/{a['id']}").status_code == 404
    # The result still holds its image, so the blob stays.
    assert client.get(f"/api/blobs/{lighthouse['blob_sha']}").status_code == 200


def test_clear_results_keeps_kept_images_and_queued_jobs(client: TestClient) -> None:
    first, second = generate(client)
    item = client.post(f"/api/results/{second['id']}/save").json()
    client.delete("/api/session")
    queued = client.post("/api/jobs", json={"spec": SPEC}).json()

    assert client.delete("/api/results").json() == {"results": 2, "jobs": 1}
    assert client.get("/api/results").json()["results"] == []
    assert [j["id"] for j in client.get("/api/jobs").json()] == [queued["id"]]
    assert client.get(f"/api/blobs/{first['blob_sha']}").status_code == 404
    # The kept image is still in the library, with its picture.
    assert client.get(f"/api/library/{item['id']}").status_code == 200
    assert client.get(f"/api/blobs/{second['blob_sha']}").status_code == 200
    assert client.delete("/api/results").json() == {"results": 0, "jobs": 0}


def test_saved_prompts(client: TestClient) -> None:
    long = "a lighthouse on a cliff at dusk, oil painting, thick impasto"
    saved = client.post("/api/prompts", json={"prompt": long, "negative_prompt": "blurry"})
    assert saved.status_code == 201
    p = saved.json()
    assert p["name"] == "a lighthouse on a cliff at…"
    assert client.post("/api/prompts", json={"prompt": "  "}).status_code == 400
    client.post("/api/prompts", json={"prompt": "a harbour", "name": "Harbour"})

    assert [x["name"] for x in client.get("/api/prompts").json()] == ["Harbour", p["name"]]
    assert [x["id"] for x in client.get("/api/prompts?q=impasto").json()] == [p["id"]]

    renamed = client.patch(f"/api/prompts/{p['id']}", json={"name": "Lighthouse"}).json()
    assert renamed["name"] == "Lighthouse"
    assert renamed["negative_prompt"] == "blurry"
    assert client.delete(f"/api/prompts/{p['id']}").json() == {"deleted": True}
    assert client.delete(f"/api/prompts/{p['id']}").status_code == 404


# -- retention -----------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="milliseconds")


def test_sweep_deletes_expired_results_but_not_kept_images(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite")
    blobs = BlobStore(tmp_path)
    blobs.ensure()
    past = datetime.now(UTC) - timedelta(days=3)

    job_row = db.insert_job(SPEC, [1, 2])
    db.update_job(job_row["id"], status="done", finished_at=_iso(past))
    shas = [blobs.put(png(i), "image/png") for i in (1, 2)]
    results = [
        db.insert_result(job_row["id"], i, s, "image/png", i, 64, 48) for i, s in enumerate(shas)
    ]
    for sha in shas:
        assert blobs.thumb(sha) is not None
    kept = db.get_result(results[1]["id"])
    assert kept is not None
    db.insert_library_item(kept, saved_config(job_row, kept))
    db.conn.execute("UPDATE results SET expires_at = ?", (_iso(past + timedelta(days=1)),))

    # An unreferenced blob written moments ago survives the grace period.
    fresh = blobs.put(png(9), "image/png")
    stale = blobs.put(png(8), "image/png")
    stale_path = blobs.path(stale)
    assert stale_path is not None
    os.utime(stale_path, (time.time() - 7200, time.time() - 7200))
    for sha in shas:
        path = blobs.path(sha)
        assert path is not None
        os.utime(path, (time.time() - 7200, time.time() - 7200))

    counts = sweep(db, blobs)
    assert counts == {"results": 2, "jobs": 1, "refs": 0, "blobs": 2}
    assert db.list_results() == []
    assert db.get_job(job_row["id"]) is None
    assert blobs.path(shas[0]) is None
    assert not (blobs.thumbs / f"{shas[0]}.webp").exists()
    assert blobs.path(shas[1]) is not None  # kept in the library
    assert blobs.path(fresh) is not None
    assert blobs.path(stale) is None
    db.close()


def test_sweep_keeps_running_sessions_results(tmp_path: Path) -> None:
    db = Database(tmp_path / "db.sqlite")
    blobs = BlobStore(tmp_path)
    job_row = db.insert_job({"params": {}}, [1])
    db.update_job(job_row["id"], status="done", finished_at=_iso(datetime.now(UTC)))
    db.insert_result(job_row["id"], 0, "a" * 64, "image/png", 1, 1, 1)
    assert sweep(db, blobs)["results"] == 0  # no expiry while the session runs
    db.close()


def test_input_blobs() -> None:
    spec = {
        "inputs": {
            "source": "sha256:aa",
            "mask": "sha256:bb",
            "transforms": {"sha256:aa": {"original": "sha256:cc", "ops": []}},
        },
        "control": [{"image": "sha256:dd", "preprocessor": {"id": "depth", "source": "sha256:ee"}}],
    }
    assert input_blobs(spec) == ["aa", "bb", "cc", "dd", "ee"]
