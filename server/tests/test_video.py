import base64
import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from degas.families import FAMILIES
from degas.library import staged_blobs
from degas.media import MediaError, apply_ops, fit_ops, validate_ops

from .conftest import WAN_5B, WAN_I2V, Harness, services
from .test_api import job, wait_for

WAN_SPEC: dict[str, Any] = {
    "family": "wan22",
    "variant": "ti2v-5b",
    "mode": "t2v",
    "model": {"path": WAN_5B["path"]},
    "params": {"prompt": "waves on a harbour wall", "seed": 7},
}


def image(w: int, h: int, fmt: str = "PNG", mode: str = "RGB", exif: Any = None) -> bytes:
    buf = io.BytesIO()
    im = Image.new(mode, (w, h), (200, 30, 60, 255)[: len(mode)])
    im.save(buf, format=fmt, **({"exif": exif} if exif is not None else {}))
    return buf.getvalue()


def upload(client: TestClient, data: bytes, content_type: str = "image/png") -> dict[str, Any]:
    resp = client.post("/api/blobs", content=data, headers={"Content-Type": content_type})
    assert resp.status_code == 201, resp.text
    body: dict[str, Any] = resp.json()
    return body


def run(client: TestClient, spec: dict[str, Any]) -> dict[str, Any]:
    resp = client.post("/api/jobs", json={"spec": spec})
    assert resp.status_code == 201, resp.text
    submitted: dict[str, Any] = resp.json()
    client.post("/api/session", json={"gpu": "L4"})
    return wait_for(lambda: (j := job(client, submitted["id"]))["status"] == "done" and j)  # type: ignore[no-any-return]


def results(client: TestClient, job_id: str) -> list[dict[str, Any]]:
    found = client.get(f"/api/results?job={job_id}").json()["results"]
    return sorted(found, key=lambda r: r["item_index"])


# -- transforms ----------------------------------------------------------------------------


def test_fit_ops() -> None:
    assert fit_ops(1280, 704, 1280, 704, "crop") == []
    # A portrait photo cropped to landscape: the centred band, then resized.
    assert fit_ops(2000, 3000, 1280, 704, "crop") == [
        {"op": "crop", "x": 0, "y": 950, "w": 2000, "h": 1100},
        {"op": "resize", "w": 1280, "h": 704, "filter": "lanczos"},
    ]
    assert fit_ops(640, 640, 1280, 704, "pad") == [
        {"op": "resize", "w": 704, "h": 704, "filter": "lanczos"},
        {"op": "pad", "w": 1280, "h": 704},
    ]
    assert fit_ops(10, 10, 20, 30, "stretch") == [
        {"op": "resize", "w": 20, "h": 30, "filter": "lanczos"}
    ]


def test_apply_ops() -> None:
    ops = validate_ops(
        [
            {"op": "rotate", "deg": 90},
            {"op": "flip_h"},
            {"op": "crop", "x": 1, "y": 2, "w": 20, "h": 30.4},
            {"op": "resize", "w": 40, "h": 60},
        ]
    )
    assert ops[2] == {"op": "crop", "x": 1, "y": 2, "w": 20, "h": 30}
    data, w, h = apply_ops(image(64, 32), ops)
    assert (w, h) == (40, 60)
    assert apply_ops(image(64, 32), ops)[0] == data  # deterministic, so content-addressed
    with pytest.raises(MediaError, match="outside"):
        apply_ops(image(64, 32), [{"op": "crop", "x": 50, "y": 0, "w": 20, "h": 10}])
    with pytest.raises(MediaError, match="Unknown"):
        validate_ops([{"op": "blur"}])
    with pytest.raises(MediaError, match="deg"):
        validate_ops([{"op": "rotate", "deg": 45}])


def test_uploads_are_normalized(client: TestClient) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # stored landscape, displayed rotated 90°
    photo = upload(client, image(40, 20, "JPEG", exif=exif), "image/jpeg")
    assert (photo["media_type"], photo["width"], photo["height"]) == ("image/jpeg", 20, 40)
    stored = client.get(f"/api/blobs/{photo['sha256']}").content
    with Image.open(io.BytesIO(stored)) as im:
        assert im.size == (20, 40)
        assert 0x0112 not in im.getexif()
    alpha = upload(client, image(8, 8, mode="RGBA"))
    assert alpha["media_type"] == "image/png"

    resp = client.post("/api/blobs", content=b"not an image", headers={"Content-Type": "x/y"})
    assert resp.status_code == 400
    assert "isn't an image" in resp.json()["detail"]


def test_import_from_a_link(client: TestClient) -> None:
    uri = "data:image/png;base64," + base64.b64encode(image(12, 10)).decode()
    got = client.post("/api/blobs/from-url", json={"url": uri}).json()
    assert (got["width"], got["height"]) == (12, 10)
    resp = client.post("/api/blobs/from-url", json={"url": "ftp://example.com/a.png"})
    assert resp.status_code == 400


def test_transforms_are_non_destructive(client: TestClient) -> None:
    original = upload(client, image(64, 48))["sha256"]
    ops = [{"op": "rotate", "deg": 90}, {"op": "crop", "x": 0, "y": 0, "w": 48, "h": 48}]
    derived = client.post(f"/api/blobs/{original}/transform", json={"ops": ops}).json()
    assert (derived["width"], derived["height"]) == (48, 48)
    record = client.get(f"/api/blobs/{derived['sha256']}/transform").json()
    assert record["original"] == original
    assert record["ops"][0] == {"op": "rotate", "deg": 90}

    # Editing the derived image again replaces the operations on the original.
    again = client.post(
        f"/api/blobs/{derived['sha256']}/transform",
        json={"ops": [{"op": "resize", "w": 32, "h": 24}]},
    ).json()
    assert client.get(f"/api/blobs/{again['sha256']}/transform").json()["original"] == original
    assert (again["width"], again["height"]) == (32, 24)

    assert client.get(f"/api/blobs/{original}/transform").json() == {
        "original": original,
        "ops": [],
    }
    assert client.get(f"/api/blobs/{'0' * 64}/transform").status_code == 404
    bad = client.post(f"/api/blobs/{original}/transform", json={"ops": [{"op": "crop"}]})
    assert bad.status_code == 400


# -- video jobs ----------------------------------------------------------------------------


def test_text_to_video(client: TestClient, harness: Harness) -> None:
    done = run(client, WAN_SPEC)
    assert done["spec"]["params"]["num_frames"] == 121
    (clip,) = results(client, done["id"])
    assert clip["media_type"] == "video/mp4"
    assert (clip["width"], clip["height"]) == (1280, 704)
    assert clip["duration"] == pytest.approx(5 / 24, abs=0.05)
    poster = client.get(f"/api/thumbs/{clip['blob_sha']}")
    assert poster.headers["content-type"] == "image/webp"
    video = client.get(f"/api/blobs/{clip['blob_sha']}", headers={"Range": "bytes=0-99"})
    assert video.status_code == 206  # iOS plays video only with Range support

    frame = client.post(f"/api/blobs/{clip['blob_sha']}/frame", json={"at": "last"}).json()
    assert (frame["media_type"], frame["width"], frame["height"]) == ("image/png", 1280, 704)
    assert client.post(f"/api/blobs/{frame['sha256']}/frame", json={}).status_code == 400

    item = client.post(f"/api/results/{clip['id']}/save").json()
    assert (item["kind"], item["duration"]) == ("video", clip["duration"])


def test_image_to_video_fits_the_source(client: TestClient, harness: Harness) -> None:
    source = upload(client, image(600, 900))  # portrait, for a landscape clip
    spec = {**WAN_SPEC, "mode": "i2v", "inputs": {"source": f"sha256:{source['sha256']}"}}
    done = run(client, spec)
    inputs = done["spec"]["inputs"]
    assert "fit" not in inputs
    assert inputs["source"] != f"sha256:{source['sha256']}"
    transform = inputs["transforms"][inputs["source"]]
    assert transform["original"] == f"sha256:{source['sha256']}"
    assert [op["op"] for op in transform["ops"]] == ["crop", "resize"]
    assert harness.wan.sources == [(1280, 704)]

    gone = {**spec, "inputs": {"source": "sha256:" + "0" * 64}}
    resp = client.post("/api/jobs", json={"spec": gone})
    assert resp.status_code == 400
    assert "no longer stored" in resp.json()["detail"]


def test_first_and_last_frame(client: TestClient, harness: Harness) -> None:
    first = upload(client, image(600, 900))
    last = upload(client, image(1280, 720))
    spec = {
        **WAN_SPEC,
        "variant": "i2v-a14b",
        "mode": "flf2v",
        "model": {"path": WAN_I2V["path"]},
        "inputs": {"source": f"sha256:{first['sha256']}", "end": f"sha256:{last['sha256']}"},
    }
    done = run(client, spec)
    inputs = done["spec"]["inputs"]
    assert inputs["source"] in inputs["transforms"]
    assert inputs["end"] == f"sha256:{last['sha256']}"
    assert (harness.wan.sources, harness.wan.ends) == ([(1280, 720)], [(1280, 720)])
    assert last["sha256"] in staged_blobs(done["spec"])


def test_extend_a_clip_and_stitch_the_chain(client: TestClient, harness: Harness) -> None:
    first = run(client, WAN_SPEC)
    (clip,) = results(client, first["id"])

    extension = client.post(f"/api/results/{clip['id']}/extend").json()
    spec = extension["spec"]
    assert (spec["variant"], spec["mode"]) == ("ti2v-5b", "i2v")
    assert spec["inputs"]["extends"] == f"sha256:{clip['blob_sha']}"
    assert spec["inputs"]["source"] == f"sha256:{extension['source']['sha256']}"
    assert spec["params"]["seed"] == -1
    assert spec["params"]["prompt"] == WAN_SPEC["params"]["prompt"]

    second = run(client, spec)
    continuation, chain = results(client, second["id"])
    assert continuation["segments"] is None
    assert chain["item_index"] == 1
    assert [s["mode"] for s in chain["segments"]] == ["t2v", "i2v"]
    # The shared boundary frame appears once: 5 + (5 - 1) frames.
    assert chain["duration"] == pytest.approx(9 / 24, abs=0.05)

    kept = client.post(f"/api/results/{chain['id']}/save").json()
    assert len(kept["config"]["segments"]) == 2
    assert kept["config"]["inputs"]["extends"] == spec["inputs"]["extends"]

    # A14B clips continue with the I2V model from the index.
    a14b = {**spec, "variant": "t2v-a14b", "model": {"path": "models/wan22/t2v-a14b/x"}}
    svc = services(client)
    ext = client.portal.call(svc.inputs.extend, FAMILIES["wan22"], clip["blob_sha"], a14b)  # type: ignore[attr-defined]
    assert ext["spec"]["variant"] == "i2v-a14b"
    assert ext["spec"]["model"]["path"].startswith("models/wan22/i2v-a14b/")
    assert "cfg_low" not in ext["spec"]["params"]  # carried over only if the source had it

    image_only = client.post("/api/results/nope/extend")
    assert image_only.status_code == 404
