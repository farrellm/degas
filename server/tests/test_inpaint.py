"""Phase 6: image-to-image, inpaint, outpaint, masks and SAM selection."""

import io
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageChops

from degas.families.base import SpecError
from degas.families.sdxl import Sdxl
from degas.media import _apply, _png, apply_mask_ops, invert_ops, ops_size

from .conftest import INPAINT_MODEL, MODEL, Harness
from .test_api import session_state, wait_for
from .test_video import image, results, run, upload

SPEC: dict[str, Any] = {
    "family": "sdxl",
    "variant": "base",
    "mode": "i2i",
    "model": {"path": MODEL["path"]},
    "params": {"prompt": "a lighthouse in fog", "width": 1024, "height": 1024, "seed": 3},
}


def mask_png(w: int, h: int, box: tuple[int, int, int, int], alpha: bool = False) -> bytes:
    """A painted mask: white (or opaque, for a canvas export) inside `box`."""
    if alpha:
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        im.paste((255, 255, 255, 255), box)
    else:
        im = Image.new("L", (w, h), 0)
        im.paste(255, box)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def read(client: TestClient, sha: str) -> Image.Image:
    return Image.open(io.BytesIO(client.get(f"/api/blobs/{sha}").content))


def put_mask(client: TestClient, source: str, data: bytes) -> dict[str, Any]:
    resp = client.post(f"/api/blobs/{source}/mask", content=data)
    assert resp.status_code == 201, resp.text
    body: dict[str, Any] = resp.json()
    return body


# -- descriptor ----------------------------------------------------------------------------


def test_modes_and_their_parameters() -> None:
    sdxl = Sdxl()
    props = sdxl.param_schema("base", "t2i")["properties"]
    assert "strength" not in props
    assert sdxl.param_schema("base", "i2i")["properties"]["strength"]["default"] == 0.6
    inpaint = sdxl.param_schema("inpaint", "inpaint")["properties"]
    assert inpaint["strength"]["default"] == 1.0
    assert inpaint["inpaint_area"]["enum"] == ["whole", "masked"]
    outpaint = sdxl.param_schema("base", "outpaint")["properties"]
    assert "strength" not in outpaint
    assert "blend" in outpaint
    with pytest.raises(SpecError, match="can't do 't2i'"):
        sdxl.param_schema("inpaint", "t2i")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "inpaint", "inputs": {"source": "sha256:" + "a" * 64}}, "Paint the area"),
        ({"mode": "i2i", "inputs": {}}, "Choose a source image"),
        ({"model": {"path": INPAINT_MODEL["path"]}}, "can only inpaint or outpaint"),
        (
            {"variant": "inpaint", "mode": "inpaint", "model": {"path": MODEL["path"]}},
            "isn't a SDXL inpainting model",
        ),
    ],
)
def test_validation(change: dict[str, Any], message: str) -> None:
    spec = {**SPEC, "inputs": {"source": "sha256:" + "a" * 64}, **change}
    with pytest.raises(SpecError, match=message):
        Sdxl().validate(spec)


@pytest.mark.parametrize(
    ("place", "message"),
    [
        (None, "Place the image"),
        ({"x": 0, "y": 0, "w": 32, "h": 512}, "at least 64 px"),
        ({"x": 600, "y": 0, "w": 512, "h": 512}, "inside the 1024x1024 canvas"),
        ({"x": 0, "y": 0, "w": 1024, "h": 1024}, "nothing to outpaint"),
    ],
)
def test_outpaint_placement(place: Any, message: str) -> None:
    spec = {
        **SPEC,
        "mode": "outpaint",
        "inputs": {"source": "sha256:" + "a" * 64, "place": place},
    }
    with pytest.raises(SpecError, match=message):
        Sdxl().validate(spec)
    spec["inputs"]["place"] = {"x": 256, "y": 0, "w": 512.4, "h": 1024}
    assert Sdxl().validate(spec)["inputs"]["place"] == {"x": 256, "y": 0, "w": 512, "h": 1024}


# -- mask geometry -------------------------------------------------------------------------


def test_invert_ops_returns_to_the_original_frame() -> None:
    ops = [
        {"op": "rotate", "deg": 90},
        {"op": "flip_h"},
        {"op": "crop", "x": 10, "y": 20, "w": 40, "h": 60},
        {"op": "resize", "w": 80, "h": 120, "filter": "lanczos"},
    ]
    assert ops_size(ops, 100, 200) == (80, 120)
    back = invert_ops(ops, 100, 200)
    assert ops_size(back, 80, 120) == (100, 200)
    assert back == [
        {"op": "resize", "w": 40, "h": 60, "filter": "lanczos"},
        {"op": "paste", "x": 10, "y": 20, "w": 200, "h": 100},
        {"op": "flip_h"},
        {"op": "rotate", "deg": 270},
    ]
    # Without the (lossy) resize, a mask comes back pixel for pixel where the crop came
    # from, and empty where it was cropped away.
    lossless, lossless_back = ops[:3], invert_ops(ops[:3], 100, 200)
    original = Image.new("L", (100, 200), 0)
    original.paste(255, (0, 0, 60, 30))  # the crop's footprint is x 20..80, y 10..50
    restored = Image.open(
        io.BytesIO(apply_mask_ops(_png(_apply(original, lossless)), lossless_back))
    )
    footprint = Image.open(
        io.BytesIO(apply_mask_ops(_png(Image.new("L", (40, 60), 255)), lossless_back))
    )
    assert restored.size == (100, 200)
    assert (
        ImageChops.difference(restored, ImageChops.multiply(original, footprint)).getbbox() is None
    )
    assert restored.getbbox() == (20, 10, 60, 30)


def test_masks_are_stored_at_their_image_size(client: TestClient) -> None:
    src = upload(client, image(400, 300))
    # A canvas export at a smaller working size, read from its alpha channel.
    mask = put_mask(client, src["sha256"], mask_png(200, 150, (0, 0, 100, 75), alpha=True))
    im = read(client, mask["sha256"])
    assert (im.mode, im.size) == ("L", (400, 300))
    left, top, right, bottom = im.getbbox() or (0, 0, 0, 0)
    assert (left, top) == (0, 0)
    assert abs(right - 200) <= 1
    assert abs(bottom - 150) <= 1
    resp = client.post(f"/api/blobs/{src['sha256']}/mask", content=mask_png(100, 300, (0, 0, 1, 1)))
    assert resp.status_code == 400
    assert "shape" in resp.json()["detail"]


def test_a_mask_follows_its_image_through_a_new_crop(client: TestClient) -> None:
    src = upload(client, image(400, 300))
    first = client.post(
        f"/api/blobs/{src['sha256']}/transform",
        json={"ops": [{"op": "crop", "x": 0, "y": 0, "w": 200, "h": 300}]},
    ).json()
    mask = put_mask(client, first["sha256"], mask_png(200, 300, (150, 0, 200, 50)))
    second = client.post(
        f"/api/blobs/{first['sha256']}/transform",
        json={"ops": [{"op": "crop", "x": 100, "y": 0, "w": 300, "h": 300}]},
    ).json()
    moved = client.post(
        f"/api/blobs/{mask['sha256']}/remap",
        json={"source": first["sha256"], "to": second["sha256"]},
    ).json()
    assert moved["empty"] is False
    im = read(client, moved["sha256"])
    assert im.size == (300, 300)
    assert im.getbbox() == (50, 0, 100, 50)
    # A crop that leaves the painted area out empties the mask.
    third = client.post(
        f"/api/blobs/{first['sha256']}/transform",
        json={"ops": [{"op": "crop", "x": 250, "y": 100, "w": 150, "h": 200}]},
    ).json()
    gone = client.post(
        f"/api/blobs/{mask['sha256']}/remap",
        json={"source": first["sha256"], "to": third["sha256"]},
    ).json()
    assert gone["empty"] is True
    other = upload(client, image(50, 50))
    resp = client.post(
        f"/api/blobs/{mask['sha256']}/remap",
        json={"source": first["sha256"], "to": other["sha256"]},
    )
    assert resp.status_code == 400


# -- jobs ----------------------------------------------------------------------------------


def test_inpaint_fits_the_mask_with_its_source(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(2048, 1024))
    mask = put_mask(client, src["sha256"], mask_png(2048, 1024, (0, 0, 800, 1024)))
    spec = {
        **SPEC,
        "mode": "inpaint",
        "inputs": {"source": f"sha256:{src['sha256']}", "mask": f"sha256:{mask['sha256']}"},
    }
    done = run(client, spec)
    assert harness.runner.inputs[-1] == {"source": (1024, 1024), "mask": (1024, 1024)}
    inputs = done["spec"]["inputs"]
    assert inputs["mask"] != f"sha256:{mask['sha256']}"
    assert inputs["transforms"][inputs["mask"]]["original"] == f"sha256:{mask['sha256']}"
    # Keeping the result keeps the painted mask as well as the fitted one.
    result = results(client, done["id"])[0]
    saved = client.post(f"/api/results/{result['id']}/save").json()
    assert saved["config"]["inputs"]["mask"] == inputs["mask"]


def test_an_empty_mask_is_refused(client: TestClient) -> None:
    src = upload(client, image(1024, 1024))
    mask = put_mask(client, src["sha256"], mask_png(1024, 1024, (0, 0, 0, 0)))
    spec = {
        **SPEC,
        "mode": "inpaint",
        "inputs": {"source": f"sha256:{src['sha256']}", "mask": f"sha256:{mask['sha256']}"},
    }
    resp = client.post("/api/jobs", json={"spec": spec})
    assert resp.status_code == 400
    assert "mask is empty" in resp.json()["detail"]


def test_outpaint_fits_the_source_to_its_place(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(600, 600))
    spec = {
        **SPEC,
        "variant": "inpaint",
        "mode": "outpaint",
        "model": {"path": INPAINT_MODEL["path"]},
        "params": {**SPEC["params"], "width": 1536, "height": 1024},
        "inputs": {
            "source": f"sha256:{src['sha256']}",
            "place": {"x": 256, "y": 0, "w": 1024, "h": 1024},
        },
    }
    done = run(client, spec)
    assert harness.runner.inputs[-1] == {"source": (1024, 1024)}
    assert done["spec"]["inputs"]["place"] == {"x": 256, "y": 0, "w": 1024, "h": 1024}
    # The 9-channel UNet loads with the inpainting repo's configs.
    assert done["spec"]["config"]["path"] == "configs/sdxl/stable-diffusion-xl-1.0-inpainting-0.1"


# -- SAM -----------------------------------------------------------------------------------


def test_select_needs_a_session(client: TestClient) -> None:
    src = upload(client, image(200, 100))
    resp = client.post(
        "/api/preprocess",
        json={"id": "sam", "image": src["sha256"], "params": {"points": [{"x": 5, "y": 5}]}},
    )
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Start a session to use Select"


def test_select_returns_candidate_masks(client: TestClient, harness: Harness) -> None:
    harness.worker_app.state.cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    client.post("/api/session", json={"gpu": "T4"})
    wait_for(lambda: session_state(client) == "ready")
    src = upload(client, image(200, 100))
    body = client.post(
        "/api/preprocess",
        json={
            "id": "sam",
            "image": f"sha256:{src['sha256']}",
            "params": {"points": [{"x": 50, "y": 50}, {"x": 150, "y": 50, "include": False}]},
        },
    )
    assert body.status_code == 201, body.text
    boxes = [read(client, c["sha256"]).getbbox() for c in body.json()["candidates"]]
    assert boxes == [(48, 48, 52, 52), (44, 44, 56, 56), (38, 38, 62, 62)]  # small to large
    assert body.json()["chosen"] == 1  # the best-scoring one
    assert harness.sam.calls[0]["model"].name == "sam3"
    text = client.post(
        "/api/preprocess",
        json={"id": "sam", "image": src["sha256"], "params": {"text": "  the left half "}},
    ).json()
    assert len(text["candidates"]) == 1
    assert harness.sam.calls[-1]["text"] == "the left half"
    resp = client.post(
        "/api/preprocess",
        json={
            "id": "sam",
            "image": src["sha256"],
            "params": {"points": [{"x": 1, "y": 1, "include": False}]},
        },
    )
    assert resp.status_code == 400
