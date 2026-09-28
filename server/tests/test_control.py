"""Phase 7: ControlNet units, their areas, and the depth, pose and edges traces."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, spec_assets
from degas.families.sdxl import Sdxl

from .conftest import CONFIGS, CONTROLNET, MODEL, VAE, Harness
from .test_api import session_state, wait_for
from .test_inpaint import mask_png, put_mask, read
from .test_video import image, results, run, upload

SHA = "sha256:" + "a" * 64
SPEC: dict[str, Any] = {
    "family": "sdxl",
    "variant": "base",
    "mode": "t2i",
    "model": {"path": MODEL["path"]},
    "params": {"prompt": "a dancer on a stage", "width": 1024, "height": 1024, "seed": 3},
}


def unit(**change: Any) -> dict[str, Any]:
    return {"controlnet": {"path": CONTROLNET["path"]}, "image": SHA, **change}


def start_session(client: TestClient, harness: Harness) -> None:
    harness.worker_app.state.cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    client.post("/api/session", json={"gpu": "T4"})
    wait_for(lambda: session_state(client) == "ready")


# -- descriptor ----------------------------------------------------------------------------


def test_units_are_normalized() -> None:
    spec = Sdxl().validate(
        {
            **SPEC,
            "control": [
                unit(
                    scale=5,
                    end=0.8,
                    mask=SHA,
                    preprocessor={"id": "depth", "source": SHA},
                )
            ],
        }
    )
    assert spec["control"] == [
        {
            "controlnet": {"path": CONTROLNET["path"], "size": None},
            "image": SHA,
            "fit": "crop",
            "scale": 2,  # clamped
            "start": 0.0,
            "end": 0.8,
            "mask": SHA,
            "preprocessor": {"id": "depth", "source": SHA, "params": {}},
        }
    ]
    assert spec_assets(spec)[-1] == {"path": CONTROLNET["path"], "size": None, "kind": "controlnet"}


@pytest.mark.parametrize(
    ("control", "message"),
    [
        ([unit(controlnet=None)], "choose a model"),
        ([unit(image=None)], "choose a control image"),
        ([unit(controlnet={"path": "controlnets/flux/x.safetensors"})], "isn't a ControlNet for"),
        ([unit(), unit()], "used twice"),
        ([unit(start=0.5, end=0.5)], "start before they end"),
        ([unit(preprocessor={"id": "sam", "source": SHA})], "preprocessor: must be one of"),
        (
            [unit(controlnet={"path": f"controlnets/sdxl/{n}"}) for n in range(4)],
            "At most 3",
        ),
    ],
)
def test_unit_validation(control: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Sdxl().validate({**SPEC, "control": control})


# -- jobs ----------------------------------------------------------------------------------


def test_control_images_and_areas_are_fitted(client: TestClient, harness: Harness) -> None:
    photo = upload(client, image(1600, 1200))
    trace = upload(client, image(1600, 1200))
    area = put_mask(client, trace["sha256"], mask_png(1600, 1200, (0, 0, 800, 1200)))
    control = unit(
        image=f"sha256:{trace['sha256']}",
        mask=f"sha256:{area['sha256']}",
        preprocessor={"id": "canny", "source": f"sha256:{photo['sha256']}", "params": {"low": 50}},
        scale=0.6,
    )
    done = run(client, {**SPEC, "control": [control]})
    assert harness.runner.inputs[-1] == {"control": [(1024, 1024)], "areas": [(1024, 1024)]}
    fitted = done["spec"]["control"][0]
    assert fitted["controlnet"]["size"] == CONTROLNET["size"]
    assert "fit" not in fitted
    transforms = done["spec"]["inputs"]["transforms"]
    assert transforms[fitted["image"]]["original"] == f"sha256:{trace['sha256']}"
    assert transforms[fitted["mask"]]["original"] == f"sha256:{area['sha256']}"
    # The area stays on the left half after the centre crop and resize.
    left, _, right, _ = read(client, fitted["mask"].removeprefix("sha256:")).getbbox() or (0,) * 4
    assert left == 0
    assert 490 < right < 530
    # Keeping the result keeps the photo the trace was made from.
    result = results(client, done["id"])[0]
    saved = client.post(f"/api/results/{result['id']}/save").json()
    assert saved["config"]["control"][0]["preprocessor"]["source"] == f"sha256:{photo['sha256']}"


def test_control_is_checked_before_queueing(client: TestClient) -> None:
    trace = upload(client, image(1024, 1024))
    ref = f"sha256:{trace['sha256']}"
    resp = client.post(
        "/api/jobs",
        json={"spec": {**SPEC, "control": [unit(image=ref, controlnet="controlnets/sdxl/gone")]}},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "ControlNet controlnets/sdxl/gone is not in the Drive index"
    empty = put_mask(client, trace["sha256"], mask_png(1024, 1024, (0, 0, 0, 0)))
    resp = client.post(
        "/api/jobs",
        json={"spec": {**SPEC, "control": [unit(image=ref, mask=f"sha256:{empty['sha256']}")]}},
    )
    assert resp.status_code == 400
    assert "area is empty" in resp.json()["detail"]
    # An area must have been painted over the control image (or its source photo).
    other = put_mask(
        client, upload(client, image(512, 512))["sha256"], mask_png(512, 512, (0, 0, 9, 9))
    )
    resp = client.post(
        "/api/jobs",
        json={"spec": {**SPEC, "control": [unit(image=ref, mask=f"sha256:{other['sha256']}")]}},
    )
    assert resp.status_code == 400
    assert "different image" in resp.json()["detail"]


def test_controlnets_are_fetched_with_the_model(client: TestClient, harness: Harness) -> None:
    svc = client.app.state.services  # type: ignore[attr-defined]
    svc.db.replace_assets([{**MODEL, "size": 10}, {**CONTROLNET, "size": 10}, VAE, *CONFIGS])
    harness.worker_app.state.cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    harness.runner.fetch = True
    trace = upload(client, image(1024, 1024))
    run(client, {**SPEC, "control": [unit(image=f"sha256:{trace['sha256']}")]})
    cache = wait_for(lambda: client.get("/api/session").json()["worker"]["cache"]["files"])
    assert {f["path"] for f in cache} == {
        MODEL["path"],
        CONTROLNET["path"],
        VAE["path"],
        CONFIGS[0]["path"],
    }


# -- traces --------------------------------------------------------------------------------


def test_traces_need_a_session(client: TestClient) -> None:
    src = upload(client, image(200, 100))
    resp = client.post("/api/preprocess", json={"id": "depth", "image": src["sha256"]})
    assert resp.status_code == 409
    assert resp.json()["detail"] == "Start a session to trace depth"


def test_traces_are_stored_at_their_image_size(client: TestClient, harness: Harness) -> None:
    start_session(client, harness)
    src = upload(client, image(300, 200))
    body = client.post("/api/preprocess", json={"id": "depth", "image": src["sha256"]})
    assert body.status_code == 201, body.text
    traced = body.json()["image"]
    assert (traced["width"], traced["height"]) == (300, 200)
    assert read(client, traced["sha256"]).size == (300, 200)
    assert harness.trace.calls[-1]["model"].name == "depth-anything-v2"
    # Edges need no model, and take thresholds.
    body = client.post(
        "/api/preprocess",
        json={"id": "canny", "image": src["sha256"], "params": {"low": 20, "high": 90}},
    )
    assert body.status_code == 201, body.text
    assert harness.trace.calls[-1] == {"model": None, "low": 20, "high": 90}
    resp = client.post(
        "/api/preprocess",
        json={"id": "canny", "image": src["sha256"], "params": {"low": 200, "high": 90}},
    )
    assert resp.status_code == 400
    # DWPose isn't in this Drive.
    resp = client.post("/api/preprocess", json={"id": "pose", "image": src["sha256"]})
    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "DWPose isn't in Drive. Put it under degas/preprocessors/dwpose/, then rescan."
    )
