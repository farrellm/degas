"""Phase 8: Qwen-Image 2.1 (text-to-image, edit with references, inpaint)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError
from degas.families.qwen21 import Qwen21
from degas.library import input_blobs

from .conftest import QWEN, Harness
from .test_inpaint import mask_png, put_mask
from .test_video import image, results, run, upload

SPEC: dict[str, Any] = {
    "family": "qwen21",
    "variant": "base",
    "mode": "t2i",
    "model": {"path": QWEN["path"]},
    "params": {"prompt": "a lighthouse in fog"},
}
SHA = "sha256:" + "a" * 64


def test_defaults_are_qwens() -> None:
    spec = Qwen21().validate(SPEC)
    params = spec["params"]
    assert (params["width"], params["height"], params["steps"]) == (2048, 2048, 40)
    assert (params["cfg"], params["schedule"], params["degrid"]) == (1.0, "default", True)
    assert spec["inputs"] == {}
    assert "mask_blur" not in params


def test_sizes_snap_to_32() -> None:
    spec = Qwen21().validate({**SPEC, "params": {"prompt": "x", "width": 1000, "height": 1400}})
    assert (spec["params"]["width"], spec["params"]["height"]) == (992, 1376)


def test_edit_keeps_its_references_in_order() -> None:
    refs = ["sha256:" + c * 64 for c in "bcd"]
    spec = Qwen21().validate(
        {**SPEC, "mode": "edit", "inputs": {"source": SHA, "refs": refs, "fit": "pad"}}
    )
    assert spec["inputs"] == {"source": SHA, "fit": "pad", "refs": refs}
    # Text-to-image has no images at all.
    assert Qwen21().validate({**SPEC, "inputs": {"source": SHA, "refs": refs}})["inputs"] == {}


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "edit"}, "Choose a source image"),
        ({"mode": "inpaint", "inputs": {"source": SHA}}, "Paint the area"),
        ({"mode": "edit", "inputs": {"source": SHA, "refs": "x"}}, "refs: expected a list"),
        ({"mode": "edit", "inputs": {"source": SHA, "refs": ["x"]}}, "Image 2: expected"),
        ({"mode": "edit", "inputs": {"source": SHA, "refs": [SHA] * 10}}, "At most 9 images"),
        ({"mode": "i2i"}, "can't do 'i2i'"),
        ({"control": [{"controlnet": "x"}]}, "doesn't take control units"),
    ],
)
def test_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Qwen21().validate({**SPEC, **change})


def test_input_blobs_include_the_references() -> None:
    refs = ["sha256:" + c * 64 for c in "bc"]
    assert input_blobs({"inputs": {"source": SHA, "refs": refs}}) == ["a" * 64, "b" * 64, "c" * 64]


# -- jobs ----------------------------------------------------------------------------------


def test_edit_fits_the_source_and_sends_references_as_they_are(
    client: TestClient, harness: Harness
) -> None:
    src = upload(client, image(2048, 1024))
    ref2 = upload(client, image(300, 500))
    ref3 = upload(client, image(640, 480))
    refs = [f"sha256:{ref2['sha256']}", f"sha256:{ref3['sha256']}"]
    spec = {
        **SPEC,
        "mode": "edit",
        "params": {"prompt": "put the hat from image 2 on her", "width": 1024, "height": 1024},
        "inputs": {"source": f"sha256:{src['sha256']}", "refs": refs},
    }
    done = run(client, spec)
    assert harness.qwen.calls[-1] == {
        "mode": "edit",
        "images": [(1024, 1024), (300, 500), (640, 480)],
    }
    assert done["spec"]["inputs"]["refs"] == refs
    # Keeping the result keeps the references with it.
    result = results(client, done["id"])[0]
    saved = client.post(f"/api/results/{result['id']}/save").json()
    assert saved["config"]["inputs"]["refs"] == refs


def test_a_cropped_reference_keeps_its_original(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(1024, 1024))
    photo = upload(client, image(800, 600))
    crop = [{"op": "crop", "x": 100, "y": 50, "w": 300, "h": 400}]
    face = client.post(f"/api/blobs/{photo['sha256']}/transform", json={"ops": crop}).json()
    spec = {
        **SPEC,
        "mode": "edit",
        "params": {"prompt": "her face from image 2", "width": 1024, "height": 1024},
        "inputs": {"source": f"sha256:{src['sha256']}", "refs": [f"sha256:{face['sha256']}"]},
    }
    done = run(client, spec)
    assert harness.qwen.calls[-1]["images"] == [(1024, 1024), (300, 400)]
    assert done["spec"]["inputs"]["transforms"] == {
        f"sha256:{face['sha256']}": {"original": f"sha256:{photo['sha256']}", "ops": crop}
    }
    # Keeping the result keeps the original too.
    result = results(client, done["id"])[0]
    saved = client.post(f"/api/results/{result['id']}/save").json()
    assert photo["sha256"] in input_blobs(saved["config"])


def test_inpaint_sends_the_mask_and_references(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(1024, 1024))
    mask = put_mask(client, src["sha256"], mask_png(1024, 1024, (0, 0, 512, 512)))
    ref = upload(client, image(256, 256))
    spec = {
        **SPEC,
        "mode": "inpaint",
        "params": {"prompt": "a red scarf", "width": 1024, "height": 1024},
        "inputs": {
            "source": f"sha256:{src['sha256']}",
            "mask": f"sha256:{mask['sha256']}",
            "refs": [f"sha256:{ref['sha256']}"],
        },
    }
    run(client, spec)
    assert harness.qwen.calls[-1] == {
        "mode": "inpaint",
        "images": [(1024, 1024), (256, 256)],
        "mask": (1024, 1024),
    }


def test_a_missing_reference_is_refused(client: TestClient) -> None:
    src = upload(client, image(1024, 1024))
    spec = {
        **SPEC,
        "mode": "edit",
        "inputs": {"source": f"sha256:{src['sha256']}", "refs": [SHA]},
    }
    resp = client.post("/api/jobs", json={"spec": spec})
    assert resp.status_code == 400
    assert resp.json()["detail"] == "Image 2 is no longer stored. Choose it again."
