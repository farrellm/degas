"""Phase 15: FLUX 3 Image Klein Base (text-to-image, and edit with references), with its prompt
upsampler and the scene caption it writes."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, describe, spec_assets
from degas.families.flux3 import UPSAMPLER, Flux3

from .conftest import FLUX3, Harness, services
from .test_video import image, results, run, upload

SHA = "sha256:" + "a" * 64
T2I: dict[str, Any] = {
    "family": "flux3",
    "variant": "klein-base",
    "mode": "t2i",
    "model": {"path": FLUX3["path"]},
    "params": {"prompt": "a man walking his dog"},
}
EDIT: dict[str, Any] = {**T2I, "mode": "edit", "inputs": {"source": SHA}}


def test_flux3_defaults_to_the_base_recipe_and_the_upsampler() -> None:
    spec = Flux3().validate(T2I)
    params = spec["params"]
    assert (params["width"], params["height"]) == (1024, 1024)
    assert (params["steps"], params["cfg"], params["upsample"]) == (50, 3.5, True)
    assert spec["inputs"] == {}
    assert spec["upsampler"] == {"path": UPSAMPLER, "size": None}
    assert {"path": UPSAMPLER, "size": None, "kind": "upsampler"} in spec_assets(spec)


def test_without_upsampling_the_upsampler_is_not_needed() -> None:
    spec = Flux3().validate({**T2I, "params": {"prompt": "x", "upsample": False}})
    assert "upsampler" not in spec
    assert [a["kind"] for a in spec_assets(spec)] == ["model"]


def test_a_flux3_edit_takes_up_to_nine_references() -> None:
    spec = Flux3().validate({**EDIT, "inputs": {"source": SHA, "refs": [SHA] * 9}})
    assert spec["inputs"]["refs"] == [SHA] * 9
    variant = describe(Flux3())["variants"][0]
    assert (variant["max_refs"], variant["ref_max_pixels"]) == (9, 1024 * 1024)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "inpaint"}, "can't do 'inpaint'"),
        ({"inputs": {}}, "Choose a source image"),
        ({"inputs": {"source": SHA, "refs": [SHA] * 10}}, "At most 9 images"),
        ({"params": {"prompt": "x", "guidance": 4}}, "Unknown parameters"),
        ({"control": [{"kind": "depth"}]}, "doesn't take control units"),
    ],
)
def test_flux3_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Flux3().validate({**EDIT, **change})


# -- jobs ----------------------------------------------------------------------------------


def test_a_flux3_job_keeps_the_upsampled_caption(client: TestClient, harness: Harness) -> None:
    done = run(client, T2I)
    assert harness.flux3.calls[-1] == {"mode": "t2i", "images": [], "upsampler": UPSAMPLER}
    assert done["spec"]["upsampler"]["size"] == 19
    [result] = results(client, done["id"])
    assert result["caption"] == "A full scene caption of a man walking his dog"

    item = client.post(f"/api/results/{result['id']}/save").json()
    assert item["config"]["caption"] == result["caption"]
    assert item["config"]["params"]["prompt"] == "a man walking his dog"


def test_a_flux3_job_without_upsampling_has_no_caption(
    client: TestClient, harness: Harness
) -> None:
    done = run(client, {**T2I, "params": {"prompt": "x", "upsample": False}})
    assert harness.flux3.calls[-1]["upsampler"] is None
    [result] = results(client, done["id"])
    assert result["caption"] is None
    item = client.post(f"/api/results/{result['id']}/save").json()
    assert "caption" not in item["config"]


def test_a_flux3_job_without_the_upsampler_is_refused(client: TestClient) -> None:
    services(client).db.replace_assets([FLUX3])
    resp = client.post("/api/jobs", json={"spec": T2I})
    assert resp.status_code == 400
    assert f"degas/{UPSAMPLER}/" in resp.json()["detail"]
    assert "Upsample prompt" in resp.json()["detail"]


def test_a_flux3_edit_sends_its_references(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(1024, 1024))
    ref = upload(client, image(300, 500))
    spec = {
        **EDIT,
        "inputs": {"source": f"sha256:{src['sha256']}", "refs": [f"sha256:{ref['sha256']}"]},
    }
    run(client, spec)
    assert harness.flux3.calls[-1]["images"] == [(1024, 1024), (300, 500)]
