"""Phase 10: FLUX.1 [dev] (text-to-image) and FLUX.2 [klein] (edit with references)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, describe, spec_assets
from degas.families.flux1 import Flux1
from degas.families.klein import Klein
from degas.families.qwen21 import Qwen21

from .conftest import FLUX1, KLEIN, Harness
from .test_video import image, run, upload

FLUX: dict[str, Any] = {
    "family": "flux1",
    "variant": "dev",
    "mode": "t2i",
    "model": {"path": FLUX1["path"]},
    "params": {"prompt": "a lighthouse in fog"},
}
SHA = "sha256:" + "a" * 64
EDIT: dict[str, Any] = {
    "family": "klein",
    "variant": "9b",
    "mode": "edit",
    "model": {"path": KLEIN["path"]},
    "params": {"prompt": "make it night"},
    "inputs": {"source": SHA},
}


def test_flux_defaults() -> None:
    spec = Flux1().validate(FLUX)
    params = spec["params"]
    assert (params["width"], params["height"], params["steps"]) == (1024, 1024, 28)
    assert params["guidance"] == 3.5
    assert "negative_prompt" not in params
    assert spec["inputs"] == {}


def test_a_single_file_flux_checkpoint_brings_the_base_folder() -> None:
    spec = Flux1().validate(FLUX)
    assert spec["config"] == {"path": "configs/flux1/FLUX.1-dev", "size": None}
    assert [a["kind"] for a in spec_assets(spec)] == ["model", "config"]
    # A whole diffusers folder has its own encoders.
    folder = Flux1().validate({**FLUX, "model": {"path": "models/flux1/FLUX.1-dev"}})
    assert "config" not in folder


def test_flux_sizes_snap_to_16() -> None:
    spec = Flux1().validate({**FLUX, "params": {"prompt": "x", "width": 1000, "height": 1400}})
    assert (spec["params"]["width"], spec["params"]["height"]) == (992, 1392)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "edit"}, "can't do 'edit'"),
        ({"params": {"prompt": "x", "negative_prompt": "y"}}, "Unknown parameters"),
        ({"params": {"prompt": "x", "width": 2048, "height": 2048}}, "pixel count"),
        ({"control": [{"controlnet": "x"}]}, "doesn't take control units"),
    ],
)
def test_flux_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Flux1().validate({**FLUX, **change})


def test_klein_defaults_to_four_steps() -> None:
    spec = Klein().validate(EDIT)
    params = spec["params"]
    assert (params["width"], params["height"], params["steps"]) == (1024, 1024, 4)
    assert spec["inputs"] == {"source": SHA, "fit": "crop"}
    assert "config" not in spec


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "t2i"}, "can't do 't2i'"),
        ({"inputs": {}}, "Choose a source image"),
        ({"inputs": {"source": SHA, "refs": [SHA] * 4}}, "At most 3 images"),
        ({"params": {"prompt": "x", "cfg": 4}}, "Unknown parameters"),
    ],
)
def test_klein_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Klein().validate({**EDIT, **change})


def test_variants_say_how_many_references_they_read() -> None:
    assert [v["max_refs"] for v in describe(Klein())["variants"]] == [3]
    assert [v["max_refs"] for v in describe(Qwen21())["variants"]] == [9]
    assert [v["max_refs"] for v in describe(Flux1())["variants"]] == [0]
    # klein only scales references down; Qwen scales them to the output's size.
    assert describe(Klein())["variants"][0]["ref_max_pixels"] == 1024 * 1024
    assert describe(Qwen21())["variants"][0]["ref_max_pixels"] is None


# -- jobs ----------------------------------------------------------------------------------


def test_a_flux_job_names_the_base_folder(client: TestClient, harness: Harness) -> None:
    done = run(client, FLUX)
    assert done["status"] == "done"
    assert harness.flux.calls[-1] == {
        "mode": "t2i",
        "images": [],
        "config": "configs/flux1/FLUX.1-dev",
    }
    assert done["spec"]["config"]["size"] == 11


def test_a_flux_job_without_the_base_folder_is_refused(client: TestClient) -> None:
    client.app.state.services.db.replace_assets([FLUX1])  # type: ignore[attr-defined]
    resp = client.post("/api/jobs", json={"spec": FLUX})
    assert resp.status_code == 400
    assert "degas/configs/flux1/FLUX.1-dev/" in resp.json()["detail"]


def test_a_klein_edit_fits_the_source_and_sends_references(
    client: TestClient, harness: Harness
) -> None:
    src = upload(client, image(2048, 1024))
    ref = upload(client, image(300, 500))
    spec = {
        **EDIT,
        "params": {"prompt": "the coat from image 2", "width": 1024, "height": 1024},
        "inputs": {"source": f"sha256:{src['sha256']}", "refs": [f"sha256:{ref['sha256']}"]},
    }
    done = run(client, spec)
    assert harness.klein.calls[-1] == {"mode": "edit", "images": [(1024, 1024), (300, 500)]}
    assert done["spec"]["inputs"]["refs"] == [f"sha256:{ref['sha256']}"]
