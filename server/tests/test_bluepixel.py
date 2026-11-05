"""Phase 20: BluePixel-Turbo (instruction editing of one image at up to 4K)."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, describe, spec_assets
from degas.families.bluepixel import BluePixel

from .conftest import BLUEPIXEL, Harness
from .test_video import image, run, upload

SHA = "sha256:" + "a" * 64
EDIT: dict[str, Any] = {
    "family": "bluepixel",
    "variant": "turbo",
    "mode": "edit",
    "model": {"path": BLUEPIXEL["path"]},
    "inputs": {"source": SHA},
    "params": {"prompt": "make the sky a sunset"},
}


def test_defaults_to_vivo_s_recipe_at_4k() -> None:
    spec = BluePixel().validate(EDIT)
    params = spec["params"]
    assert (params["width"], params["height"]) == (4096, 4096)
    assert (params["steps"], params["cfg"]) == (4, 3.0)
    assert (spec["loras"], spec["control"]) == ([], [])
    assert [(a["path"], a["kind"]) for a in spec_assets(spec)] == [(BLUEPIXEL["path"], "model")]


def test_edits_only_with_no_references() -> None:
    (variant,) = describe(BluePixel())["variants"]
    assert (variant["modes"], variant["max_refs"], variant["min_gpu"]) == (["edit"], 0, "A100")
    size = variant["size_constraints"]
    assert (size["multiple_of"], size["max_pixels"]) == (64, 4096 * 4096)
    assert all(w % 64 == 0 and h % 64 == 0 and w * h <= 4096 * 4096 for w, h in size["presets"])


def test_sizes_snap_to_multiples_of_64() -> None:
    spec = BluePixel().validate({**EDIT, "params": {"prompt": "x", "width": 3000, "height": 2000}})
    assert (spec["params"]["width"], spec["params"]["height"]) == (2944, 1984)
    spec = BluePixel().validate({**EDIT, "params": {"prompt": "x", "cfg": 9}})
    assert spec["params"]["cfg"] == 8


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "t2i"}, "can't do 't2i'"),
        ({"inputs": {}}, "Choose a source image"),
        ({"inputs": {"source": SHA, "refs": [SHA]}}, "reads no references"),
        ({"loras": [{"path": "loras/bluepixel/x.safetensors"}]}, "doesn't take LoRAs"),
        ({"control": [{"kind": "depth"}]}, "doesn't take control units"),
        ({"params": {"prompt": "x", "width": 5440, "height": 5440}}, "pixel count"),
    ],
)
def test_bluepixel_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        BluePixel().validate({**EDIT, **change})


def test_an_edit_fits_the_source_to_the_output(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(1200, 800))
    spec = {
        **EDIT,
        "params": {"prompt": "add warm light", "width": 1536, "height": 1024},
        "inputs": {"source": f"sha256:{src['sha256']}"},
    }
    done = run(client, spec)
    assert done["status"] == "done"
    assert harness.bluepixel.calls[-1] == {"mode": "edit", "images": [(1536, 1024)]}
