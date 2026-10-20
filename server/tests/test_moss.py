"""Phase 17: MOSS-Image (text-to-image), from the Base, SFT or Kromoss checkpoint."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, describe
from degas.families.moss import Moss

from .conftest import MOSS_BASE, MOSS_KROMOSS, MOSS_SFT, MOSS_TURBO, Harness
from .test_video import results, run

T2I: dict[str, Any] = {
    "family": "moss",
    "variant": "base",
    "mode": "t2i",
    "model": {"path": MOSS_BASE["path"]},
    "params": {"prompt": "a lighthouse knitted from wool"},
}
SFT: dict[str, Any] = {**T2I, "variant": "sft", "model": {"path": MOSS_SFT["path"]}}
TURBO: dict[str, Any] = {
    **T2I,
    "variant": "kromoss",
    "model": {"path": MOSS_KROMOSS["path"]},
    "loras": [{"path": MOSS_TURBO["path"]}],
    "params": {**T2I["params"], "steps": 8, "cfg": 1},
}


def test_moss_defaults_to_openmoss_recipe() -> None:
    spec = Moss().validate(T2I)
    params = spec["params"]
    assert (params["width"], params["height"]) == (1024, 1024)
    assert (params["steps"], params["cfg"], params["shift"]) == (50, 3.0, 5.0)
    assert (params["skip_layer"], params["negative_prompt"]) == ("22", "")
    assert (spec["loras"], spec["inputs"], spec["control"]) == ([], {}, [])


def test_moss_variants_read_their_own_folders() -> None:
    variants = {v["id"]: v for v in describe(Moss())["variants"]}
    assert variants["base"]["model_dir"] == "models/moss/MOSS-Image-Base"
    assert variants["sft"]["model_dir"] == "models/moss/MOSS-Image-SFT"
    assert variants["kromoss"]["model_dir"] == "models/moss/Kromoss"
    assert all(v["modes"] == ["t2i"] for v in variants.values())
    assert Moss().validate(SFT)["variant"] == "sft"


def test_moss_takes_single_file_loras() -> None:
    spec = Moss().validate(TURBO)
    assert spec["loras"] == [{"path": MOSS_TURBO["path"], "weight": 1.0, "size": None}]
    assert (spec["params"]["steps"], spec["params"]["cfg"]) == (8, 1.0)


def test_moss_snaps_sizes_to_16() -> None:
    spec = Moss().validate({**T2I, "params": {"prompt": "x", "width": 1000, "height": 1450}})
    assert spec["params"]["width"] % 16 == 0
    assert spec["params"]["height"] % 16 == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": {"path": MOSS_BASE["path"]}, "variant": "sft"}, "isn't a MOSS-Image SFT"),
        ({"mode": "edit"}, "can't do 'edit'"),
        ({"loras": [{"path": f"loras/moss/{i}.safetensors"} for i in range(5)]}, "At most 4"),
        ({"control": [{"kind": "depth"}]}, "doesn't take control units"),
        ({"params": {"prompt": "x", "skip_layer": "23"}}, "skip_layer"),
    ],
)
def test_moss_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Moss().validate({**T2I, **change})


def test_a_moss_job_runs_on_its_runner(client: TestClient, harness: Harness) -> None:
    done = run(client, SFT)
    assert harness.moss.calls[-1] == {"mode": "t2i", "images": []}
    assert done["spec"]["model"]["size"] == MOSS_SFT["size"]
    assert len(results(client, done["id"])) == 1


def test_a_kromoss_job_with_the_turbo_lora_runs(client: TestClient, harness: Harness) -> None:
    done = run(client, TURBO)
    assert harness.moss.calls[-1] == {"mode": "t2i", "images": []}
    assert done["spec"]["model"]["size"] == MOSS_KROMOSS["size"]
    assert done["spec"]["loras"][0]["size"] == MOSS_TURBO["size"]
