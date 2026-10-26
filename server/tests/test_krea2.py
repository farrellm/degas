"""Phase 19: Krea 2 (text-to-image, and edits through conradlocke's edit LoRAs), Raw and Turbo."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, describe, spec_assets
from degas.families.krea2 import EDIT_LORAS, Krea2

from .conftest import KREA2_GENERAL, KREA2_IDENTITY, KREA2_RAW, KREA2_TURBO, Harness, services
from .test_video import image, results, run, upload

SHA = "sha256:" + "a" * 64
T2I: dict[str, Any] = {
    "family": "krea2",
    "variant": "raw",
    "mode": "t2i",
    "model": {"path": KREA2_RAW["path"]},
    "params": {"prompt": "a fox in the snow"},
}
TURBO: dict[str, Any] = {**T2I, "variant": "turbo", "model": {"path": KREA2_TURBO["path"]}}
EDIT: dict[str, Any] = {
    **T2I,
    "mode": "edit",
    "inputs": {"source": SHA},
    "params": {"prompt": "make the fox black"},
}


def test_raw_defaults_to_krea_s_base_recipe() -> None:
    spec = Krea2().validate(T2I)
    params = spec["params"]
    assert (params["width"], params["height"]) == (1024, 1024)
    assert (params["steps"], params["cfg"], params["negative_prompt"]) == (28, 5.5, "")
    assert (spec["loras"], spec["inputs"], spec["control"]) == ([], {}, [])
    assert "edit_lora" not in spec
    assert "edit_lora" not in params


def test_turbo_has_no_cfg_or_negative_prompt() -> None:
    props = Krea2().param_schema("turbo", "t2i")["properties"]
    assert "cfg" not in props
    assert "negative_prompt" not in props
    assert props["steps"]["default"] == 8
    assert props["width"]["maximum"] == 2048
    assert Krea2().param_schema("raw", "t2i")["properties"]["width"]["maximum"] == 1536
    spec = Krea2().validate({**TURBO, "params": {"prompt": "x", "width": 2048, "height": 2048}})
    assert (spec["params"]["width"], spec["params"]["steps"]) == (2048, 8)


def test_raw_is_capped_below_turbo() -> None:
    spec = Krea2().validate({**T2I, "params": {"prompt": "x", "width": 2048, "height": 2048}})
    assert (spec["params"]["width"], spec["params"]["height"]) == (1536, 1536)


def test_variants_read_their_own_folders() -> None:
    variants = {v["id"]: v for v in describe(Krea2())["variants"]}
    assert variants["raw"]["model_dir"] == "models/krea2/Krea-2-Raw"
    assert variants["turbo"]["model_dir"] == "models/krea2/Krea-2-Turbo"
    assert all(v["modes"] == ["t2i", "edit"] for v in variants.values())
    assert all(v["min_gpu"] == "A100" and v["max_refs"] == 1 for v in variants.values())


def test_an_edit_adds_the_general_edit_lora() -> None:
    spec = Krea2().validate(EDIT)
    params = spec["params"]
    assert (params["steps"], params["cfg"], params["edit_lora"]) == (20, 3.0, "general")
    assert spec["edit_lora"] == {"path": EDIT_LORAS["general"], "size": None, "weight": 1.0}
    assert spec["loras"] == []
    assert [(a["path"], a["kind"]) for a in spec_assets(spec)] == [
        (KREA2_RAW["path"], "model"),
        (EDIT_LORAS["general"], "lora"),
    ]


def test_identity_reads_a_second_image() -> None:
    params = {"prompt": "she sits by the campfire", "edit_lora": "identity"}
    spec = Krea2().validate({**EDIT, "params": params, "inputs": {"source": SHA, "refs": [SHA]}})
    assert spec["inputs"]["refs"] == [SHA]
    assert spec["edit_lora"]["path"] == EDIT_LORAS["identity"]


def test_a_chosen_edit_lora_keeps_its_weight() -> None:
    loras = [{"path": EDIT_LORAS["general"], "weight": 0.6}]
    spec = Krea2().validate({**EDIT, "loras": loras})
    assert "edit_lora" not in spec
    assert spec["loras"] == [{"path": EDIT_LORAS["general"], "weight": 0.6, "size": None}]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"model": {"path": KREA2_RAW["path"]}, "variant": "turbo"}, "isn't a Krea 2 Turbo"),
        ({"mode": "inpaint"}, "can't do 'inpaint'"),
        ({"inputs": {}}, "Choose a source image"),
        ({"inputs": {"source": SHA, "refs": [SHA]}}, "Only Identity"),
        (
            {
                "inputs": {"source": SHA, "refs": [SHA, SHA]},
                "params": {"prompt": "x", "edit_lora": "identity"},
            },
            "At most 1 images",
        ),
        ({"params": {"prompt": "x", "edit_lora": "style"}}, "edit_lora"),
        ({"control": [{"kind": "depth"}]}, "doesn't take control units"),
        ({"loras": [{"path": f"loras/krea2/{i}.safetensors"} for i in range(9)]}, "At most 8"),
    ],
)
def test_krea2_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Krea2().validate({**EDIT, **change})


def test_a_krea2_job_runs_on_its_runner(client: TestClient, harness: Harness) -> None:
    done = run(client, TURBO)
    assert harness.krea2.calls[-1] == {"mode": "t2i", "images": []}
    assert done["spec"]["model"]["size"] == KREA2_TURBO["size"]
    assert len(results(client, done["id"])) == 1


def test_a_krea2_edit_sends_its_images_and_edit_lora(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(1024, 1024))
    person = upload(client, image(300, 500))
    spec = {
        **EDIT,
        "params": {"prompt": "she sits by the campfire", "edit_lora": "identity"},
        "inputs": {
            "source": f"sha256:{src['sha256']}",
            "refs": [f"sha256:{person['sha256']}"],
        },
    }
    done = run(client, spec)
    call = harness.krea2.calls[-1]
    assert call["images"] == [(1024, 1024), (300, 500)]
    assert call["edit_lora"] == KREA2_IDENTITY["path"]
    assert done["spec"]["edit_lora"]["size"] == KREA2_IDENTITY["size"]


def test_an_edit_without_its_edit_lora_is_refused(client: TestClient) -> None:
    services(client).db.replace_assets([KREA2_RAW, KREA2_IDENTITY])
    src = upload(client, image(1024, 1024))
    spec = {**EDIT, "inputs": {"source": f"sha256:{src['sha256']}"}}
    resp = client.post("/api/jobs", json={"spec": spec})
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "edit LoRA" in detail
    assert f"degas/{KREA2_GENERAL['path']}" in detail
