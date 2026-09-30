"""Image prompts (IP-Adapter) for SDXL: docs/ip-adapter.md."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, spec_assets
from degas.families.sdxl import Sdxl

from .conftest import CONFIGS, IMAGE_ENCODER, IP_ADAPTER, MODEL, VAE, Harness
from .test_api import wait_for
from .test_inpaint import mask_png, put_mask, read
from .test_video import image, results, run, upload

SHA = "sha256:" + "a" * 64
SPEC: dict[str, Any] = {
    "family": "sdxl",
    "variant": "base",
    "mode": "t2i",
    "model": {"path": MODEL["path"]},
    "params": {"prompt": "a cat", "width": 1024, "height": 768, "seed": 3},
}


def unit(**change: Any) -> dict[str, Any]:
    return {"adapter": {"path": IP_ADAPTER["path"]}, "images": [SHA], **change}


# -- descriptor ----------------------------------------------------------------------------


def test_units_are_normalized() -> None:
    spec = Sdxl().validate(
        {**SPEC, "image_prompts": [unit(purpose="style", weight=5, end=0.8, mask=SHA)]}
    )
    assert spec["image_prompts"] == [
        {
            "adapter": {"path": IP_ADAPTER["path"], "size": None},
            "images": [SHA],
            "fit": "crop",
            "purpose": "style",
            "weight": 2,  # clamped
            "start": 0.0,
            "end": 0.8,
            "mask": SHA,
        }
    ]
    assert spec["image_encoder"] == {"path": IMAGE_ENCODER["path"], "size": None}
    kinds = [(a["path"], a["kind"]) for a in spec_assets(spec)]
    assert (IP_ADAPTER["path"], "ip_adapter") in kinds
    assert kinds[-1] == (IMAGE_ENCODER["path"], "image_encoder")


def test_a_spec_without_image_prompts_is_unchanged() -> None:
    spec = Sdxl().validate({**SPEC, "image_prompts": []})
    assert "image_prompts" not in spec
    assert "image_encoder" not in spec


def test_h94s_first_adapter_reads_with_vit_bigg() -> None:
    spec = Sdxl().validate(
        {**SPEC, "image_prompts": [unit(adapter={"path": "ip_adapters/sdxl/ip-adapter_sdxl.bin"})]}
    )
    assert spec["image_encoder"]["path"] == "image_encoders/sdxl/clip-vit-bigg-14"


@pytest.mark.parametrize(
    ("prompts", "message"),
    [
        ([unit(adapter=None)], "choose a model"),
        ([unit(images=[])], "add a picture"),
        ([unit(images=[SHA] * 5)], "at most 4 pictures"),
        ([unit(adapter={"path": "controlnets/sdxl/x.safetensors"})], "isn't an image prompt"),
        ([unit(purpose="face")], "purpose: must be one of"),
        ([unit(start=0.5, end=0.5)], "start before they end"),
        ([unit(), unit(), unit()], "At most 2"),
        (
            [unit(), unit(adapter={"path": "ip_adapters/sdxl/ip-adapter_sdxl.safetensors"})],
            "different encoders",
        ),
    ],
)
def test_unit_validation(prompts: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Sdxl().validate({**SPEC, "image_prompts": prompts})


def test_a_file_can_guide_two_units() -> None:
    spec = Sdxl().validate(
        {**SPEC, "image_prompts": [unit(purpose="style"), unit(purpose="layout")]}
    )
    assert len(spec["image_prompts"]) == 2
    assert [a["kind"] for a in spec_assets(spec)].count("ip_adapter") == 1


def test_families_say_whether_they_take_image_prompts(client: TestClient) -> None:
    families = {f["id"]: f for f in client.get("/api/families").json()}
    assert families["sdxl"]["supports_image_prompts"] is True
    assert families["qwen21"]["supports_image_prompts"] is False


# -- jobs ----------------------------------------------------------------------------------


def test_pictures_are_squared_and_areas_fitted(client: TestClient, harness: Harness) -> None:
    wide = upload(client, image(1600, 1200))
    square = upload(client, image(512, 512))
    canvas = upload(client, image(1024, 768))
    area = put_mask(client, canvas["sha256"], mask_png(1024, 768, (0, 0, 512, 768)))
    prompt = unit(
        images=[f"sha256:{wide['sha256']}", f"sha256:{square['sha256']}"],
        mask=f"sha256:{area['sha256']}",
        purpose="style_layout",
    )
    done = run(client, {**SPEC, "image_prompts": [prompt]})
    assert harness.runner.inputs[-1] == {
        "prompts": [[(1024, 1024), (512, 512)]],
        "prompt_areas": [(1024, 768)],
    }
    fitted = done["spec"]["image_prompts"][0]
    assert "fit" not in fitted
    assert fitted["adapter"]["size"] == IP_ADAPTER["size"]
    assert done["spec"]["image_encoder"]["size"] == IMAGE_ENCODER["size"]
    assert fitted["images"][1] == f"sha256:{square['sha256']}"  # already square
    transforms = done["spec"]["inputs"]["transforms"]
    assert transforms[fitted["images"][0]]["original"] == f"sha256:{wide['sha256']}"
    assert read(client, fitted["images"][0].removeprefix("sha256:")).size == (1024, 1024)
    # Painted at the output's size, the area is used as it is.
    assert fitted["mask"] == f"sha256:{area['sha256']}"
    # Keeping the result keeps the pictures, and the config can be replayed.
    result = results(client, done["id"])[0]
    saved = client.post(f"/api/results/{result['id']}/save").json()
    assert saved["config"]["image_prompts"][0]["images"] == fitted["images"]
    replay = {k: v for k, v in saved["config"].items() if k not in ("degas_version", "runtime")}
    assert run(client, replay)["status"] == "done"


def test_letterboxed_pictures_keep_their_edges(client: TestClient, harness: Harness) -> None:
    tall = upload(client, image(300, 600))
    run(client, {**SPEC, "image_prompts": [unit(images=[f"sha256:{tall['sha256']}"], fit="pad")]})
    assert harness.runner.inputs[-1] == {"prompts": [[(600, 600)]]}


def test_image_prompts_are_checked_before_queueing(client: TestClient) -> None:
    picture = f"sha256:{upload(client, image(512, 512))['sha256']}"
    resp = client.post(
        "/api/jobs",
        json={
            "spec": {
                **SPEC,
                "image_prompts": [
                    unit(images=[picture], adapter={"path": "ip_adapters/sdxl/gone.safetensors"})
                ],
            }
        },
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == (
        "Image prompt model ip_adapters/sdxl/gone.safetensors is not in the Drive index"
    )
    svc = client.app.state.services  # type: ignore[attr-defined]
    svc.db.replace_assets([MODEL, VAE, *CONFIGS, IP_ADAPTER])
    resp = client.post(
        "/api/jobs", json={"spec": {**SPEC, "image_prompts": [unit(images=[picture])]}}
    )
    assert resp.status_code == 400
    assert "need the CLIP image encoder" in resp.json()["detail"]


def test_an_empty_area_is_refused(client: TestClient) -> None:
    picture = upload(client, image(512, 512))
    empty = put_mask(client, picture["sha256"], mask_png(512, 512, (0, 0, 0, 0)))
    resp = client.post(
        "/api/jobs",
        json={
            "spec": {
                **SPEC,
                "image_prompts": [
                    unit(images=[f"sha256:{picture['sha256']}"], mask=f"sha256:{empty['sha256']}")
                ],
            }
        },
    )
    assert resp.status_code == 400
    assert "Image prompt 1's area is empty" in resp.json()["detail"]


def test_adapters_are_fetched_with_the_model(client: TestClient, harness: Harness) -> None:
    svc = client.app.state.services  # type: ignore[attr-defined]
    svc.db.replace_assets(
        [
            {**MODEL, "size": 10},
            {**IP_ADAPTER, "size": 10},
            {**IMAGE_ENCODER, "size": 10},
            VAE,
            *CONFIGS,
        ]
    )
    harness.worker_app.state.cache.set_token("tok", "2026-09-27T12:00:00Z", "degas")
    harness.runner.fetch = True
    picture = upload(client, image(512, 512))
    run(client, {**SPEC, "image_prompts": [unit(images=[f"sha256:{picture['sha256']}"])]})
    cache = wait_for(lambda: client.get("/api/session").json()["worker"]["cache"]["files"])
    assert {f["path"] for f in cache} == {
        MODEL["path"],
        IP_ADAPTER["path"],
        IMAGE_ENCODER["path"],
        VAE["path"],
        CONFIGS[0]["path"],
    }
