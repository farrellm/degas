"""Phase 18: Fibo 2.0 (text-to-image, from an image, and edit with references), with Bria's
prompt-to-JSON VLMs and the caption they write."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas.families.base import SpecError, describe, spec_assets
from degas.families.fibo2 import (
    AUSBOSS_HERETIC,
    AUSBOSS_UPSAMPLER,
    EDIT_UPSAMPLER,
    TEXT_ENCODER,
    UPSAMPLER,
    Fibo2,
)

from .conftest import FIBO2, FIBO2_VLM, LORA, QWEN, Harness, services
from .test_video import image, results, run, upload

SHA = "sha256:" + "a" * 64
T2I: dict[str, Any] = {
    "family": "fibo2",
    "variant": "dev",
    "mode": "t2i",
    "model": {"path": FIBO2["path"]},
    "params": {"prompt": "a man walking his dog"},
}
EDIT: dict[str, Any] = {**T2I, "mode": "edit", "inputs": {"source": SHA}}
I2I: dict[str, Any] = {**T2I, "mode": "i2i", "params": {}, "inputs": {"source": SHA}}


def test_fibo2_defaults_to_the_dev_recipe_and_the_vlm() -> None:
    spec = Fibo2().validate(T2I)
    params = spec["params"]
    assert (params["width"], params["height"]) == (1024, 1024)
    assert (params["steps"], params["cfg"], params["upsample"]) == (50, 5.0, True)
    assert spec["inputs"] == {}
    assert spec["loras"] == []
    assert spec["upsampler"] == {"path": UPSAMPLER, "size": None}
    assert {"path": UPSAMPLER, "size": None, "kind": "upsampler"} in spec_assets(spec)


def test_an_edit_of_one_image_uses_the_edit_vlm() -> None:
    spec = Fibo2().validate(EDIT)
    assert spec["upsampler"] == {"path": EDIT_UPSAMPLER, "size": None}


def test_an_edit_with_references_sends_the_instruction_as_written() -> None:
    spec = Fibo2().validate({**EDIT, "inputs": {"source": SHA, "refs": [SHA] * 4}})
    assert spec["inputs"]["refs"] == [SHA] * 4
    assert "upsampler" not in spec
    variant = describe(Fibo2())["variants"][0]
    assert (variant["max_refs"], variant["ref_max_pixels"]) == (4, 1024 * 1024)


def test_from_an_image_fibo_vlm_takes_the_source_and_an_optional_prompt() -> None:
    spec = Fibo2().validate(I2I)
    assert spec["params"]["prompt"] == ""
    assert "upsample" not in spec["params"]
    assert spec["upsampler"] == {"path": UPSAMPLER, "size": None}
    assert spec["inputs"] == {"source": SHA, "fit": "crop"}
    schema = Fibo2().param_schema("dev", "i2i")
    assert "minLength" not in schema["properties"]["prompt"]
    assert schema["required"] == []
    assert "upsample" not in schema["properties"]
    assert "minLength" in Fibo2().param_schema("dev", "t2i")["properties"]["prompt"]


def test_from_an_image_with_a_prompt_uses_the_edit_vlm() -> None:
    spec = Fibo2().validate({**I2I, "params": {"prompt": "make it snow"}})
    assert spec["upsampler"] == {"path": EDIT_UPSAMPLER, "size": None}
    spec = Fibo2().validate({**I2I, "params": {"prompt": "  "}})
    assert spec["upsampler"] == {"path": UPSAMPLER, "size": None}


def test_from_an_image_takes_no_references() -> None:
    spec = Fibo2().validate({**I2I, "inputs": {"source": SHA, "refs": [SHA]}})
    assert "refs" not in spec["inputs"]
    with pytest.raises(SpecError, match="Unknown parameters"):
        Fibo2().validate({**I2I, "params": {"upsample": False}})
    with pytest.raises(SpecError, match="Choose a source image"):
        Fibo2().validate({**I2I, "inputs": {}})


def test_without_upsampling_no_vlm_is_needed() -> None:
    spec = Fibo2().validate({**T2I, "params": {"prompt": "x", "upsample": False}})
    assert "upsampler" not in spec
    assert [a["kind"] for a in spec_assets(spec)] == ["model"]


def test_uncensored_swaps_the_text_encoder_in_every_mode() -> None:
    for base in (T2I, EDIT, I2I):
        spec = Fibo2().validate(base)
        assert spec["params"]["uncensored"] is False
        assert "text_encoder" not in spec
        params = {**base["params"], "uncensored": True}
        spec = Fibo2().validate({**base, "params": params})
        assert spec["text_encoder"] == {"path": TEXT_ENCODER, "size": None}
        need = {"path": TEXT_ENCODER, "size": None, "kind": "text_encoder"}
        assert need in spec_assets(spec)
    # Bria's VLMs still write the caption.
    assert spec["upsampler"] == {"path": UPSAMPLER, "size": None}


def test_the_ausboss_style_writes_with_qwen_image_or_its_heretic() -> None:
    params = {"prompt": "x", "upsample_style": "ausboss"}
    for base in (T2I, EDIT):
        spec = Fibo2().validate({**base, "params": params})
        assert spec["upsampler"] == {"path": AUSBOSS_UPSAMPLER, "size": None}
        # Qwen-Image 2.1's model folder, whose text encoder writes.
        need = {"path": AUSBOSS_UPSAMPLER, "size": None, "kind": "model", "family": "qwen21"}
        assert need in spec_assets(spec)
        spec = Fibo2().validate({**base, "params": {**params, "uncensored": True}})
        assert spec["upsampler"] == {"path": AUSBOSS_HERETIC, "size": None}
        need = {"path": AUSBOSS_HERETIC, "size": None, "kind": "upsampler", "family": "qwen21"}
        assert need in spec_assets(spec)
    spec = Fibo2().validate({**EDIT, "params": params, "inputs": {"source": SHA, "refs": [SHA]}})
    assert "upsampler" not in spec
    spec = Fibo2().validate({**T2I, "params": {**params, "upsample": False}})
    assert "upsampler" not in spec
    assert Fibo2().validate(T2I)["params"]["upsample_style"] == "bria"
    # From an image, Bria's VLMs are the only writers.
    assert "upsample_style" not in Fibo2().param_schema("dev", "i2i")["properties"]
    with pytest.raises(SpecError, match="Unknown parameters"):
        Fibo2().validate({**I2I, "params": {"upsample_style": "ausboss"}})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"mode": "inpaint"}, "can't do 'inpaint'"),
        ({"inputs": {}}, "Choose a source image"),
        ({"inputs": {"source": SHA, "refs": [SHA] * 5}}, "At most 4 images"),
        ({"params": {"prompt": "x", "negative_prompt": "blur"}}, "Unknown parameters"),
        ({"control": [{"kind": "depth"}]}, "doesn't take control units"),
        ({"loras": [{"path": LORA["path"]}]}, "doesn't take LoRAs"),
    ],
)
def test_fibo2_validation(change: dict[str, Any], message: str) -> None:
    with pytest.raises(SpecError, match=message):
        Fibo2().validate({**EDIT, **change})


# -- jobs ----------------------------------------------------------------------------------


def test_a_fibo2_job_keeps_the_caption(client: TestClient, harness: Harness) -> None:
    done = run(client, T2I)
    assert harness.fibo2.calls[-1] == {"mode": "t2i", "images": [], "upsampler": UPSAMPLER}
    [result] = results(client, done["id"])
    assert result["caption"] == "A full scene caption of a man walking his dog"
    item = client.post(f"/api/results/{result['id']}/save").json()
    assert item["config"]["caption"] == result["caption"]


def test_a_fibo2_job_without_its_vlm_is_refused(client: TestClient) -> None:
    services(client).db.replace_assets([FIBO2])
    resp = client.post("/api/jobs", json={"spec": T2I})
    assert resp.status_code == 400
    assert f"degas/{UPSAMPLER}/" in resp.json()["detail"]
    assert "Upsample prompt" in resp.json()["detail"]


def test_a_fibo2_edit_sends_its_references(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(1024, 1024))
    ref = upload(client, image(300, 500))
    spec = {
        **EDIT,
        "inputs": {"source": f"sha256:{src['sha256']}", "refs": [f"sha256:{ref['sha256']}"]},
    }
    run(client, spec)
    assert harness.fibo2.calls[-1]["images"] == [(1024, 1024), (300, 500)]
    assert harness.fibo2.calls[-1]["upsampler"] is None


def test_a_fibo2_edit_of_one_image_uses_the_edit_vlm(client: TestClient, harness: Harness) -> None:
    src = upload(client, image(1024, 1024))
    run(client, {**EDIT, "inputs": {"source": f"sha256:{src['sha256']}"}})
    assert harness.fibo2.calls[-1]["upsampler"] == EDIT_UPSAMPLER


def test_a_fibo2_job_from_an_image_has_fibo_vlm_see_it(
    client: TestClient, harness: Harness
) -> None:
    src = upload(client, image(640, 480))
    done = run(client, {**I2I, "inputs": {"source": f"sha256:{src['sha256']}"}})
    call = harness.fibo2.calls[-1]
    assert (call["mode"], call["upsampler"]) == ("i2i", UPSAMPLER)
    assert len(call["images"]) == 1
    [result] = results(client, done["id"])
    assert result["caption"] is not None


def test_an_uncensored_ausboss_fibo2_job_names_both_heretics(
    client: TestClient, harness: Harness
) -> None:
    params = {"prompt": "x", "upsample_style": "ausboss", "uncensored": True}
    done = run(client, {**T2I, "params": params})
    call = harness.fibo2.calls[-1]
    assert (call["upsampler"], call["text_encoder"]) == (AUSBOSS_HERETIC, TEXT_ENCODER)
    assert (done["spec"]["upsampler"]["size"], done["spec"]["text_encoder"]["size"]) == (17, 9)


def test_the_ausboss_style_uses_qwen_images_text_encoder(
    client: TestClient, harness: Harness
) -> None:
    done = run(client, {**T2I, "params": {"prompt": "x", "upsample_style": "ausboss"}})
    assert harness.fibo2.calls[-1]["upsampler"] == AUSBOSS_UPSAMPLER
    assert done["spec"]["upsampler"]["size"] == QWEN["size"]


def test_the_ausboss_style_without_qwen_image_is_refused(client: TestClient) -> None:
    services(client).db.replace_assets([FIBO2, FIBO2_VLM])
    params = {"prompt": "x", "upsample_style": "ausboss"}
    resp = client.post("/api/jobs", json={"spec": {**T2I, "params": params}})
    assert resp.status_code == 400
    assert f"degas/{AUSBOSS_UPSAMPLER}/" in resp.json()["detail"]
    assert "pick another style" in resp.json()["detail"]


def test_uncensored_without_the_heretic_is_refused(client: TestClient) -> None:
    services(client).db.replace_assets([FIBO2, FIBO2_VLM])
    resp = client.post(
        "/api/jobs", json={"spec": {**T2I, "params": {"prompt": "x", "uncensored": True}}}
    )
    assert resp.status_code == 400
    assert f"degas/{TEXT_ENCODER}/" in resp.json()["detail"]
    assert "Uncensored" in resp.json()["detail"]
