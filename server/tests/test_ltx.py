import asyncio
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from degas import media
from degas.families.base import SpecError
from degas.families.ltx2 import Ltx2
from degas.library import input_blobs, staged_blobs

from .conftest import LTX25, Harness
from .test_video import image, results, run, upload

LTX_SPEC: dict[str, Any] = {
    "family": "ltx2",
    "variant": "ltx25",
    "mode": "t2v",
    "model": {"path": LTX25["path"]},
    "params": {"prompt": "rain on a tin roof", "seed": 3},
}
LTX23 = {**LTX_SPEC, "variant": "ltx23", "model": {"path": "models/ltx2/ltx23/LTX-2.3"}}


def probe(client: TestClient, sha: str) -> dict[str, Any]:
    svc = client.app.state.services  # type: ignore[attr-defined]
    info = asyncio.run(media.probe(svc.blobs.path(sha)))
    assert info is not None
    return info


def test_schema_per_variant(client: TestClient) -> None:
    families = {f["id"]: f for f in client.get("/api/families").json()}
    assert [v["id"] for v in families["ltx2"]["variants"]] == ["ltx25", "ltx23-distilled", "ltx23"]
    assert families["ltx2"]["variants"][0]["modes"] == ["t2v", "i2v", "flf2v"]
    assert (families["ltx2"]["extendable"], families["wan22"]["extendable"]) == (True, True)
    assert families["sdxl"]["extendable"] is False

    distilled = client.get("/api/families/ltx2/schema?variant=ltx25&mode=t2v").json()
    props = distilled["properties"]
    assert props["upscale"]["default"] is False
    assert "steps" not in props
    assert "cfg" not in props
    assert props["num_frames"]["x-step"] == 8

    dev = Ltx2().param_schema("ltx23", "flf2v")["properties"]
    assert "upscale" not in dev
    assert (dev["steps"]["default"], dev["cfg"]["default"], dev["audio_cfg"]["default"]) == (
        30,
        3.0,
        7.0,
    )


def test_validate_snaps_frames_and_size() -> None:
    ltx = Ltx2()
    params = {"prompt": "x", "width": 1000, "height": 560, "num_frames": 100}
    spec = ltx.validate({**LTX_SPEC, "params": params})
    assert (spec["params"]["width"], spec["params"]["height"]) == (992, 544)
    assert spec["params"]["num_frames"] == 97  # 8k + 1
    upscaled = ltx.validate({**LTX_SPEC, "params": {**params, "upscale": True}})
    assert (upscaled["params"]["width"], upscaled["params"]["height"]) == (960, 512)

    with pytest.raises(SpecError, match="Unknown parameters: steps"):
        ltx.validate({**LTX_SPEC, "params": {"prompt": "x", "steps": 30}})
    with pytest.raises(SpecError, match="Unknown parameters: upscale"):
        ltx.validate({**LTX23, "params": {"prompt": "x", "upscale": True}})
    with pytest.raises(SpecError, match="control units"):
        ltx.validate({**LTX_SPEC, "control": [{}]})
    with pytest.raises(SpecError, match=r"isn't a LTX-2\.5 model"):
        ltx.validate({**LTX_SPEC, "model": {"path": "models/ltx2/ltx23/LTX-2.3"}})


def test_flf2v_needs_the_last_frame() -> None:
    source = "sha256:" + "a" * 64
    spec = {**LTX_SPEC, "mode": "flf2v", "inputs": {"source": source}}
    with pytest.raises(SpecError, match="Choose the last frame"):
        Ltx2().validate(spec)
    end = "sha256:" + "b" * 64
    inputs = Ltx2().validate({**spec, "inputs": {"source": source, "end": end}})["inputs"]
    assert inputs == {"source": source, "end": end, "fit": "crop"}
    # Other modes drop it.
    i2v = Ltx2().validate({**spec, "mode": "i2v", "inputs": {"source": source, "end": end}})
    assert "end" not in i2v["inputs"]


def test_text_to_video_has_sound(client: TestClient, harness: Harness) -> None:
    done = run(client, LTX_SPEC)
    assert done["spec"]["params"]["num_frames"] == 121
    (clip,) = results(client, done["id"])
    assert (clip["media_type"], clip["width"], clip["height"]) == ("video/mp4", 960, 544)
    assert probe(client, clip["blob_sha"])["audio"] is True


def test_first_and_last_frame(client: TestClient, harness: Harness) -> None:
    first = upload(client, image(600, 900))
    last = upload(client, image(960, 544))
    spec = {
        **LTX_SPEC,
        "mode": "flf2v",
        "inputs": {"source": f"sha256:{first['sha256']}", "end": f"sha256:{last['sha256']}"},
    }
    done = run(client, spec)
    inputs = done["spec"]["inputs"]
    # The portrait first frame is cropped to the clip's size; the last frame already fits.
    assert inputs["source"] in inputs["transforms"]
    assert inputs["end"] == f"sha256:{last['sha256']}"
    assert harness.ltx.images == [{"source": (960, 544), "end": (960, 544)}]
    assert last["sha256"] in staged_blobs(done["spec"])

    (clip,) = results(client, done["id"])
    kept = client.post(f"/api/results/{clip['id']}/save").json()
    assert kept["config"]["inputs"]["end"] == inputs["end"]
    assert last["sha256"] in input_blobs(kept["config"])

    gone = {**spec, "inputs": {**spec["inputs"], "end": "sha256:" + "0" * 64}}
    resp = client.post("/api/jobs", json={"spec": gone})
    assert resp.status_code == 400
    assert "last frame is no longer stored" in resp.json()["detail"]


def test_extend_keeps_the_variant_and_the_sound(client: TestClient, harness: Harness) -> None:
    first = run(client, LTX_SPEC)
    (clip,) = results(client, first["id"])
    spec = client.post(f"/api/results/{clip['id']}/extend").json()["spec"]
    assert (spec["family"], spec["variant"], spec["mode"]) == ("ltx2", "ltx25", "i2v")

    second = run(client, spec)
    _, chain = results(client, second["id"])
    assert chain["duration"] == pytest.approx(9 / 24, abs=0.05)
    assert probe(client, chain["blob_sha"])["audio"] is True


async def _clip(path: Path, seconds: float, *, sound: bool) -> Path:
    inputs = ["-f", "lavfi", "-i", f"testsrc=size=64x32:rate=24:duration={seconds}"]
    if sound:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=24000:duration={seconds}"]
    await media._run("ffmpeg", "-v", "error", "-y", *inputs, "-pix_fmt", "yuv420p", str(path))
    return path


@pytest.mark.parametrize(
    ("first_sound", "second_sound"), [(True, True), (True, False), (False, True), (False, False)]
)
def test_stitch_keeps_sound(tmp_path: Path, first_sound: bool, second_sound: bool) -> None:
    async def go() -> dict[str, Any] | None:
        first = await _clip(tmp_path / "a.mp4", 2, sound=first_sound)
        second = await _clip(tmp_path / "b.mp4", 1, sound=second_sound)
        out = tmp_path / "chain.mp4"
        out.write_bytes(await media.stitch(first, second, 24))
        return await media.probe(out)

    info = asyncio.run(go())
    assert info is not None
    assert info["audio"] is (first_sound or second_sound)
    assert info["frames"] == 48 + 23
