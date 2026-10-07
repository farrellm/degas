"""Civitai imports: link parsing, planning, streaming into Drive, backfill and the API."""

import asyncio
import copy
import hashlib
import io
import stat
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from degas.app import create_app
from degas.civitai.client import Civitai, CivitaiError, parse_ref, still_url
from degas.civitai.importer import CivitaiImportError, Importer
from degas.civitai.plan import PlanError, plan_import, slug
from degas.drive import parse_sidecar
from degas.families import FAMILIES
from degas.rclone import AsyncRclone, RcloneError

from .conftest import Harness

WEIGHTS = b"safetensors weights " * 1000
SHA = hashlib.sha256(WEIGHTS).hexdigest()
IMAGE_URL = "https://image.civitai.com/xG1/abc-123/original=true/1917130.jpeg"

# Shaped like GET /api/v1/model-versions/200002 (trimmed).
VERSION: dict[str, Any] = {
    "id": 200002,
    "modelId": 100001,
    "name": "v1.0",
    "baseModel": "SDXL 1.0",
    "trainedWords": ["detailed", " "],
    "model": {"name": "Test Style XL", "type": "LORA", "nsfw": False},
    "files": [
        {
            "id": 300003,
            "name": "test-style-xl.safetensors",
            "type": "Model",
            "sizeKB": len(WEIGHTS) / 1024,
            "pickleScanResult": "Success",
            "virusScanResult": "Success",
            "hashes": {"SHA256": SHA.upper()},
            "downloadUrl": "https://civitai.com/api/download/models/200002?fileId=300003",
            "primary": True,
        },
        {"id": 5, "name": "training.zip", "type": "Training Data", "sizeKB": 10},
    ],
    "images": [{"url": IMAGE_URL, "type": "image"}],
    "downloadUrl": "https://civitai.com/api/download/models/200002",
}


def version(**changes: Any) -> dict[str, Any]:
    v = copy.deepcopy(VERSION)
    v.update(changes)
    return v


def wan_file(name: str, fid: int, primary: bool = False) -> dict[str, Any]:
    return {**VERSION["files"][0], "id": fid, "name": name, "primary": primary}


def plan(v: dict[str, Any], **options: Any) -> Any:
    return plan_import(v, families=set(FAMILIES), **options)


# -- links ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("200002", (None, 200002)),
        ("https://civitai.com/models/100001/test-style-xl", (100001, None)),
        ("https://civitai.red/models/100001?modelVersionId=200002", (100001, 200002)),
        ("www.civitai.com/models/100001", (100001, None)),
        ("https://civitai.com/api/download/models/200002?fileId=300003", (None, 200002)),
        ("https://civitai.com/api/v1/model-versions/200002", (None, 200002)),
        ("urn:air:sdxl:lora:civitai:100001@200002", (100001, 200002)),
    ],
)
def test_parse_ref(ref: str, expected: tuple[int | None, int | None]) -> None:
    assert parse_ref(ref) == expected


@pytest.mark.parametrize("ref", ["https://example.com/models/1", "https://civitai.com/user/x"])
def test_parse_ref_rejects(ref: str) -> None:
    with pytest.raises(CivitaiError):
        parse_ref(ref)


def test_still_url() -> None:
    video = "https://image.civitai.com/xG1/f3c1/original=true/135268953.mp4"
    assert still_url(video) == (
        "https://image.civitai.com/xG1/f3c1/anim=false,transcode=true,width=768/135268953.jpeg"
    )
    assert still_url("https://example.com/a.png") == "https://example.com/a.png"


# -- planning ------------------------------------------------------------------------------


def test_plan_sdxl() -> None:
    p = plan(version())
    assert p.family == "sdxl"
    assert [f.path for f in p.files] == ["loras/sdxl/test_style_xl_v1.0.safetensors"]
    assert p.files[0].sha256 == SHA
    assert p.files[0].size == len(WEIGHTS)
    assert p.preview_url
    assert "anim=false" in p.preview_url
    sidecar = parse_sidecar(p.sidecar())
    assert sidecar == {
        "label": "Test Style XL",
        "trigger_words": ["detailed"],
        "default_weight": 0.8,
        "notes": "SDXL 1.0 LoRA from Civitai (v1.0)",
        "source": "https://civitai.com/models/100001?modelVersionId=200002",
    }


def test_plan_notes_the_base_of_an_sdxl_fine_tune() -> None:
    p = plan(version(baseModel="Pony"), name="derived", weight=0.6)
    assert p.files[0].path == "loras/sdxl/derived.safetensors"
    assert "Pony LoRA" in p.sidecar()
    assert "default_weight: 0.6" in p.sidecar()


def test_plan_refuses_an_unknown_base_unless_told_the_family() -> None:
    with pytest.raises(PlanError, match=r"for SD 1\.5, which Degas doesn't run"):
        plan(version(baseModel="SD 1.5"))
    p = plan(version(baseModel="SD 1.5"), family="sdxl")
    assert p.family == "sdxl"
    assert p.warnings == ["It's for SD 1.5; importing it for sdxl anyway"]
    with pytest.raises(PlanError, match="Unknown family"):
        plan(version(), family="sd15")


def test_plan_refuses_what_isnt_a_safe_lora() -> None:
    with pytest.raises(PlanError, match="is a Checkpoint, not a LoRA"):
        plan(version(model={"name": "X", "type": "Checkpoint"}))
    with pytest.raises(PlanError, match=r"No \.safetensors .*training\.zip"):
        plan(version(files=[VERSION["files"][1]]))
    flagged = {**VERSION["files"][0], "virusScanResult": "Danger"}
    with pytest.raises(PlanError, match="scan flagged"):
        plan(version(files=[flagged]))
    with pytest.raises(PlanError, match="no SHA-256"):
        plan(version(files=[{**VERSION["files"][0], "hashes": {}}]))


def test_plan_wan_pair_in_one_version() -> None:
    v = version(
        baseModel="Wan Video 2.2 I2V-A14B",
        model={"name": "Side Step", "type": "LORA"},
        files=[wan_file("step_i2v-LOW.safetensors", 2), wan_file("step_i2v-high.safetensors", 1)],
    )
    p = plan(v)
    assert [(f.civitai_name, f.path, f.half) for f in p.files] == [
        ("step_i2v-high.safetensors", "loras/wan22/side_step_high_noise.safetensors", "high"),
        ("step_i2v-LOW.safetensors", "loras/wan22/side_step_low_noise.safetensors", "low"),
    ]
    assert p.variants == ["i2v-a14b"]
    assert parse_sidecar(p.sidecar())["variants"] == ["i2v-a14b"]
    assert not p.warnings


def test_plan_wan_half_from_the_version_name() -> None:
    v = version(
        name="v1.0 Low Noise",
        baseModel="Wan Video 2.2 T2V-A14B",
        model={"name": "Char", "type": "LORA"},
        files=[wan_file("char_wan22.safetensors", 1)],
    )
    p = plan(v)
    assert [f.path for f in p.files] == ["loras/wan22/char_low_noise.safetensors"]
    assert "import the high-noise one" in p.warnings[0]
    with pytest.raises(PlanError, match="high- or low-noise"):
        plan({**v, "name": "v1.0"})


def tilt(vid: int, name: str, file: str, sha: str = SHA) -> dict[str, Any]:
    """Shaped like a version of model 900000, whose Wan halves are separate versions."""
    f = {
        **wan_file(f"{file}.safetensors", vid, primary=True),
        "hashes": {"SHA256": sha},
        "downloadUrl": f"https://civitai.com/api/download/models/{vid}?fileId={vid}",
    }
    return version(
        id=vid,
        modelId=900000,
        name=name,
        baseModel="Wan Video 2.2 I2V-A14B",
        model={"name": "Camera Tilt-down Slow-慢速", "type": "LORA"},
        files=[f],
    )


TILT = [
    tilt(900080, "LownoiseV2.0", "V2_Camera Tilt-down Slow-慢速_low_e28", "b" * 64),
    tilt(900023, "highnoiseV2.0", "V2_high_Camera Tilt-down Slow-慢速_e48", "a" * 64),
    tilt(800026, "LownoiseV1.0", "Lownoise_Camera Tilt-down Slow_慢速", "d" * 64),
    tilt(800020, "highnoiseV1.0", "Highnoise_Camera Tilt-down Slow_慢速", "c" * 64),
]


def test_plan_wan_half_from_a_noise_version_name() -> None:
    p = plan({**TILT[0], "files": [wan_file("tilt.safetensors", 1)]})
    assert [f.half for f in p.files] == ["low"]
    p = plan({**TILT[1], "files": [wan_file("tilt.safetensors", 1)]})
    assert [f.half for f in p.files] == ["high"]


@pytest.mark.parametrize("linked", [TILT[1], TILT[0]])
def test_plan_wan_pair_from_two_versions(linked: dict[str, Any]) -> None:
    p = plan(linked, siblings=TILT)
    folder = "loras/wan22/camera_tilt_down_slow"
    assert [(f.path, f.sha256, f.half, f.from_sibling) for f in p.files] == [
        (f"{folder}_high_noise.safetensors", "a" * 64, "high", linked is TILT[0]),
        (f"{folder}_low_noise.safetensors", "b" * 64, "low", linked is TILT[1]),
    ]
    assert [f.url for f in p.files] == [
        "https://civitai.com/api/download/models/900023?fileId=900023",
        "https://civitai.com/api/download/models/900080?fileId=900080",
    ]
    assert not p.warnings
    other = TILT[0] if linked is TILT[1] else TILT[1]
    notes = parse_sidecar(p.sidecar())["notes"]
    assert notes.endswith(f"({linked['name']} + {other['name']})")
    assert p.source.endswith(f"modelVersionId={linked['id']}")


def test_plan_wan_pair_needs_one_matching_version() -> None:
    twice = [*TILT, {**TILT[0], "id": 1}]
    other_base = [TILT[1], {**TILT[0], "baseModel": "Wan Video 2.2 T2V-A14B"}]
    for siblings in (twice, other_base, TILT[1:]):
        p = plan(TILT[1], siblings=siblings)
        assert [f.half for f in p.files] == ["high"]
        assert "import the low-noise one" in p.warnings[0]


def test_plan_wan_5b_is_a_single_file() -> None:
    v = version(baseModel="Wan Video 2.2 TI2V-5B", files=[wan_file("fire_5b.safetensors", 1)])
    p = plan(v)
    assert len(p.files) == 1
    assert p.files[0].half is None
    assert p.variants == ["ti2v-5b"]


def test_plan_wan21_i2v_is_a_single_file() -> None:
    v = version(baseModel="Wan Video 14B i2v 720p", files=[wan_file("spin_i2v.safetensors", 1)])
    p = plan(v)
    assert [(f.path.split("/")[1], f.half) for f in p.files] == [("wan22", None)]
    assert p.variants == ["wan21-i2v-14b", "wan21-flf2v-14b"]


def test_plan_ltx_loras_name_their_variants() -> None:
    v = version(baseModel="LTXV 2.3", files=[wan_file("dolly_ltx23.safetensors", 1)])
    p = plan(v)
    assert [(f.path.split("/")[1], f.half) for f in p.files] == [("ltx2", None)]
    assert p.variants == ["ltx23", "ltx23-distilled"]
    assert plan({**v, "baseModel": "LTXV 2.5"}).variants == ["ltx25"]
    with pytest.raises(PlanError, match="which Degas doesn't run"):
        plan({**v, "baseModel": "LTXV2"})


def test_slug() -> None:
    assert slug("Tést Style XL") == "test_style_xl"
    assert slug("Slap (and Self Slap) - Wan 2.2") == "slap_and_self_slap_wan_2.2"
    assert slug("娃娃") == ""
    unnamed = plan(version(model={"name": "娃娃", "type": "LORA"}))
    assert unnamed.files[0].path == "loras/sdxl/civitai_200002.safetensors"


def test_sidecar_keeps_source() -> None:
    assert parse_sidecar("source: https://civitai.com/models/1\n") == {
        "source": "https://civitai.com/models/1"
    }


# -- importing -----------------------------------------------------------------------------


class FakeRemote:
    """rclone over a dict of Drive paths."""

    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.calls: list[tuple[str, ...]] = []
        self.corrupt = False

    async def run(self, *args: str, stdin: bytes | None = None) -> str:
        self.calls.append(args)
        match args[0]:
            case "rcat":
                self.files[args[1]] = stdin or b""
            case "md5sum":
                data = self.files[args[1]] + (b"x" if self.corrupt else b"")
                return f"{hashlib.md5(data).hexdigest()}  {args[1].rsplit('/', 1)[-1]}\n"
            case "deletefile":
                del self.files[args[1]]
            case "lsf":
                folder = args[-1] + "/"
                return "".join(
                    f"{p.removeprefix(folder)}\n"
                    for p in sorted(self.files)
                    if p.startswith(folder) and "/" not in p.removeprefix(folder)
                )
        return ""

    async def rcat(self, target: str, chunks: AsyncIterator[bytes], size: int | None) -> None:
        self.calls.append(("rcat", target))
        self.files[target] = b"".join([c async for c in chunks])


def jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (1200, 900), "red").save(buf, format="JPEG")
    return buf.getvalue()


class FakeCivitai:
    """Civitai's API, its download redirect and the image CDN."""

    def __init__(self, versions: list[dict[str, Any]] | None = None) -> None:
        self.versions = {v["id"]: v for v in versions or [VERSION]}
        self.weights = WEIGHTS
        self.download_status = 200
        self.auth: dict[str, str | None] = {}

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        url = request.url
        self.auth[url.host] = request.headers.get("authorization")
        path = url.path
        if path.startswith("/api/v1/model-versions/by-hash/"):
            sha = path.rsplit("/", 1)[1]
            found = [
                v
                for v in self.versions.values()
                if v["files"][0]["hashes"]["SHA256"].lower() == sha
            ]
            return httpx2.Response(200, json=found[0]) if found else httpx2.Response(404)
        if path.startswith("/api/v1/model-versions/"):
            v = self.versions.get(int(path.rsplit("/", 1)[1]))
            return httpx2.Response(200, json=v) if v else httpx2.Response(404)
        if path.startswith("/api/v1/models/"):
            newest = sorted(self.versions.values(), key=lambda v: v["id"], reverse=True)
            return httpx2.Response(200, json={"modelVersions": newest})
        if path.startswith("/api/download/models/"):
            if self.download_status != 200:
                return httpx2.Response(self.download_status)
            return httpx2.Response(307, headers={"location": "https://storage.example/w?sig=1"})
        if url.host == "storage.example":
            return httpx2.Response(200, content=self.weights)
        if url.host == "image.civitai.com":
            return httpx2.Response(200, content=jpeg())
        return httpx2.Response(404)


def make_importer(fake: FakeCivitai, remote: FakeRemote, token: Path | None = None) -> Importer:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(fake))
    return Importer(Civitai(token, http=http), remote, "gdrive:", "degas", set(FAMILIES))


async def test_import(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("secret\n")
    fake, remote = FakeCivitai(), FakeRemote()
    imp = make_importer(fake, remote, token)
    p = await imp.plan("https://civitai.com/models/100001", [])
    seen: list[str] = []
    paths = await imp.run(p, lambda stage, done, total: seen.append(stage))
    assert paths == ["loras/sdxl/test_style_xl_v1.0.safetensors"]
    folder = "gdrive:degas/loras/sdxl"
    assert remote.calls[0] == ("mkdir", folder)
    assert remote.files[f"{folder}/test_style_xl_v1.0.safetensors"] == WEIGHTS
    assert "Test Style XL" in remote.files[f"{folder}/test_style_xl_v1.0.yaml"].decode()
    preview = remote.files[f"{folder}/test_style_xl_v1.0.jpg"]
    with Image.open(io.BytesIO(preview)) as im:
        assert max(im.size) == 768
    assert seen[0] == "download"
    assert seen[-2:] == ["sidecar", "preview"]
    # The token goes to Civitai, never to the storage host a download redirects to.
    assert fake.auth["civitai.com"] == "Bearer secret"
    assert fake.auth["storage.example"] is None


async def test_import_wan_pair_names_the_preview_after_the_high_half() -> None:
    v = version(
        baseModel="Wan Video 2.2 I2V-A14B",
        files=[wan_file("x_high.safetensors", 1), wan_file("x_low.safetensors", 2)],
    )
    remote = FakeRemote()
    imp = make_importer(FakeCivitai([v]), remote)
    await imp.run(await imp.plan("200002", []))
    folder = "gdrive:degas/loras/wan22"
    assert sorted(remote.files) == sorted(
        f"{folder}/test_style_xl_{name}"
        for name in ("high_noise.safetensors", "low_noise.safetensors", "high_noise.yaml",
                     "low_noise.yaml", "high_noise.jpg")
    )  # fmt: skip


async def test_import_wan_pair_from_two_versions() -> None:
    remote = FakeRemote()
    served = [  # every download is WEIGHTS
        {**v, "files": [{**v["files"][0], "hashes": {"SHA256": SHA}}]} for v in TILT
    ]
    imp = make_importer(FakeCivitai(served), remote)
    p = await imp.plan("https://civitai.com/models/900000?modelVersionId=900023", [])
    await imp.run(p)
    folder = "gdrive:degas/loras/wan22/camera_tilt_down_slow"
    assert sorted(remote.files) == [
        f"{folder}_{name}"
        for name in ("high_noise.jpg", "high_noise.safetensors", "high_noise.yaml",
                     "low_noise.safetensors", "low_noise.yaml")
    ]  # fmt: skip


async def test_import_wan_half_pairs_with_the_one_in_drive() -> None:
    imp = make_importer(FakeCivitai(TILT), FakeRemote())
    index = [{"path": "loras/wan22/tilt_high_noise.safetensors", "sha256": "a" * 64}]
    p = await imp.plan("900080", index)
    assert [f.path for f in p.files] == ["loras/wan22/tilt_low_noise.safetensors"]
    assert p.warnings == [
        "The high-noise half is already in Drive as loras/wan22/tilt_high_noise.safetensors"
    ]
    assert "+" not in p.sidecar()
    # A name given keeps its name, and linking the half that's in Drive is still refused.
    p = await imp.plan("900080", index, name="mine")
    assert [f.path for f in p.files] == ["loras/wan22/mine_low_noise.safetensors"]
    with pytest.raises(PlanError, match="already in Drive"):
        await imp.plan("900023", index)


async def test_import_refuses_duplicates_unless_forced() -> None:
    imp = make_importer(FakeCivitai(), FakeRemote())
    index = [{"path": "loras/sdxl/mine.safetensors", "sha256": SHA}]
    with pytest.raises(PlanError, match=r"already in Drive as loras/sdxl/mine\.safetensors"):
        await imp.plan("200002", index)
    same_name = [{"path": "loras/sdxl/test_style_xl_v1.0.safetensors", "sha256": None}]
    with pytest.raises(PlanError, match="already exists"):
        await imp.plan("200002", same_name)
    p = await imp.plan("200002", index, force=True)
    warning = (
        "Going ahead anyway: test-style-xl.safetensors is already in Drive"
        " as loras/sdxl/mine.safetensors"
    )
    assert p.warnings == [warning]


async def test_import_deletes_a_file_that_fails_its_checks() -> None:
    fake, remote = FakeCivitai(), FakeRemote()
    imp = make_importer(fake, remote)
    p = await imp.plan("200002", [])
    fake.weights = WEIGHTS[:-1] + b"!"
    with pytest.raises(CivitaiImportError, match="SHA-256"):
        await imp.run(p)
    assert remote.files == {}
    assert ("deletefile", "gdrive:degas/loras/sdxl/test_style_xl_v1.0.safetensors") in (
        remote.calls
    )
    fake.weights = WEIGHTS
    remote.corrupt = True
    with pytest.raises(CivitaiImportError, match="Upload check failed"):
        await imp.run(p)
    assert remote.files == {}


async def test_download_refused_without_a_token(tmp_path: Path) -> None:
    fake = FakeCivitai()
    fake.download_status = 401
    imp = make_importer(fake, FakeRemote(), tmp_path / "missing")
    p = await imp.plan("200002", [])
    with pytest.raises(CivitaiError, match=r"\(401\); put an API token in .*missing"):
        await imp.run(p)


async def test_backfill() -> None:
    remote = FakeRemote()
    imp = make_importer(FakeCivitai(), remote)
    asset = {"path": "loras/sdxl/sub/mine.safetensors", "family": "sdxl", "sha256": SHA}
    p = await imp.backfill(asset)
    assert p is not None
    assert "label: Test Style XL" in remote.files["gdrive:degas/loras/sdxl/sub/mine.yaml"].decode()
    assert "gdrive:degas/loras/sdxl/sub/mine.jpg" in remote.files
    remote.files.clear()
    await imp.backfill({**asset, "preview_thumb": "sha256:x"})  # keeps its own preview
    assert list(remote.files) == ["gdrive:degas/loras/sdxl/sub/mine.yaml"]
    assert await imp.backfill({**asset, "sha256": "0" * 64}) is None
    assert await imp.backfill({**asset, "sha256": None}) is None


# -- rclone --------------------------------------------------------------------------------

# Stand-in for rclone: `rcat [--size N] DEST` writes stdin to DEST; anything else fails.
FAKE_RCLONE = """#!/bin/sh
if [ "$1" = rcat ]; then
    eval "dest=\\${$#}"
    cat > "$dest"
else
    echo "no $1 here" >&2
    exit 3
fi
"""


async def test_async_rclone(tmp_path: Path) -> None:
    binary = tmp_path / "rclone"
    binary.write_text(FAKE_RCLONE)
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    rclone = AsyncRclone(str(binary))

    async def chunks() -> AsyncIterator[bytes]:
        yield b"abc"
        yield b"def"

    await rclone.rcat(str(tmp_path / "out"), chunks(), 6)
    assert (tmp_path / "out").read_bytes() == b"abcdef"
    with pytest.raises(RcloneError, match="no lsf here"):
        await rclone.run("lsf", "x")

    async def failing() -> AsyncIterator[bytes]:
        yield b"abc"
        raise OSError("connection reset")

    with pytest.raises(OSError, match="connection reset"):
        await rclone.rcat(str(tmp_path / "partial"), failing(), None)


# -- API -----------------------------------------------------------------------------------


@pytest.fixture
def civitai_client(harness: Harness) -> Iterator[tuple[TestClient, FakeCivitai, FakeRemote]]:
    fake, remote = FakeCivitai(), FakeRemote()

    def build(config: Any) -> Any:
        svc = harness.build(config)
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(fake))
        svc.imports.importer = Importer(
            Civitai(None, http=http), remote, "gdrive:", "degas", set(FAMILIES)
        )
        svc.assets.remote = remote
        return svc

    with TestClient(create_app(harness.config, build)) as client:
        yield client, fake, remote


def test_api_plan_and_import(civitai_client: tuple[TestClient, FakeCivitai, FakeRemote]) -> None:
    client, _fake, remote = civitai_client
    resp = client.post("/api/civitai/plan", json={"url": "https://civitai.com/models/100001"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["family"] == "sdxl"
    assert body["files"][0]["path"] == "loras/sdxl/test_style_xl_v1.0.safetensors"
    assert "Test Style XL" in body["sidecar"]

    bad = client.post("/api/civitai/plan", json={"url": "https://example.com/x"})
    assert bad.status_code == 422
    assert "Not a Civitai link" in bad.json()["detail"]

    resp = client.post("/api/civitai/import", json={"url": "200002", "weight": 0.5})
    assert resp.status_code == 202, resp.text
    assert resp.json()["state"] == "copying"
    deadline = time.monotonic() + 5
    state = None
    while time.monotonic() < deadline:
        state = client.get("/api/civitai/import").json()
        if state["state"] in ("done", "failed"):
            break
        time.sleep(0.02)
    assert state is not None
    assert state["state"] == "done", state
    assert state["paths"] == ["loras/sdxl/test_style_xl_v1.0.safetensors"]
    # There's no Drive in tests, so the rescan afterwards fails and says so.
    assert state["warnings"] == ["Imported, but Drive couldn't be rescanned"]
    sidecar = yaml.safe_load(remote.files["gdrive:degas/loras/sdxl/test_style_xl_v1.0.yaml"])
    assert sidecar["default_weight"] == 0.5


def test_api_import_refuses_while_one_runs(
    civitai_client: tuple[TestClient, FakeCivitai, FakeRemote],
) -> None:
    client, _fake, remote = civitai_client
    gate = asyncio.Event()
    original = remote.rcat

    async def slow_rcat(target: str, chunks: AsyncIterator[bytes], size: int | None) -> None:
        await gate.wait()
        await original(target, chunks, size)

    remote.rcat = slow_rcat  # type: ignore[method-assign]
    assert client.post("/api/civitai/import", json={"url": "200002"}).status_code == 202
    resp = client.post("/api/civitai/import", json={"url": "200002", "name": "other"})
    assert resp.status_code == 409


def test_api_delete_lora(civitai_client: tuple[TestClient, FakeCivitai, FakeRemote]) -> None:
    client, _fake, remote = civitai_client
    svc = client.app.state.services  # type: ignore[attr-defined]
    base = "gdrive:degas/loras/wan22"
    for name in [
        "motion_high_noise.safetensors",
        "motion_high_noise.yaml",
        "motion_high_noise.jpg",
        "motion_low_noise.safetensors",
        "motion_low_noise.yaml",
        "motion_v2.safetensors",
        "motion_v2.jpg",
    ]:
        remote.files[f"{base}/{name}"] = b"x"
    preview = svc.blobs.put(jpeg(), "image/jpeg")
    lora = {"family": "wan22", "kind": "lora", "size": 1}
    svc.db.replace_assets(
        [
            {
                **lora,
                "path": "loras/wan22/motion_high_noise.safetensors",
                "drive_file_id": "h",
                "preview_thumb": preview,
            },
            {**lora, "path": "loras/wan22/motion_low_noise.safetensors", "drive_file_id": "l"},
            {**lora, "path": "loras/wan22/motion_v2.safetensors", "drive_file_id": "v"},
            {
                "path": "models/sdxl/base.safetensors",
                "family": "sdxl",
                "kind": "model",
                "drive_file_id": "m",
            },
        ]
    )

    resp = client.delete(
        "/api/assets",
        params=[
            ("path", "loras/wan22/motion_high_noise.safetensors"),
            ("path", "loras/wan22/motion_low_noise.safetensors"),
        ],
    )
    assert resp.status_code == 200, resp.text
    # Both halves go, with their sidecars and preview; a LoRA with a similar name stays.
    assert sorted(remote.files) == [f"{base}/motion_v2.jpg", f"{base}/motion_v2.safetensors"]
    assert [a["path"] for a in svc.db.list_assets(kind="lora")] == [
        "loras/wan22/motion_v2.safetensors"
    ]
    assert svc.blobs.path(preview) is None

    model = client.delete("/api/assets", params={"path": "models/sdxl/base.safetensors"})
    assert model.status_code == 400
    missing = client.delete("/api/assets", params={"path": "loras/wan22/gone.safetensors"})
    assert missing.status_code == 404
