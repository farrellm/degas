"""Hugging Face imports: link parsing, planning, and streaming into Drive."""

import copy
import hashlib
import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from degas import media
from degas.app import create_app
from degas.civitai.client import Civitai
from degas.civitai.huggingface import HfRef, HuggingFace, HuggingFaceError, parse_hf_ref
from degas.civitai.importer import CivitaiImportError, Importer
from degas.civitai.plan import PlanError, plan_hf_import, readme_triggers
from degas.families import FAMILIES

from .conftest import Harness
from .test_civitai import FakeRemote, jpeg

WEIGHTS = b"lora weights " * 1000
SHA = hashlib.sha256(WEIGHTS).hexdigest()
COMMIT = "81681346c128adc529ef4249024024dd1f6b6b00"
FILE = "LTX-2.3 - Tin Robot v1.1.safetensors"
LINK = "https://huggingface.co/someone/LTX-2.3-TinRobot/blob/main/LTX-2.3%20-%20Tin%20Robot%20v1.1.safetensors"
README = """---
license: apache-2.0
---

Triggers: 2d animation, Tin the robot

<video controls width="300"></video>
"""


def lfs(name: str, data: bytes = WEIGHTS) -> dict[str, Any]:
    sha = hashlib.sha256(data).hexdigest()
    return {"rfilename": name, "size": len(data), "lfs": {"sha256": sha, "size": len(data)}}


# Shaped like GET /api/models/someone/LTX-2.3-TinRobot/revision/main?blobs=true (trimmed).
INFO: dict[str, Any] = {
    "id": "someone/LTX-2.3-TinRobot",
    "sha": COMMIT,
    "cardData": {"license": "apache-2.0"},
    "siblings": [
        {"rfilename": ".gitattributes", "size": 1696},
        lfs(FILE),
        lfs("LTX_2.3_t2v_01506_.mp4", b"m" * 900),
        lfs("LTX_2.3_t2v_01507_.mp4", b"m" * 800),
        {"rfilename": "README.md", "size": len(README)},
    ],
}


def info(**changes: Any) -> dict[str, Any]:
    i = copy.deepcopy(INFO)
    i.update(changes)
    return i


def plan(i: dict[str, Any], ref: HfRef | str = LINK, **options: Any) -> Any:
    if isinstance(ref, str):
        ref = parse_hf_ref(ref)
    return plan_hf_import(i, ref, families=set(FAMILIES), **options)


# -- links ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (LINK, HfRef("someone/LTX-2.3-TinRobot", "main", FILE)),
        (
            "huggingface.co/a/b/resolve/v2/sub/x%20y.safetensors?download=true",
            HfRef("a/b", "v2", "sub/x y.safetensors"),
        ),
        ("https://hf.co/a/b", HfRef("a/b", "main")),
        ("https://huggingface.co/a/b/tree/main/loras", HfRef("a/b", "main", folder="loras")),
        ("https://huggingface.co/a/b/tree/refs%2Fpr%2F1", HfRef("a/b", "refs/pr/1")),
    ],
)
def test_parse_hf_ref(ref: str, expected: HfRef) -> None:
    assert parse_hf_ref(ref) == expected


@pytest.mark.parametrize(
    "ref",
    [
        "https://civitai.com/models/1",
        "https://huggingface.co/datasets/a/b",
        "https://huggingface.co/a",
        "https://huggingface.co/a/b/blob/main",
        "https://huggingface.co/a/b/commits/main",
    ],
)
def test_parse_hf_ref_rejects(ref: str) -> None:
    with pytest.raises(HuggingFaceError):
        parse_hf_ref(ref)


# -- planning ------------------------------------------------------------------------------


def test_plan_ltx_from_its_name() -> None:
    p = plan(INFO, readme=README)
    assert (p.family, p.base_model, p.variants) == ("ltx2", "LTX-2.3", ["ltx23", "ltx23-distilled"])
    assert p.origin == "huggingface"
    assert p.label == "LTX-2.3 - Tin Robot v1.1"
    assert p.trigger_words == ["2d animation", "Tin the robot"]
    assert p.warnings == []
    (f,) = p.files
    assert f.path == "loras/ltx2/ltx_2.3_tin_robot_v1.1.safetensors"
    assert (f.sha256, f.size) == (SHA, len(WEIGHTS))
    # Pinned to the commit, so the file can't change under its hash.
    assert f.url == (
        f"https://huggingface.co/someone/LTX-2.3-TinRobot/resolve/{COMMIT}/"
        "LTX-2.3%20-%20Tin%20Robot%20v1.1.safetensors"
    )
    # The smallest example video.
    assert p.preview_url is not None
    assert p.preview_url.endswith("LTX_2.3_t2v_01507_.mp4")
    sidecar = yaml.safe_load(p.sidecar())
    assert sidecar["notes"] == "LTX-2.3 LoRA from Hugging Face (someone/LTX-2.3-TinRobot)"
    assert sidecar["source"] == LINK
    assert sidecar["variants"] == ["ltx23", "ltx23-distilled"]


def test_plan_prefers_an_image_for_the_preview() -> None:
    i = info(siblings=[*INFO["siblings"], lfs("sample.png", b"p")])
    assert plan(i).preview_url.endswith("/sample.png")


def test_plan_family_from_the_card() -> None:
    i = info(siblings=[lfs("my_style.safetensors")])
    i["cardData"]["base_model"] = ["black-forest-labs/FLUX.1-dev"]
    p = plan(i, "https://huggingface.co/someone/style")
    assert (p.family, p.base_model, p.variants) == ("flux1", "FLUX.1", None)
    assert p.files[0].path == "loras/flux1/my_style.safetensors"
    assert p.preview_url is None


def test_plan_falls_back_to_the_hint() -> None:
    i = info(siblings=[lfs("my_style.safetensors")])
    with pytest.raises(PlanError, match="choose its family"):
        plan(i, "https://huggingface.co/someone/style")
    p = plan(i, "https://huggingface.co/someone/style", hint="sdxl")
    assert (p.family, p.base_model) == ("sdxl", "")
    assert p.warnings == ["Its page doesn't say which model it's for; importing it for sdxl"]
    assert "LoRA from Hugging Face" in yaml.safe_load(p.sidecar())["notes"]


def test_plan_family_override() -> None:
    p = plan(INFO, family="wan22", hint="sdxl")
    assert (p.family, p.variants) == ("wan22", None)
    assert p.warnings == ["It looks like it's for LTX-2.3; importing it for wan22"]
    with pytest.raises(PlanError, match="Unknown family"):
        plan(INFO, family="nope")


def test_plan_instance_prompt_beats_the_readme() -> None:
    i = info()
    i["cardData"]["instance_prompt"] = "bndr robot"
    assert plan(i, readme=README).trigger_words == ["bndr robot"]


@pytest.mark.parametrize(
    ("readme", "words"),
    [
        ("Triggers: a, b", ["a", "b"]),
        ("## Trigger words\n\n**Trigger words:** `ohwx`, `man`.", ["ohwx", "man"]),
        ("- Trigger: xyz style", ["xyz style"]),
        ("Use it with prompts about triggers", []),
    ],
)
def test_readme_triggers(readme: str, words: list[str]) -> None:
    assert readme_triggers(readme) == words


def test_plan_picks_the_repo_s_one_lora_or_refuses() -> None:
    p = plan(INFO, "https://huggingface.co/someone/LTX-2.3-TinRobot")
    assert p.files[0].path == "loras/ltx2/ltx_2.3_tin_robot_v1.1.safetensors"
    two = info(siblings=[*INFO["siblings"], lfs("other.safetensors", b"x")])
    with pytest.raises(PlanError, match="More than one LoRA"):
        plan(two, "https://huggingface.co/someone/LTX-2.3-TinRobot")
    with pytest.raises(PlanError, match="has no file"):
        plan(INFO, "https://huggingface.co/someone/LTX-2.3-TinRobot/blob/main/x.safetensors")
    with pytest.raises(PlanError, match=r"isn't a \.safetensors"):
        plan(INFO, "https://huggingface.co/someone/LTX-2.3-TinRobot/blob/main/README.md")
    no_hash = info(siblings=[{"rfilename": FILE, "size": 10}])
    with pytest.raises(PlanError, match="no SHA-256"):
        plan(no_hash)


WAN = info(
    id="someone/Wan2.2-Loras",
    siblings=[
        lfs("wan2.2_i2v_A14b_high_noise_lora_v1.safetensors", b"ih"),
        lfs("wan2.2_i2v_A14b_low_noise_lora_v1.safetensors", b"il"),
        lfs("wan2.2_t2v_A14b_high_noise_lora_v2.safetensors", b"th"),
        lfs("wan2.2_t2v_A14b_low_noise_lora_v2.safetensors", b"tl"),
    ],
)
WAN_REPO = "https://huggingface.co/someone/Wan2.2-Loras"


def test_plan_wan_pair_from_one_half_s_link() -> None:
    p = plan(WAN, f"{WAN_REPO}/blob/main/wan2.2_t2v_A14b_low_noise_lora_v2.safetensors")
    assert (p.family, p.variants) == ("wan22", ["t2v-a14b"])
    assert p.label == "wan2.2_t2v_A14b_lora_v2"
    assert [(f.half, f.civitai_name, f.path) for f in p.files] == [
        (
            "high",
            "wan2.2_t2v_A14b_high_noise_lora_v2.safetensors",
            "loras/wan22/wan2.2_t2v_a14b_lora_v2_high_noise.safetensors",
        ),
        (
            "low",
            "wan2.2_t2v_A14b_low_noise_lora_v2.safetensors",
            "loras/wan22/wan2.2_t2v_a14b_lora_v2_low_noise.safetensors",
        ),
    ]
    with pytest.raises(PlanError, match="More than one LoRA"):
        plan(WAN, WAN_REPO)


def test_plan_wan_half_alone() -> None:
    i = info(siblings=[lfs("wan22_i2v_style_HIGH.safetensors")])
    p = plan(i, "https://huggingface.co/a/wan22-i2v-style")
    assert [f.half for f in p.files] == ["high"]
    assert "only the high-noise half" in p.warnings[0]


# -- importing -----------------------------------------------------------------------------


async def mp4(tmp_path: Path) -> bytes:
    path = tmp_path / "x.mp4"
    lavfi = ["-f", "lavfi", "-i", "testsrc=size=64x32:rate=24:duration=0.2"]
    await media._run("ffmpeg", "-v", "error", "-y", *lavfi, "-pix_fmt", "yuv420p", str(path))
    return path.read_bytes()


class FakeHub:
    """Hugging Face's API, its download redirect and the CDN."""

    def __init__(self, video: bytes = b"") -> None:
        self.info = info(
            siblings=[
                *INFO["siblings"][:2],
                lfs("LTX_2.3_t2v_01506_.mp4", video),
                INFO["siblings"][-1],
            ]
        )
        self.weights = WEIGHTS
        self.video = video
        self.status = 200
        self.auth: dict[str, str | None] = {}

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        url = request.url
        self.auth[url.host] = request.headers.get("authorization")
        path = url.path
        if path.startswith("/api/models/someone/LTX-2.3-TinRobot/revision/"):
            return httpx2.Response(200, json=self.info)
        if path.startswith("/api/models/"):
            return httpx2.Response(401)
        if "/resolve/" in path:
            if self.status != 200:
                return httpx2.Response(self.status)
            name = path.rsplit("/", 1)[1]
            return httpx2.Response(302, headers={"location": f"https://cdn.example/{name}"})
        if url.host == "cdn.example":
            if path.endswith(".safetensors"):
                return httpx2.Response(200, content=self.weights)
            if path.endswith(".mp4"):
                return httpx2.Response(200, content=self.video)
            if path.endswith(".png"):
                return httpx2.Response(200, content=jpeg())
            if path.endswith("README.md"):
                return httpx2.Response(200, content=README.encode())
        return httpx2.Response(404)


def make_importer(hub: FakeHub, remote: FakeRemote, token: Path | None = None) -> Importer:
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(hub))
    civitai = Civitai(None, http=httpx2.AsyncClient(transport=httpx2.MockTransport(hub)))
    hf = HuggingFace(token, http=http)
    return Importer(civitai, remote, "gdrive:", "degas", set(FAMILIES), hf)


async def test_import(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("hf_secret\n")
    hub, remote = FakeHub(await mp4(tmp_path)), FakeRemote()
    imp = make_importer(hub, remote, token)
    p = await imp.plan(LINK, [])
    assert p.trigger_words == ["2d animation", "Tin the robot"]
    paths = await imp.run(p)
    assert paths == ["loras/ltx2/ltx_2.3_tin_robot_v1.1.safetensors"]
    folder = "gdrive:degas/loras/ltx2"
    assert remote.files[f"{folder}/ltx_2.3_tin_robot_v1.1.safetensors"] == WEIGHTS
    assert "Tin the robot" in remote.files[f"{folder}/ltx_2.3_tin_robot_v1.1.yaml"].decode()
    # The preview is the example video's first frame.
    with Image.open(io.BytesIO(remote.files[f"{folder}/ltx_2.3_tin_robot_v1.1.jpg"])) as im:
        assert im.size == (64, 32)
    # The token goes to Hugging Face, never to the CDN a download redirects to.
    assert hub.auth["huggingface.co"] == "Bearer hf_secret"
    assert hub.auth["cdn.example"] is None


async def test_import_skips_a_preview_it_cant_read() -> None:
    hub, remote = FakeHub(b"not a video"), FakeRemote()
    imp = make_importer(hub, remote)
    await imp.run(await imp.plan(LINK, []))
    assert not any(p.endswith(".jpg") for p in remote.files)
    assert any(p.endswith(".yaml") for p in remote.files)


async def test_import_deletes_a_file_that_fails_its_hash() -> None:
    hub, remote = FakeHub(), FakeRemote()
    imp = make_importer(hub, remote)
    p = await imp.plan(LINK, [])
    hub.weights = WEIGHTS[:-1] + b"!"
    with pytest.raises(CivitaiImportError, match="doesn't match"):
        await imp.run(p)
    assert not any(p.endswith(".safetensors") for p in remote.files)


async def test_import_refuses_duplicates_and_gated_downloads() -> None:
    hub = FakeHub()
    imp = make_importer(hub, FakeRemote())
    index = [{"path": "loras/ltx2/tin_robot.safetensors", "sha256": SHA}]
    with pytest.raises(PlanError, match="already in Drive"):
        await imp.plan(LINK, index)
    hub.status = 401
    p = await imp.plan(LINK, [])  # the README is skipped
    assert p.trigger_words == []
    with pytest.raises(HuggingFaceError, match="accept its license"):
        await imp.run(p)
    with pytest.raises(HuggingFaceError, match="no repo"):
        await imp.plan("https://huggingface.co/a/missing", [])


# -- API -----------------------------------------------------------------------------------


@pytest.fixture
def hf_client(harness: Harness) -> Iterator[tuple[TestClient, FakeHub]]:
    hub = FakeHub()

    def build(config: Any) -> Any:
        svc = harness.build(config)
        svc.imports.importer = make_importer(hub, FakeRemote())
        return svc

    with TestClient(create_app(harness.config, build)) as client:
        yield client, hub


def test_api_plan(hf_client: tuple[TestClient, FakeHub]) -> None:
    client, _hub = hf_client
    resp = client.post("/api/civitai/plan", json={"url": LINK, "hint": "sdxl"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["origin"], body["family"], body["model_id"]) == ("huggingface", "ltx2", None)

    resp = client.post("/api/civitai/plan", json={"url": "https://huggingface.co/a/missing"})
    assert resp.status_code == 422
    assert "no repo a/missing" in resp.json()["detail"]
