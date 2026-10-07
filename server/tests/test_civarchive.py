"""CivArchive imports: link parsing, Civitai's shape, mirrors, and streaming into Drive."""

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

from degas.app import create_app
from degas.civitai.civarchive import (
    CivArchive,
    CivArchiveError,
    civitai_version,
    live_mirrors,
    parse_civarchive_ref,
)
from degas.civitai.client import Civitai
from degas.civitai.huggingface import HuggingFace
from degas.civitai.importer import CivitaiImportError, Importer
from degas.civitai.plan import PlanError, plan_import
from degas.families import FAMILIES

from .conftest import Harness
from .test_civitai import FakeRemote, jpeg

WEIGHTS = b"wan lora weights " * 1000
SHA = hashlib.sha256(WEIGHTS).hexdigest()
FILE = "i2v_480p_wave_v1.safetensors"
LINK = "https://civitaiarchive.com/models/400004?modelVersionId=500005"
ARCHIVE = "https://huggingface.co/someone/civitai-mirror/resolve/main/400004/500005"
COPY = "https://huggingface.co/someone/wan_i2v/resolve/main/loras"


def mirror(url: str, source: str = "huggingface", deleted: str | None = None) -> dict[str, Any]:
    return {
        "filename": url.rsplit("/", 1)[-1],
        "url": url,
        "source": source,
        "deletedAt": deleted,
        "is_gated": False,
        "is_paid": False,
    }


def model_file(name: str, sha: str, mirrors: list[dict[str, Any]], **more: Any) -> dict[str, Any]:
    return {
        "name": name,
        "type": "Model",
        "sizeKB": len(WEIGHTS) / 1024,
        "sha256": sha,
        "is_primary": False,
        "mirrors": mirrors,
        **more,
    }


# Shaped like GET https://civarchive.com/api/models/400004?modelVersionId=500005 (trimmed).
MODEL: dict[str, Any] = {
    "id": 400004,
    "name": "Wave Motion",
    "type": "LORA",
    "versions": [{"id": 500005, "name": "Img2Vid - v1.0"}],
    "version": {
        "id": "500005",
        "modelId": 400004,
        "name": "Img2Vid - v1.0",
        "baseModel": "Wan Video 14B i2v 480p",
        "files": [
            {
                "name": "training_data.zip",
                "type": "Training Data",
                "sizeKB": 7.76,
                "sha256": "197b9a585c733acb020780bcde50e7668f1268f84d46ea04a45207274016ab01",
                "is_primary": False,
                "mirrors": [mirror(f"{ARCHIVE}/training_data.zip")],
            },
            model_file(
                FILE,
                SHA,
                [
                    mirror(
                        "https://civitai.com/api/download/models/500005",
                        "civitai",
                        "2026-06-23T02:09:02.792Z",
                    ),
                    mirror(f"https://huggingface.co/gone/x/resolve/main/{FILE}", deleted="2025"),
                    mirror(f"{ARCHIVE}/{FILE}"),
                    mirror(f"{COPY}/{FILE}"),
                    mirror("/tensorart/models/889838563821118245/versions/1", "tensorart"),
                ],
            ),
        ],
        "images": [
            {
                "url": "https://c.genur.art/6ed8_small.mp4",
                "type": "video",
                "video_url": "https://c.genur.art/6ed8_small.mp4",
                "image_url": "https://c.genur.art/6ed8_small.webp",
            }
        ],
        "trigger": ["performing the wave_x move", "performs the wave_x move"],
        "download_url": "/api/download/models/500005",
        "mirrors": [],
    },
}


def model(**changes: Any) -> dict[str, Any]:
    m = copy.deepcopy(MODEL)
    m["version"].update(changes)
    return m


# -- links and Civitai's shape -------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (LINK, (400004, 500005)),
        ("civarchive.com/models/400004", (400004, None)),
        ("https://www.civarchive.com/models/400004/wave?modelVersionId=x", (400004, None)),
    ],
)
def test_parse_civarchive_ref(ref: str, expected: tuple[int, int | None]) -> None:
    assert parse_civarchive_ref(ref) == expected


@pytest.mark.parametrize(
    "ref", ["https://civitai.com/models/1", "https://civarchive.com/users/someone"]
)
def test_parse_civarchive_ref_rejects(ref: str) -> None:
    with pytest.raises(CivArchiveError):
        parse_civarchive_ref(ref)


def test_live_mirrors_skip_deleted_paid_and_other_sites() -> None:
    f = MODEL["version"]["files"][1]
    assert live_mirrors(f) == [f"{ARCHIVE}/{FILE}", f"{COPY}/{FILE}"]
    live = mirror("https://civitai.com/api/download/models/500005", "civitai")
    paid = {**mirror("https://huggingface.co/a/b/resolve/main/x"), "is_paid": True}
    assert live_mirrors({"mirrors": [paid, *f["mirrors"], live]})[0] == live["url"]


def test_civitai_version() -> None:
    v = civitai_version(MODEL)
    assert (v["id"], v["modelId"], v["baseModel"]) == (500005, 400004, "Wan Video 14B i2v 480p")
    assert v["model"] == {"name": "Wave Motion", "type": "LORA"}
    assert v["trainedWords"] == ["performing the wave_x move", "performs the wave_x move"]
    assert v["images"] == [{"url": "https://c.genur.art/6ed8_small.webp"}]
    f = v["files"][1]
    assert (f["hashes"], f["downloadUrl"]) == ({"SHA256": SHA}, f"{ARCHIVE}/{FILE}")


def test_plan_a_civarchive_version() -> None:
    p = plan_import(civitai_version(MODEL), families=set(FAMILIES))
    assert (p.family, p.variants) == ("wan22", ["wan21-i2v-14b", "wan21-flf2v-14b"])
    (f,) = p.files
    assert f.path == "loras/wan22/wave_motion_img2vid_v1.0.safetensors"
    assert (f.url, f.mirrors) == (f"{ARCHIVE}/{FILE}", [f"{COPY}/{FILE}"])
    assert p.preview_url == "https://c.genur.art/6ed8_small.webp"


def test_plan_refuses_a_file_with_no_copy_left() -> None:
    m = model()
    m["version"]["files"][1]["mirrors"] = m["version"]["files"][1]["mirrors"][:2]
    with pytest.raises(PlanError, match="No copy of i2v_480p"):
        plan_import(civitai_version(m), families=set(FAMILIES))


# -- importing -----------------------------------------------------------------------------


class FakeArchive:
    """CivArchive's API, the Hugging Face mirrors (and their CDN), and the image host."""

    def __init__(self, models: list[dict[str, Any]] | None = None) -> None:
        self.models = {int(m["version"]["id"]): m for m in models or [MODEL]}
        self.files: dict[str, bytes] = {f"{ARCHIVE}/{FILE}": WEIGHTS, f"{COPY}/{FILE}": WEIGHTS}
        self.auth: dict[str, str | None] = {}
        self.requested: list[str] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        url = request.url
        self.auth[url.host] = request.headers.get("authorization")
        self.requested.append(str(url))
        if url.host == "civarchive.com" and url.path.startswith("/api/models/"):
            vid = url.params.get("modelVersionId")
            found = self.models.get(int(vid)) if vid else max(self.models.items())[1]
            return httpx2.Response(200, json=found) if found else httpx2.Response(404)
        if url.host == "huggingface.co":
            plain = str(url.copy_with(query=None))
            if plain not in self.files:
                return httpx2.Response(404)
            return httpx2.Response(302, headers={"location": f"https://cdn.example/{plain}"})
        if url.host == "cdn.example":
            return httpx2.Response(200, content=self.files[url.path.lstrip("/")])
        if url.host == "c.genur.art":
            return httpx2.Response(200, content=jpeg())
        return httpx2.Response(404)


def make_importer(fake: FakeArchive, remote: FakeRemote, token: Path | None = None) -> Importer:
    def http() -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(fake))

    return Importer(
        Civitai(None, http=http()),
        remote,
        "gdrive:",
        "degas",
        set(FAMILIES),
        HuggingFace(token, http=http()),
        CivArchive(http=http()),
    )


async def test_import(tmp_path: Path) -> None:
    token = tmp_path / "token"
    token.write_text("hf_secret\n")
    fake, remote = FakeArchive(), FakeRemote()
    imp = make_importer(fake, remote, token)
    p = await imp.plan(LINK, [])
    assert p.origin == "civarchive"
    paths = await imp.run(p)
    assert paths == ["loras/wan22/wave_motion_img2vid_v1.0.safetensors"]
    folder = "gdrive:degas/loras/wan22"
    assert remote.files[f"{folder}/wave_motion_img2vid_v1.0.safetensors"] == WEIGHTS
    sidecar = yaml.safe_load(remote.files[f"{folder}/wave_motion_img2vid_v1.0.yaml"])
    assert sidecar["notes"] == "Wan Video 14B i2v 480p LoRA from CivArchive (Img2Vid - v1.0)"
    assert sidecar["source"] == "https://civarchive.com/models/400004?modelVersionId=500005"
    assert sidecar["trigger_words"] == [
        "performing the wave_x move",
        "performs the wave_x move",
    ]
    with Image.open(io.BytesIO(remote.files[f"{folder}/wave_motion_img2vid_v1.0.jpg"])) as im:
        assert im.size == (768, 576)
    # The Hugging Face token goes to Hugging Face only.
    assert fake.auth["huggingface.co"] == "Bearer hf_secret"
    assert fake.auth["cdn.example"] is None
    assert fake.auth["c.genur.art"] is None


async def test_import_falls_back_to_the_next_mirror() -> None:
    fake, remote = FakeArchive(), FakeRemote()
    imp = make_importer(fake, remote)
    p = await imp.plan(LINK, [])
    fake.files[f"{ARCHIVE}/{FILE}"] = WEIGHTS[:-1] + b"!"  # a different file under that name
    await imp.run(p)
    assert remote.files["gdrive:degas/loras/wan22/wave_motion_img2vid_v1.0.safetensors"] == WEIGHTS
    del fake.files[f"{ARCHIVE}/{FILE}"]  # gone
    fake.requested.clear()
    await imp.run(p)
    hf = [u for u in fake.requested if u.startswith("https://huggingface.co/")]
    assert hf == [f"{ARCHIVE}/{FILE}", f"{COPY}/{FILE}"]

    fake.files[f"{COPY}/{FILE}"] = b"wrong"
    remote.files.clear()
    with pytest.raises(CivitaiImportError, match="expected about"):
        await imp.run(p)
    assert not any(k.endswith(".safetensors") for k in remote.files)


def half(vid: int, name: str, file: str, sha: str) -> dict[str, Any]:
    m = model(id=str(vid), name=name, baseModel="Wan Video 2.2 I2V-A14B")
    m["version"]["files"] = [model_file(file, sha, [mirror(f"{ARCHIVE}/{file}")])]
    m["versions"] = [{"id": 11, "name": "HIGH v1"}, {"id": 12, "name": "LOW v1"}]
    return m


async def test_import_a_wan_pair_from_two_versions() -> None:
    low = b"L" * len(WEIGHTS)
    fake = FakeArchive(
        [
            half(11, "HIGH v1", "wave_high.safetensors", SHA),
            half(12, "LOW v1", "wave_low.safetensors", hashlib.sha256(low).hexdigest()),
        ]
    )
    fake.files[f"{ARCHIVE}/wave_high.safetensors"] = WEIGHTS
    fake.files[f"{ARCHIVE}/wave_low.safetensors"] = low
    remote = FakeRemote()
    imp = make_importer(fake, remote)
    p = await imp.plan("https://civarchive.com/models/400004?modelVersionId=11", [])
    assert [f.path for f in p.files] == [
        "loras/wan22/wave_motion_high_noise.safetensors",
        "loras/wan22/wave_motion_low_noise.safetensors",
    ]
    assert p.paired_version == "LOW v1"
    await imp.run(p)
    assert remote.files["gdrive:degas/loras/wan22/wave_motion_low_noise.safetensors"] == low


# -- API -----------------------------------------------------------------------------------


@pytest.fixture
def archive_client(harness: Harness) -> Iterator[TestClient]:
    fake = FakeArchive()

    def build(config: Any) -> Any:
        svc = harness.build(config)
        svc.imports.importer = make_importer(fake, FakeRemote())
        return svc

    with TestClient(create_app(harness.config, build)) as client:
        yield client


def test_api_plan(archive_client: TestClient) -> None:
    resp = archive_client.post("/api/civitai/plan", json={"url": LINK})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert (body["origin"], body["family"], body["version_id"]) == ("civarchive", "wan22", 500005)

    resp = archive_client.post(
        "/api/civitai/plan", json={"url": "https://civarchive.com/models/400004?modelVersionId=9"}
    )
    assert resp.status_code == 422
    assert "no such model" in resp.json()["detail"]
