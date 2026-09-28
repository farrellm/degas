import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from degas.drive import (
    FOLDER,
    DriveAuth,
    DriveAuthError,
    DriveIndexer,
    parse_sidecar,
    save_refresh_token,
)

# folder id → children
TREE: dict[str, list[dict[str, Any]]] = {
    "root": [{"id": "degas", "name": "degas", "mimeType": FOLDER}],
    "degas": [
        {"id": "models", "name": "models", "mimeType": FOLDER},
        {"id": "loras", "name": "loras", "mimeType": FOLDER},
        {"id": "pre", "name": "preprocessors", "mimeType": FOLDER},
        {"id": "junk", "name": "notes.txt", "mimeType": "text/plain"},
    ],
    "models": [
        {"id": "m-sdxl", "name": "sdxl", "mimeType": FOLDER},
        {"id": "m-wan", "name": "wan22", "mimeType": FOLDER},
    ],
    "m-sdxl": [
        {
            "id": "f1",
            "name": "juggernaut.safetensors",
            "mimeType": "application/octet-stream",
            "size": "6938040682",
            "modifiedTime": "2026-09-01T00:00:00Z",
            "md5Checksum": "abc",
        },
        {"id": "f2", "name": "juggernaut.yaml", "mimeType": "text/yaml", "size": "10"},
    ],
    "m-wan": [{"id": "d-5b", "name": "ti2v-5b", "mimeType": FOLDER}],
    "d-5b": [
        {"id": "mi", "name": "model_index.json", "mimeType": "application/json", "size": "5"},
        {"id": "sub", "name": "transformer", "mimeType": FOLDER},
    ],
    "pre": [{"id": "sam3", "name": "sam3", "mimeType": FOLDER}],
    "sam3": [
        {"id": "sc", "name": "config.json", "mimeType": "application/json", "size": "7"},
        {"id": "sw", "name": "model.safetensors", "mimeType": "x", "size": "3000"},
    ],
    "sub": [{"id": "w", "name": "weights.safetensors", "mimeType": "x", "size": "1000"}],
    "loras": [{"id": "l-sdxl", "name": "sdxl", "mimeType": FOLDER}],
    "l-sdxl": [
        {"id": "l1", "name": "film.safetensors", "mimeType": "application/octet-stream"},
        {"id": "l1y", "name": "film.yaml", "mimeType": "text/yaml", "md5Checksum": "y1"},
        {"id": "l1p", "name": "film-sample.jpg", "mimeType": "image/jpeg", "md5Checksum": "p1"},
        {"id": "l2", "name": "detail.safetensors", "mimeType": "application/octet-stream"},
        {"id": "l2p", "name": "detail.png", "mimeType": "image/png", "md5Checksum": "p2"},
    ],
}

# file id → content (alt=media downloads)
CONTENT = {
    "f2": b"label: Juggernaut XL v10\n",
    "l1y": b"label: Film Grain v3\ntrigger_words: [filmgrain]\ndefault_weight: 0.8\n"
    b"preview: film-sample.jpg\nbogus: 1\n",
    "l1p": b"jpeg bytes",
    "l2p": b"png bytes",
}
downloads: list[str] = []


def drive_api(request: httpx2.Request) -> httpx2.Response:
    if request.url.host == "oauth2.googleapis.com":
        body = dict(x.split("=") for x in request.content.decode().split("&"))
        if body["refresh_token"] != "refresh-1":
            return httpx2.Response(400, json={"error": "invalid_grant"})
        return httpx2.Response(200, json={"access_token": "access-1", "expires_in": 3600})
    assert request.headers["authorization"] == "Bearer access-1"
    if request.url.params.get("alt") == "media":
        file_id = request.url.path.rsplit("/", 1)[1]
        downloads.append(file_id)
        return httpx2.Response(200, content=CONTENT[file_id])
    q = request.url.params["q"]
    parent = q.split("'")[1]
    files = TREE.get(parent, [])
    if "name = '" in q:
        name = q.split("name = '")[1].split("'")[0]
        files = [f for f in files if f["name"] == name]
    return httpx2.Response(200, json={"files": files})


@pytest.fixture
def auth(tmp_path: Path) -> DriveAuth:
    client = tmp_path / "client.json"
    client.write_text(json.dumps({"installed": {"client_id": "id", "client_secret": "s"}}))
    token = tmp_path / "token.json"
    save_refresh_token(token, "refresh-1")
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(drive_api))
    return DriveAuth(client, token, http=http)


async def test_access_token_is_cached(auth: DriveAuth) -> None:
    first = await auth.access_token()
    assert first.token == "access-1"
    assert await auth.access_token() is first
    assert first.expiry_rfc3339.endswith("Z")


async def test_revoked_refresh_token(auth: DriveAuth) -> None:
    save_refresh_token(auth.token_file, "revoked")
    with pytest.raises(DriveAuthError):
        await auth.access_token()
    assert auth.error is not None
    assert auth.token_file.stat().st_mode & 0o077 == 0


async def test_scan(auth: DriveAuth) -> None:
    indexer = DriveIndexer(
        auth, "degas", http=httpx2.AsyncClient(transport=httpx2.MockTransport(drive_api))
    )
    assets = {a["path"]: a for a in await indexer.scan()}
    assert set(assets) == {
        "models/sdxl/juggernaut.safetensors",
        "models/wan22/ti2v-5b",
        "loras/sdxl/film.safetensors",
        "loras/sdxl/detail.safetensors",
        "preprocessors/sam3",
    }
    sam = assets["preprocessors/sam3"]
    assert (sam["kind"], sam["family"], sam["size"]) == ("preprocessor", None, 3007)
    model = assets["models/sdxl/juggernaut.safetensors"]
    assert (model["family"], model["kind"], model["size"]) == ("sdxl", "model", 6938040682)
    assert model["md5"] == "abc"
    assert assets["models/wan22/ti2v-5b"]["family"] == "wan22"
    assert assets["models/wan22/ti2v-5b"]["size"] == 1005  # includes transformer/
    assert assets["loras/sdxl/film.safetensors"]["kind"] == "lora"


async def test_scan_missing_root(auth: DriveAuth) -> None:
    indexer = DriveIndexer(
        auth, "elsewhere", http=httpx2.AsyncClient(transport=httpx2.MockTransport(drive_api))
    )
    with pytest.raises(Exception, match="not found"):
        await indexer.scan()


async def test_sidecars_and_previews(auth: DriveAuth) -> None:
    indexer = DriveIndexer(
        auth, "degas", http=httpx2.AsyncClient(transport=httpx2.MockTransport(drive_api))
    )
    stored: dict[str, bytes] = {}

    def store(data: bytes, media_type: str) -> str:
        sha = f"sha-{len(stored)}-{media_type}"
        stored[sha] = data
        return sha

    downloads.clear()
    assets = await indexer.scan()
    await indexer.enrich(assets, {}, store)
    by_path = {a["path"]: a for a in assets}
    film = by_path["loras/sdxl/film.safetensors"]
    assert film["sidecar"] == {
        "label": "Film Grain v3",
        "trigger_words": ["filmgrain"],
        "default_weight": 0.8,
        "preview": "film-sample.jpg",
    }
    # The sidecar names the preview; otherwise it is the image with the same name.
    assert stored[film["preview_thumb"]] == b"jpeg bytes"
    detail = by_path["loras/sdxl/detail.safetensors"]
    assert detail["sidecar"] is None
    assert stored[detail["preview_thumb"]] == b"png bytes"
    model = by_path["models/sdxl/juggernaut.safetensors"]
    assert (model["sidecar"], model["preview_thumb"]) == ({"label": "Juggernaut XL v10"}, None)
    assert sorted(downloads) == ["f2", "l1p", "l1y", "l2p"]

    # A rescan reuses unchanged sidecars and previews without downloading them again.
    downloads.clear()
    again = await indexer.scan()
    await indexer.enrich(again, by_path, store)
    assert downloads == []
    assert {a["path"]: a["preview_thumb"] for a in again} == {
        a["path"]: a["preview_thumb"] for a in assets
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", {}),
        ("trigger_words: 'a, b ,'\n", {"trigger_words": ["a", "b"]}),
        ("label: 3\ndefault_weight: true\n", {"label": "3"}),
        (
            "pair: {high: h.safetensors, low: l.safetensors}\nvariants: [t2v-a14b]\n",
            {"pair": {"high": "h.safetensors", "low": "l.safetensors"}, "variants": ["t2v-a14b"]},
        ),
    ],
)
def test_parse_sidecar(text: str, expected: dict[str, Any]) -> None:
    assert parse_sidecar(text) == expected


def test_parse_sidecar_rejects_non_mappings() -> None:
    with pytest.raises(ValueError, match="mapping"):
        parse_sidecar("- a\n- b\n")
