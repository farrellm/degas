import json
from pathlib import Path
from typing import Any

import httpx2
import pytest

from degas.drive import FOLDER, DriveAuth, DriveAuthError, DriveIndexer, save_refresh_token

# folder id → children
TREE: dict[str, list[dict[str, Any]]] = {
    "root": [{"id": "degas", "name": "degas", "mimeType": FOLDER}],
    "degas": [
        {"id": "models", "name": "models", "mimeType": FOLDER},
        {"id": "loras", "name": "loras", "mimeType": FOLDER},
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
    "loras": [{"id": "l-sdxl", "name": "sdxl", "mimeType": FOLDER}],
    "l-sdxl": [
        {"id": "l1", "name": "film.safetensors", "mimeType": "application/octet-stream"},
    ],
}


def drive_api(request: httpx2.Request) -> httpx2.Response:
    if request.url.host == "oauth2.googleapis.com":
        body = dict(x.split("=") for x in request.content.decode().split("&"))
        if body["refresh_token"] != "refresh-1":
            return httpx2.Response(400, json={"error": "invalid_grant"})
        return httpx2.Response(200, json={"access_token": "access-1", "expires_in": 3600})
    assert request.headers["authorization"] == "Bearer access-1"
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
    }
    model = assets["models/sdxl/juggernaut.safetensors"]
    assert (model["family"], model["kind"], model["size"]) == ("sdxl", "model", 6938040682)
    assert model["md5"] == "abc"
    assert assets["models/wan22/ti2v-5b"]["family"] == "wan22"
    assert assets["loras/sdxl/film.safetensors"]["kind"] == "lora"


async def test_scan_missing_root(auth: DriveAuth) -> None:
    indexer = DriveIndexer(
        auth, "elsewhere", http=httpx2.AsyncClient(transport=httpx2.MockTransport(drive_api))
    )
    with pytest.raises(Exception, match="not found"):
        await indexer.scan()
