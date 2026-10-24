"""The MCP server: Qwen-Image 2.1's create_image and edit_image, over streamable HTTP."""

import base64
import io
import json
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from degas import generate
from degas.families.base import SpecError

from .conftest import QWEN, services
from .test_video import image, upload


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "POLL_S", 0.02)


def rpc(client: TestClient, method: str, params: dict[str, Any]) -> dict[str, Any]:
    """A request to the stateless endpoint: the result of its last SSE message."""
    resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert resp.status_code == 200, resp.text
    data = [line[5:] for line in resp.text.splitlines() if line.startswith("data:")]
    result: dict[str, Any] = json.loads(data[-1])["result"]
    return result


def call(client: TestClient, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    return rpc(client, "tools/call", {"name": tool, "arguments": args})


def made(result: dict[str, Any]) -> tuple[Image.Image, dict[str, Any]]:
    assert not result.get("isError"), result
    preview, text = result["content"]
    assert (preview["type"], preview["mimeType"]) == ("image", "image/jpeg")
    im = Image.open(io.BytesIO(base64.b64decode(preview["data"])))
    return im, json.loads(text["text"])


def job_spec(client: TestClient, job_id: str) -> dict[str, Any]:
    job = services(client).db.get_job(job_id)
    assert job is not None
    spec: dict[str, Any] = job["spec"]
    return spec


def test_sizes() -> None:
    assert generate.preset_size("1:1", "2K") == (2048, 2048)
    assert generate.preset_size("16:9", "1K") == (1376, 768)
    assert generate.preset_size("2:3", "2K") == (1696, 2528)
    with pytest.raises(SpecError, match="aspect_ratio"):
        generate.preset_size("5:4", "2K")
    # An image's own shape, at the resolution's pixel count, in multiples of 32.
    assert generate.matching_size(3000, 2000, "2K") == (2496, 1664)
    assert generate.matching_size(500, 500, "1K") == (1024, 1024)
    w, h = generate.matching_size(1000, 9000, "2K")
    assert (w % 32, h % 32, h) == (0, 0, generate.MAX_SIDE)


def test_create_without_images_is_text_to_image(client: TestClient) -> None:
    result = call(client, "create_image", {"prompt": "a lighthouse in fog", "resolution": "1K"})
    preview, info = made(result)
    assert max(preview.size) <= generate.PREVIEW_SIDE
    # No session was running, so the tool started one on Qwen's smallest GPU.
    session = services(client).sessions.session
    assert session is not None
    assert session["gpu"] == "L4"
    spec = job_spec(client, info["job_id"])
    assert (spec["mode"], spec["model"]["path"]) == ("t2i", QWEN["path"])
    assert (spec["params"]["width"], spec["params"]["height"]) == (1024, 1024)
    assert not spec.get("inputs")
    assert info["sha256"].startswith("sha256:")
    assert info["url"].endswith("/api/blobs/" + info["sha256"].removeprefix("sha256:"))
    assert client.get(info["url"]).status_code == 200


def test_create_with_references_edits_from_image_1(client: TestClient) -> None:
    first = "sha256:" + upload(client, image(600, 400))["sha256"]
    second = "sha256:" + upload(client, image(300, 300))["sha256"]
    args = {
        "prompt": "the woman from image 1 wearing the jacket from image 2",
        "reference_images": [first, second],
        "aspect_ratio": "3:4",
        "seed": 5,
    }
    _, info = made(call(client, "create_image", args))
    spec = job_spec(client, info["job_id"])
    assert spec["mode"] == "edit"
    assert (spec["params"]["width"], spec["params"]["height"]) == (1792, 2400)
    assert spec["inputs"]["refs"] == [second]
    # Image 1 is padded to the shape asked for, not cropped.
    ops = spec["inputs"]["transforms"][spec["inputs"]["source"]]["ops"]
    assert ops[-1]["op"] == "pad"
    assert info["seed"] == 5


def test_edit_keeps_the_images_shape_and_numbers_references(client: TestClient) -> None:
    _, first = made(call(client, "create_image", {"prompt": "a man", "resolution": "1K"}))
    hat = upload(client, image(200, 300))["sha256"]
    source = upload(client, image(1500, 1000))["sha256"]
    args = {
        "image": source,  # a bare hash works too
        "prompt": "put the hat from image 2 on the man in image 1",
        "reference_images": [f"sha256:{hat}", first["sha256"]],
    }
    _, info = made(call(client, "edit_image", args))
    spec = job_spec(client, info["job_id"])
    assert spec["mode"] == "edit"
    assert (spec["params"]["width"], spec["params"]["height"]) == (2496, 1664)
    assert spec["inputs"]["refs"] == [f"sha256:{hat}", first["sha256"]]


def test_a_bad_image_is_a_tool_error(client: TestClient) -> None:
    result = call(client, "edit_image", {"image": "sha256:" + "a" * 64, "prompt": "x"})
    assert result["isError"]
    assert "Image 1" in result["content"][0]["text"]
    assert services(client).db.list_jobs() == []
    too_many = ["sha256:" + upload(client, image(64, 64))["sha256"]] * 11
    result = call(client, "create_image", {"prompt": "x", "reference_images": too_many})
    assert result["isError"]


def test_tools_are_listed(client: TestClient) -> None:
    tools = {t["name"]: t for t in rpc(client, "tools/list", {})["tools"]}
    assert set(tools) == {"create_image", "edit_image"}
    assert "image 2" in tools["edit_image"]["description"]
    schema = tools["create_image"]["inputSchema"]["properties"]
    assert schema["aspect_ratio"]["enum"] == list(generate.ASPECTS)
