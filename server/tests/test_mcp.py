"""The MCP server: create_image and edit_image on the configured model (Qwen-Image 2.1 by
default), over streamable HTTP."""

import base64
import io
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from degas import generate
from degas.app import create_app
from degas.config import Config, McpConfig, McpLora
from degas.families.base import SpecError

from .conftest import KLEIN, QWEN, QWEN_LORA, Harness, services
from .test_video import image, upload


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(generate, "POLL_S", 0.02)


@pytest.fixture
def mcp_config() -> McpConfig:
    """`[mcp]`; a test parametrizes it to try another."""
    return McpConfig()


@pytest.fixture
def harness(harness: Harness, mcp_config: McpConfig) -> Harness:
    harness.config = harness.config.model_copy(update={"mcp": mcp_config})
    return harness


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
    qwen = generate.resolve_target(McpConfig())
    # Qwen's own presets for its aspects, at 2K and 1K.
    assert generate.preset_size(qwen, "1:1", "2K") == (2048, 2048)
    assert generate.preset_size(qwen, "16:9", "1K") == (1376, 768)
    assert generate.preset_size(qwen, "2:3", "2K") == (1696, 2528)
    with pytest.raises(SpecError, match="aspect_ratio"):
        generate.preset_size(qwen, "5:4", "2K")
    # An image's own shape, at the resolution's pixel count, in multiples of 32.
    assert generate.matching_size(qwen, 3000, 2000, "2K") == (2496, 1664)
    assert generate.matching_size(qwen, 500, 500, "1K") == (1024, 1024)
    w, h = generate.matching_size(qwen, 1000, 9000, "2K")
    assert (w % 32, h % 32, h) == (0, 0, qwen.sides[1])
    # Klein has 1K presets but none at 2K for 16:9: the size is worked out, keeping the shape
    # within its 2048 px sides.
    klein = generate.resolve_target(McpConfig(family="klein"))
    assert generate.preset_size(klein, "3:4", "1K") == (896, 1152)
    assert generate.preset_size(klein, "16:9", "2K") == (2048, 1152)
    # Krea 2 Raw tops out at 1536 square, so 2K is capped there.
    raw = generate.resolve_target(McpConfig(family="krea2", variant="raw"))
    assert generate.preset_size(raw, "1:1", "2K") == (1536, 1536)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        (McpConfig(family="nope"), "family 'nope'"),
        (McpConfig(family="wan22"), "makes videos"),
        (McpConfig(family="sdxl", variant="inpaint"), "doesn't do text-to-image"),
        (McpConfig(family="krea2", variant="huge"), "no variant 'huge'"),
        (McpConfig(gpu="V100"), "gpu must be"),
        (McpConfig(family="krea2", variant="raw", model="models/krea2/Krea-2-Turbo"), "isn't"),
        (McpConfig(params={"stpes": 30}), "params: unknown parameters stpes"),
        (McpConfig(params={"steps": "many"}), "steps"),
        (McpConfig(family="flux1", edit_params={"steps": 9}), "can't do 'edit'"),
        (McpConfig(family="krea2", t2i_params={"edit_lora": "identity"}), "t2i_params"),
    ],
)
def test_a_bad_config_fails_when_the_app_is_built(
    config: McpConfig, error: str, tmp_path: Path
) -> None:
    with pytest.raises(generate.McpConfigError, match=error):
        create_app(Config(data_dir=tmp_path, mcp=config))


def test_create_without_images_is_text_to_image(client: TestClient) -> None:
    result = call(client, "create_image", {"prompt": "a lighthouse in fog"})
    preview, info = made(result)
    assert max(preview.size) <= generate.PREVIEW_SIDE
    # No session was running, so the tool started one on Qwen's smallest GPU.
    session = services(client).sessions.session
    assert session is not None
    assert session["gpu"] == "L4"
    spec = job_spec(client, info["job_id"])
    assert (spec["mode"], spec["model"]["path"]) == ("t2i", QWEN["path"])
    assert (spec["params"]["width"], spec["params"]["height"]) == (2048, 2048)
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
    # With images, 1K unless asked: 2K runs out of memory on an L4.
    assert (spec["params"]["width"], spec["params"]["height"]) == (864, 1152)
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
    assert (spec["params"]["width"], spec["params"]["height"]) == (1248, 832)
    assert spec["inputs"]["refs"] == [f"sha256:{hat}", first["sha256"]]
    _, info = made(call(client, "edit_image", {**args, "resolution": "2K"}))
    spec = job_spec(client, info["job_id"])
    assert (spec["params"]["width"], spec["params"]["height"]) == (2496, 1664)


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


@pytest.mark.parametrize(
    "mcp_config",
    [
        McpConfig(
            loras=[McpLora(path=QWEN_LORA["path"], weight=0.7)],
            gpu="A100",
            high_mem=True,
            resolution="1K",
            resolution_with_images="2K",
            params={"steps": 20, "sampler": "er_sde"},
            t2i_params={"cfg": 2.5},
            edit_params={"steps": 30, "upsample": True},
        )
    ],
)
def test_the_config_sets_the_job(client: TestClient) -> None:
    _, info = made(call(client, "create_image", {"prompt": "a fox"}))
    session = services(client).sessions.session
    assert session is not None
    assert (session["gpu"], session["high_mem"]) == ("A100", True)
    spec = job_spec(client, info["job_id"])
    assert spec["loras"] == [{"path": QWEN_LORA["path"], "weight": 0.7, "size": 5}]
    params = spec["params"]
    assert (params["width"], params["steps"], params["sampler"], params["cfg"]) == (
        1024,
        20,
        "er_sde",
        2.5,
    )
    assert params["upsample"] is False
    source = upload(client, image(400, 400))["sha256"]
    _, info = made(call(client, "edit_image", {"image": source, "prompt": "make it night"}))
    params = job_spec(client, info["job_id"])["params"]
    assert (params["width"], params["steps"], params["upsample"]) == (2048, 30, True)
    # A tool argument wins over the config.
    args = {"image": source, "prompt": "x", "upsample_prompt": False}
    _, info = made(call(client, "edit_image", args))
    assert job_spec(client, info["job_id"])["params"]["upsample"] is False


@pytest.mark.parametrize("mcp_config", [McpConfig(family="klein", params={"steps": 6})])
def test_another_family(client: TestClient) -> None:
    tools = {t["name"]: t for t in rpc(client, "tools/list", {})["tools"]}
    assert "FLUX.2 [klein] 9B" in tools["create_image"]["description"]
    # Klein takes a source and 3 references.
    refs = tools["edit_image"]["inputSchema"]["properties"]["reference_images"]
    assert refs["maxItems"] == 3
    _, info = made(call(client, "create_image", {"prompt": "a fox", "aspect_ratio": "3:4"}))
    spec = job_spec(client, info["job_id"])
    assert (spec["family"], spec["variant"], spec["model"]["path"]) == (
        "klein",
        "9b",
        KLEIN["path"],
    )
    assert (spec["params"]["width"], spec["params"]["steps"]) == (1536, 6)
    # Klein has no upsampler or negative prompt.
    result = call(client, "create_image", {"prompt": "x", "negative_prompt": "blur"})
    assert result["isError"]
    assert "negative prompt" in result["content"][0]["text"]
    result = call(client, "create_image", {"prompt": "x", "upsample_prompt": True})
    assert result["isError"]


@pytest.mark.parametrize("mcp_config", [McpConfig(family="flux1")])
def test_a_text_only_model_has_no_edit_tool(client: TestClient) -> None:
    tools = {t["name"]: t for t in rpc(client, "tools/list", {})["tools"]}
    assert set(tools) == {"create_image"}
    assert tools["create_image"]["inputSchema"]["properties"]["reference_images"]["maxItems"] == 0


@pytest.mark.parametrize("mcp_config", [McpConfig(model="models/qwen21/Missing")])
def test_a_model_missing_from_drive_is_a_tool_error(client: TestClient) -> None:
    result = call(client, "create_image", {"prompt": "x"})
    assert result["isError"]
    assert "models/qwen21/Missing" in result["content"][0]["text"]
