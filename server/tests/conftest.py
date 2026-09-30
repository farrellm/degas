"""Fakes for the Colab CLI and SSH tunnel; the worker is the real app over ASGI."""

import io
import json
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from degas.app import create_app
from degas.colab.session import STARTED_MARKER, Intervals
from degas.colab.tunnel import Tunnel
from degas.colab.worker_client import WorkerClient
from degas.config import Config
from degas.services import Services, build_services
from degas_worker import masks
from degas_worker.app import create_app as create_worker_app
from degas_worker.families.base import FamilyRunner, Output, RunContext
from degas_worker.paths import Paths
from degas_worker.preprocess.base import trace
from degas_worker.video import encode_mp4


class FakeColab:
    session = "degas"

    def __init__(self) -> None:
        self.alive = False
        self.calls: list[str] = []
        self.exec_output = f"{STARTED_MARKER} 123\n"

    async def new(self, gpu: str | None, high_mem: bool) -> None:
        self.calls.append(f"new {gpu} {high_mem}")
        self.alive = True

    async def stop(self) -> None:
        self.calls.append("stop")
        self.alive = False

    async def is_alive(self) -> bool:
        return self.alive

    async def exec(self, code: str, timeout: float = 120) -> str:
        self.calls.append("exec")
        return self.exec_output

    def proxy_command(self, identity: str) -> str:
        return "true"


class FakeTunnel:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.uploads: list[str] = []
        self.opened = False

    @property
    def local_port(self) -> int:
        return 1

    async def open(self) -> None:
        self.opened = True

    async def close(self) -> None:
        self.opened = False

    async def is_open(self) -> bool:
        return self.opened

    async def run(self, command: str, timeout: float = 120) -> str:
        self.commands.append(command)
        return ""

    async def upload(self, local: Path, remote: str) -> None:
        self.uploads.append(remote)


def png(seed: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (seed % 256, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


# Stand-in for rclone: `copyto SRC DEST ...` writes 10 bytes to DEST.
FAKE_RCLONE = """#!/bin/sh
mkdir -p "$(dirname "$3")"
printf '0123456789' > "$3"
echo '{"level":"notice","msg":"stats","stats":{"bytes":10,"totalBytes":10}}' >&2
"""


class FakeSdxl:
    fail: str | None = None
    fetch = False  # copy the spec's model and LoRAs into the VM cache, like the real runner

    def __init__(self) -> None:
        self.inputs: list[dict[str, Any]] = []  # sizes of the staged inputs

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        staged = {}
        for key in ("source", "mask"):
            ref = (spec.get("inputs") or {}).get(key)
            if ref:
                with Image.open(ctx.blob(ref)) as im:
                    staged[key] = im.size
        for unit in spec.get("control") or []:
            with Image.open(ctx.blob(unit["image"])) as im:
                size = im.size
            if unit.get("mask"):
                with Image.open(ctx.blob(unit["mask"])) as im:
                    staged.setdefault("areas", []).append(im.size)
            staged.setdefault("control", []).append(size)
        for unit in spec.get("image_prompts") or []:
            pictures = []
            for ref in unit["images"]:
                with Image.open(ctx.blob(ref)) as im:
                    pictures.append(im.size)
            staged.setdefault("prompts", []).append(pictures)
            if unit.get("mask"):
                with Image.open(ctx.blob(unit["mask"])) as im:
                    staged.setdefault("prompt_areas", []).append(im.size)
        if staged:
            self.inputs.append(staged)
        if self.fetch:
            nets = [u["controlnet"] for u in spec.get("control") or []]
            nets += [u["adapter"] for u in spec.get("image_prompts") or []]
            extra = [spec[k] for k in ("vae", "image_encoder") if spec.get(k)]
            for asset in [spec["model"], spec["config"], *extra, *spec.get("loras", []), *nets]:
                ctx.fetch_asset(asset["path"], asset.get("size"))
            ctx.progress(0, "load", 1, 1)
        for item, seed in enumerate(seeds):
            for step in range(2):
                ctx.check_cancelled()
                ctx.progress(item, "denoise", step + 1, 2)
            if self.fail:
                raise RuntimeError(self.fail)
            yield Output(item, seed, png(seed), "image/png", "png")

    def unload(self) -> None:
        pass


class FakeWan:
    """Encodes a few solid frames at the spec's size; i2v checks its source was staged."""

    frames = 5
    sources: list[tuple[int, int]]

    def __init__(self) -> None:
        self.sources = []

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        params = spec["params"]
        w, h = params["width"], params["height"]
        if spec["mode"] == "i2v":
            with Image.open(ctx.blob(spec["inputs"]["source"])) as im:
                self.sources.append(im.size)
        for item, seed in enumerate(seeds):
            ctx.progress(item, "denoise", 1, 1)
            frame = bytes([seed % 256, 80, 120]) * (w * h)
            data = encode_mp4([frame] * self.frames, w, h, params["fps"])
            yield Output(item, seed, data, "video/mp4", "mp4")

    def unload(self) -> None:
        pass


class FakeQwen:
    """Records the sizes of the condition images (source first, then references) and mask.

    Also stands in for FLUX.1 and FLUX.2 [klein], recording the assets each job names."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        inputs = spec.get("inputs") or {}
        staged: dict[str, Any] = {"mode": spec["mode"], "images": []}
        if spec.get("config"):
            staged["config"] = spec["config"]["path"]
        for ref in [inputs.get("source"), *(inputs.get("refs") or [])]:
            if ref:
                with Image.open(ctx.blob(ref)) as im:
                    staged["images"].append(im.size)
        if inputs.get("mask"):
            with Image.open(ctx.blob(inputs["mask"])) as im:
                staged["mask"] = im.size
        self.calls.append(staged)
        for item, seed in enumerate(seeds):
            ctx.progress(item, "denoise", 1, 1)
            yield Output(item, seed, png(seed), "image/png", "png")

    def unload(self) -> None:
        pass


class FakeSam:
    """Selects a rectangle around each included point; a description selects the left half."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"model": model, **params})
        with Image.open(image) as im:
            w, h = im.size
        found: list[Image.Image] = []
        if params.get("text"):
            mask = Image.new("L", (w, h), 0)
            mask.paste(255, (0, 0, w // 2, h))
            found.append(mask)
        else:
            for r in (2, 6, 12):
                mask = Image.new("L", (w, h), 0)
                for p in params["points"]:
                    if p["include"]:
                        x, y = int(p["x"]), int(p["y"])
                        mask.paste(255, (max(0, x - r), max(0, y - r), x + r, y + r))
                found.append(mask)
        return masks.candidates(found, [0.5, 0.9, 0.7][: len(found)])

    def unload(self) -> None:
        pass


class FakeTrace:
    """A depth, pose or edges trace: the image's own size, grey."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, model: Path | None, image: Path, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"model": model, **params})
        with Image.open(image) as im:
            return trace(Image.new("RGB", im.size, (128, 128, 128)))

    def unload(self) -> None:
        pass


class FastIntervals(Intervals):
    tick = 0.02
    health = 0.05
    colab_status = 0.1
    exec_heartbeat = 3600
    drive_token = 3600
    health_wait = 5


class Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.colab = FakeColab()
        self.tunnels: list[FakeTunnel] = []
        self.runner = FakeSdxl()
        self.wan = FakeWan()
        self.qwen = FakeQwen()
        self.flux = FakeQwen()
        self.klein = FakeQwen()
        self.sam = FakeSam()
        self.trace = FakeTrace()
        self.worker_paths = Paths(home=tmp_path / "vm", models=tmp_path / "vm-models")
        self.worker_paths.ensure()

        def runner_factory() -> FamilyRunner:
            return self.runner

        self.worker_exits = 0

        def exit_process() -> None:
            self.worker_exits += 1

        rclone = tmp_path / "rclone"
        rclone.write_text(FAKE_RCLONE)
        rclone.chmod(rclone.stat().st_mode | stat.S_IEXEC)
        self.worker_app = create_worker_app(
            self.worker_paths,
            {
                "sdxl": runner_factory,
                "wan22": lambda: self.wan,
                "qwen21": lambda: self.qwen,
                "flux1": lambda: self.flux,
                "klein": lambda: self.klein,
            },
            rclone=str(rclone),
            exit_process=exit_process,
            preprocessors={
                "sam": lambda: self.sam,
                **{kind: (lambda: self.trace) for kind in ("depth", "pose", "canny")},
            },
        )
        self.config = Config(data_dir=tmp_path / "data", web_dist=tmp_path / "no-web")
        key = self.config.ssh_key
        key.parent.mkdir(parents=True)
        key.write_text("fake key")
        self.model = dict(MODEL)
        # Web Push deliveries: (endpoint, payload); `push_status` is the push service's reply.
        self.pushed: list[tuple[str, dict[str, Any]]] = []
        self.push_status = 201

    async def send_push(self, subscription: dict[str, Any], payload: str) -> int | None:
        self.pushed.append((subscription["endpoint"], json.loads(payload)))
        return self.push_status

    def tunnel_factory(self) -> Tunnel:
        tunnel = FakeTunnel()
        self.tunnels.append(tunnel)
        return tunnel

    def worker_factory(self, _tunnel: Tunnel) -> WorkerClient:
        return WorkerClient("http://worker", transport=httpx2.ASGITransport(app=self.worker_app))

    def build(self, config: Config | None = None) -> Services:
        svc = build_services(
            config or self.config,
            colab=self.colab,
            tunnel_factory=self.tunnel_factory,
            worker_factory=self.worker_factory,
            push_sender=self.send_push,
        )
        svc.sessions.iv = FastIntervals()
        return svc

    def services_factory(self) -> Callable[[Config], Services]:
        return self.build


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


MODEL = {
    "path": "models/sdxl/juggernaut.safetensors",
    "family": "sdxl",
    "kind": "model",
    "drive_file_id": "f1",
    "size": 1234,
}
LORA = {
    "path": "loras/sdxl/film.safetensors",
    "family": "sdxl",
    "kind": "lora",
    "drive_file_id": "l1",
    "size": 10,
    "sidecar": {"label": "Film Grain v3", "trigger_words": ["filmgrain"]},
}
VAE = {
    "path": "vae/sdxl/sdxl-vae-fp16-fix",
    "family": "sdxl",
    "kind": "vae",
    "drive_file_id": "v1",
    "size": 10,
}
CONFIGS = [
    {
        "path": "configs/sdxl/stable-diffusion-xl-base-1.0",
        "family": "sdxl",
        "kind": "config",
        "drive_file_id": "c1",
        "size": 10,
    },
    {
        "path": "configs/sdxl/stable-diffusion-xl-1.0-inpainting-0.1",
        "family": "sdxl",
        "kind": "config",
        "drive_file_id": "c2",
        "size": 10,
    },
]


SAM = {
    "path": "preprocessors/sam3",
    "family": None,
    "kind": "preprocessor",
    "drive_file_id": "s3",
    "size": 10,  # what the fake rclone writes
}
DEPTH = {
    "path": "preprocessors/depth-anything-v2",
    "family": None,
    "kind": "preprocessor",
    "drive_file_id": "da2",
    "size": 10,
}
CONTROLNET = {
    "path": "controlnets/sdxl/depth-xl",
    "family": "sdxl",
    "kind": "controlnet",
    "drive_file_id": "cn1",
    "size": 25,
    "sidecar": {"label": "Depth XL", "control": "depth"},
}
IP_ADAPTER = {
    "path": "ip_adapters/sdxl/ip-adapter-plus_sdxl_vit-h.safetensors",
    "family": "sdxl",
    "kind": "ip_adapter",
    "drive_file_id": "ip1",
    "size": 26,
}
IMAGE_ENCODER = {
    "path": "image_encoders/sdxl/clip-vit-h-14",
    "family": "sdxl",
    "kind": "image_encoder",
    "drive_file_id": "ie1",
    "size": 27,
}
INPAINT_MODEL = {
    "path": "models/sdxl/inpaint/sdxl-inpaint.safetensors",
    "family": "sdxl",
    "kind": "model",
    "drive_file_id": "f9",
    "size": 1234,
}


WAN_5B = {
    "path": "models/wan22/ti2v-5b",
    "family": "wan22",
    "kind": "model",
    "drive_file_id": "w5",
    "size": 20,
}
QWEN = {
    "path": "models/qwen21/Qwen-Image-2.1",
    "family": "qwen21",
    "kind": "model",
    "drive_file_id": "q21",
    "size": 10,
}
FLUX1 = {
    "path": "models/flux1/flux1-dev-fp8.safetensors",
    "family": "flux1",
    "kind": "model",
    "drive_file_id": "fx1",
    "size": 12,
}
FLUX1_BASE = {
    "path": "configs/flux1/FLUX.1-dev",
    "family": "flux1",
    "kind": "config",
    "drive_file_id": "fx1c",
    "size": 11,
}
KLEIN = {
    "path": "models/klein/FLUX.2-klein-9B",
    "family": "klein",
    "kind": "model",
    "drive_file_id": "k9",
    "size": 35,
}
WAN_I2V = {
    "path": "models/wan22/i2v-a14b/Wan2.2-I2V-A14B-Diffusers",
    "family": "wan22",
    "kind": "model",
    "drive_file_id": "w14",
    "size": 30,
}


@pytest.fixture
def client(harness: Harness) -> Iterator[TestClient]:
    app = create_app(harness.config, harness.services_factory())
    with TestClient(app) as c:
        c.app.state.services.db.replace_assets(  # type: ignore[attr-defined]
            [
                harness.model,
                LORA,
                VAE,
                *CONFIGS,
                WAN_5B,
                WAN_I2V,
                QWEN,
                FLUX1,
                FLUX1_BASE,
                KLEIN,
                INPAINT_MODEL,
                SAM,
                DEPTH,
                CONTROLNET,
                IP_ADAPTER,
                IMAGE_ENCODER,
            ]
        )
        yield c
