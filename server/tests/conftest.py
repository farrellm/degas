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
from degas_worker.app import create_app as create_worker_app
from degas_worker.families.base import FamilyRunner, Output, RunContext
from degas_worker.paths import Paths
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

    def run(self, spec: dict[str, Any], seeds: list[int], ctx: RunContext) -> Iterator[Output]:
        if self.fetch:
            for asset in [spec["model"], *spec.get("loras", [])]:
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
            {"sdxl": runner_factory, "wan22": lambda: self.wan},
            rclone=str(rclone),
            exit_process=exit_process,
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


WAN_5B = {
    "path": "models/wan22/ti2v-5b",
    "family": "wan22",
    "kind": "model",
    "drive_file_id": "w5",
    "size": 20,
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
            [harness.model, LORA, WAN_5B, WAN_I2V]
        )
        yield c
