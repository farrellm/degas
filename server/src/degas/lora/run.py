"""Run a training job on a Colab VM, and follow it from here.

Training runs detached on the VM (started from the kernel, like the worker), so a lost
connection or a Ctrl-C here doesn't stop it: `attach` picks it up again from `state.json`.
Checkpoints and sample images are copied back as each epoch writes them.
"""

import asyncio
import json
import re
import shlex
import tarfile
import tempfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel

from degas.colab.cli import Colab
from degas.colab.session import RCLONE_URL
from degas.colab.tunnel import Tunnel
from degas.drive import AccessToken
from degas.errors import DegasError
from degas.lora import recipe
from degas.lora.recipe import LORA_HOME, RemoteRun
from degas.lora.settings import LoraSettings

Status = Literal["starting", "training", "done", "failed", "stopped"]

# tqdm's line, e.g. "steps:  12%|█▏  | 240/2000 [05:10<37:50,  1.29s/it, avr_loss=0.0921]"
_PROGRESS = re.compile(
    r"(\d+)/(\d+) \[([\d:]+)<([\d:?]+),\s*([\d.?]+)\s*(s/it|it/s)(?:,\s*avr_loss=([\d.]+))?"
)


class RunError(DegasError):
    pass


class BusyError(RunError):
    """The session's VM is already training another run."""


class Remote(Tunnel, Protocol):
    async def download(self, remote: str, local: Path) -> None: ...


class TokenSource(Protocol):
    async def access_token(self, min_valid_s: float = 300) -> AccessToken: ...


class RunState(BaseModel):
    run: str
    session: str
    settings: LoraSettings
    images: int
    steps: int
    created_at: str
    status: Status = "starting"
    error: str | None = None
    downloaded: list[str] = []  # paths under the remote output folder


def new_run_id(name: str, now: datetime | None = None) -> str:
    return f"{name}-{(now or datetime.now(UTC)).strftime('%Y%m%d-%H%M%S')}"


class RunDir:
    """`<data_dir>/lora-runs/<run>/`: state, the prepared dataset, downloads."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def state_file(self) -> Path:
        return self.path / "state.json"

    @property
    def dataset(self) -> Path:
        return self.path / "dataset"

    @property
    def checkpoints(self) -> Path:
        return self.path / "checkpoints"

    @property
    def samples(self) -> Path:
        return self.path / "samples"

    @property
    def log(self) -> Path:
        return self.path / "train.log"

    def load(self) -> RunState:
        if not self.state_file.exists():
            raise RunError(f"No run at {self.path}")
        return RunState.model_validate_json(self.state_file.read_text())

    def save(self, state: RunState) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(state.model_dump_json(indent=2) + "\n")
        tmp.replace(self.state_file)

    def local_for(self, rel: str) -> Path:
        """Where a file from the remote output folder goes."""
        name = Path(rel).name
        return (self.samples if rel.startswith("sample/") else self.checkpoints) / name


def format_progress(log_tail: str) -> str | None:
    """The latest tqdm progress in a log tail, as "step 240/2000 · 1.29s/it · …"."""
    matches = list(_PROGRESS.finditer(log_tail))
    if not matches:
        return None
    m = matches[-1]
    step, total, _, left, rate, unit, loss = m.groups()
    text = f"step {step}/{total} · {rate}{unit} · {left} left"
    return text + (f" · loss {loss}" if loss else "")


class Trainer:
    def __init__(
        self,
        run_dir: RunDir,
        state: RunState,
        colab: Colab,
        tunnel: Remote,
        drive: TokenSource,
        drive_root: str,
        rclone_binary: Path | None = None,
        say: Callable[[str], None] = print,
        poll_s: float = 15,
        heartbeat_s: float = 300,
    ) -> None:
        self.dir = run_dir
        self.state = state
        self.colab = colab
        self.tunnel = tunnel
        self.drive = drive
        self.drive_root = drive_root.strip("/")
        self.rclone_binary = rclone_binary
        self.say = say
        self.poll_s = poll_s
        self.heartbeat_s = heartbeat_s
        self.remote = RemoteRun(state.run)
        self._sizes: dict[str, int] = {}  # remote file → size at the last poll

    def set_status(self, status: Status, error: str | None = None) -> None:
        self.state.status = status
        self.state.error = error
        self.dir.save(self.state)

    # -- start ------------------------------------------------------------------------------

    async def start(self) -> None:
        """Get a VM (or reuse the session's), stage everything and start training."""
        s = self.state.settings
        try:
            if await self.colab.is_alive():
                self.say(f"Reusing the running Colab session {self.colab.session!r}")
                await self.tunnel.open()
                busy = await self.tunnel.run(
                    "pgrep -f sdxl_train_network.py >/dev/null && echo busy; true"
                )
                if busy.strip() == "busy":
                    raise BusyError("That VM is already training: attach to that run, or stop it")
            else:
                self.say(f"Starting a {s.gpu} VM ({self.colab.session})")
                await self.colab.new(s.gpu, s.high_mem)
                await self.tunnel.open()
            self.say(f"Installing sd-scripts {recipe.SD_SCRIPTS_TAG}")
            await self._install()
            self.say(f"Uploading {self.state.images} images")
            await self._stage()
            self.say(f"Copying {s.base} from Drive")
            await self._fetch_base()
            self.say("Starting training")
            out = await self.colab.exec(recipe.launch_code(self.remote), timeout=120)
            if recipe.STARTED_MARKER not in out:
                raise RunError(f"Training did not start: {out.strip()[-500:]}")
        except Exception as e:
            self.set_status("failed", str(e))
            raise
        self.set_status("training")

    async def _install(self) -> None:
        t = self.tunnel
        repo, tag = recipe.SD_SCRIPTS_REPO, recipe.SD_SCRIPTS_TAG
        await t.run(f"mkdir -p {LORA_HOME}/bin {LORA_HOME}/models {LORA_HOME}/runs")
        await t.run(
            f'cd {LORA_HOME} && if [ "$(git -C sd-scripts describe --tags 2>/dev/null)" != {tag} ];'
            f" then rm -rf sd-scripts && git clone -q --depth 1 -b {tag} {repo} sd-scripts; fi",
            timeout=300,
        )
        # Moves transformers/accelerate/huggingface-hub a patch or two; torch stays Colab's.
        await t.run(
            f"cd {LORA_HOME}/sd-scripts && (test -f .degas-installed"
            " || (python3 -m pip install -q -r requirements.txt && touch .degas-installed))",
            timeout=900,
        )
        rclone = f"{LORA_HOME}/bin/rclone"
        if self.rclone_binary is not None:
            has = await t.run(f"test -x {rclone} && echo yes; true")
            if has.strip() != "yes":
                await t.upload(self.rclone_binary, rclone)
                await t.run(f"chmod +x {rclone}")
        else:
            await t.run(
                f"test -x {rclone} || (cd /tmp && curl -fsSLo rclone.zip {RCLONE_URL}"
                f" && unzip -oq rclone.zip && cp rclone-*-linux-amd64/rclone {rclone}"
                f" && chmod +x {rclone})",
                timeout=300,
            )

    async def _stage(self) -> None:
        s = self.state.settings
        files = {
            "dataset.toml": recipe.dataset_toml(s, self.remote, self.state.images),
            "prompts.txt": recipe.sample_prompts(s),
            "run.sh": recipe.run_script(s, self.remote),
        }
        for name, text in files.items():  # kept with the run, for the record
            (self.dir.path / name).write_text(text)
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "run.tar"
            with tarfile.open(bundle, "w") as tar:
                tar.add(self.dir.dataset, arcname="dataset")
                for name in files:
                    tar.add(self.dir.path / name, arcname=name)
            await self.tunnel.run(f"rm -rf {self.remote.dir} && mkdir -p {self.remote.dir}")
            await self.tunnel.upload(bundle, f"{self.remote.dir}/run.tar")
        await self.tunnel.run(f"cd {self.remote.dir} && tar xf run.tar && rm run.tar")

    async def _fetch_base(self) -> None:
        token = await self.drive.access_token(min_valid_s=20 * 60)
        body = json.dumps(
            {"access_token": token.token, "token_type": "Bearer", "expiry": token.expiry_rfc3339}
        )
        conf = f"{LORA_HOME}/rclone.conf"
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp) / "rclone.conf"
            local.write_text(f"[drive]\ntype = drive\nscope = drive.readonly\ntoken = {body}\n")
            local.chmod(0o600)
            await self.tunnel.upload(local, conf)
        base = self.state.settings.base
        src = shlex.quote(f"drive:{self.drive_root}/{base}")
        dest = shlex.quote(recipe.model_path(base))
        # copyto skips a file that's already there with the same size and time.
        await self.tunnel.run(
            f"{LORA_HOME}/bin/rclone copyto {src} {dest} --config {conf}"
            f" --multi-thread-streams 8; code=$?; rm -f {conf}; exit $code",
            timeout=1800,
        )

    # -- follow -----------------------------------------------------------------------------

    async def follow(self) -> Status:
        """Report progress and download new files until training exits."""
        last_beat = time.monotonic()
        last_line = None
        while True:
            exit_code, files, tail = await self._poll()
            line = format_progress(tail)
            if line and line != last_line:
                self.say(line)
                last_line = line
            await self._download_new(files, final=exit_code is not None)
            if exit_code is not None:
                return await self._finish(exit_code)
            if time.monotonic() - last_beat >= self.heartbeat_s:
                # The kernel is idle while training runs detached; keep Colab from reclaiming it.
                try:
                    await self.colab.exec("pass", timeout=60)
                except Exception as e:  # a missed heartbeat isn't fatal
                    self.say(f"(heartbeat failed: {e})")
                last_beat = time.monotonic()
            await asyncio.sleep(self.poll_s)

    async def _poll(self) -> tuple[int | None, dict[str, int], str]:
        r = self.remote
        out = await self.tunnel.run(
            f"echo @@exit; cat {r.exit_code} 2>/dev/null; echo; echo @@files;"
            f" (cd {r.output} 2>/dev/null && find . -maxdepth 2 -type f"
            " \\( -name '*.safetensors' -o -name '*.png' \\) -printf '%s %P\\n');"
            f" echo @@log; tail -c 4000 {r.log} 2>/dev/null | tr '\\r' '\\n'; true",
            timeout=60,
        )
        sections: dict[str, list[str]] = {}
        current = ""
        for raw in out.splitlines():
            if raw.startswith("@@"):
                current = raw[2:]
                sections[current] = []
            elif current:
                sections[current].append(raw)
        code_text = "".join(sections.get("exit", [])).strip()
        exit_code = int(code_text) if code_text.lstrip("-").isdigit() else None
        files: dict[str, int] = {}
        for entry in sections.get("files", []):
            size, _, rel = entry.partition(" ")
            if rel and size.isdigit():
                files[rel] = int(size)
        return exit_code, files, "\n".join(sections.get("log", []))

    async def _download_new(self, files: dict[str, int], final: bool) -> None:
        """Download files whose size held still since the last poll (or all, when done)."""
        for rel, size in sorted(files.items()):
            if rel in self.state.downloaded:
                continue
            if not final and self._sizes.get(rel) != size:
                continue
            local = self.dir.local_for(rel)
            await self.tunnel.download(f"{self.remote.output}/{rel}", local)
            self.state.downloaded.append(rel)
            self.dir.save(self.state)
            if rel.endswith(".safetensors"):
                self.say(f"Saved {local}")
        self._sizes = files

    async def _finish(self, exit_code: int) -> Status:
        await self.tunnel.download(self.remote.log, self.dir.log)
        if exit_code == 0:
            self.set_status("done")
            self.say(f"Training finished: {self.dir.checkpoints}, samples in {self.dir.samples}")
            return "done"
        lines = self.dir.log.read_text(errors="replace").replace("\r", "\n").splitlines()
        tail = "\n".join(line for line in lines if line.strip())[-3000:]
        self.set_status("failed", f"sd-scripts exited with {exit_code}")
        self.say(f"Training failed (exit {exit_code}). End of {self.dir.log}:\n{tail}")
        return "failed"

    async def stop_vm(self) -> None:
        await self.tunnel.close()
        await self.colab.stop()
        if self.state.status in ("starting", "training"):
            self.set_status("stopped")


def checkpoint_for(run_dir: RunDir, state: RunState, epoch: int | None) -> Path:
    """An epoch's checkpoint; the last epoch's is saved without a number."""
    name = state.settings.name
    epochs = state.settings.train.epochs
    if epoch is None or epoch == epochs:
        path = run_dir.checkpoints / f"{name}.safetensors"
    else:
        path = run_dir.checkpoints / f"{name}-{epoch:06d}.safetensors"
    if not path.exists():
        have = sorted(p.name for p in run_dir.checkpoints.glob("*.safetensors"))
        raise RunError(f"No checkpoint {path.name}; have: {', '.join(have) or 'none'}")
    return path


def samples_for(run_dir: RunDir, state: RunState, epoch: int | None) -> list[Path]:
    """An epoch's sample images, in prompt order."""
    n = state.settings.train.epochs if epoch is None else epoch
    return sorted(run_dir.samples.glob(f"{state.settings.name}_e{n:06d}_*.png"))
