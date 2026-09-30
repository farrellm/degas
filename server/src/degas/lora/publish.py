"""Publish a run's checkpoint to `loras/sdxl/` in Drive, with a sidecar and a preview.

Degas's own Drive token is read-only, so this uploads with a local rclone remote that can
write to the same Drive (`lora.rclone_remote`, `gdrive:` by default).
"""

import hashlib
import io
import subprocess
from collections.abc import Callable
from pathlib import Path

import yaml
from PIL import Image

from degas.lora.run import RunDir, RunError, RunState, checkpoint_for, samples_for

PREVIEW_SIDE = 768

Rclone = Callable[..., str]


def rclone(*args: str, stdin: bytes | None = None) -> str:
    proc = subprocess.run(  # noqa: S603 - fixed argv
        ["rclone", *args],  # noqa: S607 - the user's rclone, from PATH
        input=stdin,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RunError(f"rclone {args[0]} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc.stdout.decode()


def sidecar(state: RunState, epoch: int, label: str | None, weight: float) -> str:
    s = state.settings
    data = {
        "label": label or s.name,
        "trigger_words": [s.subject],
        "default_weight": weight,
        "notes": f"Trained on {Path(s.base).name}, run {state.run}, epoch {epoch}/{s.train.epochs}",
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def preview_jpeg(sample: Path) -> bytes:
    with Image.open(sample) as im:
        rgb = im.convert("RGB")
        rgb.thumbnail((PREVIEW_SIDE, PREVIEW_SIDE), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        rgb.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


def publish(
    run_dir: RunDir,
    state: RunState,
    *,
    remote: str,
    drive_root: str,
    epoch: int | None,
    label: str | None = None,
    weight: float = 0.8,
    preview_index: int = 0,
    force: bool = False,
    run_rclone: Rclone = rclone,
    say: Callable[[str], None] = print,
) -> str:
    """Upload and return the LoRA's Drive path (relative to the root)."""
    s = state.settings
    epoch = s.train.epochs if epoch is None else epoch
    ckpt = checkpoint_for(run_dir, state, epoch)
    folder = f"{remote}{drive_root.strip('/')}/loras/sdxl"
    target = f"{folder}/{s.name}.safetensors"
    run_rclone("mkdir", folder)  # before any upload: parallel ones would make duplicate folders
    existing = run_rclone("lsf", "--files-only", folder).splitlines()
    if f"{s.name}.safetensors" in existing and not force:
        raise RunError(f"{target} already exists; pass --force to replace it")
    say(f"Uploading {ckpt.name} → {target}")
    run_rclone("copyto", str(ckpt), target)
    local_md5 = hashlib.md5(ckpt.read_bytes(), usedforsecurity=False).hexdigest()
    remote_md5 = run_rclone("md5sum", target).split(" ")[0].strip()
    if remote_md5 != local_md5:
        raise RunError(f"Upload check failed: md5 {remote_md5} in Drive, {local_md5} here")
    run_rclone(
        "rcat", f"{folder}/{s.name}.yaml", stdin=sidecar(state, epoch, label, weight).encode()
    )
    samples = samples_for(run_dir, state, epoch)
    if samples:
        sample = samples[min(preview_index, len(samples) - 1)]
        run_rclone("rcat", f"{folder}/{s.name}.jpg", stdin=preview_jpeg(sample))
    return f"loras/sdxl/{s.name}.safetensors"
