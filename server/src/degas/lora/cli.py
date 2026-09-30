"""`degas lora …` subcommands."""

import argparse
import asyncio
import contextlib
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx2
from pydantic import ValidationError

from degas.colab.cli import ColabCli, ColabError
from degas.colab.session import ensure_key
from degas.colab.tunnel import SshTunnel
from degas.config import Config
from degas.db import Database
from degas.drive import DriveAuth
from degas.lora import dataset as ds
from degas.lora.publish import publish
from degas.lora.run import BusyError, RunDir, RunError, RunState, Trainer, new_run_id
from degas.lora.settings import LoraSettings, load_settings
from degas.rclone import RcloneError


def add_parser(sub: Any) -> None:
    lora = sub.add_parser("lora", help="train an SDXL character LoRA on a Colab GPU")
    cmd = lora.add_subparsers(dest="lora_command", required=True)

    def dataset_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("dataset", type=Path, help="folder of images, .txt captions and lora.toml")
        p.add_argument("--base", help="checkpoint under the Drive root (overrides lora.toml)")
        p.add_argument("--gpu", choices=["T4", "L4", "G4", "A100", "H100"])

    dataset_args(cmd.add_parser("check", help="check a dataset and show the training plan"))
    train = cmd.add_parser("train", help="train on a Colab VM and follow it")
    dataset_args(train)
    train.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    attach = cmd.add_parser("attach", help="follow a run again after a disconnect")
    attach.add_argument("run", help="run id, or a LoRA name for its latest run")
    attach.add_argument("--keep", action="store_true", help="leave the VM running afterwards")
    stop = cmd.add_parser("stop", help="stop a run's VM")
    stop.add_argument("run")
    cmd.add_parser("list", help="list runs")
    pub = cmd.add_parser("publish", help="upload a checkpoint to loras/sdxl/ in Drive")
    pub.add_argument("run")
    pub.add_argument("--epoch", type=int, help="default: the last one")
    pub.add_argument("--label")
    pub.add_argument("--weight", type=float, default=0.8, help="the sidecar's default weight")
    pub.add_argument("--preview", type=int, default=0, help="which sample prompt's image to use")
    pub.add_argument("--force", action="store_true", help="replace a LoRA with the same name")


def runs_root(config: Config) -> Path:
    return config.data_dir / "lora-runs"


def find_run(config: Config, ref: str) -> RunDir:
    root = runs_root(config)
    if (root / ref / "state.json").exists():
        return RunDir(root / ref)
    matches = sorted(p for p in root.glob(f"{ref}-*") if (p / "state.json").exists())
    if not matches:
        raise RunError(f"No run {ref!r} in {root}")
    return RunDir(matches[-1])


def trainer(config: Config, run_dir: RunDir, state: RunState) -> Trainer:
    cli = ColabCli(config.colab.binary, state.session, config.colab.auth)
    tunnel = SshTunnel(
        cli.proxy_command(str(config.ssh_key)),
        config.ssh_key,
        config.colab.worker_port,
        log_file=config.data_dir / "ssh-lora.log",
        name=state.session,
    )
    drive = DriveAuth(config.drive.client_file, config.drive_token_file)
    return Trainer(
        run_dir, state, cli, tunnel, drive, config.drive.root, config.colab.rclone_binary
    )


def _settings(args: argparse.Namespace) -> LoraSettings:
    try:
        return load_settings(args.dataset, {"base": args.base, "gpu": args.gpu})
    except ValidationError as e:
        sys.exit(f"{args.dataset / 'lora.toml'}: {e}")


def _check(config: Config, settings: LoraSettings, dataset: Path) -> ds.Report:
    report = ds.check(dataset, settings)
    index = config.data_dir / "degas.sqlite"
    if index.exists():
        db = Database(index)
        try:
            models = [
                a["path"]
                for a in db.list_assets("sdxl", "model")
                if not a["path"].startswith("models/sdxl/inpaint/")  # 9-channel UNets
            ]
        finally:
            db.close()
        if settings.base not in models:
            report.errors.append(
                f"{settings.base} isn't an SDXL checkpoint in the Drive index"
                f" (have: {', '.join(models) or 'none'}; run `degas rescan` after adding one)"
            )
    else:
        report.warnings.append("no Drive index yet, so the base checkpoint wasn't checked")
    for w in report.warnings:
        print(f"warning: {w}")
    for e in report.errors:
        print(f"error: {e}")
    n = len(report.items)
    if n:
        t = settings.train
        print(
            f"{n} images x {t.repeats_for(n)} repeats x {t.epochs} epochs, batch {t.batch_size}"
            f" ≈ {t.steps_for(n)} steps · {t.optimizer} lr {t.lr:g} {t.scheduler}"
            f" · rank {t.network_dim}/{t.network_alpha:g} · {settings.gpu}"
            f" {settings.mixed_precision} · base {settings.base}"
        )
        print(f"Trigger: {settings.subject!r} · output {settings.name}.safetensors")
    return report


async def _train(config: Config, args: argparse.Namespace) -> None:
    settings = _settings(args)
    report = _check(config, settings, args.dataset)
    if not report.ok:
        sys.exit("Fix the errors above first.")
    if not DriveAuth(config.drive.client_file, config.drive_token_file).authorized:
        sys.exit("Drive is not authorized: run `degas auth drive` first")
    run = new_run_id(settings.name)
    run_dir = RunDir(runs_root(config) / run)
    ds.prepare(report, run_dir.dataset)
    n = len(report.items)
    state = RunState(
        run=run,
        session=config.lora.session_name,
        settings=settings,
        images=n,
        steps=settings.train.steps_for(n),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    run_dir.save(state)
    print(f"Run {run} ({run_dir.path})")
    await ensure_key(config.ssh_key)
    t = trainer(config, run_dir, state)
    try:
        await t.start()
    except BusyError as e:
        sys.exit(str(e))
    except Exception as e:
        print(f"Start failed: {e}; stopping the VM", file=sys.stderr)
        with contextlib.suppress(ColabError):
            await t.stop_vm()
        sys.exit(1)
    await _follow(t, args.keep)


async def _follow(t: Trainer, keep: bool) -> None:
    run = t.state.run
    try:
        status = await t.follow()
    except (KeyboardInterrupt, asyncio.CancelledError):
        print(
            f"\nTraining keeps running on the VM (still using compute units)."
            f"\n  follow it: degas lora attach {run}\n  stop it:   degas lora stop {run}"
        )
        raise SystemExit(130) from None
    if keep:
        print(f"The VM is still running: degas lora stop {run} when you're done.")
    else:
        print("Stopping the VM")
        await t.stop_vm()
    if status == "done":
        print(f"Compare the samples, then: degas lora publish {run} --epoch N")
    else:
        sys.exit(1)


async def _attach(config: Config, args: argparse.Namespace) -> None:
    run_dir = find_run(config, args.run)
    state = run_dir.load()
    t = trainer(config, run_dir, state)
    if not await t.colab.is_alive():
        if state.status in ("starting", "training"):
            t.set_status("stopped", "the VM was gone")
        sys.exit(f"Run {state.run}: its VM ({state.session}) is no longer running")
    await t.tunnel.open()
    print(f"Following {state.run}")
    await _follow(t, args.keep)


async def _stop(config: Config, args: argparse.Namespace) -> None:
    run_dir = find_run(config, args.run)
    t = trainer(config, run_dir, run_dir.load())
    await t.stop_vm()
    print(f"Stopped {t.state.session}")


def _list(config: Config) -> None:
    root = runs_root(config)
    for path in sorted(root.glob("*/state.json")) if root.exists() else []:
        state = RunDir(path.parent).load()
        epochs = len(list((path.parent / "checkpoints").glob("*.safetensors")))
        print(f"{state.run}  {state.status:<8}  {epochs}/{state.settings.train.epochs} epochs")


def _publish(config: Config, args: argparse.Namespace) -> None:
    run_dir = find_run(config, args.run)
    state = run_dir.load()
    path = publish(
        run_dir,
        state,
        remote=config.lora.rclone_remote,
        drive_root=config.drive.root,
        epoch=args.epoch,
        label=args.label,
        weight=args.weight,
        preview_index=args.preview,
        force=args.force,
    )
    print(f"Published {path}")
    try:
        resp = httpx2.post(f"http://{config.host}:{config.port}/api/assets/rescan", timeout=120)
        resp.raise_for_status()
        print("Rescanned Drive")
    except httpx2.HTTPError:
        print("The Degas server didn't answer: run `degas rescan` (or Rescan Drive in the app)")


def run(config: Config, args: argparse.Namespace) -> None:
    try:
        match args.lora_command:
            case "check":
                if not _check(config, _settings(args), args.dataset).ok:
                    sys.exit(1)
            case "train":
                asyncio.run(_train(config, args))
            case "attach":
                asyncio.run(_attach(config, args))
            case "stop":
                asyncio.run(_stop(config, args))
            case "list":
                _list(config)
            case "publish":
                _publish(config, args)
    except (RunError, ColabError, RcloneError) as e:
        sys.exit(str(e))
