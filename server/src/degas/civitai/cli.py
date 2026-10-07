"""`degas civitai …` subcommands."""

import argparse
import asyncio
import logging
import sys

import httpx2

from degas.civitai.client import Civitai, CivitaiError
from degas.civitai.huggingface import HuggingFace, HuggingFaceError
from degas.civitai.importer import CivitaiImportError, Importer
from degas.civitai.plan import ImportPlan, PlanError
from degas.config import Config
from degas.db import AssetRow, Database
from degas.families import FAMILIES
from degas.rclone import AsyncRclone, RcloneError

GB = 1000**3
MB = 1000**2


def add_parser(sub: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    civitai = sub.add_parser("civitai", help="import LoRAs from Civitai or Hugging Face")
    civitai.set_defaults(run=run)
    cmd = civitai.add_subparsers(dest="civitai_command", required=True)
    imp = cmd.add_parser("import", help="copy a LoRA into loras/<family>/ with a sidecar")
    imp.add_argument(
        "url",
        help="a Civitai model or version link, an AIR, a version id, or a Hugging Face link",
    )
    imp.add_argument(
        "--family", choices=sorted(FAMILIES), help="default: from its base model (or name)"
    )
    imp.add_argument("--name", help="file name without .safetensors (default: from its name)")
    imp.add_argument("--weight", type=float, help="the sidecar's default weight (0.8)")
    imp.add_argument("--dry-run", action="store_true", help="show the plan and stop")
    imp.add_argument("--force", action="store_true", help="import even if it's already there")
    back = cmd.add_parser("backfill", help="write sidecars for LoRAs in Drive that have none")
    back.add_argument("--family", choices=sorted(FAMILIES))
    back.add_argument("--dry-run", action="store_true", help="look them up but write nothing")


def importer(config: Config) -> Importer:
    return Importer(
        Civitai(config.civitai.token_file, config.civitai.api_base),
        AsyncRclone(),
        config.lora.rclone_remote,
        config.drive.root,
        set(FAMILIES),
        HuggingFace(config.huggingface.token_file, config.huggingface.api_base),
    )


def load_index(config: Config) -> list[AssetRow]:
    path = config.data_dir / "degas.sqlite"
    if not path.exists():
        print("warning: no Drive index yet, so duplicates weren't checked")
        return []
    db = Database(path)
    try:
        return db.list_assets(kind="lora")
    finally:
        db.close()


def size(n: int) -> str:
    return f"{n / GB:.1f} GB" if n >= GB else f"{max(1, round(n / MB))} MB"


def show(plan: ImportPlan) -> None:
    base = plan.base_model or "unknown base"
    print(f"{plan.model_name} · {plan.version_name} · {base} → {plan.family}")
    for f in plan.files:
        print(f"  {f.civitai_name} ({size(f.size)}) → {f.path}")
    print(f"  Trigger words: {', '.join(plan.trigger_words) or 'none'}")
    print(
        f"  Weight {plan.weight:g}{' · preview from the first example' if plan.preview_url else ''}"
    )
    for w in plan.warnings:
        print(f"warning: {w}")


def rescan(config: Config) -> None:
    try:
        resp = httpx2.post(f"http://{config.host}:{config.port}/api/assets/rescan", timeout=120)
        resp.raise_for_status()
        print("Rescanned Drive")
    except httpx2.HTTPError:
        print("The Degas server didn't answer: run `degas rescan` (or Rescan Drive in the app)")


async def _import(config: Config, args: argparse.Namespace) -> None:
    imp = importer(config)
    try:
        plan = await imp.plan(
            args.url, load_index(config), family=args.family, name=args.name,
            weight=args.weight, force=args.force,
        )  # fmt: skip
        show(plan)
        if args.dry_run:
            return

        def progress(stage: str, done: int, total: int) -> None:
            if stage == "download":
                pct = 100 * done // total if total else 0
                print(f"\r  Copying to Drive: {pct}% of {size(total)}", end="", flush=True)
            elif stage == "sidecar":
                print()

        paths = await imp.run(plan, progress)
    finally:
        await imp.civitai.aclose()
        await imp.hf.aclose()
    for p in paths:
        print(f"Imported {p}")
    rescan(config)


async def _backfill(config: Config, args: argparse.Namespace) -> None:
    imp = importer(config)
    loras = [
        a
        for a in load_index(config)
        if not a.get("sidecar") and (args.family is None or a["family"] == args.family)
    ]
    written = 0
    try:
        for asset in loras:
            sha256 = asset.get("sha256")
            if not sha256:
                print(f"{asset['path']}: no SHA-256 in the index (rescan Drive first)")
                continue
            if args.dry_run:
                version = await imp.civitai.by_hash(sha256)
                found = (
                    version
                    and f"{(version.get('model') or {}).get('name')} ({version['baseModel']})"
                )
                print(f"{asset['path']}: {found or 'not on Civitai'}")
                continue
            try:
                plan = await imp.backfill(asset)
            except PlanError as e:
                print(f"{asset['path']}: {e}")
                continue
            print(f"{asset['path']}: {plan.model_name if plan else 'not on Civitai'}")
            written += plan is not None
    finally:
        await imp.civitai.aclose()
    if not loras:
        print("Every LoRA in the index has a sidecar")
    if written:
        rescan(config)


def run(config: Config, args: argparse.Namespace) -> None:
    logging.getLogger("httpx2").setLevel(logging.WARNING)  # a line per request otherwise
    try:
        match args.civitai_command:
            case "import":
                asyncio.run(_import(config, args))
            case "backfill":
                asyncio.run(_backfill(config, args))
    except (CivitaiError, HuggingFaceError, PlanError, CivitaiImportError, RcloneError) as e:
        sys.exit(str(e))
