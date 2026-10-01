"""`degas` command-line entry point."""

import argparse
import asyncio
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import uvicorn

from degas.app import create_app
from degas.civitai import cli as civitai_cli
from degas.config import Config, load_config
from degas.db import Database
from degas.drive import DriveAuth, DriveIndexer, authorize_interactive
from degas.lora import cli as lora_cli


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args.run(config, args)


def build_parser() -> argparse.ArgumentParser:
    """Each subcommand sets `run`: the function that carries it out, given the config."""
    parser = argparse.ArgumentParser(prog="degas")
    parser.add_argument("--config", type=Path, help="path to degas.toml")
    parser.set_defaults(run=serve)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the server (default)").set_defaults(run=serve)
    auth = sub.add_parser("auth", help="one-time authorization")
    auth.add_argument("what", choices=["drive"])
    auth.add_argument("--port", type=int, default=0, help="loopback port for the OAuth redirect")
    auth.set_defaults(run=authorize)
    sub.add_parser("rescan", help="re-index the Drive folder").set_defaults(run=rescan)
    lora_cli.add_parser(sub)
    civitai_cli.add_parser(sub)
    return parser


def serve(config: Config, _args: argparse.Namespace) -> None:
    uvicorn.run(create_app(config), host=config.host, port=config.port)


def authorize(config: Config, args: argparse.Namespace) -> None:
    if config.drive.client_file is None:
        sys.exit("Set drive.client_file in degas.toml to your OAuth client JSON first.")
    authorize_interactive(config.drive.client_file, config.drive_token_file, port=args.port)
    print(f"Saved the Drive refresh token to {config.drive_token_file}")


def rescan(config: Config, _args: argparse.Namespace) -> None:
    async def scan() -> int:
        config.data_dir.mkdir(parents=True, exist_ok=True)
        db = Database(config.data_dir / "degas.sqlite")
        drive = DriveAuth(config.drive.client_file, config.drive_token_file)
        indexer = DriveIndexer(drive, config.drive.root)
        try:
            return db.replace_assets(await indexer.scan())
        finally:
            await indexer.aclose()
            await drive.aclose()
            db.close()

    print(f"Indexed {asyncio.run(scan())} assets")
