"""`degas` command-line entry point."""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import uvicorn

from degas.config import load_config
from degas.db import Database
from degas.drive import DriveAuth, DriveIndexer, authorize_interactive


def main() -> None:
    parser = argparse.ArgumentParser(prog="degas")
    parser.add_argument("--config", type=Path, help="path to degas.toml")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the server (default)")
    auth = sub.add_parser("auth", help="one-time authorization")
    auth.add_argument("what", choices=["drive"])
    auth.add_argument("--port", type=int, default=0, help="loopback port for the OAuth redirect")
    sub.add_parser("rescan", help="re-index the Drive folder")
    args = parser.parse_args()
    config = load_config(args.config)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if args.command in (None, "serve"):
        if args.config:
            os.environ["DEGAS_CONFIG"] = str(args.config)
        uvicorn.run("degas.app:app", host=config.host, port=config.port)
    elif args.command == "auth":
        if config.drive.client_file is None:
            sys.exit("Set drive.client_file in degas.toml to your OAuth client JSON first.")
        authorize_interactive(config.drive.client_file, config.drive_token_file, port=args.port)
        print(f"Saved the Drive refresh token to {config.drive_token_file}")
    elif args.command == "rescan":

        async def rescan() -> int:
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

        print(f"Indexed {asyncio.run(rescan())} assets")
