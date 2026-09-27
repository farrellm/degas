"""Server configuration, read from `degas.toml`."""

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

# Repository checkout: <repo>/server/src/degas/config.py
_REPO = Path(__file__).resolve().parents[3]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColabConfig(_Section):
    binary: str = "colab"
    auth: Literal["oauth2", "adc"] | None = None  # None: the CLI's default
    session_name: str = "degas"
    ssh_key: Path | None = None  # default: <data_dir>/ssh/id_ed25519 (generated)
    # Static linux-amd64 rclone to upload. If unset, the VM downloads the official build.
    rclone_binary: Path | None = None
    worker_port: int = 8765
    # The VM's model cache is evicted least-recently-used above this size (it has ~190 GB free).
    cache_budget_gb: float = 150
    # Trivial `colab exec` every few minutes in case an idle kernel gets the VM reclaimed.
    exec_heartbeat: bool = True


class DriveConfig(_Section):
    root: str = "degas"  # folder under My Drive
    client_file: Path | None = None  # OAuth client JSON (desktop app)
    token_file: Path | None = None  # default: <data_dir>/drive_token.json


class Config(_Section):
    data_dir: Path = Path("data")
    host: str = "127.0.0.1"
    port: int = 8420
    idle_timeout_min: float = 15
    web_dist: Path | None = None  # default: <repo>/web/dist
    colab: ColabConfig = ColabConfig()
    drive: DriveConfig = DriveConfig()

    @property
    def ssh_key(self) -> Path:
        return self.colab.ssh_key or self.data_dir / "ssh" / "id_ed25519"

    @property
    def drive_token_file(self) -> Path:
        return self.drive.token_file or self.data_dir / "drive_token.json"

    @property
    def web_dist_dir(self) -> Path:
        return self.web_dist or _REPO / "web" / "dist"


def load_config(path: Path | None = None) -> Config:
    """Load `path`, else `$DEGAS_CONFIG`, else `./degas.toml`; defaults if none exists.

    Relative paths in the file are resolved against the file's directory.
    """
    if path is None:
        env = os.environ.get("DEGAS_CONFIG")
        path = Path(env) if env else Path("degas.toml")
        if not env and not path.exists():
            return Config()
    with path.open("rb") as f:
        config = Config.model_validate(tomllib.load(f))
    base = path.resolve().parent

    def resolve(p: Path | None) -> Path | None:
        return None if p is None else (base / p.expanduser())

    return config.model_copy(
        update={
            "data_dir": resolve(config.data_dir),
            "web_dist": resolve(config.web_dist),
            "colab": config.colab.model_copy(
                update={
                    "ssh_key": resolve(config.colab.ssh_key),
                    "rclone_binary": resolve(config.colab.rclone_binary),
                }
            ),
            "drive": config.drive.model_copy(
                update={
                    "client_file": resolve(config.drive.client_file),
                    "token_file": resolve(config.drive.token_file),
                }
            ),
        }
    )
