"""Filesystem layout on the VM."""

import os
import re
from dataclasses import dataclass
from pathlib import Path

_SHA256 = re.compile(r"[0-9a-f]{64}")


def is_sha256(value: str) -> bool:
    """Whether `value` can name a blob (and so is safe as a file name)."""
    return _SHA256.fullmatch(value) is not None


@dataclass(frozen=True)
class Paths:
    home: Path  # /content/degas: bundle, logs, outputs, blobs, rclone
    models: Path  # /content/models: local copy of Drive assets, same relative paths

    @property
    def outputs(self) -> Path:
        return self.home / "outputs"

    @property
    def blobs(self) -> Path:
        return self.home / "blobs"

    @property
    def rclone_conf(self) -> Path:
        return self.home / "rclone.conf"

    @property
    def rclone_bin(self) -> Path:
        return self.home / "bin" / "rclone"

    def ensure(self) -> None:
        for d in (self.home, self.models, self.outputs, self.blobs):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "Paths":
        return cls(
            home=Path(os.environ.get("DEGAS_WORKER_HOME", "/content/degas")),
            models=Path(os.environ.get("DEGAS_MODELS", "/content/models")),
        )
