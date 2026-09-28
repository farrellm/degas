"""Packages the Colab image lacks, installed the first time something needs them."""

import importlib
import subprocess
import sys
import threading
from types import ModuleType

_lock = threading.Lock()
INSTALL_TIMEOUT_S = 600


def ensure(module: str, package: str) -> ModuleType:
    """Import `module`, installing `package` with pip first if it's missing."""
    with _lock:
        try:
            return importlib.import_module(module)
        except ImportError:
            pass
        done = subprocess.run(  # noqa: S603 - fixed arguments
            [sys.executable, "-m", "pip", "install", "-q", package],
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT_S,
            check=False,
        )
        if done.returncode != 0:
            tail = (done.stderr or done.stdout).strip().splitlines()[-1:]
            raise ValueError(f"Could not install {package}: {' '.join(tail)}")
        importlib.invalidate_caches()
        return importlib.import_module(module)
