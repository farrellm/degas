"""SSH to the Colab VM through `colab ssh --proxy-mode`, with a forwarded worker port.

All SSH traffic goes through one ControlMaster connection. The transport is kept
behind this one interface in case `colab ssh` changes (design §11).
"""

import asyncio
import contextlib
import logging
import os
import shlex
import socket
from pathlib import Path
from typing import Protocol

from degas.errors import DegasError

log = logging.getLogger(__name__)

HOST = "root@colab-runtime"


class TunnelError(DegasError):
    pass


class Tunnel(Protocol):
    @property
    def local_port(self) -> int: ...

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def is_open(self) -> bool: ...

    async def run(self, command: str, timeout: float = 120) -> str:
        """Run a shell command on the VM; raise TunnelError on a non-zero exit."""
        ...

    async def upload(self, local: Path, remote: str) -> None: ...


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def control_dir() -> Path:
    # ControlPath must stay under 108 bytes, hence the short runtime dir.
    return Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")  # noqa: S108


class SshTunnel:
    def __init__(
        self,
        proxy_command: str,
        key: Path,
        remote_port: int,
        log_file: Path | None = None,
        name: str = "degas",
    ) -> None:
        self.proxy_command = proxy_command
        # Every VM is `root@colab-runtime`, so each Colab session needs its own ControlPath.
        self.name = name
        self.key = key
        self.remote_port = remote_port
        self.log_file = log_file
        self._local_port = 0

    @property
    def local_port(self) -> int:
        return self._local_port

    def _opts(self) -> list[str]:
        return [
            "-o", f"ControlPath={control_dir()}/{self.name}-%C",
            "-o", f"ProxyCommand={self.proxy_command}",
            "-o", f"IdentityFile={self.key}",
            "-o", "IdentitiesOnly=yes",
            # Every VM has a fresh host key; the proxy connection is already authenticated.
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
            "-o", "BatchMode=yes",
        ]  # fmt: skip

    async def _exec(self, argv: list[str], timeout: float, detach: bool = False) -> tuple[int, str]:
        log.debug("run %s", argv[:1] + argv[-3:])
        if detach:
            # `ssh -f` forks a background master that keeps inherited fds open, so its
            # output must not go to a pipe we wait on.
            err_path = self.log_file or Path(os.devnull)
            with err_path.open("ab") as err:
                proc = await asyncio.create_subprocess_exec(
                    *argv, stdin=asyncio.subprocess.DEVNULL, stdout=err, stderr=err
                )
                async with asyncio.timeout(timeout):
                    code = await proc.wait()
            return code, ""
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            async with asyncio.timeout(timeout):
                out, _ = await proc.communicate()
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TunnelError(f"SSH command timed out after {timeout:.0f}s") from None
        assert proc.returncode is not None
        return proc.returncode, out.decode(errors="replace")

    async def open(self) -> None:
        await self.close()
        self._local_port = free_port()
        argv = [
            "ssh",
            *self._opts(),
            "-o", "ControlMaster=yes",
            "-o", "ExitOnForwardFailure=yes",
            "-f", "-N",
            "-L", f"127.0.0.1:{self._local_port}:127.0.0.1:{self.remote_port}",
            HOST,
        ]  # fmt: skip
        try:
            code, _ = await self._exec(argv, timeout=90, detach=True)
        except TimeoutError:
            raise TunnelError("Timed out opening the SSH tunnel") from None
        if code != 0:
            raise TunnelError(f"Could not open the SSH tunnel (exit {code}); see {self.log_file}")

    async def is_open(self) -> bool:
        code, _ = await self._exec(["ssh", *self._opts(), "-O", "check", HOST], timeout=15)
        return code == 0

    async def close(self) -> None:
        with contextlib.suppress(TunnelError, OSError):
            await self._exec(["ssh", *self._opts(), "-O", "exit", HOST], timeout=15)

    async def run(self, command: str, timeout: float = 120) -> str:
        code, out = await self._exec(["ssh", *self._opts(), HOST, command], timeout)
        if code != 0:
            raise TunnelError(f"Remote command failed (exit {code}): {command}\n{out.strip()}")
        return out

    async def upload(self, local: Path, remote: str) -> None:
        argv = ["scp", "-q", *self._opts(), str(local), f"{HOST}:{shlex.quote(remote)}"]
        code, out = await self._exec(argv, timeout=600)
        if code != 0:
            raise TunnelError(f"scp {local} failed (exit {code}): {out.strip()}")

    async def download(self, remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        argv = ["scp", "-q", *self._opts(), f"{HOST}:{shlex.quote(remote)}", str(local)]
        code, out = await self._exec(argv, timeout=600)
        if code != 0:
            raise TunnelError(f"scp {remote} failed (exit {code}): {out.strip()}")
