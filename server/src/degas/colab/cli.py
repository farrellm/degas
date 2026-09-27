"""Async wrapper around the `colab` CLI.

Phase 0: exit codes are unreliable (`exec` exits 0 when the code raises; `status`
exits 0 for an unknown session), so results are judged from the output text.
"""

import asyncio
import logging
import re
from typing import Protocol

log = logging.getLogger(__name__)

_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class ColabError(RuntimeError):
    pass


class Colab(Protocol):
    session: str

    async def new(self, gpu: str | None, high_mem: bool) -> None: ...

    async def stop(self) -> None: ...

    async def is_alive(self) -> bool: ...

    async def exec(self, code: str, timeout: float = 120) -> str: ...

    def proxy_command(self, identity: str) -> str: ...


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


class ColabCli:
    def __init__(self, binary: str = "colab", session: str = "degas", auth: str | None = None):
        self.binary = binary
        self.session = session
        self.auth = auth

    def _argv(self, *args: str) -> list[str]:
        global_opts = ["--auth", self.auth] if self.auth else []
        return [self.binary, *global_opts, *args]

    async def _run(self, *args: str, stdin: str | None = None, timeout: float = 300) -> str:
        argv = self._argv(*args)
        log.debug("run %s", argv)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            async with asyncio.timeout(timeout):
                out, _ = await proc.communicate(stdin.encode() if stdin is not None else None)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ColabError(f"`colab {args[0]}` timed out after {timeout:.0f}s") from None
        text = strip_ansi(out.decode(errors="replace"))
        if proc.returncode != 0:
            raise ColabError(f"`colab {args[0]}` failed (exit {proc.returncode}): {text.strip()}")
        return text

    async def new(self, gpu: str | None, high_mem: bool) -> None:
        args = ["new", "-s", self.session]
        if gpu:
            args += ["--gpu", gpu]
        if high_mem:
            args.append("--high-mem")
        out = await self._run(*args, timeout=600)
        if "error" in out.lower() and not await self.is_alive():
            raise ColabError(out.strip())

    async def stop(self) -> None:
        await self._run("stop", "-s", self.session, timeout=120)

    async def is_alive(self) -> bool:
        """True if the backend still has the VM (`status` syncs and prunes stale sessions)."""
        out = await self._run("status", "-s", self.session, timeout=60)
        return any(line.startswith(f"[{self.session}] ") for line in out.splitlines())

    async def exec(self, code: str, timeout: float = 120) -> str:
        return await self._run("exec", "-s", self.session, stdin=code, timeout=timeout)

    def proxy_command(self, identity: str) -> str:
        return " ".join([*self._argv("ssh", "--proxy-mode", "-s", self.session, "-i", identity)])
