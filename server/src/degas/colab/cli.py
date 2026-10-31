"""Degas's one Colab session over `colabkit`: the `Colab` protocol is the seam tests fake."""

from typing import Protocol

from colabkit import AsyncColabCli, ColabError, Usage

__all__ = ["Colab", "ColabCli", "ColabError", "Usage"]


class Colab(Protocol):
    session: str

    async def new(self, gpu: str | None, high_mem: bool) -> None: ...

    async def stop(self) -> None: ...

    async def is_alive(self) -> bool: ...

    async def exec(self, code: str, timeout: float = 120) -> str: ...

    async def usage(self) -> Usage: ...

    def proxy_command(self, identity: str) -> str: ...


class ColabCli:
    """`colabkit.AsyncColabCli` bound to one session name."""

    def __init__(self, binary: str = "colab", session: str = "degas", auth: str | None = None):
        self.cli = AsyncColabCli(binary, auth)
        self.session = session

    async def new(self, gpu: str | None, high_mem: bool) -> None:
        await self.cli.new(self.session, gpu, high_mem)

    async def stop(self) -> None:
        await self.cli.stop(self.session)

    async def is_alive(self) -> bool:
        """True if the backend still has the VM (`status` syncs and prunes stale sessions)."""
        return await self.cli.status(self.session)

    async def exec(self, code: str, timeout: float = 120) -> str:
        return await self.cli.exec(self.session, code, timeout=timeout)

    async def usage(self) -> Usage:
        return await self.cli.usage()

    def proxy_command(self, identity: str) -> str:
        return self.cli.proxy_command(self.session, identity)
