"""`degas` command-line entry point."""

import uvicorn


def main() -> None:
    uvicorn.run("degas.app:app", host="127.0.0.1", port=8420)
