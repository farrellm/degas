.PHONY: install fmt lint typecheck test check dev-server dev-web build

install:
	uv sync
	cd web && pnpm install

fmt:
	uv run ruff format .
	uv run ruff check --fix .
	cd web && pnpm format

lint:
	uv run ruff check .
	uv run ruff format --check .
	cd web && pnpm lint && pnpm format:check

typecheck:
	uv run mypy
	cd web && pnpm typecheck

test:
	uv run pytest
	cd web && pnpm test

# Everything CI runs.
check: lint typecheck test

build:
	cd web && pnpm build

dev-server:
	uv run uvicorn degas.app:app --reload --host 127.0.0.1 --port 8420

dev-web:
	cd web && pnpm dev
