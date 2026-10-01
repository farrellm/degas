.PHONY: install fmt lint typecheck test check dev-server dev-web build deploy logs

install:
	uv sync
	cd web && pnpm install
	uv run pre-commit install

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
	uv run uvicorn --factory degas.app:create_app --reload --host 127.0.0.1 --port 8420

dev-web:
	cd web && pnpm dev

# Production: a systemd user unit runs the server (which also serves web/dist)
# on 127.0.0.1:8420 and publishes it with `tailscale serve --https=8448`.
# Survives reboot only with `sudo loginctl enable-linger $USER`.
UNIT_DIR := $(HOME)/.config/systemd/user

deploy: build
	uv sync --frozen
	install -Dm644 deploy/degas.service $(UNIT_DIR)/degas.service
	systemctl --user daemon-reload
	systemctl --user enable degas
	systemctl --user restart degas

logs:
	journalctl --user -u degas -f
