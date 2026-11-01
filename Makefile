.PHONY: help tools fmt fmt-check lint test check build dev-server dev-web deploy logs

SHELL := bash
.SHELLFLAGS := -euo pipefail -c

BIN := $(CURDIR)/bin
export PATH := $(BIN):$(PATH)

# Pinned tool versions; Renovate bumps them via the comments.
# renovate: datasource=github-releases depName=gitleaks/gitleaks
GITLEAKS_VERSION   := v8.30.1
# renovate: datasource=github-releases depName=evilmartians/lefthook
LEFTHOOK_VERSION   := v2.2.2
# renovate: datasource=github-releases depName=koalaman/shellcheck
SHELLCHECK_VERSION := v0.11.0
# renovate: datasource=github-releases depName=mvdan/sh
SHFMT_VERSION      := v3.14.1
# renovate: datasource=pypi depName=pip-audit
PIP_AUDIT_VERSION  := 2.10.1

PRETTIER := web/node_modules/.bin/prettier
SH_FILES := $(shell git ls-files --cached --others --exclude-standard '*.sh')

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────

web/node_modules: web/pnpm-lock.yaml
	cd web && pnpm install --frozen-lockfile
	@touch $@

tools: ## Install pinned tools into ./bin, the Python and web packages, and the git hooks
	@mkdir -p $(BIN)
	curl -sSfL https://github.com/gitleaks/gitleaks/releases/download/$(GITLEAKS_VERSION)/gitleaks_$(GITLEAKS_VERSION:v%=%)_linux_x64.tar.gz \
		| tar -xz -C $(BIN) gitleaks
	curl -sSfL https://github.com/evilmartians/lefthook/releases/download/$(LEFTHOOK_VERSION)/lefthook_$(LEFTHOOK_VERSION:v%=%)_Linux_x86_64.gz \
		| gunzip > $(BIN)/lefthook
	curl -sSfL https://github.com/koalaman/shellcheck/releases/download/$(SHELLCHECK_VERSION)/shellcheck-$(SHELLCHECK_VERSION).linux.x86_64.tar.xz \
		| tar -xJ -C $(BIN) --strip-components=1 shellcheck-$(SHELLCHECK_VERSION)/shellcheck
	curl -sSfL -o $(BIN)/shfmt \
		https://github.com/mvdan/sh/releases/download/$(SHFMT_VERSION)/shfmt_$(SHFMT_VERSION)_linux_amd64
	chmod +x $(BIN)/lefthook $(BIN)/shfmt
	uv sync --locked
	cd web && pnpm install --frozen-lockfile
	lefthook install

# ── Checks ────────────────────────────────────────────────────────────────────

fmt: web/node_modules ## Format everything in place
	uv run ruff format .
	uv run ruff check --fix .
	$(PRETTIER) --write --log-level warn .
	shfmt -w $(SH_FILES)

fmt-check: web/node_modules ## Fail if anything needs formatting
	uv run ruff format --check .
	$(PRETTIER) --check --log-level warn .
	shfmt -d $(SH_FILES)

lint: web/node_modules ## Run every linter and type checker
	uv run ruff check .
	uv run mypy
	# colabkit is our own git source: no hash to require, nothing in the advisory DB
	uv export --frozen --all-packages --no-emit-workspace --no-emit-package colabkit -q \
		| uvx pip-audit==$(PIP_AUDIT_VERSION) --disable-pip --require-hashes -r /dev/stdin --progress-spinner off
	cd web && pnpm -s lint
	shellcheck $(SH_FILES)

test: web/node_modules ## Run the Python and web tests
	uv run pytest
	cd web && pnpm -s test

# Not `build`: the prod server serves web/dist straight from disk.
check: fmt-check lint test ## Everything CI runs
	gitleaks git --no-banner --redact .

# ── Run ───────────────────────────────────────────────────────────────────────

build: web/node_modules ## Build the PWA into web/dist (live at once: the server serves it)
	cd web && pnpm build

dev-server: ## Run the API with reload on 127.0.0.1:8420
	uv run uvicorn --factory degas.app:create_app --reload --host 127.0.0.1 --port 8420

dev-web: web/node_modules ## Run the Vite dev server (proxies /api to the API)
	cd web && pnpm dev

# ── Deploy ────────────────────────────────────────────────────────────────────

# Production: a systemd user unit runs the server (which also serves web/dist)
# on 127.0.0.1:8420 and publishes it with `tailscale serve --https=8448`.
# Survives reboot only with `sudo loginctl enable-linger $USER`.
UNIT_DIR := $(HOME)/.config/systemd/user

deploy: build ## Build, sync the venv, and (re)start the systemd user unit
	uv sync --frozen
	install -Dm644 deploy/degas.service $(UNIT_DIR)/degas.service
	systemctl --user daemon-reload
	systemctl --user enable degas
	systemctl --user restart degas

logs: ## Follow the deployed unit's logs
	journalctl --user -u degas -f
