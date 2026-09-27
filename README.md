# Degas

Personal generative image/video web app: an iPhone PWA over Tailscale, with inference on Google Colab GPUs driven by the [Colab CLI](https://github.com/googlecolab/google-colab-cli).

- Design: [docs/design.md](docs/design.md)
- UX and visual design: [docs/ux.md](docs/ux.md)
- Phase 0 (Colab CLI spike): [docs/phase0-findings.md](docs/phase0-findings.md)

## Layout

| Path | What |
|---|---|
| `server/` | Home server (FastAPI): API, job queue, Colab session manager — package `degas` |
|  | `colab/` CLI wrapper, SSH tunnel, worker client, session manager · `dispatcher.py` job loop · `drive.py` OAuth + index · `families/` descriptors |
| `worker/` | GPU worker (FastAPI) that runs on the Colab VM — package `degas_worker` (`jobs.py`, `cache.py` rclone, `families/` runners) |
| `web/` | React + Vite PWA |
| `spike/` | Throwaway Phase 0 scripts, kept for reference (not linted) |

## Running

Phases 1–2 are implemented: SDXL text-to-image with LoRAs, and a model cache on the GPU VM. One-time setup on the home server:

1. Install and authenticate the Colab CLI: `uv tool install google-colab-cli`, then run any `colab` command once to sign in.
2. Create a Google Cloud OAuth client of type *Desktop app* with the Drive API enabled, and download its JSON.
3. `cp degas.toml.example degas.toml` and point `drive.client_file` at that JSON.
4. `uv run degas auth drive`: open the printed URL in a browser on the server (or forward the loopback port over SSH) and approve read-only Drive access.
5. Put SDXL checkpoints in Drive under `My Drive/degas/models/sdxl/` and LoRAs under `My Drive/degas/loras/sdxl/`, optionally each with a `<name>.yaml` sidecar (label, trigger words, default weight) and a `<name>.jpg` preview. Then run `uv run degas rescan` (or **Rescan Drive** in the app).
6. `make build && uv run degas` serves the API and the PWA on `127.0.0.1:8420`; `tailscale serve --bg https / http://127.0.0.1:8420` exposes it on the tailnet.

In the app: tap **No GPU** in the header to start a GPU session, write a prompt in **Create** → Generate, and watch images arrive in **Results**. The session stops itself after `idle_timeout_min` without activity.

## Development

Requires [uv](https://docs.astral.sh/uv/), Node ≥ 24 and pnpm.

```sh
make install     # uv sync + pnpm install + git hooks
make check       # lint + typecheck + test (what CI runs)
make fmt         # ruff format/fix + prettier
make dev-server  # API on 127.0.0.1:8420
make dev-web     # Vite dev server, proxies /api to the API
```

Git hooks ([pre-commit](https://pre-commit.com), config in `.pre-commit-config.yaml`): on commit, ruff / Prettier / ESLint fix staged files plus generic file checks; on push, `make check`. Bypass once with `--no-verify`.

Tooling: ruff (lint + format), mypy (strict), pytest · ESLint (typescript-eslint strict, type-checked), Prettier, Vitest + Testing Library.
