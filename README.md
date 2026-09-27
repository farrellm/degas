# Degas

Personal generative image/video web app: an iPhone PWA over Tailscale, with inference on Google Colab GPUs driven by the [Colab CLI](https://github.com/googlecolab/google-colab-cli).

- Design: [docs/design.md](docs/design.md)
- Phase 0 (Colab CLI spike): [docs/phase0-findings.md](docs/phase0-findings.md)

## Layout

| Path | What |
|---|---|
| `server/` | Home server (FastAPI): API, job queue, Colab session manager — package `degas` |
| `worker/` | GPU worker (FastAPI) that runs on the Colab VM — package `degas_worker` |
| `web/` | React + Vite PWA |
| `spike/` | Throwaway Phase 0 scripts, kept for reference (not linted) |

## Development

Requires [uv](https://docs.astral.sh/uv/), Node ≥ 24 and pnpm.

```sh
make install     # uv sync + pnpm install
make check       # lint + typecheck + test (what CI runs)
make fmt         # ruff format/fix + prettier
make dev-server  # API on 127.0.0.1:8420
make dev-web     # Vite dev server, proxies /api to the API
```

Tooling: ruff (lint + format), mypy (strict), pytest · ESLint (typescript-eslint strict, type-checked), Prettier, Vitest + Testing Library.
