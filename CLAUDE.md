# Degas

Personal gen-image/video PWA (iPhone over Tailscale); inference on Colab GPUs via the `colab` CLI.
Read `README.md` for layout/setup, `docs/design.md` for architecture and the phase plan (§10),
`docs/ux.md` for tokens, vocabulary and per-phase UX. Read both docs before starting a phase.

## Commands

```sh
make check        # lint + typecheck + test — exactly what CI runs; run before declaring done
make fmt          # ruff format/fix + prettier
make dev-server   # FastAPI on 127.0.0.1:8420
make dev-web      # Vite; proxies /api to dev-server
uv run pytest server/tests/test_api.py -k name   # single Python test
cd web && pnpm vitest run src/prompt.test.ts      # single web test
make deploy / make logs   # prod: systemd user unit + `tailscale serve --https=8448`
```

## Architecture

- `server/src/degas` (home server): API, SQLite, blob store, dispatcher, Colab session manager
  (`colab/`: CLI wrapper, SSH tunnel, worker client, bundle).
- `worker/src/degas_worker` (runs on the Colab VM, behind the SSH tunnel): tarred by
  `degas/colab/bundle.py` and scp'd over — must never import `degas`.
- `web/`: React 19 + Vite + TanStack Query. The Create form is rendered from each family's
  JSON Schema (`SchemaForm.tsx`), so new families usually need no frontend changes.

## Gotchas

- A model family = descriptor `server/src/degas/families/<id>.py` + runner
  `worker/src/degas_worker/families/<id>.py`, same name. Register the descriptor in `FAMILIES`
  (`server/src/degas/families/__init__.py`) and the runner in `RUNNERS`
  (`worker/src/degas_worker/families/__init__.py`) via a factory that imports the module lazily.
- torch/diffusers/transformers/peft/accelerate are not installed locally or in CI (mypy
  `ignore_missing_imports`). Runner modules may import them at top level, but nothing else may
  import a runner module eagerly — tests and the worker app must load without GPU deps.
- Tests fake the Colab CLI and tunnel but run the real worker app over ASGI
  (`server/tests/conftest.py`); no GPU needed.
- `spike/` is a frozen Phase 0 record — excluded from lint; don't edit.
- Pre-commit: hooks fix staged files on commit; `make check` runs in GitHub CI only.

## Workflow

- Phase work is committed as `Phase N: <title>`; update `docs/design.md` §10 status
  (and record live GPU test results there) when a phase lands.
