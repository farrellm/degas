# Degas

Personal gen-image/video PWA (iPhone over Tailscale); inference on Colab GPUs via the `colab` CLI.
Read `README.md` for layout/setup, `docs/design.md` for architecture and the phase plan (§10),
`docs/ux.md` for tokens, vocabulary and per-phase UX. Read both docs before starting a phase.

## Commands

```sh
make check        # lint + typecheck + test; run before declaring done (CI adds `pnpm build`)
uv run ruff check . && uv run mypy && uv run pytest   # Python only (web untouched)
uv run pytest server/tests/test_api.py -k name   # single Python test
cd web && pnpm vitest run src/lib/schema.test.ts # single web test
```

## Architecture

- `server/src/degas` (home server): `api/` (a router per resource, request schemas, and
  `errors.py`, the one table mapping `DegasError`s to HTTP statuses), `db/` (schema, queries,
  row TypedDicts), blob store, dispatcher, Colab session manager (`colab/`: CLI wrapper, SSH
  tunnel, worker client, bundle). `services.py` wires them; `degas.app:create_app` is a
  factory (no module-level app).
- `worker/src/degas_worker` (runs on the Colab VM, behind the SSH tunnel): tarred by
  `degas/colab/bundle.py` and scp'd over — must never import `degas`. `degas_worker/spec.py`
  types the job spec for both sides (the server imports it).
- `web/`: React 19 + Vite + TanStack Query; its conventions are in `web/CLAUDE.md`. The Create
  form is rendered from each family's JSON Schema (`features/create/schema-form/`), so new
  families usually need no frontend changes.

## Gotchas

- A model family = descriptor `server/src/degas/families/<id>.py` + runner
  `worker/src/degas_worker/families/<id>.py` (or a package `<id>/`, as SDXL is), same name.
  Shared pieces: `families/validation.py` and `schema.py` for descriptors,
  `families/runtime.py` for runners. Register the descriptor in `FAMILIES`
  (`server/src/degas/families/__init__.py`) and the runner in `RUNNERS`
  (`worker/src/degas_worker/families/__init__.py`) via a factory that imports the module lazily.
- torch/diffusers/transformers/peft/accelerate are not installed locally or in CI (mypy
  `ignore_missing_imports`). Runner modules may import them at top level, but nothing else may
  import a runner module eagerly — tests and the worker app must load without GPU deps.
- A route that returns a stored row is annotated `Mapping[str, Any]`, not the row's TypedDict:
  FastAPI would make the TypedDict a response model and drop keys the route adds.
- Tests fake the Colab CLI and tunnel but run the real worker app over ASGI
  (`server/tests/conftest.py`); no GPU needed.
- `spike/` is a frozen Phase 0 record — excluded from lint; don't edit.
- The prod service serves `web/dist` from disk, so `pnpm build` / `make build` changes what
  the phone gets at once, with no restart. To check a build without deploying, build in a
  scratch `git worktree`. To check the UI live, run `cd web && pnpm dev --port 5199`: it
  proxies `/api` to the running server (real data) and leaves `dist` alone.
- The prod service runs this checkout's Python (editable install) and builds the worker
  bundle from it at each session start, so half-done Python work here can ship to a GPU
  session. Do multi-commit Python work in a `git worktree` beside the repo (inside it, ruff
  lints the copy too); `systemctl --user restart degas` picks up merged Python changes.
- Never start a second server on the real `degas.toml` (same SQLite file and Colab
  session). For a smoke test or a live GPU check, use a scratch config: its own `data_dir`,
  `port` and `colab.session_name`, with `drive.client_file` / `token_file` pointing at the
  real ones in `data/` (only read).
- Runner changes are only proven on a GPU: after touching `degas_worker/families` or
  `preprocess`, run one job per family and each preprocessor through the API on a scratch
  server. An A100 runs every family.
- Pre-commit: hooks fix staged files on commit; `make check` runs in GitHub CI only.

## Workflow

- Phase work is committed as `Phase N: <title>`; update `docs/design.md` §10 status
  (and record live GPU test results there) when a phase lands.
