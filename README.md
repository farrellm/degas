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

Phases 1–6 are implemented: SDXL text-to-image, image-to-image, inpainting (with SAM 3 selection) and outpainting with LoRAs, a model cache on the GPU VM, a library of kept images and saved prompts with remix, Wan 2.2 text- and image-to-video with clip extension, an image picker and a crop editor, and a reorderable queue with an installable PWA and push notifications. One-time setup on the home server:

0. Install `ffmpeg` (video posters, frames and clip stitching).
1. Install and authenticate the Colab CLI: `uv tool install google-colab-cli`, then run any `colab` command once to sign in.
2. Create a Google Cloud OAuth client of type *Desktop app* with the Drive API enabled, and download its JSON.
3. `cp degas.toml.example degas.toml` and point `drive.client_file` at that JSON.
4. `uv run degas auth drive`: open the printed URL in a browser on the server (or forward the loopback port over SSH) and approve read-only Drive access.
5. Put SDXL checkpoints in Drive under `My Drive/degas/models/sdxl/` and LoRAs under `My Drive/degas/loras/sdxl/`. Wan 2.2 models are diffusers folders under `models/wan22/ti2v-5b/`, `models/wan22/t2v-a14b/` or `models/wan22/i2v-a14b/`, and their LoRAs go under `loras/wan22/` (A14B pairs named `…_high_noise` / `…_low_noise`). Inpainting checkpoints go in `models/sdxl/inpaint/`, so they're only offered for inpaint and outpaint. For *Select* in the mask editor, put SAM 3 under `preprocessors/sam3/`: the `facebook/sam3` repository from Hugging Face, which is gated, so request access there, then copy its files (`config.json`, the weights and the processor files) into that folder. Each can have a `<name>.yaml` sidecar (label, trigger words, default weight) and a `<name>.jpg` preview. Then run `uv run degas rescan` (or **Rescan Drive** in the app).
6. `make deploy` builds the PWA, installs the systemd user unit `deploy/degas.service` and (re)starts it: the server listens on `127.0.0.1:8420` and `tailscale serve --https=8448` publishes it on the tailnet (`tailscale serve status` prints the URL). `make logs` follows its journal. To survive a reboot it needs `sudo loginctl enable-linger $USER`. For a one-off foreground run, use `make build && uv run degas`.

In the app: tap **No GPU** in the header to start a GPU session, write a prompt in **Create** → Generate, and watch images arrive in **Results**. Open an image and choose **Keep** to save it, with its settings, to the **Library**; **Remix** loads those settings back into Create. Images you don't keep are deleted 24 hours after the session ends. The session stops itself after `idle_timeout_min` without activity.

On the iPhone, open the app in Safari, then Share → **Add to Home Screen**. Notifications (jobs finishing, and a warning 2 minutes before an idle session stops) only work from the installed app; turn them on in the GPU session sheet. Set `push.subject` in `degas.toml` to a `mailto:` address you read, because push services use it as a contact.

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
