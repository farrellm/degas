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

Phases 1–7 are implemented: SDXL text-to-image, image-to-image, inpainting (with SAM 3 selection) and outpainting with LoRAs and ControlNets (depth, pose and edge traces, limited to an area if you like), a model cache on the GPU VM, a library of kept images and saved prompts with remix, Wan 2.2 text- and image-to-video with clip extension, an image picker and a crop editor, and a reorderable queue with an installable PWA and push notifications. One-time setup on the home server:

0. Install `ffmpeg` (video posters, frames and clip stitching).
1. Install and authenticate the Colab CLI: `uv tool install google-colab-cli`, then run any `colab` command once to sign in.
2. Create a Google Cloud OAuth client of type *Desktop app* with the Drive API enabled, and download its JSON.
3. `cp degas.toml.example degas.toml` and point `drive.client_file` at that JSON.
4. `uv run degas auth drive`: open the printed URL in a browser on the server (or forward the loopback port over SSH) and approve read-only Drive access.
5. Put SDXL checkpoints in Drive under `My Drive/degas/models/sdxl/` and LoRAs under `My Drive/degas/loras/sdxl/`. Wan 2.2 models are diffusers folders under `models/wan22/ti2v-5b/`, `models/wan22/t2v-a14b/` or `models/wan22/i2v-a14b/`, and their LoRAs go under `loras/wan22/` (A14B pairs named `…_high_noise` / `…_low_noise`). Inpainting checkpoints go in `models/sdxl/inpaint/`, so they're only offered for inpaint and outpaint. SDXL needs its diffusers configs in Drive so the GPU never reaches Hugging Face: copy the `*.json`, `*.txt` and `*.model` files (keeping their folders) from `stabilityai/stable-diffusion-xl-base-1.0` into `configs/sdxl/stable-diffusion-xl-base-1.0/`, and from `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` into `configs/sdxl/stable-diffusion-xl-1.0-inpainting-0.1/`. SDXL jobs decode with the fp16-fix VAE: put `config.json` and `diffusion_pytorch_model.safetensors` from `madebyollin/sdxl-vae-fp16-fix` in `vae/sdxl/sdxl-vae-fp16-fix/` (or tick *Built-in VAE in float32* under More settings). For *Select* in the mask editor, put SAM 3 under `preprocessors/sam3/`: the `facebook/sam3` repository from Hugging Face, which is gated, so request access there, then copy its files (`config.json`, the weights and the processor files) into that folder. SDXL ControlNets go in `controlnets/sdxl/` (a `.safetensors` file or a diffusers folder); a sidecar line `control: depth` (or `pose`, `canny`) says what they read, otherwise the file name is used. To trace depth, put the transformers-format `depth-anything/Depth-Anything-V2-Large-hf` files in `preprocessors/depth-anything-v2/`; to trace poses, put `yolox_l.onnx` and `dw-ll_ucoco_384.onnx` from `yzd-v/DWPose` in `preprocessors/dwpose/`. Edges need nothing. For image prompts (IP-Adapter, SDXL), put adapters from `h94/IP-Adapter`'s `sdxl_models/` (e.g. `ip-adapter-plus_sdxl_vit-h.safetensors`, `ip-adapter-plus-face_sdxl_vit-h.safetensors`) in `ip_adapters/sdxl/`, and `config.json` and `model.safetensors` from its `models/image_encoder/` (CLIP ViT-H) in `image_encoders/sdxl/clip-vit-h-14/`. For Qwen-Image 2.1 (Phase 8, in progress), copy the whole `Qwen/Qwen-Image-2.1` repository (about 36 GB) into `models/qwen21/Qwen-Image-2.1/`, and put its LoRAs under `loras/qwen21/`. It needs an L4 with high memory, or an A100 or H100. Each can have a `<name>.yaml` sidecar (label, trigger words, default weight) and a `<name>.jpg` preview. Then run `uv run degas rescan` (or **Rescan Drive** in the app).
6. `make deploy` builds the PWA, installs the systemd user unit `deploy/degas.service` and (re)starts it: the server listens on `127.0.0.1:8420` and `tailscale serve --https=8448` publishes it on the tailnet (`tailscale serve status` prints the URL). `make logs` follows its journal. To survive a reboot it needs `sudo loginctl enable-linger $USER`. For a one-off foreground run, use `make build && uv run degas`.

In the app: tap **No GPU** in the header to start a GPU session, write a prompt in **Create** → Generate, and watch images arrive in **Results**. Open an image and choose **Keep** to save it, with its settings, to the **Library**; **Remix** loads those settings back into Create. Images you don't keep are deleted 24 hours after the session ends. The session stops itself after `idle_timeout_min` without activity.

On the iPhone, open the app in Safari, then Share → **Add to Home Screen**. Notifications (jobs finishing, and a warning 2 minutes before an idle session stops) only work from the installed app; turn them on in the GPU session sheet. Set `push.subject` in `degas.toml` to a `mailto:` address you read, because push services use it as a contact.

## Training a character LoRA

`degas lora` trains an SDXL LoRA with [kohya sd-scripts](https://github.com/kohya-ss/sd-scripts) (v0.12.0) on a Colab GPU. It runs in its own Colab session (`lora.session_name`, `degas-lora` by default), so it doesn't disturb the app's. Make a folder with 20–40 photos of the person (vary the angle, expression, framing, lighting and clothes; crop out other people). Give each photo a `<name>.txt` caption. Start with the trigger and class (`photo of ohwx woman, …`), then describe only what changes from photo to photo, like pose, clothes, setting and framing. Don't describe the face, hair or build: what the captions leave out is what the LoRA learns. Then add a `lora.toml`:

```toml
trigger = "ohwx"
class_word = "woman"
base = "models/sdxl/my_checkpoint.safetensors"  # train on the checkpoint you'll use
# name = "jane"      # the LoRA's file name; default: the folder's name
# gpu = "L4"         # T4 works (fp16, slow); A100 is faster
# [train]            # defaults: rank 32/16, U-Net only, AdamW8bit 1e-4 cosine, ~2000 steps / 10 epochs
# optimizer = "Prodigy"
# [samples]
# prompts = ["photo of {subject}, …"]   # {subject} = "ohwx woman"
```

```sh
uv run degas lora check ~/lora/jane    # checks the captions and images, prints the plan
uv run degas lora train ~/lora/jane    # starts a VM, trains, copies each epoch back, stops the VM
uv run degas lora attach jane          # follow it again after a disconnect or Ctrl-C
uv run degas lora stop jane            # stop the VM
uv run degas lora publish jane --epoch 8   # upload to loras/sdxl/ with a sidecar and preview
```

Runs live in `data/lora-runs/<name>-<time>/`: `checkpoints/` (one per epoch; the last is `<name>.safetensors`), `samples/` (each epoch's sample prompts, same seeds; `e000000` is before training) and `train.log`. Compare the epochs' samples and publish the latest one that still follows the unusual prompts (the astronaut, the painting) and doesn't leak into the prompt without the trigger. That's often not the last epoch. `publish` uploads with your own rclone remote (`lora.rclone_remote`, `gdrive:`), because Degas's Drive token is read-only, then rescans. Use the LoRA at a weight of about 0.6–0.9.

## Importing LoRAs from Civitai

Paste a Civitai link into **Import from Civitai** at the foot of the LoRA picker, or:

```sh
uv run degas civitai import https://civitai.com/models/<id> --dry-run   # show the plan
uv run degas civitai import https://civitai.com/models/<id>             # copy it into Drive
uv run degas civitai backfill            # sidecars for LoRAs already in Drive, found by hash
```

The base model on Civitai picks the folder (`loras/sdxl/`, `loras/flux1/`, `loras/wan22/` …); `--family` overrides it, and `--name` and `--weight` set the file name and the sidecar's weight. The file streams from Civitai to Drive through your rclone remote (`lora.rclone_remote`, as for `degas lora publish`) and is checked against Civitai's SHA-256. It gets a sidecar with the trigger words and the base model, and a preview from the first example image. Wan 2.2 A14B LoRAs often put the high- and low-noise halves in separate versions: import both links and they pair up. Many downloads need an API key (civitai.com → Account settings → API keys) in `~/.config/civitai/token` (`civitai.token_file`).

## Development

Requires [uv](https://docs.astral.sh/uv/), Node ≥ 24 and pnpm.

```sh
make install     # uv sync + pnpm install + git hooks
make check       # lint + typecheck + test (what CI runs)
make fmt         # ruff format/fix + prettier
make dev-server  # API on 127.0.0.1:8420
make dev-web     # Vite dev server, proxies /api to the API
```

Git hooks ([pre-commit](https://pre-commit.com), config in `.pre-commit-config.yaml`): on commit, ruff / Prettier / ESLint fix staged files plus generic file checks. `make check` runs in GitHub CI. Bypass once with `--no-verify`.

Tooling: ruff (lint + format), mypy (strict), pytest · ESLint (typescript-eslint strict, type-checked), Prettier, Vitest + Testing Library.
