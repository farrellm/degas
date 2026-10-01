# Degas — Design Document

Status: Draft v1 · 2026-09-27

## 1. Overview

Degas is a personal web app for generating images and videos with generative models. It is used primarily from an iPhone over Tailscale. The app runs on a home server. All inference runs on Google Colab GPU runtimes, which the server drives through the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli). Models, LoRAs, ControlNets and preprocessors are loaded from Google Drive.

The initial model families are Stable Diffusion XL (images) and Wan 2.2 (video). The architecture treats model families as plugins so that new ones can be added without changing the UI.

### Goals

- Mobile-first PWA that feels good on iPhone Safari and can be installed to the home screen.
- Text-to-image, image-to-image, inpainting and outpainting with SDXL.
- Text-to-video, image-to-video and video extension (last frame → i2v) with Wan 2.2.
- LoRA support for every family, including Wan 2.2 A14B high/low-noise LoRA pairs.
- SDXL ControlNet with:
  - control images: uploaded, or generated as depth, pose or canny;
  - masks: hand-drawn or SAM-assisted;
  - regional ControlNet (a control signal restricted to a mask).
- Pick source images for i2i, i2v and control from recent results, the saved library, the camera roll, or a URL.
- Crop, resize, rotate and flip source, control and mask-base images before use, with presets that match each model's resolution constraints.
- Results are ephemeral by default. Explicit save buttons exist for images, videos and prompts. Saving media also saves the full generation configuration, so the result can be reproduced or remixed.
- A job queue with batch generation, cancel and reorder, and a push notification when a job completes.
- Explicit control over GPU spend: sessions are started manually and shut down on idle.

### Non-goals (v1)

- Multi-user support, or any auth beyond tailnet membership.
- Video ControlNet (Wan VACE / Fun-Control). The schema is designed so it can be added later.
- Regional LoRA.
- Training LoRAs in the app. (`degas lora`, a separate CLI, trains SDXL LoRAs on Colab with kohya sd-scripts: see the README.)
- Local (non-Colab) inference.

## 2. Architecture

```
┌──────────────────┐   HTTPS (tailscale serve)   ┌─────────────────────────────┐  colab CLI (new/stop/exec)  ┌───────────────────────────┐
│ iPhone PWA       │ ──────────────────────────▶ │ Degas server (home box)     │ ──────────────────────────▶ │ Colab VM (GPU)            │
│ React + Vite     │ ◀── SSE progress, Web Push  │ FastAPI + SQLite            │  ssh (colab ssh ProxyCmd)   │ • kernel: bootstrap only  │
└──────────────────┘                             │ • REST API + static UI      │  -L tunnel ⇄ worker HTTP    │ • degas_worker (uvicorn,  │
                                                 │ • session manager           │ ◀─────────────────────────▶ │   127.0.0.1:8765)         │
                                                 │ • job queue / dispatcher    │                             │ • diffusers pipelines     │
                                                 │ • Drive OAuth + asset index │  Drive access tokens ──▶    │ • rclone → /content/models│
                                                 │ • blob store, library       │                             └────────────┬──────────────┘
                                                 │ • retention sweeper         │ ── Drive API (index) ──┐                  │ rclone copy
                                                 └─────────────────────────────┘                        ▼                  ▼
                                                                                                   Google Drive  ◀─────────┘
```

- **Frontend**: a React + Vite single-page PWA. The FastAPI server serves it as static files.
- **Server**: Python 3.13 (matching the Colab runtime) and FastAPI. SQLite (via SQLModel or plain SQLAlchemy) holds the metadata. Media lives on the local filesystem. The server is the only component that invokes the `colab` CLI, using asyncio subprocesses.
- **Worker**: a Python package (`degas_worker`) that runs a small FastAPI/uvicorn HTTP server on the VM, bound to `127.0.0.1`. The server reaches it through an SSH tunnel (§3.2). Loaded pipelines stay in the worker's memory between jobs. The worker process is started from the Colab kernel, so it inherits the kernel's CUDA environment (Phase 0 finding 6).

### Network and security

- The server binds to the Tailscale interface only, or to localhost behind `tailscale serve`.
- `tailscale serve` provides HTTPS with a `*.ts.net` certificate. Both PWA install and iOS Web Push require HTTPS.
- There is no application-level auth. Access control is tailnet membership.
- The `colab` CLI authenticates with ADC or OAuth2 on the server. Credentials never reach the browser.
- The worker only listens on the VM's loopback interface, and is reachable only through the server's SSH tunnel.
- The worker receives short-lived Drive access tokens (about 1 h), never the refresh token.

## 3. Colab integration

### 3.1 Session lifecycle

The user starts a session from the UI and chooses a GPU type: T4, L4, A100 or H100, optionally with `--high-mem`. The server runs these steps and streams their progress to the UI:

1. `colab new -s degas --gpu <GPU> [--high-mem]`. `--high-mem` is preselected for Wan A14B, because standard shapes have only 12 GB of RAM.
2. Open the SSH master connection:
   ```
   ssh -o ControlMaster=yes -o ControlPath=$XDG_RUNTIME_DIR/degas-%C \
       -o ProxyCommand="colab ssh --proxy-mode -s degas -i <key>" \
       -f -N -L <local_port>:127.0.0.1:8765 root@colab-runtime
   ```
   Degas has its own ed25519 key, set in `degas.toml`. The ControlPath must be under 108 bytes, which is why it lives in `$XDG_RUNTIME_DIR`.
3. `scp` the worker bundle and the `rclone` binary to `/content/degas/`. The bundle is only re-sent if its content hash has changed.
4. Install packages only if something is missing. The Colab image already has torch, diffusers, transformers, peft, fastapi, uvicorn and ffmpeg, so this is usually a no-op. Extra packages such as DWPose's dependencies are installed lazily, the first time they're used.
5. `colab exec -s degas` with a bootstrap snippet. It starts the worker with `subprocess.Popen([... "uvicorn", "degas_worker.app:app", "--host", "127.0.0.1", "--port", "8765"], start_new_session=True)` and returns immediately. Starting from the kernel means the worker inherits `LD_LIBRARY_PATH=/usr/lib64-nvidia` and the rest of the CUDA environment.
6. Poll `GET /health` through the tunnel until the worker reports the GPU name and free VRAM and free disk. Then `POST /drive-token` with a fresh access token. The session is now `ready`.

There is no Drive FUSE mount: see §5 for how Drive is accessed.

**Session states:** `starting → ready ⇄ busy → stopping → stopped`, plus `error`.

**Idle shutdown:** if no job has run for `idle_timeout` minutes (default 15, configurable in settings), the server stops the session. The UI shows a countdown, and any user interaction or queued job resets it. The user can also stop the session manually.

Only one session can be active at a time in v1. Changing GPU type means stopping and restarting.

**Jobs without a session.** Jobs can be submitted when no session is running. They wait in the queue, and the Create and Queue screens show a "Start session" prompt. When a session reaches `ready`, the dispatcher starts on the queue.

**Liveness and heartbeat.**
- While a session is `ready` or `busy`, the server calls `GET /health` every 30 s. It also runs `colab status` every 5 min to detect reclamation.
- If the tunnel drops, the server re-establishes the SSH master and re-checks `/health`.
- If the VM is gone, `colab` reports `Session '…' not found` and exits with code 1. The session then moves to `error`.
- Whether a kernel exec heartbeat is also needed is still unknown. The Phase 0 liveness test was inconclusive: the VM was still alive 15 minutes after the kernel went idle. The question is whether a VM whose kernel is idle, with only the SSH-launched worker active, gets reclaimed. If it does, the server also sends a trivial `colab exec` every few minutes while the session is active.

**Failure and restart recovery.**
- If the VM is reclaimed or the worker dies, the session moves to `error` and `ended_at` is set. The running job is marked `error`. Queued jobs stay queued.
- On server startup, the server compares its session row with `colab sessions`:
  - If the VM is still alive, the server re-opens the tunnel and calls `GET /state`. That returns any in-flight job and its progress, plus any finished outputs the server hasn't fetched yet, so nothing is lost. If the worker isn't running, the server restarts it (bootstrap step 5).
  - If the VM is gone, the session is marked stopped.
  - Either way, the idle timer restarts from startup.
  This keeps a server crash from leaving a GPU VM running and consuming compute units. Colab's own idle reclaim is a final backstop.

Each model variant declares a minimum GPU. If you submit a job whose model needs more than the current session provides, the UI warns you. The job is not rejected, because offloading may still make it work, only slowly.

### 3.2 Worker protocol (HTTP over the SSH tunnel)

The server talks to the worker through `http://127.0.0.1:<local_port>` on the forwarded port. A request's round trip is about 0.12 s. Calls are serialized per GPU by the worker. The worker has one generation slot. Preprocessing requests are also accepted while a job is running, but they wait for the GPU between denoising steps.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | GPU name, VRAM free/total, disk free, loaded family/variant, worker version, model cache (used, budget, files) |
| GET | `/state` | Current job (id, status, progress) and outputs not yet acknowledged (for reattach) |
| PUT | `/blobs/{sha256}` | Upload an input blob (source, control or mask image); `HEAD` to check whether it's already present |
| POST | `/jobs` | Start a job: `{job_id, spec}`; `409` if a job is already running |
| GET | `/jobs/{id}/events` | SSE stream of `progress`, `output`, `done`, `error`, `cancelled` events |
| POST | `/jobs/{id}/cancel` | Cooperative cancel |
| GET | `/outputs/{job}/{item}` | Download one output; `DELETE` acknowledges it and frees disk |
| POST | `/preprocess` | Run a preprocessor on a staged blob: `{id, image, asset?: {path, size}, params}` (canny has no asset). SAM answers `{candidates: [{mask (base64 PNG), score}], chosen}`, smallest mask first; depth, pose and canny answer `{image}`, a base64 PNG at the image's size |
| POST | `/drive-token` | Store a fresh Drive access token |
| POST | `/assets/fetch` | Copy the given Drive paths into the local model cache (progress is reported via SSE) |
| POST | `/shutdown` | Unload models and exit (used before `colab stop`) |

**Per job:**

1. **Stage inputs.** `HEAD` each input blob and `PUT` any that are missing. Blobs are content-addressed, so each is sent at most once per session.
2. **Start.** `POST /jobs` with the job spec. The job spec uses the same schema as the saved generation config (§6.4), except that `runtime` is left out. A client-submitted spec may contain `fit` modes; the server resolves them into transforms before dispatch.
3. **Stream.** The server subscribes to `/jobs/{id}/events` and relays the events to the phone over its own SSE stream.
   ```
   {"t":"progress","job":"…","item":0,"phase":"copy|load|denoise|decode|encode","step":12,"steps":30}
   {"t":"output","job":"…","item":0,"seed":1234,"media_type":"image/png"}
   {"t":"done","job":"…"}
   {"t":"error","job":"…","message":"CUDA out of memory"}
   ```
4. **Fetch outputs.** On each `output` event, the server GETs the file, stores it as a blob, and then DELETEs it on the worker. Finished batch items appear on the phone before the whole batch is done.

The job's final state comes from the terminal event (`done`, `error` or `cancelled`). If the stream ends without one, the server treats the job as failed and reconciles via `GET /state`.

**Cancellation** is cooperative. `POST /jobs/{id}/cancel` sets a flag that the step callback (`callback_on_step_end`) checks. The worker then aborts and emits `cancelled`. Phase 0 showed that killing a client never stops running code on the VM. If the worker is stuck, the last resort is **Force reset worker**: kill the worker process over SSH and restart it. This reloads the models. `colab restart-kernel` is only needed if the kernel itself is wedged.

**Interactive preprocessing.** SAM prompts and the depth, pose and edge traces use `POST /preprocess`. The server stages the image blob first (as for a job), names the preprocessor's model asset, and the worker copies that asset into its cache on first use, so the first SAM request of a session waits for the copy. For taps, the worker caches the image embedding by image hash, so each extra tap only runs the lightweight mask decoder. The expected round trip is 0.12 s of transport plus the inference time. Preprocessing runs in the worker's threadpool, one request at a time, alongside a generation job on the same GPU rather than waiting for it. One preprocessor is resident at a time: running another unloads it.

### 3.3 Phase 0 results

See [phase0-findings.md](phase0-findings.md). Summary:

- Kernel state persists across `exec` calls, and `exec` streams output. However, `exec` has 1.3–1.5 s of overhead per call, `--timeout` has no effect, and it exits 0 even when the code raises.
- `colab ssh` works as a `ProxyCommand`. A port-forwarded HTTP round trip takes about 0.12 s, and SSE streams through the tunnel. This is why the transport above uses SSH, not per-job `exec`.
- `drivemount` needs interactive consent on every new VM. This is why Drive access doesn't use the FUSE mount (§5).
- rclone with a server-supplied token copies a cold file from Drive at about 65 MB/s.

## 4. Model plugin system

A model family is implemented as a pair of modules that share a name:

- `server/src/degas/families/<family>.py`: the **descriptor** for the family.
- `worker/src/degas_worker/families/<family>.py`: the **runner** for the family.

### 4.1 Descriptor (server)

```python
class FamilyDescriptor:
    id: str                          # "sdxl", "wan22"
    label: str
    media: Literal["image", "video"]
    variants: list[Variant]          # e.g. wan22: ti2v-5b, t2v-a14b, i2v-a14b
    modes: list[Mode]                # t2i, i2i, edit, inpaint, outpaint, t2v, i2v
    lora_format: Literal["single", "paired_hi_lo"]
    supports_control: bool
    supports_image_prompts: bool     # IP-Adapter (SDXL)
    def param_schema(self, variant, mode) -> JsonSchema   # drives the UI form
    def size_constraints(self, variant) -> SizeConstraints # multiple_of, min/max pixels, presets
    def validate(self, spec) -> Spec                       # fill defaults, clamp values
```

Each `Variant` declares:

- `min_gpu`, the minimum GPU for the variant;
- `default_params`;
- which Drive paths make up the variant (the base checkpoint, plus a VAE and refiner if it uses them).

`param_schema` returns JSON Schema with UI hints for widget type, grouping and an "advanced" flag. The frontend renders the generation form from this schema. Adding a new family therefore needs no frontend changes, except for a family that needs genuinely new widget types.

### 4.2 Runner (worker)

```python
class FamilyRunner:
    def load(self, variant, ctx) -> None        # build pipeline; reuse if already loaded
    def apply_loras(self, loras) -> None        # diff against currently-applied set
    def run(self, spec, emit) -> Iterator[Output]
    def unload(self) -> None
```

- **Memory.** The worker keeps at most one family's pipeline resident. Switching family unloads the current pipeline with `del` + `torch.cuda.empty_cache()`.
- **Variants.** Switching variant within a family reuses shared components where possible, such as text encoders and the VAE.
- **LoRAs.** Applied with diffusers `load_lora_weights` and `set_adapters` using per-adapter weights. The runner tracks the set currently applied, so it only loads new adapters and only removes adapters that are no longer requested.
- **Offloading.** Chosen automatically from free VRAM: none, `enable_model_cpu_offload`, or sequential offload.

### 4.3 v1 families

**SDXL (`sdxl`)**

- **Modes:**
  - `t2i` and `i2i`: `StableDiffusionXLPipeline` and `StableDiffusionXLImg2ImgPipeline`.
  - `inpaint`: `StableDiffusionXLInpaintPipeline`.
  - `outpaint`: an inpaint on a larger canvas. The canvas is the form's width and height, and `inputs.place` (`{x, y, w, h}` in canvas pixels) says where the source goes; the server fits the source to `w × h`. The worker fills the margins with a blurred stretch of the source, and masks the margins plus a *blend* band (32 px by default) inside the source's edges that face a margin. Outpaint always runs at strength 1.
  - Regular checkpoints do all four modes: the text-to-image pipeline is loaded, and the image-to-image and inpaint pipelines are made from it with `from_pipe`, sharing its weights and LoRAs. Inpainting checkpoints (9-channel UNets) are the `inpaint` variant, found under `models/sdxl/inpaint/`, and only inpaint and outpaint.
  - After inpaint and outpaint, the worker pastes the original back outside the blurred mask, so the untouched area skips the VAE round trip. *Redraw: around the mask* uses diffusers' `padding_mask_crop` (the masked area plus *Space around the mask*, redrawn at full resolution).
- **Checkpoints:** any SDXL-architecture `.safetensors` file on Drive, including Pony and Illustrious derivatives. Loaded with `from_single_file`, given the diffusers configs and tokenizers from Drive (`configs/sdxl/stable-diffusion-xl-base-1.0/`, or `…/stable-diffusion-xl-1.0-inpainting-0.1/` for the `inpaint` variant: the repos' `*.json`, `*.txt` and `*.model` files), so loading doesn't reach Hugging Face. Single-file ControlNets still fetch their config from Hugging Face; diffusers-folder ones don't.
- **VAE:** the stock SDXL VAE overflows in float16, so diffusers moves a checkpoint's own VAE to float32 for every encode and decode (`force_upcast`). Instead, jobs use Ollin's fp16-fix VAE (`madebyollin/sdxl-vae-fp16-fix`, its `config.json` and `diffusion_pytorch_model.safetensors` in `vae/sdxl/sdxl-vae-fp16-fix/`), which stays in float16. Its outputs differ very slightly. *Built-in VAE in float32* (More settings, `vae_fp32`) uses the checkpoint's own VAE, for checkpoints that ship a custom one. Changing between them reloads the pipeline.
- **ControlNet:** one or more ControlNet units. Each unit has:
  - a ControlNet model,
  - a control image,
  - a conditioning scale,
  - a start and end fraction,
  - an optional mask (regional control).

  The ControlNet variants of the pipelines (`StableDiffusionXLControlNet{,Img2Img,Inpaint}Pipeline`) are made from the loaded pipeline with `from_pipe`, and a list of ControlNets becomes a `MultiControlNetModel`. A job has at most 3 units, and a ControlNet model guides one unit. The runner keeps the requested ControlNets resident and drops the rest. A ControlNet is a single `.safetensors` file (`from_single_file`) or a diffusers folder with `config.json` (`from_pretrained`). Union ControlNets (`ControlNetUnionModel`) are refused with a clear error for now.
- **Regional ControlNet:** each unit's down-block and mid-block residuals are multiplied by its area mask, downsampled to each residual's size (`area` interpolation). The runner replaces `forward` on the ControlNet instance while the pipeline runs (`limit_to_areas` in the SDXL runner), rather than wrapping it in another module, so the pipelines' `isinstance` checks and accelerate's offload hook keep working. With *Around the mask*, area masks get the same crop as the image.
- **Image prompts (IP-Adapter, Phase 11).** Up to 2 units, each an IP-Adapter from `ip_adapters/sdxl/` with up to 4 pictures, a purpose, a weight (0–2), a step range and an optional area of the output. The purpose picks the UNet blocks the adapter acts in (InstantStyle): `all`, `style` (up block 0, attention 1), `layout` (down block 2, attention 1) or `style_layout`. The adapters share one CLIP image encoder, chosen from the file name (ViT-H, or ViT-bigG for h94's `ip-adapter_sdxl`) and put in the spec as `image_encoder`. The runner loads them into the shared UNet with `load_ip_adapter` and registers the encoder, unloads them for a job without image prompts (the UNet refuses IP layers without pictures), remakes the `from_pipe` pipelines when the set changes, encodes the pictures once per job, sets each unit's scale again between steps for its range, and passes areas as `ip_adapter_masks`. Research, variants and the later plans (FaceID, FLUX.1 Redux): [ip-adapter.md](ip-adapter.md).
- **FaceID (Phase 12).** A unit whose adapter's name has `faceid` (FaceID Plus v2) reads who a face is. The runner finds each picture's biggest face with InsightFace (`preprocessors/insightface/`: SCRFD `det_10g.onnx` and ArcFace `w600k_r50.onnx` on onnxruntime, `degas_worker/preprocess/face.py`), passes the 512-number identities as the unit's embeddings, and sets the projection's `clip_embeds` from CLIP's reading of the aligned face; v2's `shortcut_scale` is the unit's `structure`. The model's own LoRA, which diffusers loads from the `.bin` as `faceid_<i>`, is weighted by the unit's `lora_weight` alongside the job's LoRAs. The spec gains `face_detector`, prefetched like the encoder. `/preprocess` `face` returns the aligned crop, so the editor can show which face was found.
- **Parameters:** prompt, negative prompt, width and height (with SDXL aspect-ratio presets), steps, CFG, sampler and noise schedule (Default/Karras/Exponential), seed, clip skip, denoise strength (i2i and inpaint), mask blur and padding (inpaint), and an optional refiner.

**Wan 2.2 (`wan22`)**

- **Variants:**
  - `ti2v-5b`: text-to-video and image-to-video, minimum GPU L4.
  - `t2v-a14b`: text-to-video, minimum GPU A100.
  - `i2v-a14b`: image-to-video, minimum GPU A100.
- **Pipelines:** diffusers `WanPipeline` and `WanImageToVideoPipeline`.
- **LoRAs:** `paired_hi_lo` for the A14B variants. A LoRA entry names a high-noise file and a low-noise file, and each has its own weight. A LoRA entry for the 5B variant is a single file.
- **Parameters:** prompt, negative prompt, resolution preset, frame count, fps, steps, CFG (with separate values for the two experts on A14B), boundary ratio (A14B), seed, and a source image for i2v.
- **Output:** an MP4 encoded with H.264 in `yuv420p` pixel format, so it plays inline on iOS. A poster frame is extracted, and so is the last frame, which is used for video extension.

**Qwen-Image 2.1 (`qwen21`)**

- **Model.** A 7B single-stream DiT (32 layers, block-causal attention), a 16× RGBA VAE, and `Qwen3-VL-8B-Instruct` as the text encoder. The encoder reads the prompt and the reference images together, so one model does text-to-image and editing with up to 10 reference images. Released 2026-09-20 under the Qwen Research License (non-commercial use only). diffusers' `QwenImage21Pipeline` comes from PR #14804. That PR isn't in a release yet (0.40.0 lacks it), so the bootstrap installs a pinned diffusers commit when the image's diffusers can't import the pipeline (§11).
- **Checkpoint.** The official diffusers folder `Qwen/Qwen-Image-2.1` under `models/qwen21/` (about 36 GB: transformer 13.6 GB, text encoder 16.7 GB, VAE 1.3 GB, all bf16). ComfyUI's `int8_convrot` files don't load in diffusers.
- **Variant `base`**, minimum GPU L4. A T4 has no usable bf16 and too little memory. An H100 holds all of it. On an L4 or a 40 GB A100 the runner uses model CPU offload, which needs a high-memory VM: the text encoder runs first, then moves off the GPU for the transformer.
- **Memory.** An edit's KV cache (every block's keys and values for the prompt and condition images, 512 KB a token, about 10 GB for one 2K image) lets later steps run only the output's tokens. The pipeline holds it until it returns, so it's still there during the VAE decode. The runner turns the cache off when the transformer, the cache and an estimate of the activations wouldn't fit in 90% of the GPU, e.g. a 2K edit with a reference on an A100. It also tiles the VAE (512 px tiles) and starts the worker with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Untiled 2K decodes and fragmentation ran A100 edits out of memory.
- **Modes:**
  - `t2i`: text-to-image.
  - `edit`: the source is image 1, and `inputs.refs` adds up to 9 more, in order. Order matters, because each image attends only to the ones before it (block-causal), and prompts refer to images by position ("the jacket from image 2"). The source is fitted to the output size like an i2i source. The pipeline sizes each reference to the output's pixel count at its own aspect ratio, so references aren't fitted on the server. A reference can be cropped freely (any shape, no resize), which gives the model more pixels of what matters; the crop is recorded in `inputs.transforms` like the source's.
  - `inpaint`: an edit of the source (with any references) that is pasted back outside the blurred mask, as SDXL does. The model itself doesn't see the mask. Its native editing by painted marks or masks is future work (§12.1).
- **Sizes:** multiples of 32. The VAE shrinks 16× and the transformer doesn't patch, so 16 would be enough in principle, but the pipeline rounds down to 32. The 2K presets are Qwen's table (2048², 2400×1792, 2528×1696, 2752×1536 and their portraits); the 1K presets are 1024², 1152×864, 1248×832 and 1376×768, and portraits. The default is 2048², which is what Qwen recommends.
- **Guidance.** `true_cfg_scale`, default 1 (off). Above 1, the negative prompt is used and each step does twice the work. Past about 2, images over-saturate.
- **Schedule.** Flow-match Euler with Qwen's exponential dynamic shift, 40 steps by default. *Beta* (`use_beta_sigmas`) is offered because a ComfyUI comparison preferred Euler/Beta at 30 steps. ComfyUI's "simple" isn't Qwen's schedule, though, so the defaults stay until a live A/B (Phase 8).
- **Grid removal.** The VAE leaves a faint 2 px lattice, most visible on skin and flat areas (Hugging Face discussion #12). *Remove VAE grid* (on by default) runs the notch filter from ComfyUI-DeGrid (Apache-2.0, ported to `degas_worker/degrid.py`) after the decode. The filter detects the lattice's phase and does nothing to an image without one.
- **LoRAs:** single files, applied with the pipeline's `QwenImageLoraLoaderMixin` (transformer only).
- **Not in v1:** transparent output, native mask editing, sizing references by their longest edge, the prompt-rewriting models, and faster attention. §12.1 says why for each, and what adding it would take.

**FLUX.1 [dev] (`flux1`)**

- **Model.** A 12B guidance-distilled rectified-flow transformer, T5-XXL and CLIP-L text encoders, and a 16-channel VAE. Non-commercial license. diffusers' `FluxPipeline`.
- **Variant `dev`**, mode `t2i` only, minimum GPU L4.
- **Checkpoints.** A single `.safetensors` file in `models/flux1/` holds the transformer. `flux1-dev-fp8.safetensors` from `Kijai/flux-fp8` (11.9 GB, fp8 e4m3fn) is the one to use. Comfy-Org's file of the same name (17.2 GB) bundles an fp8 T5, CLIP and the VAE with the transformer; it loads too, but only its transformer is used, so its extra 5 GB is copied for nothing. Civitai fine-tunes load the same way. The transformer is loaded with `FluxTransformer2DModel.from_single_file`. Everything else comes from `configs/flux1/FLUX.1-dev/`, the official diffusers folder without the transformer's weights (about 10 GB, mostly T5), which the job names as its `config`. A whole diffusers folder in `models/flux1/` loads by itself with no `config`.
- **fp8.** `from_single_file` casts fp8 weights up to bf16, which would make the transformer 24 GB again. The runner reads the checkpoint's header (`degas_worker/safetensors_info.py`). If the transformer blocks are stored in fp8, it calls `enable_layerwise_casting(storage_dtype=float8, compute_dtype=bfloat16)`, so the weights stay 12 GB on the GPU and each layer is cast up as it runs. Casting fp8 weights up and back loses nothing. A bf16 transformer that wouldn't fit the GPU even by itself (an L4) is stored in fp8 too, which does lose a little precision. Placement then follows the rule the Qwen runner uses (`families/offload.py`): model CPU offload when the weights exceed 70% of the GPU. On an L4 that means T5 moves off the GPU before denoising; on an A100 everything stays on the GPU.
- **Parameters:** prompt, width and height (multiples of 16, up to 2.4 MP; about-1-megapixel presets plus 1536²), steps (default 28), *Guidance* (the distilled guidance, default 3.5) and seed. The model is guidance-distilled, so there's no negative prompt and no CFG.
- **Image prompts (FLUX.1 Redux, Phase 13).** Up to 2 units, each up to 4 pictures, a weight and *How closely* (`downsample`). The runner loads Redux's SigLIP and embedder from `ip_adapters/flux1/FLUX.1-Redux-dev/`, encodes the prompt once with the pipeline's own encoders, and passes `prompt_embeds`: the prompt's T5 tokens followed by each picture's 729 tokens, averaged down to 27, 13, 9, 6 or 5 a side and scaled by the weight (ip-adapter.md §6.1). No areas, step ranges or purposes. Families describe what their image prompts can do (`image_prompt_options`), and the form follows it.
- **LoRAs:** single files, in `loras/flux1/`. The kohya, xlabs and diffusers formats all load through `FluxLoraLoaderMixin`. CLIP keys are renamed to match transformers 5's flattened `CLIPTextModel`, as for SDXL. PEFT gives a new adapter its base layer's dtype, which with fp8 storage would put the LoRA in fp8. The runner moves each adapter back to bf16 before the LoRA's weights are copied into it (`_adapters_in_bf16`). Flux Control LoRAs aren't supported.

**FLUX.2 [klein] (`klein`)**

- **Model.** FLUX.2 [klein] 9B: a 9B flow transformer with Qwen3-8B as its text encoder and the FLUX.2 VAE, step-distilled to 4 steps. Released January 2026 under the FLUX Non-Commercial License. diffusers' `Flux2KleinPipeline`, which is in diffusers 0.40 and in the pinned commit.
- **Checkpoint.** The official diffusers folder `black-forest-labs/FLUX.2-klein-9B` under `models/klein/` (about 35 GB: transformer 18.2 GB, text encoder 16.4 GB, all bf16).
- **Variant `9b`**, modes `t2i` and `edit`, minimum GPU L4. It uses the same offload rule as Qwen: offloaded on an L4 or A100, and fully on the GPU on an H100.
- **Text-to-image.** The same pipeline with no condition images.
- **Edit.** The source is image 1 and `inputs.refs` adds up to 3 more: BFL's model table allows klein 4 images, and the pipeline itself sets no limit. The pipeline scales each condition image down to at most 1 megapixel and never up, so the crop editor doesn't warn that a small reference will be enlarged (`Variant.ref_max_pixels`). Images with alpha are flattened onto white.
- **Parameters:** prompt, width and height (multiples of 16, up to 4 MP, default 1024²), steps (default 4) and seed. There's no CFG or negative prompt, so guidance is fixed at 1.
- **LoRAs:** single files, in `loras/klein/`, loaded by the pipeline's `Flux2LoraLoaderMixin`.
- **Not in v1:** `inpaint` (it could work like Qwen's, as an edit pasted back through the mask); the 4B and undistilled base models.

### 4.4 Preprocessors

Preprocessors run on the GPU, in the worker, as a family-independent registry:

| id | Model | Output |
|---|---|---|
| `depth` | Depth Anything V2 (`preprocessors/depth-anything-v2/`, the transformers-format `…-hf` folder) | Depth map image, near is white |
| `pose` | DWPose (`preprocessors/dwpose/`: `yolox_l.onnx` and `dw-ll_ucoco_384.onnx`; `onnxruntime-gpu` is installed on first use) | Pose skeleton image in OpenPose format, drawn at 512 px on the short side and scaled up |
| `canny` | OpenCV, no model | Edge image; `{low, high}` thresholds |
| `sam` | SAM 3 (`preprocessors/sam3/`, the transformers-format `facebook/sam3` folder) | Masks from included and excluded taps (`Sam3TrackerModel`, three candidates), or from a description (`Sam3Model`: every match, as one mask; taps then keep or drop matches) |

Preprocessor outputs are ordinary images. They go into the blob store and can be edited, saved, or reused.

## 5. Google Drive layout

```
MyDrive/degas/
  models/
    sdxl/          *.safetensors (+ optional *.yaml sidecar)
      inpaint/     inpainting checkpoints (9-channel UNet): the `inpaint` variant
    wan22/         <variant>/…   (diffusers-format directories)
    qwen21/        Qwen-Image-2.1/ (the official diffusers folder)
    flux1/         *.safetensors (single-file transformers, e.g. flux1-dev-fp8)
    klein/         FLUX.2-klein-9B/ (the official diffusers folder)
  loras/
    sdxl/          *.safetensors (+ *.yaml, preview *.jpg/png)
    wan22/         *.safetensors; A14B pairs named *_high_noise.safetensors / *_low_noise.safetensors or declared in sidecar
    qwen21/        *.safetensors
    flux1/         *.safetensors
    klein/         *.safetensors
  controlnets/
    sdxl/          *.safetensors or diffusers dirs
  ip_adapters/
    sdxl/          IP-Adapter *.safetensors / *.bin (+ *.yaml sidecar: `purpose: subject|face|faceid|composition`)
    flux1/         FLUX.1-Redux-dev/ (the diffusers folder: image_encoder/, feature_extractor/, image_embedder/)
  image_encoders/
    sdxl/          clip-vit-h-14/ (h94/IP-Adapter models/image_encoder: config.json + model.safetensors),
                   clip-vit-bigg-14/ (sdxl_models/image_encoder, only for ip-adapter_sdxl)
  preprocessors/   one folder per preprocessor: sam3/ and depth-anything-v2/ (transformers folders), dwpose/ (two ONNX files),
                   insightface/ (det_10g.onnx and w600k_r50.onnx from buffalo_l, for FaceID)
  vae/
    sdxl/          sdxl-vae-fp16-fix/ (the fp16-fix VAE, diffusers folder)
  configs/
    sdxl/          stable-diffusion-xl-base-1.0/, stable-diffusion-xl-1.0-inpainting-0.1/
                   (the repos' configs and tokenizers, no weights)
    flux1/         FLUX.1-dev/ (the diffusers folder without transformer weights: T5, CLIP,
                   VAE, scheduler, and transformer/config.json)
```

**Sidecar YAML** (optional; every field is optional):

```yaml
label: "Film Grain v3"
trigger_words: ["filmgrain"]
default_weight: 0.8
variants: [t2v-a14b, i2v-a14b]      # wan22 only
pair: { high: foo_high_noise.safetensors, low: foo_low_noise.safetensors }
preview: foo.jpg
notes: "Works best with CFG 3–4"
source: "https://civitai.com/models/<id>?modelVersionId=<version>"   # where it came from
```

**Civitai imports** (`degas/civitai/`; `degas civitai import <link>`, or *Import from Civitai* in the LoRA picker). The version's `baseModel` picks the family: SDXL 1.0 and its fine-tunes (Pony, Illustrious, NoobAI) → `sdxl`, Flux.1 D/S/Krea → `flux1`, Flux.2 Klein 9B → `klein`, Qwen 2/2.1 → `qwen21`, and the Wan Video 2.2 bases → `wan22` with `variants` set. Anything else is refused unless a family is given. A Wan A14B file is named `<name>_high_noise` or `_low_noise` from `high`/`low` in its file or version name, so halves imported from two versions still pair up. The file streams from Civitai into `rclone rcat` on the server's writable remote (`lora.rclone_remote`), with its SHA-256 checked against Civitai's and its md5 against Drive's. A mismatch deletes it. Then a sidecar (label, trigger words, weight 0.8, the base model in `notes`, `source`) and a 768 px preview from the first example image (a still, for a video) are written, and Drive is rescanned. A file whose SHA-256 is already in the index, or whose name is taken, is refused unless forced. `degas civitai backfill` writes sidecars for LoRAs already in Drive, looked up by SHA-256.

**Drive access.** Degas does not use `drivemount`, because it needs interactive consent on every new VM (Phase 0). Instead:

- **OAuth.** Degas has its own Google Cloud OAuth client (desktop type, scope `drive.readonly`). A one-time setup command (`degas auth drive`) stores the refresh token on the server. rclone's shared client_id is being retired during 2026, so Degas doesn't depend on it.
- **Tokens to the worker.** The server mints access tokens, which last about 1 h. It pushes one to the worker (`POST /drive-token`) at session start and every 45 min after that. The worker writes it into an rclone config that has no refresh token.
- **Copying.** The worker copies assets with `rclone copy --multi-thread-streams 8`. A cold copy runs at about 65 MB/s: about 100 s for an SDXL checkpoint, and about 7 min for the Wan A14B pair.

**Asset index.** The server indexes `MyDrive/degas/` itself, using the Drive API with the same OAuth client. Indexing happens on demand ("Rescan Drive") and every 6 hours. The index holds each asset's path, Drive file ID, size, modification time, md5, parsed sidecar, and a thumbnail of any preview image. It is cached in SQLite, so the model and LoRA pickers, and rescans, work with no GPU session running. Pickers show a "last indexed" timestamp.

**Local cache.** Before first use in a session, the worker copies a model or LoRA into `/content/models/<same relative path>`, via `POST /assets/fetch` or implicitly when a job needs it. Copy progress is reported with `progress` events, where `phase: "copy"`. The server knows what each job needs, so it can prefetch the next queued job's assets while the current job is running. The VM has about 190 GB of free disk. The cache is evicted least-recently-used above 150 GB.

## 6. Data model and storage

### 6.1 Filesystem

```
data/
  degas.sqlite
  blobs/<sha256[:2]>/<sha256>.<ext>    # content-addressed; all images/videos/masks
  thumbs/<sha256>.webp                 # generated thumbnails and video posters
```

Every piece of media, whether a result, upload, mask, control image or preprocessor output, is a content-addressed blob. A mask is a single-channel 8-bit PNG at the same pixel size as the image it belongs to. White means "affected" (regenerate, or apply control), and grey values are allowed for soft edges. Whether a blob is ephemeral or saved is tracked in the database by reference, not by directory. A blob is deleted once nothing references it.

**Images from a URL.** `POST /api/blobs/from-url` fetches the URL on the server using `httpx`. The server:

- accepts only `http` and `https` URLs, follows at most 5 redirects, and times out after 20 s;
- stops the download if it exceeds 50 MB;
- sends a browser-like `User-Agent`, because some image hosts reject clients without one;
- decodes the downloaded bytes with Pillow to confirm they are an image, whatever the `Content-Type` header says (WebP, AVIF and HEIC are supported via `pillow-heif` and `pillow-avif`);
- applies the EXIF orientation and then strips EXIF, as is also done for camera-roll uploads, so what the app shows is what the model receives;
- normalizes the image to PNG, or to JPEG when it has no alpha channel, then stores it as an ordinary blob;
- also accepts `data:` URIs.

Direct links to video files (MP4/WebM) are accepted as a video source. The frame picker extracts frames from them the same way it does for result videos. Pages that are not media, such as a web page that embeds an image, are not scraped in v1; the endpoint returns a clear error instead. There is no SSRF filtering, because the server is single-user and only reachable on the tailnet.

### 6.2 Tables

- **`sessions`**: id, gpu, high_mem, state, started_at, ended_at, last_activity_at, error.
- **`jobs`**: id, session_id, status (`queued|running|done|cancelled|error`), queue_position, spec (JSON), created_at, started_at, finished_at, error, log.
- **`results`**: id, job_id, item_index, blob_sha, media_type, seed, width, height, duration, created_at, expires_at.
- **`library_items`**: id, kind (`image|video`), blob_sha, config (JSON, the full generation config), title, tags, created_at, source_result_id.
- **`prompts`**: id, name, prompt, negative_prompt, family (nullable), tags, created_at.
- **`blob_refs`**: blob_sha, ref_type (`result|library|job|draft|derived`), ref_id, expires_at (nullable). Used for reference counting and retention.
- **`blob_transforms`**: derived_sha, original_sha, ops (JSON). A derived blob holds a reference to its original (see §6.5).
- **`assets`**: family, kind (`model|lora|controlnet|ip_adapter|image_encoder|vae|config|preprocessor`), path, drive_file_id, size, mtime, md5, sha256 (Drive's, for Civitai lookups), sidecar (JSON), preview_thumb, indexed_at.
- **`push_subscriptions`**: endpoint, keys, created_at.
- **`settings`**: key, value. Holds idle timeout, default GPU, retention, and similar settings.

### 6.3 Retention

- A result's `expires_at` is set to `session.ended_at + 24h` when its session ends. While the session is active, `expires_at` is null.
- A sweeper runs every hour. It deletes expired results and then removes blobs that no longer have references.
- Saving a result creates a `library_items` row. That row takes its own references to the result's blob and to every input blob (source, control and mask images), so the saved item survives the sweep.
- Input blobs that no result or saved item references also have an expiry: uploads, URL fetches, derived (transformed) images, masks and preprocessor outputs. While they're referenced by the current Create form draft or a queued or running job, they are held. Otherwise they expire under the same rule as results: 24h after the end of the session they were created in, or 24h after creation if no session was active. This keeps an image you are still editing from being swept up.
- Jobs hold references to their input blobs until the job's results expire.
- The job queue view shows how long each unsaved result has left before it is deleted.

### 6.4 Saved generation config

Saving an image or video stores a config that is self-contained and can be replayed:

```json
{
  "degas_version": 1,
  "family": "sdxl",
  "variant": "base",
  "mode": "inpaint",
  "model": { "path": "models/sdxl/studioXL_v10.safetensors", "size": 6938040682 },
  "loras": [
    { "path": "loras/sdxl/filmgrain.safetensors", "weight": 0.8 }
  ],
  "params": {
    "prompt": "…", "negative_prompt": "…",
    "width": 1024, "height": 1024, "steps": 30, "cfg": 5.5,
    "scheduler": "dpmpp_2m", "schedule": "karras", "seed": 1234, "strength": 0.75,
    "mask_blur": 8
  },
  "inputs": {
    "source": "sha256:…",
    "mask": "sha256:…",
    "place": { "x": 0, "y": 0, "w": 1024, "h": 1024 },
    "transforms": {
      "sha256:<source>": { "original": "sha256:…",
        "ops": [ { "op": "rotate", "deg": 90 },
                 { "op": "crop", "x": 120, "y": 0, "w": 1536, "h": 1536 },
                 { "op": "resize", "w": 1024, "h": 1024, "filter": "lanczos" } ] }
    }
  },
  "control": [
    { "controlnet": { "path": "controlnets/sdxl/depth.safetensors", "size": 2502139136 },
      "image": "sha256:…", "preprocessor": { "id": "depth", "source": "sha256:…", "params": {} },
      "scale": 0.7, "start": 0.0, "end": 0.8, "mask": "sha256:…" }
  ],
  "runtime": { "gpu": "L4", "diffusers": "0.x", "torch": "2.x", "duration_s": 14.2 }
}
```

For a Wan 2.2 A14B LoRA, the entry has the form `{ "high": {path, weight}, "low": {path, weight} }`. If an input image came from a URL, the config also records `inputs.origins: { "<sha256>": "<url>" }` for provenance. Replay always uses the stored blob, so the URL is never fetched again. Video extension records the parent video in `inputs.extends` (the parent's blob hash) and the extracted frame in `inputs.source`. A Qwen-Image 2.1 edit records its extra reference images, in order, in `inputs.refs` (a list of blob hashes; the source is image 1). SDXL image prompts are a top-level `image_prompts` list, `[{adapter: {path, size}, images: [sha…], purpose, weight, start, end, mask?}]`; each picture is fitted to a square (cropped, or letterboxed with `fit: pad`) and its area to the output size, both recorded in `inputs.transforms`.

**Video extension output.** An extend job produces the new continuation clip as its result. When it completes, the server uses `ffmpeg` to create a second result: the stitched chain, which is the parent (itself possibly stitched) followed by the continuation, with the duplicated boundary frame dropped. Both results can be saved. The stitched result's config records the ordered list of segment configs, so every segment of the chain can be reproduced.

If an input was cropped or resized, `inputs.transforms` maps the derived blob to the original blob and the operations applied to it. Replay uses the derived blob. The original is kept so the crop can be adjusted during a remix.

**Remix** opens the generation form pre-filled from a saved config. If a referenced asset is no longer in the Drive index, the form flags it.

### 6.5 Image transforms (crop, resize, rotate, flip)

A transform is **non-destructive**:

- It is a list of operations (`rotate` by 90/180/270 degrees, `flip_h`, `flip_v`, `crop`, `resize`) applied to an original blob.
- The result is a new, **derived** blob. The server records the original and the operations (`blob_transforms` table: derived_sha, original_sha, ops JSON).
- Re-opening the editor on a derived image loads the original with the previous operations, so crops can be adjusted without losing quality over repeated edits.

**Where transforms run.** The phone never re-encodes pixels:

- The editor previews on the client with CSS transforms over a downscaled image.
- The editor sends the operations to the server. The server applies them with Pillow to the full-resolution original.
- Resizing uses Lanczos, for downscaling and for (non-AI) upscaling. AI upscaling is future work.
- The derived blob is content-addressed, so applying the same operations to the same original always yields the same blob.

**Constraints and presets.** Each family's `size_constraints` supplies the resolution presets and rules the editor uses:

- a `multiple_of` value: 8 for SDXL, 16 or 32 for Wan depending on the variant;
- the minimum and maximum pixel count;
- named presets, such as SDXL's aspect-ratio buckets (1024², 896×1152, 832×1216, …) and Wan's 480p/720p sizes.

**Crop modes** in the editor:

- **Match target:** the crop is locked to the aspect ratio of the output resolution currently set in the form. This is the default when the image is opened from a source slot.
- **Preset aspect:** choose from 1:1, 4:3, 3:2, 16:9, their portrait versions, or the family's buckets.
- **Free:** any rectangle. The final size snaps to `multiple_of`.

**Resize** is set in one of three ways: to the target resolution (default), to a preset, or to an explicit width × height with the aspect ratio locked. The editor shows the final pixel size, and warns when the image is being upscaled by more than 1.5×. Resizing is optional: with *Resize* off the transform ends at the crop, and auto-fit resizes it to the output size at submit.

**Auto-fit.** If the user skips the editor and the source image's aspect ratio doesn't match the target size, the job spec gets a `fit` mode:

- `crop` (default): center-crop to the target aspect, then resize.
- `pad`: letterbox. For SDXL, the editor offers to switch the job to outpaint, with the pad area as the mask.
- `stretch`: resize without keeping the aspect ratio.

The server resolves `fit` into explicit transform operations when the job is submitted. The saved config therefore always records exact operations.

**Interaction with masks and control images.**

- A mask belongs to a specific source blob, and is stored at that blob's pixel size (the editor paints at up to 2048 px a side and the server scales it back up). When the source is cropped again, the mask follows it: `POST /blobs/{mask}/remap` undoes the old crop's operations back onto the original (a crop comes back as a paste, so what it cut off is unmasked), then applies the new crop's. Whatever the new crop leaves out is dropped, and the UI says so. Picking a different image drops the mask.
- At submit, the mask gets exactly the fit operations its source gets, and is recorded in `inputs.transforms` like the source. An empty mask is refused.
- A control image that has not been edited is automatically fitted to the output resolution using the same `fit` rules.

## 7. Server API

All endpoints are under `/api`. JSON unless noted.

| Method | Path | Purpose |
|---|---|---|
| GET | `/families` | Family descriptors, variants, modes |
| GET | `/families/{id}/schema?variant=&mode=` | Param JSON Schema for form |
| GET | `/assets?family=&kind=` | Cached Drive asset index |
| POST | `/assets/rescan` | Re-index Drive via the Drive API (no session needed) |
| POST | `/civitai/plan` `{url}` | What importing a Civitai LoRA would do: family, Drive paths, sidecar (§5) |
| POST | `/civitai/import` `{url}` | Start the import (202); `import` SSE events follow it; 409 while one runs |
| GET | `/civitai/import` | The latest import's state |
| GET | `/session` | Current session state, idle countdown |
| POST | `/session` | Start `{gpu, high_mem}` |
| DELETE | `/session` | Stop |
| GET | `/jobs` | Queue + recent jobs |
| POST | `/jobs` | Submit `{spec, batch_count, seed_mode}` |
| PATCH | `/jobs/{id}` | Reorder: `{position}` in the queue, 0 runs next |
| POST | `/jobs/{id}/restore` | Undo cancelling a job that hadn't started |
| DELETE | `/jobs/{id}` | Cancel |
| GET | `/results?cursor=` | Recent ephemeral results |
| DELETE | `/results` | Delete every finished job and its results now (queued and running jobs stay; kept items keep their own refs) |
| POST | `/results/{id}/save` | Save to library (image/video + config) |
| POST | `/results/{id}/extend`, `/library/{id}/extend` | Create an i2v job spec seeded from the video's last frame (returns spec for editing, and the frame) |
| GET/PATCH/DELETE | `/library?q=&cursor=`, `/library/{id}` | Browse and search / edit title and tags / delete saved items |
| GET/POST/PATCH/DELETE | `/prompts?q=`, `/prompts/{id}` | Saved prompts (rename via PATCH) |
| POST | `/blobs` | Upload an image or video as the raw request body (with its `Content-Type`) → `{sha256, media_type, width, height, duration?}` |
| POST | `/blobs/{sha}/transform` | `{ops}` → derived blob `{sha256, width, height}` (applies to original if `sha` is itself derived) |
| GET | `/blobs/{sha}/transform` | Original sha + ops for a derived blob (to reopen editor); any other image is its own original with no ops |
| POST | `/blobs/{sha}/frame` | `{at: "first"\|"last"\|seconds}` → one frame of a video blob as an image blob |
| POST | `/blobs/from-url` | `{url}` → server fetches and stores → `{sha256, media_type, width, height}` |
| GET | `/blobs/{sha}` / `/thumbs/{sha}` | Media (Range support for video) |
| POST | `/blobs/{sha}/mask` | A mask painted over image `sha` (raw PNG body; alpha or white is redrawn) → a single-channel mask blob at the image's size |
| POST | `/blobs/{mask}/remap` | `{source, to}` → the mask carried from `source` onto `to`, another crop of the same original, plus `empty` |
| POST | `/preprocess` | `{id: "sam", image, params: {points: [{x, y, include}], text?}}` → `{candidates: [blob], chosen}`, smallest first; `{id: "depth"\|"pose"\|"canny", image, params}` → `{image: blob}`, the trace at the image's size |
| GET | `/push` | VAPID public key (`applicationServerKey`) |
| POST | `/push/subscribe`, `/push/unsubscribe` | Store or drop this device's Web Push subscription |
| GET | `/events` | SSE: session state, job progress, outputs |

**Batching and seeds.** Submitting a job with `batch_count > 1` creates one job whose worker loop produces N items. The seed for item *i* is `seed + i` in incrementing mode, or a random seed in random mode. The seed actually used is always recorded on each result. A fixed seed with N > 1 is only useful if other parameters vary, so the UI disallows it.

## 8. Frontend

### 8.1 Stack

- React 19, Vite and TypeScript.
- TanStack Query for server state, and a small Zustand store for form and editor state.
- A PWA manifest and a service worker (`vite-plugin-pwa`). The service worker caches the app shell and handles the Web Push `push` and `notificationclick` events.
- Form generation from the JSON Schema uses a thin custom renderer, not a heavy form library. The renderer has a fixed set of widget types: slider, number, select, text/prompt area, seed, aspect-ratio picker, image-input, LoRA list, and control-unit list.
- The mask editor is a plain `<canvas>` with Pointer Events. It runs at device pixel ratio and supports two-finger pinch-zoom and pan.

### 8.2 Screens

The app has a bottom tab bar with Create and Results, and Library from Phase 3. The GPU session is a chip in the header that opens the Session sheet. The visual system and per-phase screen plans are in [ux.md](ux.md).

1. **Create.** The generation form, laid out top to bottom:
   - Family, variant and mode selectors.
   - Prompt fields, with a "Saved prompts" picker and a Save-prompt button.
   - A model picker and a LoRA list: add, remove, and a weight slider per LoRA. A14B LoRAs show a pair of weight sliders.
   - A source-image slot (i2i, i2v, inpaint, outpaint), which opens the image picker.
   - Inpaint: a mask slot that opens the mask editor on the source image. Outpaint: a preview of the source on the canvas (the Size), which can be dragged, scaled and pushed to an edge.
   - Control units (SDXL): each unit has an image slot, a preprocessor button, a model, scale, a start/end range slider, and an optional mask.
   - Parameters, with the advanced ones collapsed.
   - Batch count and seed mode.
   - A Generate bar, sticky above the tab bar, with the batch size stepper. Create stays on screen after Generate.
2. **Image picker** (a sheet). Tabs for Recent results, Library, Camera roll (`<input type="file" accept="image/*">`), and URL. The URL tab has a text field and a "Paste" button that reads the clipboard (`navigator.clipboard.readText()`), then shows a preview before the image is used. Choosing a video offers its first frame, its last frame, or a scrubbed frame. After an image is picked, the sheet shows it with **Use** and **Crop / resize** buttons. A filled image slot on the Create form also has an edit button that opens the editor.
3. **Crop and resize editor** (full screen):
   - Pinch and drag to position the image under a fixed crop frame; the crop corners can also be dragged.
   - An aspect selector: Match target, the presets, or Free.
   - Rotate 90° and flip buttons.
   - A resize row showing the output size, with target, preset and custom options.
   - A live readout of the final pixel size, with an upscale warning.
   - Reset (return to the original) and Apply (send the operations to the server; the derived image goes into the slot).
4. **Mask editor** (full screen; see ux.md Phase 6 for what shipped):
   - Brush and eraser, a size slider, invert, clear, and undo/redo.
   - SAM mode: tap to add a positive point, and long-press or toggle to add a negative one. The returned mask is composited onto the current mask as add, subtract or replace.
   - Feather preview.
   - The source image appears underneath at adjustable opacity.
5. **Control editor.** Choose a control image (from the picker or from a preprocessor output), run a preprocessor with its parameters, preview the result, and attach a mask for regional control.
6. **Results** (the queue and results in one feed). Each job is a group captioned with its prompt. Running and queued jobs come first; unfinished images are drawn as hatched sketch tiles that fill in with denoising progress. Queued jobs can be cancelled, and from Phase 5 reordered. Finished groups show how long their unsaved images have left.
7. **Viewer.** A full-screen image with the generation settings as a wall label. Actions: save to Photos (share sheet) and reuse settings in Phase 1; save, save prompt, remix, use as source, use as control and extend (video) as their phases land.
8. **Library.** A grid of saved items with search by prompt text and tags, plus a saved-prompts tab. The detail view has remix, use as source, and delete.
9. **Session** (a sheet opened from the header chip, which shows the GPU and the idle countdown). Choose a GPU and start or stop the session. Shows state, bootstrap progress, idle countdown, and GPU/VRAM information. Also contains Drive status (the last index time and a Re-authorize button if the refresh token has failed), the Rescan Drive button, a **Force reset worker** button, and settings.

### 8.3 iOS specifics

- **Video:** `<video playsinline muted loop>` served with HTTP Range support, H.264 in `yuv420p`.
- **Saving to Photos:** use the Web Share API (`navigator.share({files})`) so the image or video can go to the Photos app. Fall back to a download link.
- **Reconnects:** SSE reconnects automatically. On `visibilitychange` the app refetches state, because iOS suspends background tabs.
- **App switcher:** a web page can't choose what iOS snapshots for the switcher. Discretion mode (Phase 9) raises a paper shield on `blur`, `visibilitychange` and `pagehide`, and keeps images covered at rest in case the snapshot comes first.
- **Web Push:** requires the app to be installed to the home screen (iOS 16.4+). The server uses VAPID keys and `pywebpush`. Notifications are sent when a job completes or fails, and when an idle shutdown is 2 minutes away.

## 9. Deployment

- The server and built frontend run as a single process, `degas serve`, bound to `127.0.0.1:8420`.
- `tailscale serve --bg --https=8448 http://127.0.0.1:8420` exposes it on the tailnet with HTTPS. The port is 8448, not 443, because other apps on the host already hold 443 and 8443–8447. A non-443 HTTPS origin is still a secure context, so PWA install and Web Push work.
- It runs as a systemd user service, `deploy/degas.service`. `make deploy` builds the PWA, syncs the venv, installs the unit and restarts it. The unit runs the `tailscale serve` above in `ExecStartPost` and turns it off in `ExecStopPost`.
  - A user unit cannot order against system units such as `tailscaled`. It sets `Restart=always` and `StartLimitIntervalSec=0`, so it retries until tailscaled is up after a reboot.
  - The unit starts at boot only if `sudo loginctl enable-linger $USER` has been run.
- Configuration comes from `degas.toml`:
  - the data directory;
  - the `colab` binary path and auth mode;
  - the path to Degas's SSH key (ed25519, generated at setup);
  - the Drive root (`degas/`), the OAuth client file, and the stored refresh token;
  - the VAPID keys;
  - the default idle timeout.
- The server needs `ffmpeg` (for video posters, frame extraction and stitching) and Pillow with the HEIF/AVIF plugins.
- The `colab` CLI is installed with `uv tool install google-colab-cli`, and authentication is set up once on the server.
- The server needs OpenSSH and a static `rclone` binary for linux-amd64. The rclone binary is uploaded to the VM.
- One-time setup: `degas auth drive` runs the OAuth consent flow for the Drive client.
- The worker bundle is built from `worker/`. Its content hash is checked at session start, and it is uploaded again if it has changed.

### Repository layout

```
degas/
  docs/design.md
  server/src/degas/        app.py, api/ (routers, schemas, error statuses), db/ (schema,
                           queries, row types), colab/ (CLI wrapper, session mgr,
                           tunnel.py for ssh), dispatcher.py, drive/ (OAuth, index, catalog),
                           families/ (descriptors, validation), inputs.py, library.py, push.py
  worker/src/degas_worker/ app.py (FastAPI), routers/, jobs.py, cache.py (rclone), families/,
                           preprocess/, spec.py (the job spec, shared with the server)
  worker/requirements-worker.txt
  web/                 Vite React app
  degas.toml.example
```

## 10. Implementation phases

Phases are numbered from 0.

0. **Spike.** ✅ Done: see [phase0-findings.md](phase0-findings.md). Outcome: the worker runs as an HTTP server behind an SSH tunnel, and Drive is accessed with OAuth plus rclone.
1. **Core loop.** ✅ Done. Live test on a T4 (2026-09-27, SDXL base 1.0 from Drive, 1024², 30 steps): session `ready` in 30 s; first job 174 s (Drive copy 90 s at ~77 MB/s, model load 30 s, denoise 28 s); next images ~34 s each with the model resident; same seed reproduced a byte-identical PNG. Build the session manager (colab CLI and SSH tunnel), Drive OAuth and indexing, the worker HTTP app, the dispatcher, and the SDXL family in `t2i` mode. Add the blob store, SSE, and a minimal Create/Queue/Results UI. Exit criterion: generate an image from the phone.
2. **Assets and LoRA.** ✅ Built, not yet tested on a live GPU. Sidecars, the rclone-backed model cache with prefetch and eviction, and LoRA application (with diffing) for SDXL. Model and LoRA pickers in the UI. The server prefetches the next queued job's assets once the running job is past its own copy phase. The VM cache budget is `colab.cache_budget_gb` (150 by default), and `/health` reports the cached files so the pickers can say what is already on the GPU.
3. **Save, library and remix.** ✅ Built. Keep saves a result with its replayable config (§6.4, including the runtime GPU and library versions reported by `/health`) to `library_items`, and the item holds its own blob refs. Saved prompts. An hourly retention sweeper deletes expired results, finished jobs left with no results, and unreferenced blobs (with a 1 h grace for blobs just written). Remix loads a kept image's or a result's config into Create and flags a model or LoRA that's no longer in the Drive index. Library search matches prompt text, title and tags.
4. **Wan 2.2.** ✅ Built, not yet tested on a live GPU. Wan 2.2 descriptor (TI2V 5B for t2v and i2v; T2V and I2V A14B with high/low LoRA pairs; models are diffusers folders under `models/wan22/<variant>/`, which picks the variant) and runner (`WanImageToVideoPipeline.from_pipe` for 5B i2v, each A14B LoRA half loaded into its expert, MP4 via ffmpeg in `degas_worker/video.py`). Uploads, URL/`data:` imports, video frames, non-destructive transforms and auto-fit at submit (`degas/inputs.py`, `degas/media.py`); video posters, durations and extension stitching in the dispatcher; the image picker, crop editor, video playback, Extend and Use as source in the UI. The 5B variant (t2v and i2v), then the A14B variants with paired LoRAs. Video playback, video extension and stitching. Build the full image picker here (recent results, library, camera-roll upload, URL import, frame selection from videos), together with the crop and resize editor, image transforms, and auto-fit here, since i2v is the first mode that takes a source image.
5. **Queue UX and push.** ✅ Built, not yet tested on an installed iPhone. Batch generation and seed modes, cancel, reorder, the PWA manifest and service worker, and Web Push. A batch picks *Random seeds* or *Count up* from the seed field. `PATCH /jobs/{id}` `{position}` reorders the queued jobs among their existing positions (new jobs still go last) and re-targets prefetch; `POST /jobs/{id}/restore` undoes cancelling a job that never started. The service worker is hand-written (`web/public/sw.js`) rather than `vite-plugin-pwa`: it caches the hashed build assets, falls back to the cached shell offline, never caches `/api`, and shows pushes. `degas/push.py` generates the VAPID key pair on first use (`push.key_file`, default `<data_dir>/vapid_private.pem`) and sends with `pywebpush` off the event loop, dropping subscriptions the push service reports gone (404/410). Notifications go out when a job finishes or fails (not when it's cancelled) and once per idle deadline when an idle session is 2 minutes from stopping; tapping one opens `/?tab=results` or `/?sheet=session`.
6. **i2i, inpaint and outpaint.** ✅ Done. SDXL gains `i2i`, `inpaint` and `outpaint` on regular checkpoints, and an `inpaint` variant for inpainting checkpoints in `models/sdxl/inpaint/` (§4.3). Masks are stored by `POST /blobs/{sha}/mask`, follow their source through re-crops (`POST /blobs/{mask}/remap`) and are fitted with it at submit (§6.5). Outpaint places the source on the Size canvas (`inputs.place`). SAM 3 selection moved up from Phase 7: `POST /api/preprocess` stages the image and asks the worker's new `/preprocess` route, which loads `Sam3TrackerModel` for taps or `Sam3Model` for a description (one at a time) from `preprocessors/sam3/`. Drive indexes a preprocessor folder with a `config.json` as one asset. The mask editor (brush, erase, select, undo, invert, blur preview) and the outpaint placement are in the UI. Live test on a T4 (2026-09-28, a regular checkpoint): inpaint at 1024², 30 steps, strength 0.85, with the mask made by SAM 3 selection and SAM 3 resident next to SDXL; four jobs took 29–34 s each. The first attempts ran out of memory: diffusers 0.40's `from_pipe` defaults to float32 and casts the shared modules in place, which doubled SDXL to 13.1 GiB. SDXL now passes `torch_dtype=torch.float16` (6.6 GiB loaded, 9.7 GiB peak for a 1024² inpaint), and Wan builds its i2v pipeline from the components instead of using `from_pipe`. Still untested live: i2i, outpaint, the `inpaint` variant, and Wan i2v after that change.
7. **Control.** ✅ Built, not yet tested on a live GPU. Preprocessors (depth, pose, canny), SDXL ControlNet units, and regional ControlNet. A unit is `{controlnet, image, fit, scale, start, end, mask?, preprocessor?}` (§6.4); at submit each control image is fitted to the output size by its own `fit`, and its area mask with it, both recorded in `inputs.transforms`. The Drive index takes a folder under `controlnets/<family>/` with a `config.json` as one asset, and every folder directly under `preprocessors/` as one. A ControlNet's sidecar can say `control: depth|pose|canny`, which the UI uses to pick a model for a trace and to warn about a mismatch. Depth Anything V2 and DWPose run on the worker (DWPose's ONNX pre- and post-processing are adapted from controlnet_aux); canny is OpenCV. `degas_worker/deps.py` installs a missing package the first time it's needed. Untested live: all of it, in particular `from_single_file` for SDXL ControlNets, the ControlNet pipelines built with `from_pipe`, the residual masking under CPU offload, and onnxruntime's CUDA provider on Colab.

8. **Qwen-Image 2.1.** 🚧 In progress. The `qwen21` family (§4.3): `t2i`, `edit` with up to 10 ordered images (the source plus `inputs.refs`), and `inpaint` as an edit pasted back through the mask. The server validates the references, checks they're still stored, and keeps them (and a cropped reference's original) with the job and any kept item. The runner loads `QwenImage21Pipeline` from the diffusers folder in Drive and offloads to the CPU when the weights don't fit. It maps CFG to `true_cfg_scale` and the schedule to the scheduler's settings, and runs the ported DeGrid filter. The bootstrap installs a pinned diffusers commit with the pipeline when the image lacks it. In the UI, *Edit* is a mode chip, and an *Images* row under Source adds, crops, orders and removes references. A reference's crop is free and keeps its own size. Live test to do, on an L4 (high memory) and an A100: cold copy, load, and per-image time at 1K and 2K; a 40-step Default against 30-step Beta comparison to settle the default schedule; the grid filter on skin and hair; and SDXL and Wan on the pinned diffusers. LoRA live test on an A100 (2026-10-01, diffusers 0.41.0.dev0, offloaded): two Civitai LoRAs in ComfyUI's `diffusion_model.…lora_A/B` format loaded through the pipeline's mixin, Lenovo UltraReal in `t2i` and Anything2RealCharacters in `edit`. In each mode a fixed seed gave a different image at weight 0.8 and at 1.2 (the adapter was reused, not reloaded), and removing the LoRA gave back the no-LoRA image exactly. About 48 s an image for `t2i` at 1024² and 50 s for an 832×1216 edit, 40 steps, CFG 1; a LoRA's first use added 10–20 s. Lenovo UltraReal at 1.2 is overcooked, which is the LoRA and not the loader.
9. **Discretion.** ✅ Built, not yet tested on an installed iPhone. A header switch (`web/src/lib/discretion.ts`, kept in `localStorage` and set on `<html data-discreet>` by an inline script before the first paint) covers images and prompts with a blur until tapped; ux.md Phase 9 has the behaviour. Uncovered items are held in a module store and cleared by closing the viewer, a tab change or hiding the app. On `blur`, `visibilitychange` and `pagehide` a CSS-only shield covers the viewport for the app switcher. The page tells the service worker the mode (`postMessage`, kept in the `degas-prefs` cache, which activation no longer deletes), and pushes then leave out the body. `App.test.tsx` now renders under `StrictMode`, as `main.tsx` does. To check on the phone: whether the shield is in the switcher's snapshot, from the feed and from an uncovered viewer; scrolling with ~100 covered tiles (if CSS blur is slow, use a tiny thumbnail scaled up instead); and the lock-screen notification text.
10. **Flux.** ✅ Built; LoRAs and klein `t2i` tested on an A100. Two families (§4.3): `flux1`, FLUX.1 [dev] for text-to-image, and `klein`, FLUX.2 [klein] 9B for text-to-image and editing (`t2i` added 2026-10-01). A FLUX.1 checkpoint is a single-file transformer. The base folder in `configs/flux1/FLUX.1-dev/` supplies the rest and is the job's `config`, so it's checked at submit and prefetched like SDXL's. A checkpoint stored in fp8 stays fp8 on the GPU through layerwise casting. Klein reads the source and up to 3 references. Variants now declare `max_refs` (the web *Images* row stops there) and `ref_max_pixels`. `validate_refs` moved to `families/validation.py`, and the offload rule to `degas_worker/families/offload.py`. Both models are gated on Hugging Face: accept their licenses, download with an HF token, and upload with rclone, creating the Drive folders first. Live test to do, on an L4 (high memory) and an A100:
    - load time, peak memory and per-image time for `flux1-dev-fp8` at 1024², 28 steps;
    - fp8 against bf16 on a fixed seed (an A100 holds the bf16 folder);
    - a kohya-format Civitai LoRA on the fp8 transformer, checking that the adapters stay bf16 and the output is sane;
    - that layerwise casting works under model CPU offload;
    - klein edits with 0 and 3 references, and their time on an L4, where it offloads.

    Live test on an A100 (2026-10-01, diffusers 0.41.0.dev0), each set at a fixed seed with no LoRA, the LoRA at 0.8, at 1.2, then no LoRA again. Every set gave three different images, reused the adapter for the weight change, and gave back the no-LoRA image exactly once the LoRA was removed:
    - klein `t2i` (1024²) and `edit` (832×1216, no references) with UltraReal (ComfyUI `diffusion_model.…lora_A/B` keys), offloaded: 36–39 s an image at 4 steps, and about 30 s more for the LoRA's first use. The adapter carried over from `t2i` to `edit` without a reload.
    - `flux1-dev-fp8` `t2i` at 1024², 28 steps, with a kohya-format LoRA (`lora_unet_…`, Velvet's Mythic Fantasy Styles): 17–19 s an image, no fp8 error and the output is sane (the adapters' dtype wasn't inspected directly). The first job took about 80 s to load after a 2.8 min copy.
    - Still to do: an L4, fp8 against bf16, klein with references, and a LoRA with text-encoder keys.
11. **Image prompts (IP-Adapter).** ✅ Built, not yet tested on a live GPU. SDXL image prompts (§4.3; research and plan in [ip-adapter.md](ip-adapter.md), Plan A). The Drive index takes `ip_adapters/<family>/` files and `image_encoders/<family>/` folders; sidecars can say an adapter's `purpose`. `validate_image_prompts` in `families/validation.py`; pictures are squared and areas fitted in `inputs.resolve`; kept items and staging include them. The worker's torch-free `families/ip_adapter.py` maps purposes to blocks and step ranges to scales. In the UI, an *Image prompts* row under ControlNet opens a sheet with the pictures (cut square, as the encoder sees them), *Everything / Style / Layout / Style and layout / Face* chips that pick the blocks and the model, weight, steps and an area; the crop editor gains a square mode. Live test to do (ip-adapter.md §4.8), on a T4 and an L4: VRAM with ViT-H resident next to SDXL, with and without ControlNet; Style, Layout and Everything against no image prompt on a fixed seed; a LoRA with an image prompt, then removing the image prompt (no leftover IP layers); two units with areas; a step range; `enable_model_cpu_offload` after registering the encoder; the NoobAI adapter on an Illustrious checkpoint.
12. **Faces (FaceID).** ✅ Built; tested on an A100 (below). FaceID Plus v2 image prompts (§4.3; ip-adapter.md Plan B and §5.1). InsightFace's detector and recognizer run on onnxruntime without the `insightface` package; alignment and NMS are plain Python (`degas_worker/faces.py`). The runner keeps the model's embedded LoRA weighted with the job's own, and deletes it on unload. The *Face* chip prefers FaceID, starting at 0.8, and adds *Face structure* and *Face LoRA* sliders; a ready session shows the face each picture's crop found. Live test to do: that diffusers loads the SDXL FaceID Plus v2 `.bin` and its LoRA; likeness against IP-Adapter Plus Face on the same seed; FaceID with a character LoRA; removing FaceID leaves no `faceid_*` adapter active; onnxruntime's CUDA provider next to SDXL; a picture with two faces.

    Live test on an A100 (2026-10-01, diffusers 0.41.0.dev0), 29 images at a fixed seed, kept in the library under the tag `faceid-test`; the settings compared are in ip-adapter.md §5.2. Every job ran:
    - diffusers loads `ip-adapter-faceid-plusv2_sdxl.bin` and its LoRA; about 8.5 s an image at 1024², 30 steps, on a photoreal SDXL checkpoint. It also ran on a second one and SDXL base, and in `inpaint` *Around the mask*.
    - FaceID with a style LoRA at 0.8: the LoRA-only image before and after the FaceID job is identical, so no `faceid_*` adapter stays active.
    - A picture with two faces uses the bigger one; two pictures in a unit, and FaceID beside Plus Face, both run.
    - Face detection now retries at 576 down to 320 px when 640 finds nothing (as ComfyUI_IPAdapter_plus does): before, two close crops of a face were refused with *No face found*; after, both are found.
    - Not FaceID's doing, but seen here: after an SDXL LoRA has been loaded and removed, a job without one differs slightly from a fresh worker's (mean 0.8 of 255, nothing visible), and stays that way until the worker restarts. Removing FaceID or Plus Face adds nothing to it.
    - Still to do: which onnxruntime provider InsightFace gets next to SDXL (a face takes 0.3 s either way), a T4 or L4, and a character LoRA.
13. **FLUX.1 Redux.** ✅ Built, not yet tested on a live GPU. Image prompts for FLUX.1 (§4.3; ip-adapter.md Plan C and §6.1): Redux's tokens appended to the prompt's, shrunk by *How closely* and scaled by the weight, as ComfyUI does, rather than diffusers' summing prior pipeline. Families now send `image_prompt_options`, and the *Image prompts* sheet shows only what the family's allow. Live test to do, on an L4 (high memory) and an A100: Redux's load time and memory next to the fp8 transformer, with and without offload; each *How closely* setting on a fixed seed against no image prompt; two pictures; a LoRA with Redux.

## 11. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| `colab ssh` is a new CLI feature and could change or break | High: the transport depends on it | Keep the transport behind one interface (`colab/tunnel.py`). An `exec`-based fallback is proven to work (persistent kernel, streamed output, cancel via a flag file uploaded mid-job) |
| A VM with an idle kernel may be reclaimed even while the worker is busy | Jobs killed partway through | Build the `exec` heartbeat in Phase 1 as a setting that is on by default (one trivial `exec` every 5 min, costing about 1.5 s each) |
| Drive OAuth app in "testing" mode issues refresh tokens that expire after 7 days | Weekly re-authorization | Publish the OAuth app for your own account only (drive.readonly is a restricted scope, but an unverified app used only by its owner is fine); the UI prompts to re-authorize when a refresh fails |
| SSH tunnel throughput (about 12 MB/s) | Slow transfer of large video outputs | Acceptable: a 10 MB MP4 takes about 1 s. Encode with a sensible CRF |
| Colab reclaims the VM or hits usage limits | Running job is lost | Detect the failure and move the session to `error`; queued jobs return to the queue; the notification explains what happened |
| Copying Wan A14B (two 14B experts) from Drive is slow, and it needs offloading | Cold start of about 7 min for the copy; slow generation; standard shapes have only 12 GB of RAM | Show copy progress; prefetch; keep the session warm; default to `--high-mem`; use 5B for drafts |
| Regional ControlNet needs a custom residual-masking path | Extra complexity | Isolated in `limit_to_areas`, which patches `forward` on the ControlNet instance for the length of a call |
| Session bootstrap time | Measured: `colab new` 3.5 s (CPU), SSH 1.5 s, and packages are preinstalled, so bootstrap takes about 10–20 s before model copy | Good enough; the model copy dominates |
| Web Push on iOS | Needs a home-screen install and iOS 16.4+ | Document it; the in-app SSE path works regardless |
| diffusers API churn for newer Wan and ControlNet classes | Upgrades break the worker | Pin versions in `requirements-worker.txt`; record versions in each saved config |
| fp8 layerwise casting is newer diffusers code, used here with model CPU offload and PEFT LoRAs | FLUX.1 fails to load a LoRA, or runs slowly, on an L4 | `_adapters_in_bf16` keeps adapters out of fp8; the live test covers offload and a LoRA; on an A100 and up, a bf16 folder model skips fp8 entirely |
| diffusers' IP-Adapter layers and PEFT LoRAs on the same attention modules; `unload_ip_adapter` resets every processor | A LoRA or a regional ControlNet misbehaves after image prompts are added or removed | The live test adds and removes each with the other loaded; the runner remakes derived pipelines whenever the adapter set changes |
| Qwen-Image 2.1 needs diffusers `main` (not in 0.40.0) | The pinned commit also runs SDXL and Wan, and could break them | Install the pin only when the image's diffusers lacks `QwenImage21Pipeline`; re-test SDXL and Wan on it; move to the first release that has the pipeline |

## 12. Future work

- Video control for Wan 2.2 (VACE / Fun-Control): pose- or depth-driven video, reusing the preprocessors and control-unit UI.
- Additional families such as SD3.5, FLUX.2 [dev] and Hunyuan/LTX video, added through the plugin interface.
- Upscaling / hires-fix passes, and video frame interpolation.
- Live latent previews during sampling using TAESD / TAEW.
- Regional LoRA.
- Multiple concurrent sessions, e.g. an L4 for images alongside an A100 for video.

### 12.1 Qwen-Image 2.1 after Phase 8

Phase 8 leaves these out, to get a first live test sooner:

- **Transparent output.** The VAE decodes RGBA and the model can make images with a real alpha channel. Qwen's recommended prompt: *"This is an RGBA image with transparency. [description]. The image has alpha channel and the background is transparent."* Uploads with alpha already stay PNG, and the grid filter keeps the alpha it's given. What's missing:
  - a way to ask for it (a *Transparent background* switch that adds Qwen's wording);
  - a checkerboard behind transparent images in the viewer and thumbnails;
  - inpaint, whose paste-back flattens to RGB;
  - a check of what *Save to Photos* does with alpha on iOS.
- **Native mask editing.** The model card lists editing guided by circles, painted marks, or a separate mask. The diffusers pipeline has no mask argument, so the mask would have to reach the model as a condition image (a mark drawn on the source, or the mask as its own image). The format it was trained on isn't documented yet. Until then, Degas inpaints by editing the whole picture and pasting the original back outside the mask. That keeps the unmasked area exact, but the model can't tell which area it is meant to change.
- **Sizing references by their longest edge.** The pipeline scales every reference to the output's pixel count. It rounds to 32, and a thin reference ends up much longer than a square one. A community ComfyUI workflow for Qwen-Image 2.1 scales by the longest edge instead and says results are more predictable. Doing this needs either pre-resized references with the pipeline's own resize bypassed, or an option upstream. It's worth an A/B test first. Notes for when it's picked up:
  - The pipeline sizes every condition image with the module-level `calculate_dimensions(output_resolution², ratio)` in `pipeline_qwenimage21`, looked up at call time, and has no other resize path. The runner could swap it for the length of a call (as `limit_to_areas` does for ControlNet), finding the module through `QwenImage21Pipeline.__module__` and refusing to run if the function is gone. The pipeline also calls it once for its default output size, which is ignored when `width` and `height` are passed.
  - A rule to test: the longest edge is the output's longest side, but a reference never gets more pixels than the output. Image 1, already fitted to the output, comes out at exactly its size under both rules. On a square output, square references are unchanged and thin ones get smaller (at 2048², a 1:2 reference goes from 1440×2880 to 1024×2048), and no reference gets more tokens than it does now. A fixed longest edge (1024 or 1536) is the cheaper alternative.
  - It would be an advanced `edit`/`inpaint` param (*Size references by*: *Output's pixel count* / *Longest edge*), defaulting to today's rule so saved configs replay unchanged. The A/B: fixed seeds at 2048² and 2752×1536 with a tall full-body, a wide scene and a square face as references, comparing fidelity, time and peak memory.
- **Prompt rewriting.** Qwen publishes two rewriting models, `Qwen/Qwen-Image-2.1-PE-T2I` and `-PE-I2I` (Qwen3.5-VL 9B fine-tunes). They expand a short prompt and suggest an aspect ratio. They're another ~9B model, so they'd compete with the image model for GPU memory. They would suit an *Expand prompt* action that shows the rewritten text for review before the job runs, rather than a hidden step.
- **Faster attention and smaller weights.**
  - diffusers' `QwenImage21FlexAttnProcessor` is faster, but only with `torch.compile`: uncompiled, it runs out of memory at high resolution, and compiling costs time on the first job of each session. The ComfyUI write-up reports ~30% from SageAttention, but the model uses its own attention processors, so it doesn't plug straight in.
  - ComfyUI's int8 weights (6.9 GB transformer) don't load in diffusers. Quantizing when loading could make the text encoder fit an L4 without offload. The usual tool is torchao, which the bootstrap currently uninstalls because Colab's copy is too old for peft; that would need a newer torchao instead.
