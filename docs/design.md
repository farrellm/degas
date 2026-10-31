# Degas — Design Document

Status: Draft v1 · 2026-09-27

## 1. Overview

Degas is a personal web app for generating images and videos with generative models. It is used primarily from an iPhone over Tailscale. The app runs on a home server. All inference runs on Google Colab GPU runtimes, which the server drives through the [Google Colab CLI](https://github.com/googlecolab/google-colab-cli). Models, LoRAs, ControlNets and preprocessors are loaded from Google Drive.

The initial model families are Stable Diffusion XL (images) and Wan 2.2 (video). The architecture treats model families as plugins so that new ones can be added without changing the UI.

### Goals

- Mobile-first PWA that feels good on iPhone Safari and can be installed to the home screen.
- Text-to-image, image-to-image, inpainting and outpainting with SDXL.
- Text-to-video, image-to-video, first-and-last-frame video and video extension (last frame → i2v) with Wan 2.2.
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
5. `colab exec -s degas` with a bootstrap snippet. It starts the worker with `subprocess.Popen([... "uvicorn", "--factory", "degas_worker.app:create_app", "--host", "127.0.0.1", "--port", "8765"], start_new_session=True)` and returns immediately. Starting from the kernel means the worker inherits `LD_LIBRARY_PATH=/usr/lib64-nvidia` and the rest of the CUDA environment.
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

**Compute units.** `colab usage` reports the account's balance, its rate in units an hour, and its active assignments. All three cover every VM on the account, including other projects', so the balance can't be charged to Degas. The server reads usage just before `colab new` and again after it. If one assignment was added, the difference in rate is that session's own rate (`sessions.unit_rate`, from `allocated_at`), and the sheet shows the session's spend as that rate times the time since allocation. If the rate can't be told apart, it's retried at each 5 min `colab status` check, and the sheet shows only the balance. Usage is also re-read when the session ends, and when `GET /session` finds the figure older than 5 min. The re-read runs in the background, and the new figure arrives as a `session` event.

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
  - `i2v-a14b`: image-to-video and first and last frame (`flf2v`), minimum GPU A100.
  - `wan21-i2v-14b`: Wan 2.1's 14B image-to-video model (the 480P or 720P diffusers folder), minimum GPU A100. It is here because it shares the pipeline and runner, and because LoRAs trained on it lose their image cross-attention layers (`k_img`, `v_img`) on Wan 2.2, which has no CLIP image encoder. It has one transformer, so no low-noise CFG or expert switch, and takes single-file LoRAs. Live test on an A100 40 GB (2026-10-02, diffusers 0.41.0.dev0, 480P): 656×592, 81 frames, 28 steps, CFG 5. The cold copy of the 90 GB folder and load took about 8 minutes, and a clip about 9.5 minutes (roughly 20 s a step). At a fixed seed, Remade-AI's Squish LoRA (a Wan 2.1 I2V LoRA) turned a hand pressing a rigid figurine into the squish effect, and the worker logged no unexpected LoRA keys. The pipeline needs `ftfy`, which the bootstrap installs.
  - `wan21-flf2v-14b`: Wan 2.1's 14B first-and-last-frame model (`Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`), minimum GPU A100, `flf2v` only: its image embedding expects two pictures. It is fine-tuned from the I2V 720P model, so it takes the same single-file LoRAs (Civitai's `Wan Video 14B i2v` LoRAs are tagged for both), and its clips are extended with `wan21-i2v-14b`. Defaults from diffusers' example: 50 steps, CFG 5.5.
- **Pipelines:** diffusers `WanPipeline` and `WanImageToVideoPipeline`. In `flf2v`, `inputs.end` goes in as the pipeline's `last_image`, fitted to the output like the source. The 5B doesn't offer `flf2v`: its pipeline conditions on the first frame only and would ignore the last.
- **LoRAs:** `paired_hi_lo` for the A14B variants. A LoRA entry names a high-noise file and a low-noise file, and each has its own weight. Both may be the same file: a single-file LoRA (Wan 2.1 14B's load on the A14B experts) whose sidecar lists an A14B variant is offered there and goes into both experts, still with a weight each. A LoRA entry for the 5B and Wan 2.1 variants is a single file; an untagged single file is offered to both, so tag it with `variants` in its sidecar.
- **Parameters:** prompt, negative prompt, resolution preset, frame count, fps, steps, CFG (with separate values for the two experts on A14B), boundary ratio (A14B), seed, a source image for i2v, and a first and last frame for flf2v.
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
- **Prompt upsampling.** After Kokon's "Prompt+" (GabrieleLozupone/kokon, run there on its stable-diffusion.cpp fork). The text encoder is a whole `Qwen3VLForConditionalGeneration`, LM head included, so it can write a prompt too, with no second model. *Upsample prompt* (off by default; `t2i` and `edit`, not `inpaint`, whose mask the encoder wouldn't see) has it rewrite the prompt first. For `t2i` it writes one detailed English paragraph. For an edit it sees the images (each at most 1024 px, labelled `<image1>`… when there are several) and writes one precise instruction that ends by keeping everything else. Sampling is Qwen's for Qwen3-VL, as Kokon uses it: temperature 0.7, top-p 0.8, top-k 20, and a presence penalty of 1.5 on generated tokens (a logits processor, since transformers' `repetition_penalty` is multiplicative and counts the prompt). It's seeded with the job's first seed, runs once per job (progress phase `upsample`), and stops at 1024 tokens. A leading `Prompt:` label or quotes around the whole answer are removed, and the result is kept as each image's caption. An empty answer sends the prompt as written. Offloaded, the encoder is moved to the GPU before generating, and the transformer's offload hook sends it back. *Upsample style* (under More settings) picks the system prompts. `kokon` (the default) uses Kokon's own, chosen in its blind A/Bs, verbatim in `degas_worker/families/qwen21_prompt_kokon.py`, which keeps their AGPL-3.0 license. `degas` uses ours, written from the rules Kokon's notes give. Neither names an example colour or object, because Kokon found the model copies them. *Uncensored upsampler* (under More settings, off) has Qwen3-VL 8B Heretic write instead (`DreamFast/Qwen3-VL-8B-Heretic-1.3.0`, Heretic 1.3.0 on Qwen3-VL-8B-Instruct, the encoder's base model, with its refusal direction ablated; KL 0.031, Apache-2.0), in `upsamplers/qwen21/Qwen3-VL-8B-Heretic/` with the same style, sampling and the pipeline's processor. Only the writing changes: the pipeline still encodes with its own text encoder. The writer is kept on the CPU (17 GB) between jobs, comes to the GPU only to write, and goes back before the transformer runs. Its folder has no preprocessor configs, so Qwen3-VL-8B-Instruct's two are copied in beside it, so it loads alone as Fibo 2.0's AusBoss writer too.
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

**FLUX 3 Image (`flux3`)**

- **Model.** FLUX 3 Image Klein (October 2026, FLUX Kommunity License): a 5.6B flow transformer with Qwen3-VL-4B-Instruct as its text encoder and the FLUX.2 VAE. diffusers runs it as a modular pipeline (`Flux3AutoBlocks`, `Flux3ModularPipeline`), which only diffusers `main` has (§11).
- **Checkpoint.** Klein Base, the undistilled checkpoint (50 steps, CFG 3.5): `black-forest-labs/flux-3-image-klein-base` under `models/flux3/flux-3-image-klein-base/`, with only `modular_model_index.json`, `transformer/`, `text_encoder/` and `vae/` (about 20 GB, bf16). The root `model.safetensors`, `ae.safetensors` and `variants/` hold BFL's own format of the same weights and aren't copied. The index makes the folder one asset by its `modular_model_index.json`. That file names its components by Hub repo, so the runner loads each one from its subfolder into a pipeline built from the blocks.
- **Prompt upsampler.** BFL says the model needs a full scene caption and not a short prompt. `black-forest-labs/flux-3-image-klein-pup` (a 9B Qwen3.5 VLM, 18.8 GB, Apache 2.0) writes one from the prompt, and for an edit from its images too. It lives in `upsamplers/flux3/flux-3-image-klein-pup/`, an asset of the new `upsampler` kind (a `config.json` folder). The descriptor adds it to the spec as `upsampler`, so a missing upsampler is refused at submit, and it's prefetched like any other asset. The runner leaves out diffusers' own upsampler block and calls its `upsample_prompt` and `prepare_references` itself (BFL's system prompts, greedy decoding, up to 3584 tokens), so it keeps the caption. The caption is the same for every seed, so it's written once per job (progress phase `upsample`). Each output carries it as `caption`, which goes into the result row, the saved config (§6.4) and the viewer's wall label. *Upsample prompt* off sends the prompt as written, with no upsampler asset and no caption.
- **Variant `klein-base`**, modes `t2i` and `edit`, minimum GPU L4. With the upsampler, the models come to about 39 GB. If they fit under the usual offload share, they all stay on the GPU (an 80 GB A100 or an H100). Otherwise they're chained through accelerate's offload hooks in the order they run (upsampler, text encoder, transformer, VAE), so each comes onto the GPU when it runs and sends the one before it back. The modular pipeline has no `enable_model_cpu_offload`, so the runner keeps these hooks itself, and takes them off while it loads a LoRA.
- **Edit.** The source is image 1 and `inputs.refs` adds up to 9 more (the model reads 10). The pipeline keeps each condition image at its own size, so the runner scales any above 1 megapixel down to it (`Variant.ref_max_pixels`). Images with alpha are flattened onto white.
- **Parameters:** prompt, negative prompt, width and height (multiples of 16, up to 4 MP, default 1024²), steps (default 50), CFG (default 3.5, as `ClassifierFreeGuidance`), *Upsample prompt* (on by default), *Uncensored* (off) and seed. There's no step callback, so the runner wraps the scheduler's `step` to report progress and to cancel.
- **LoRAs:** single files, in `loras/flux3/`, loaded by `Flux3LoraLoaderMixin`, which converts BFL-format keys.
- **Uncensored** (under More settings, off) loads Fibo 2.0's Qwen3-VL 4B Heretic (`text_encoders/fibo2/Qwen3-VL-4B-Heretic/`, §4.3 Fibo 2.0) as the text encoder. The spec names it as its `text_encoder`, which `spec_assets` checks as a `fibo2` asset. ComfyUI uses one encoder file for both models, and the folder keeps its own tokenizer. A job that changes it swaps only that model with `update_components`: the transformer stays loaded, and so do its LoRAs, which `Flux3LoraLoaderMixin` loads only into the transformer. BFL's upsampler (Qwen3.5 9B) has no Heretic and is unchanged.
- **Not in v1:** the distilled Klein checkpoint (16 steps, no CFG; a second variant with `Flux3DistilledAutoBlocks`), layout boxes (the upsampler's `*_bbox` modes), the fp8 and NVFP4 transformers, and `inpaint`.

**MOSS-Image (`moss`)**

- **Model.** MOSS-Image (OpenMOSS, Apache 2.0): a 5B DiT with Qwen3.5 (`Qwen3_5Model`) as its text encoder and the FLUX.2 VAE. It's a diffusers custom pipeline: the folder's `pipeline_moss_image.py` (copied into `transformer/` and `scheduler/` too) defines `MOSSImagePipeline`, `MOSSImageTransformer` and `MOSSImageScheduler`. The runner loads it with `custom_pipeline` pointing at that file and `trust_remote_code`. The text encoder and transformer load in bf16 and the VAE in fp32.
- **Checkpoints.** `OpenMOSS-Team/MOSS-Image-Base`, `OpenMOSS-Team/MOSS-Image-SFT` (fine-tuned from Base) and `lodestones/Kromoss` (a full fine-tune of Base at 1024 px that fixes its blurriness) share the architecture and recipe. Each is a diffusers folder of about 27 GB, under `models/moss/MOSS-Image-Base/`, `MOSS-Image-SFT/` and `Kromoss/`, without the repos' `assets/`. Only the transformer differs: Kromoss's text encoder, tokenizer, VAE, configs and pipeline file are byte-identical to SFT's (the same hashes), so on Drive they were copied from the SFT folder. Its repo's ComfyUI single files (`diffusion_models/`, `text_encoders/`, `vae/flux2-vae_small.safetensors`) are left out. The index makes each folder one asset by its `model_index.json`.
- **Variants `base`, `sft` and `kromoss`**, mode `t2i` only, minimum GPU L4. Each variant's `model_dir` is its folder. `place` keeps the pipeline on an 80 GB A100 or an H100, and offloads it with `enable_model_cpu_offload` (text encoder, transformer, VAE) on an L4 or a 40 GB A100. Switching variants reloads.
- **Parameters:**
  - Prompt and negative prompt (empty by default; the description gives OpenMOSS's suggested tags).
  - Width and height: multiples of 16, up to 4 MP, default 1024². The presets include the 1456² square OpenMOSS shows.
  - Steps (default 50) and CFG (default 3).
  - Under More settings: *Shift* (default 5, set by building the scheduler from its config with that shift for each job) and *Skip layer*. The skip layer is the block the negative branch skips (`uncond_skip_layers`): 22 by default, 24, 26 or off.
  - Seed.
- **LoRAs.** Up to 4 single kohya files under `loras/moss/` (`diffusion_model.<module>.lora_down/lora_up/alpha`). The custom transformer has no LoRA loader, so the runner applies each one unmerged, as forward hooks on the linears it names. Its matrices stay in fp32 on the GPU, and each hook adds `weight · alpha/rank · (x @ downᵀ) @ upᵀ` to the module's output. Kromoss's README warns against merging: the delta is close to bf16's rounding step. A module that's missing or the wrong shape fails the job (diffusers would skip it with only a log line). The hooks read the weight at each call, so changing a weight reloads nothing. Only new files are loaded, and dropped ones are unhooked. Kromoss ships `8-steps-dmd-lora.safetensors`, an 8-step turbo (DMD distillation of Base, rank 128, 1.2 GB fp32). It runs with 8 steps and CFG 1, and at CFG 1 the pipeline runs no negative branch. The CFG description says so.
- **Progress.** The pipeline takes no step callback, so the runner wraps the scheduler's `step` (`runtime.scheduler_progress`, shared with FLUX 3) to report progress and to cancel.
- **Not in v1:** layout JSON prompts and SLG on the positive branch (`slg_layers`).

**Fibo 2.0 (`fibo2`)**

- **Model.** Fibo 2.0 (Bria, October 2026, CC BY-NC 4.0): an 8B single-stream flow DiT (44 blocks) with Qwen3-VL-4B-Instruct as its text encoder and the FLUX.2 VAE. A Perceiver condenses the text encoder's hidden states into gist tokens. One model generates and edits: with images, `BriaFibo2Pipeline` (diffusers `main`, #14991, §11) edits them.
- **Checkpoint.** Dev, the full undistilled model (50 steps, CFG 5): the gated diffusers folder `briaai/Fibo-2.0-dev` under `models/fibo2/Fibo-2.0-dev/`, with `model_index.json`, `processor/`, `scheduler/`, `text_encoder/`, `transformer/`, `vae/` and `fonts/` (about 25.5 GB, bf16), without `assets/` or `fine_tuning/`. The pipeline draws each text a caption asks for (`text_render`) as a small image for the text encoder, in the repo's fonts (Aileron, then DejaVu Sans), which the runner passes as `fonts`. Comfy-Org's repackaging (`Comfy-Org/Fibo-2.0`) is in ComfyUI's key layout, which diffusers can't load. Edit Turbo, the 4-step distilled editing checkpoint, isn't used.
- **Prompt-to-JSON VLMs.** Fibo 2.0 is trained on structured JSON captions (short description, objects, background, lighting, aesthetics, photographic characteristics, style, text) and works poorly with free text, although a short plain description and a plain edit instruction are trained forms too. Bria's own prompt writer, `bria-vgl`, runs on Gemini, so Degas uses Bria's local VLMs instead, both Qwen3-VL-4B fine-tunes (8.9 GB, CC BY-NC 4.0) and `upsampler` assets: `briaai/FIBO-vlm` in `upsamplers/fibo2/FIBO-vlm/` writes the caption of a new image (`<generate>`), or of one inspired by a source image (`<inspire>`), and `briaai/FIBO-edit-vlm` in `upsamplers/fibo2/FIBO-edit-vlm/` sees the source and writes an `edit_instruction` with the result's caption (`<edit>`). The runner ports the requests and the answer's clean-up from Bria's modular blocks (torch-free in `fibo2_prompt.py`): Bria's sampling (top-p 0.9, temperature 0.2, or 0.4 for edits), seeded with the job's first seed and run once per job (progress phase `upsample`). The JSON is kept as the result's caption, as FLUX 3's is. FIBO-edit-vlm reads one image, so an edit with references sends the instruction as written, with no VLM. *Upsample prompt* off sends the prompt as written.
- **AusBoss style.** *Upsample style* (under More settings; `t2i` and `edit`) picks the caption writer: `bria` (the default) or `ausboss`, after AusBoss's ComfyUI workflow "FIBO 2.0 Generate + Edit | 4-Step Turbo, Auto Prompt" (Civitai 3013987, v1.0). There a plain Qwen3-VL-8B-Instruct, Qwen-Image 2.1's text encoder, gets a hand-written prompt as the user turn of its default template, with no system prompt. The prompt lists Bria's caption keys (plus `edit_instruction` for an edit) and its own rules: every person an adult, no invented text, a photograph unless asked otherwise. The user's words go in place of `{user}`, and an edit's picture is shown first. Both prompts are verbatim in `fibo2_prompt_ausboss.py`, with the workflow's sampling (temperature 0.2, no top-k, top-p or repetition penalty) and its five regexes that tidy the answer into JSON (code fences, text before the first brace, a missing last brace, trailing commas) before `parse_caption`. The workflow's fixed seed isn't kept: the job's first seed is used, as for Bria's VLMs. The writer is Qwen-Image 2.1's model folder (`models/qwen21/Qwen-Image-2.1`, its `text_encoder/` and `processor/`): the spec names it as its `upsampler`, which `spec_assets` checks as a `qwen21` model. The worker cache copies whole folders, so a fresh VM's first `ausboss` job copies all 36 GB, which Qwen-Image jobs then share. An edit with references still sends the prompt as written, and `i2i` keeps Bria's VLMs (AusBoss has no Inspire).
- **Uncensored** (under More settings, every mode) loads community *Heretic* abliterations of Qwen3-VL, whose refusal direction is ablated. `DreamFast/Qwen3-VL-4b-Heretic` (Heretic trial t122 of Qwen3-VL-4B-Instruct, KL 0.028, Apache-2.0) in `text_encoders/fibo2/Qwen3-VL-4B-Heretic/` (a `text_encoder` asset: `config.json`, `generation_config.json` and the two shards, 8.9 GB) takes the place of the pipeline's text encoder, the stock Qwen3-VL-4B-Instruct (ComfyUI shares one file for it between FIBO and FLUX 3 Klein). The pipeline keeps its own `processor/`. A job that changes the text encoder swaps only that model, without reloading the transformer. With the `ausboss` style, Qwen3-VL 8B Heretic writes the caption instead (`upsamplers/qwen21/Qwen3-VL-8B-Heretic/`, below). Bria's VLMs have no Heretic, so with the `bria` style only the encoder changes. The workflow's ComfyUI files (`int8_convrot`, `fp8`) don't load in diffusers, so these are the bf16 transformers folders.
- **Variant `dev`**, modes `t2i`, `i2i` and `edit`, minimum GPU L4. With a VLM, the models come to about 35 GB. They stay on the GPU if they fit under the offload share, and are otherwise chained through accelerate's offload hooks (VLM, text encoder, transformer, VAE), as FLUX 3's are. One VLM is loaded at a time, and a job that needs the other swaps it.
- **Edit.** The source is image 1 and `inputs.refs` adds up to 4 more (the pipeline reads 5), named `<image_1>`, `<image_2>` and so on in the instruction. The pipeline crops each to its trained size of about 1 megapixel. The runner first scales any larger one down to 1 megapixel (`Variant.ref_max_pixels`), and flattens alpha onto white.
- **From image (`i2i`)** is Bria's *Inspire*: a VLM sees the source and writes the caption of a new image drawn from it, and the pipeline renders that caption as text-to-image, without the image (so with the text-to-image negative). The prompt is optional (the schema drops it from `required` and `minLength`, and Create allows Generate without one) and asks for changes ("the same scene, the whale is blue"). With no prompt, FIBO-vlm writes the caption (`[image, "<inspire>"]`). FIBO-vlm reads no text in an Inspire, and none in a `<refine>` either: on the GPU, a prompt after `<inspire>` was ignored, and `<refine>` gave back its input caption unchanged (as text) or redescribed the image (with it). So with a prompt, FIBO-edit-vlm's `<edit>` request on the source writes the caption of the changed scene, its `edit_instruction` is dropped (`inspired_caption`), and the job loads FIBO-edit-vlm as its upsampler. The card's prompted Inspire is `bria-vgl`'s, on Gemini. *Upsample prompt* isn't offered, as the VLM is the only way the source reaches the model. If the VLM writes no JSON, the job fails rather than drop the image.
- **Parameters:** prompt, width and height (multiples of 16, up to 4 MP, default 1024²; the presets include Bria's 1248×832 and 1424×800), steps (default 50), CFG (default 5), *Upsample prompt* (on), *Upsample style* (`bria`), *Uncensored* (off) and seed. There's no negative prompt: text-to-image uses Bria's negative caption (soft, featureless, artifact-laden), and an edit the pipeline's default, the empty caption. At CFG 1 neither runs. Progress and cancelling use the pipeline's step callback.
- **No LoRAs.** The pipeline has no LoRA loader, so a spec with LoRAs is refused.
- **Not in v1:** masked edits (`inpaint`: the pipeline takes a `mask` and greys the region out), and Edit Turbo.

**Krea 2 (`krea2`)**

- **Model.** Krea 2 (Krea AI, June 2026, Krea 2 Community License): a 12.9B single-stream MMDiT with Qwen3-VL-4B as its text encoder (twelve tapped layers, fused in the transformer) and the Qwen-Image VAE. diffusers' `Krea2Pipeline` (#14045), which the pin already has. It does text-to-image only: Krea ships no edit model.
- **Checkpoints.** The gated diffusers folders `krea/Krea-2-Raw` and `krea/Krea-2-Turbo` under `models/krea2/Krea-2-Raw/` and `models/krea2/Krea-2-Turbo/` (about 36 GB each: transformer 26.3 GB, text encoder 8.9 GB, VAE 0.5 GB, bf16), without the root `raw.safetensors` / `turbo.safetensors` (the same transformer in Krea's own format) or `images/`. Turbo's `model_index.json` sets `is_distilled`, which gives it the fixed shift `mu = 1.15`. The repos ship only a tokenizer, so each folder has a `processor/` with the tokenizer and Qwen3-VL-4B-Instruct's `processor_config.json`, for the edit's image-grounded encode.
- **Variants `raw` and `turbo`**, modes `t2i` and `edit`, minimum GPU A100: the transformer alone overflows an L4. The usual offload rule keeps them resident on an 80 GB A100 or an H100 and offloads on a 40 GB A100. Switching variants reloads.
- **Parameters:** prompt, width and height (multiples of 16; Raw up to 1536², as Krea runs it up to 1K, and Turbo up to 2048²), steps and seed. Raw adds a negative prompt and CFG. Krea's `guidance_scale` is `cond + s · (cond − uncond)`, so the form's CFG is the usual one (1 = off) and the runner sends CFG − 1: Raw defaults to 28 steps and CFG 5.5 (diffusers' 4.5). Turbo runs 8 steps with no CFG and no negative prompt.
- **Edit.** Two community LoRAs by conradlocke (`conradlocke/krea2-identity-edit`, V2, trained on Raw; free for personal use, and a licence from the author for commercial use), in `loras/krea2/`: *General edit* (`krea2_general_edit_v2`, instructions: remove, recolour, relight, restyle) and *Identity* (`krea2_identity_edit_v2`, the person from a photo in a new scene, pose or outfit). *Edit LoRA* picks one, and the descriptor adds it to the spec as `edit_lora` (refused at submit if it isn't in Drive). A LoRA row naming it sets its weight instead of the default 1. *General edit* reads the source alone. *Identity* reads the source, and optionally a second image, the person, with the source as the scene (`max_refs` 1). Raw edits default to 20 steps and CFG 3, conradlocke's settings for removals; Turbo's 8 steps suit most edits.
- **How an edit runs.** The runner ports what comfyui-krea2edit (Apache-2.0, 2.x) does, which mirrors the trainer's `predict_velocity_edit`. The torch-free geometry and prompt are in `degas_worker/families/krea2_edit.py`.
  - *Appearance.* Each image is VAE-encoded, normalized as the pipeline's latents are, and packed. Its clean tokens go before the noisy target's on every transformer call (`prepend_refs`, which replaces `forward` on the instance as `limit_to_areas` does), told apart only by the RoPE frame axis: images 1, 2, the target 0. All tokens share the step's timestep modulation, and only the target's come back.
  - *Fit.* An image is fitted inside the output at its own aspect ratio, each side floored to 16. Within 8% of the output's ratio, it's centre-cropped to fill the output exactly instead, since a margin of a token or two fills with repeated edges. Its positions are centred on the target's grid at fractional offsets (*aligned*, how the LoRAs were trained). With Identity's two images the person is placed *beside*, to the right of the target, so the scene's layout holds while the person is restaged.
  - *Boost.* The target's attention to an image's tokens is raised by `log(boost)` in the logits: 4 for General edit, 2 for Identity's person and 1 for its scene. The boost goes into each block's attention mask, which becomes an additive float mask.
  - *Grounded encode.* The prompt is encoded by Qwen3-VL together with the images (each at most 768 px on the long side), with Krea's system prompt and the vision tokens before the prompt in the user turn, unpadded, and passed as `prompt_embeds`. With CFG, the negative is the empty prompt with the same images, which is what the LoRAs were trained with as the unconditional. Both are padded to one length afterwards, since the pipeline builds one set of position ids for both.
- **LoRAs:** single files, in `loras/krea2/`, loaded by `Krea2LoraLoaderMixin`, which converts Krea's and ai-toolkit's key names. Civitai's `Krea 2` LoRAs are offered to both variants (Krea says LoRAs trained on Raw work on Turbo), and `Krea 2 Turbo` ones to Turbo.
- **Not in v1:** the edit's knobs (boost, grounding size, placement), the reference masks (`ref_boost_mask`), Ostris's appended-reference LoRAs (`Krea2OstrisEdit`), the fp8 transformer on an L4, and `inpaint`.

**LTX-2 (`ltx2`)**

- **Model.** Lightricks' LTX-2.3 (March 2026) and LTX-2.5 (August 2026): a 22B audio-video DiT that makes a clip and its sound together. LTX-2.3's text encoder is Gemma 3 12B and LTX-2.5's is Gemma 4 12B. Both load in diffusers as `LTX2Pipeline`. That needs diffusers `main` and transformers 5.10.1 or later (for Gemma 4), which the bootstrap installs when the image's are older (§11).
- **Variants**, each a diffusers folder under `models/ltx2/<variant>/`, all minimum GPU A100 with the usual offload rule:
  - `ltx25`: `Lightricks/LTX-2.5-Diffusers` without `transformer_full/` (about 72 GB). Its default transformer is distilled.
  - `ltx23-distilled`: `diffusers/LTX-2.3-Distilled-Diffusers`.
  - `ltx23`: `diffusers/LTX-2.3-Diffusers`, the full model, at 30 steps with CFG 3, audio CFG 7 and STG 1 (spatio-temporal guidance).
- **Distilled variants** run diffusers' fixed 8-sigma schedule (`DISTILLED_SIGMA_VALUES`), with CFG, STG and modality guidance all off, so each step is one forward pass. Their form has no *Steps* or *CFG*.
- **Upscale 2×** (distilled variants only). The clip is made at half the size, `LTX2LatentUpsamplePipeline` doubles its latents, and 3 sigmas (`STAGE_2_DISTILLED_SIGMA_VALUES`) refine them at full size, with the sound's latents carried over. Sizes then step by 64 rather than 32. The upsampler is read from the model folder's `latent_upsampler/`. If that folder is missing, the job fails with a message saying to copy it in.
- **Modes:** `t2v`; `i2v` (`LTX2ImageToVideoPipeline`); and `flf2v`, first and last frame (`LTX2ConditionPipeline` with the source at frame 0 and `inputs.end` at the last frame). An upscaled `i2v` or `flf2v` passes its images to both passes. The server fits the last frame to the output size, as it does the source.
- **Parameters:** prompt, negative prompt, width and height (multiples of 32, up to 1536×1024, default 960×544), frames (8k + 1, 17–257, default 121), frame rate (default 24), seed; Upscale on the distilled variants; steps, CFG, audio CFG and STG on `ltx23`.
- **LoRAs:** single files, in `loras/ltx2/`, loaded by `LTX2LoraLoaderMixin`. Civitai's `LTXV 2.3` LoRAs are tagged for both 2.3 variants and `LTXV 2.5` for `ltx25`. LTX-2.0 (`LTXV2`, 19B) LoRAs aren't imported.
- **Output:** H.264 as for Wan, plus the vocoder's sound as AAC in the same MP4. The viewer starts it muted (iOS autoplays only muted video), and its controls unmute it. Extension and stitching keep the sound (§6.4).
- **Not in v1:** LTX-2.5's full transformer (`transformer_full/`, 30 steps with CFG), its prompt enhancer and duration head, IC-LoRAs, and upscaling on `ltx23` (stage 2 needs the distilled LoRA).

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
    wan22/         <variant>/…   (diffusers-format directories; Wan 2.1 I2V in wan21-i2v-14b/)
    qwen21/        Qwen-Image-2.1/ (the official diffusers folder)
    flux1/         *.safetensors (single-file transformers, e.g. flux1-dev-fp8)
    klein/         FLUX.2-klein-9B/ (the official diffusers folder)
    flux3/         flux-3-image-klein-base/ (the modular diffusers folder, without variants/ or BFL's root files)
    moss/          MOSS-Image-Base/, MOSS-Image-SFT/, Kromoss/ (the repos' diffusers folders, without assets/)
    fibo2/         Fibo-2.0-dev/ (the diffusers folder with fonts/, without assets/ or fine_tuning/)
    krea2/         Krea-2-Raw/, Krea-2-Turbo/ (the diffusers folders, without the root *.safetensors or images/; processor/ added)
    ltx2/          <variant>/…   (diffusers folders: ltx25/, ltx23-distilled/, ltx23/; latent_upsampler/ inside for Upscale)
  loras/
    sdxl/          *.safetensors (+ *.yaml, preview *.jpg/png)
    wan22/         *.safetensors; A14B pairs named *_high_noise.safetensors / *_low_noise.safetensors or declared in sidecar
    qwen21/        *.safetensors
    flux1/         *.safetensors
    klein/         *.safetensors
    moss/          *.safetensors (kohya; Kromoss's 8-steps-dmd-lora)
    flux3/         *.safetensors
    krea2/         *.safetensors (conradlocke's krea2_general_edit_v2 and krea2_identity_edit_v2 for edits)
    ltx2/          *.safetensors (sidecar `variants` from Civitai: ltx23 + ltx23-distilled, or ltx25)
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
  upsamplers/
    flux3/         flux-3-image-klein-pup/ (BFL's prompt upsampler, a transformers folder)
    fibo2/         FIBO-vlm/, FIBO-edit-vlm/ (Bria's prompt-to-JSON VLMs, transformers folders)
    qwen21/        Qwen3-VL-8B-Heretic/ (DreamFast's transformers folder, with Qwen3-VL-8B-Instruct's
                   preprocessor configs; Qwen-Image 2.1's uncensored upsampler and Fibo's AusBoss writer)
  text_encoders/
    fibo2/         Qwen3-VL-4B-Heretic/ (DreamFast's transformers folder, weights and configs only)
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

**Civitai imports** (`degas/civitai/`; `degas civitai import <link>`, or *Import a LoRA* in the LoRA picker). The version's `baseModel` picks the family: SDXL 1.0 and its fine-tunes (Pony, Illustrious, NoobAI) → `sdxl`, Flux.1 D/S/Krea → `flux1`, Flux.2 Klein 9B → `klein`, Qwen 2/2.1 → `qwen21`, and the Wan Video 2.2 bases, and Wan 2.1's two 14B i2v bases, → `wan22` with `variants` set. Anything else is refused unless a family is given. A Wan A14B file is named `<name>_high_noise` or `_low_noise` from `high`/`low` in its file or version name (`highnoiseV2.0` counts). When the linked version has one half, the model's other versions (`GET /models/<id>`) are searched for the other: one with the same base model, a name equal apart from high/low/noise, and only that half. If exactly one matches, both halves are imported together, and the sidecar's notes name both versions. If that half is already in Drive (by SHA-256), it's left out and the new half takes its name, so they pair. Halves imported from two links one at a time also pair up. The file streams from Civitai into `rclone rcat` on the server's writable remote (`lora.rclone_remote`), with its SHA-256 checked against Civitai's and its md5 against Drive's. A mismatch deletes it. Then a sidecar (label, trigger words, weight 0.8, the base model in `notes`, `source`) and a 768 px preview from the first example image (a still, for a video) are written, and Drive is rescanned. A file whose SHA-256 is already in the index, or whose name is taken, is refused unless forced. `degas civitai backfill` writes sidecars for LoRAs already in Drive, looked up by SHA-256.

**Hugging Face imports** (`degas/civitai/huggingface.py`, `plan_hf_import`; the same command, endpoints and sheet take a `huggingface.co` link). The repo at the link's revision comes from `GET /api/models/<repo>/revision/<rev>?blobs=true`: its commit, `cardData` and each LFS file's SHA-256 and size. A file link imports that file; a repo or `tree/` link must hold one `.safetensors` file in that folder (or one Wan pair). Files download from `resolve/<commit>/…`, so they can't change under the planned hash, and stream into Drive with the same checks. There's no reliable base-model field, so `NAMED_BASES` patterns are tried on the file's name, then the card's `base_model`, then the repo's name, and set the family and `variants` as for Civitai (`LTX-2.3` → `ltx2` with `ltx23` + `ltx23-distilled`). When nothing matches, the request's `hint` (the family Create is on) is used with a warning; without one, the plan is refused until a family is given. For a Wan A14B family, a file link also takes the file in its folder named the same apart from high/low. Trigger words: the card's `instance_prompt`, else a `Triggers:` / `Trigger words:` line in the README. Preview: the repo's first image, else the first frame (ffmpeg) of its smallest example video under 32 MB. The sidecar's `notes` read `<base> LoRA from Hugging Face (<repo>)`, and `source` is the link's page. The token (`huggingface.token_file`) is sent only to huggingface.co; gated repos need it.

**CivArchive imports** (`degas/civitai/civarchive.py`; the same command, endpoints and sheet take a `civarchive.com` / `civitaiarchive.com` model link). For LoRAs deleted from Civitai. `GET /api/models/<id>?modelVersionId=<v>` gives the model and one version, much like Civitai's API but with each file's mirrors. `civitai_version` converts it to Civitai's shape, so `plan_import` plans it as usual: family from `baseModel`, trigger words from `trigger`, preview from the first example's still (`image_url`). Each planned file keeps its live mirrors (no `deletedAt`, not paid or gated): Civitai's first, then Hugging Face's, in CivArchive's order. Other sites' links aren't direct downloads, so they're left out. A file with no live mirror is refused. The import tries the mirrors in turn, each with the token for its site; one that fails, has the wrong size or fails the SHA-256 check is skipped for the next. A Wan A14B half finds its pair among the model's other versions as on Civitai, fetching each version named the same apart from high/low. The sidecar's `notes` say *from CivArchive*, and `source` is the CivArchive page.

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

For a Wan 2.2 A14B LoRA, the entry has the form `{ "high": {path, weight}, "low": {path, weight} }`. If an input image came from a URL, the config also records `inputs.origins: { "<sha256>": "<url>" }` for provenance. Replay always uses the stored blob, so the URL is never fetched again. Video extension records the parent video in `inputs.extends` (the parent's blob hash) and the extracted frame in `inputs.source`. Any family whose descriptor gives an `extend_variant` can be extended (Wan 2.2 and LTX-2; `/families` says `extendable`). A first-and-last-frame video (`flf2v`) records its last frame in `inputs.end`. A Qwen-Image 2.1 edit records its extra reference images, in order, in `inputs.refs` (a list of blob hashes; the source is image 1). SDXL image prompts are a top-level `image_prompts` list, `[{adapter: {path, size}, images: [sha…], purpose, weight, start, end, mask?}]`; each picture is fitted to a square (cropped, or letterboxed with `fit: pad`) and its area to the output size, both recorded in `inputs.transforms`.

**Video extension output.** An extend job produces the new continuation clip as its result. When it completes, the server uses `ffmpeg` to create a second result: the stitched chain, which is the parent (itself possibly stitched) followed by the continuation, with the duplicated boundary frame dropped. If either clip has sound, the chain does too: a silent clip contributes silence, and the continuation's sound loses the length of the dropped frame. Both results can be saved. The stitched result's config records the ordered list of segment configs, so every segment of the chain can be reproduced.

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
| POST | `/civitai/plan` `{url, hint?}` | What importing a Civitai, CivArchive or Hugging Face LoRA would do: family, Drive paths, sidecar (§5); `hint` is the family for an HF LoRA nothing names the base of |
| POST | `/civitai/import` `{url, hint?}` | Start the import (202); `import` SSE events follow it; 409 while one runs |
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

**MCP server.** `/mcp` (outside `/api`) is a stateless streamable-HTTP MCP server (`mcp` 2.x's `MCPServer`, `degas/mcp_tools.py`) for agents, with two tools on Qwen-Image 2.1. `create_image` is `t2i` without images; with reference images it is an `edit` whose image 1 is the first reference, padded (not cropped) to the chosen aspect, and the rest are `inputs.refs`. `edit_image` takes the image to edit as image 1 and up to 9 references; its output keeps image 1's shape at the resolution's pixel count unless an aspect is given. Sizes are Qwen's presets: seven aspects at 2K or 1K. The resolution defaults to 2K without images and 1K with them, since a 2K edit with references runs out of memory on the L4 the tools start (live test below). An image is a `sha256:` ref, an http(s) link or a `data:` URI (imported as `/blobs/from-url` does). Each call goes through the same submission as `POST /jobs` (`submission.submit`), starts an L4 session if none is active, reports progress notifications while it waits, and answers with a ≤1024 px JPEG preview plus `{sha256, url, width, height, seed, job_id, prompt_used?}`; results are ordinary results, so they show in Results and expire unless kept. If the session ends before the job runs, the job is cancelled and the call fails; after 45 minutes the call gives up and leaves the job queued. The tool descriptions carry the prompting guide (images are referred to by number). DNS-rebinding protection is off because the Host header is the tailnet name; access is the tailnet's, as for `/api`.

Live test (2026-10-24, through the tailnet URL, on an L4 without high memory, which the tools started): `create_image` without images at 2K took 640 s from a stopped session (start, model copy, load and 40 steps), and 94 s for the next at 1K (about 1.5 s a step). With images, a 2K `create_image` with two references ran out of GPU memory at the first step (22 GB L4). At 1K, `create_image` with two references (1248×832) and `edit_image` with one (1024², keeping the source's shape) each worked in about 4 minutes, following the prompts' image numbers. Progress notifications arrived throughout.

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
- **Web Push:** requires the app to be installed to the home screen (iOS 16.4+). The server uses VAPID keys and `pywebpush`. Notifications are sent when a job completes or fails, when a new session or a reset worker is ready, and when an idle shutdown is 2 minutes away.

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
                           queries, row types), colab/ (session mgr, worker client,
                           bundle; CLI and ssh via colabkit), dispatcher.py, drive/ (OAuth,
                           index, catalog),
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
5. **Queue UX and push.** ✅ Built, not yet tested on an installed iPhone. Batch generation and seed modes, cancel, reorder, the PWA manifest and service worker, and Web Push. A batch picks *Random seeds* or *Count up* from the seed field. `PATCH /jobs/{id}` `{position}` reorders the queued jobs among their existing positions (new jobs still go last) and re-targets prefetch; `POST /jobs/{id}/restore` undoes cancelling a job that never started. The service worker is hand-written (`web/public/sw.js`) rather than `vite-plugin-pwa`: it caches the hashed build assets, falls back to the cached shell offline, never caches `/api`, and shows pushes. `degas/push.py` generates the VAPID key pair on first use (`push.key_file`, default `<data_dir>/vapid_private.pem`) and sends with `pywebpush` off the event loop, dropping subscriptions the push service reports gone (404/410). Notifications go out when a job finishes or fails (not when it's cancelled), when a newly started session or a reset worker becomes ready (*L4 is ready.*, opening the session sheet; not on a reattach or a silent recovery), and once per idle deadline when an idle session is 2 minutes from stopping; tapping one opens `/?tab=results` or `/?sheet=session`.
6. **i2i, inpaint and outpaint.** ✅ Done. SDXL gains `i2i`, `inpaint` and `outpaint` on regular checkpoints, and an `inpaint` variant for inpainting checkpoints in `models/sdxl/inpaint/` (§4.3). Masks are stored by `POST /blobs/{sha}/mask`, follow their source through re-crops (`POST /blobs/{mask}/remap`) and are fitted with it at submit (§6.5). Outpaint places the source on the Size canvas (`inputs.place`). SAM 3 selection moved up from Phase 7: `POST /api/preprocess` stages the image and asks the worker's new `/preprocess` route, which loads `Sam3TrackerModel` for taps or `Sam3Model` for a description (one at a time) from `preprocessors/sam3/`. Drive indexes a preprocessor folder with a `config.json` as one asset. The mask editor (brush, erase, select, undo, invert, blur preview) and the outpaint placement are in the UI. Live test on a T4 (2026-09-28, a regular checkpoint): inpaint at 1024², 30 steps, strength 0.85, with the mask made by SAM 3 selection and SAM 3 resident next to SDXL; four jobs took 29–34 s each. The first attempts ran out of memory: diffusers 0.40's `from_pipe` defaults to float32 and casts the shared modules in place, which doubled SDXL to 13.1 GiB. SDXL now passes `torch_dtype=torch.float16` (6.6 GiB loaded, 9.7 GiB peak for a 1024² inpaint), and Wan builds its i2v pipeline from the components instead of using `from_pipe`. Still untested live: i2i, outpaint, the `inpaint` variant, and Wan i2v after that change.
7. **Control.** ✅ Built, not yet tested on a live GPU. Preprocessors (depth, pose, canny), SDXL ControlNet units, and regional ControlNet. A unit is `{controlnet, image, fit, scale, start, end, mask?, preprocessor?}` (§6.4); at submit each control image is fitted to the output size by its own `fit`, and its area mask with it, both recorded in `inputs.transforms`. The Drive index takes a folder under `controlnets/<family>/` with a `config.json` as one asset, and every folder directly under `preprocessors/` as one. A ControlNet's sidecar can say `control: depth|pose|canny`, which the UI uses to pick a model for a trace and to warn about a mismatch. Depth Anything V2 and DWPose run on the worker (DWPose's ONNX pre- and post-processing are adapted from controlnet_aux); canny is OpenCV. `degas_worker/deps.py` installs a missing package the first time it's needed. Untested live: all of it, in particular `from_single_file` for SDXL ControlNets, the ControlNet pipelines built with `from_pipe`, the residual masking under CPU offload, and onnxruntime's CUDA provider on Colab.

8. **Qwen-Image 2.1.** 🚧 In progress. The `qwen21` family (§4.3): `t2i`, `edit` with up to 10 ordered images (the source plus `inputs.refs`), and `inpaint` as an edit pasted back through the mask. The server validates the references, checks they're still stored, and keeps them (and a cropped reference's original) with the job and any kept item. The runner loads `QwenImage21Pipeline` from the diffusers folder in Drive and offloads to the CPU when the weights don't fit. It maps CFG to `true_cfg_scale` and the schedule to the scheduler's settings, and runs the ported DeGrid filter. The bootstrap installs a pinned diffusers commit with the pipeline when the image lacks it. In the UI, *Edit* is a mode chip, and an *Images* row under Source adds, crops, orders and removes references. A reference's crop is free and keeps its own size. Live test to do, on an L4 (high memory) and an A100: cold copy, load, and per-image time at 1K and 2K; a 40-step Default against 30-step Beta comparison to settle the default schedule; the grid filter on skin and hair; and SDXL and Wan on the pinned diffusers. LoRA live test on an A100 (2026-10-01, diffusers 0.41.0.dev0, offloaded): two Civitai LoRAs in ComfyUI's `diffusion_model.…lora_A/B` format loaded through the pipeline's mixin, Lenovo UltraReal in `t2i` and Anything2RealCharacters in `edit`. In each mode a fixed seed gave a different image at weight 0.8 and at 1.2 (the adapter was reused, not reloaded), and removing the LoRA gave back the no-LoRA image exactly. About 48 s an image for `t2i` at 1024² and 50 s for an 832×1216 edit, 40 steps, CFG 1; a LoRA's first use added 10–20 s. Lenovo UltraReal at 1.2 is overcooked, which is the LoRA and not the loader.

    Added later: *Upsample prompt* (§4.3), after Kokon's Prompt+. The loaded text encoder rewrites the prompt, using Kokon's system prompts or ours. Live test on an A100 40 GB, normal memory and so offloaded (2026-10-23, diffusers 0.42.0.dev0, transformers 5.20.2, torch 2.11.0), on the real server. The five images are kept in the library under `qwen-upsample-test`.
    - `t2i`, 1024², seed 42, "a cat café in Tokyo at night, sign says "NEKO"": both styles wrote a ~130-word English paragraph (photograph, light, cats, patrons) with the sign quoted exactly, and the picture has the NEKO neon sign spelled right. With upsampling off, the same seed gave a close-up of one cat in a café and no sign. A job took 67 s with upsampling and 52 s without, so the rewrite (encoder moved to the GPU, sampled, then handed back by the transformer's hook) costs about 15 s. The first job, with the load, took 3 min.
    - A 2-image edit (Kokon style, the snowy harbour and the café picture): "Place the ginger cat from image2 onto the red and white boat in image1 … Keep … unchanged." The cat sits on the red boat and the rest of the scene is untouched. 64 s at 1376×768.
    - A 1-image edit from Italian (Degas style), "fai che sia una giornata estiva soleggiata": an English instruction to turn the snow scene into a sunny summer day that ends with "Keep everything else in the image unchanged." The boats, houses and gulls kept their places. 57 s.
    - No warnings in `worker.log`.

    Added later: *Uncensored upsampler* (§4.3), with Qwen3-VL 8B Heretic writing. Live on an A100 40 GB, normal memory (2026-10-23, the same versions), on the real server; two images under `fibo2-heretic-test`, after the Fibo jobs below.
    - `t2i` cat café (Kokon style, seed 42): a ~110-word paragraph much like the stock encoder's (photograph, cats from cream to black, the sign "NEKO" in white on black), and the sign was right. 184 s, including the switch from Fibo to Qwen-Image and the writer's load.
    - The Italian 1-image edit (Degas style): an English instruction for a sunny summer day ending "Keep the composition and the subjects unchanged", and the snow scene became a green summer harbour. 70 s.
    - The writer went back to the CPU: about 2.7 GB stayed on the GPU after the jobs. No warnings in `worker.log` beyond the old deprecation notices.
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
14. **LTX-2.** ✅ Built, not yet tested on a live GPU. The `ltx2` family (§4.3): LTX-2.5, LTX-2.3 Distilled and LTX-2.3 as variants, with `t2v`, `i2v` and `flf2v` (first and last frame), and *Upscale 2×* on the distilled variants. Video now has sound: `encode_mp4` muxes the vocoder's waveform as AAC, `media.probe` reports `audio`, and stitching carries it (silence for a silent clip). Extension is no longer Wan-only: descriptors give `extend_variant`, and the UI asks `extendable`. `flf2v` is a generic mode with an `inputs.end` image (validated, fitted, staged and kept like the source), shown as a *Last frame* row in Create. Civitai `LTXV 2.3` / `LTXV 2.5` LoRAs import to `loras/ltx2/`. The diffusers pin moved to a `main` commit with LTX-2.5, and the bootstrap upgrades transformers when it lacks Gemma 4. Live test to do, on an A100 (high memory): the cold copy, load and per-clip time for each variant at 960×544 and 121 frames; `t2v`, `i2v` and `flf2v` (that the clip ends on the last frame); Upscale on both distilled variants, including `flf2v` with its conditions in the refine pass; sound in sync, after extending and stitching too; a Civitai LTX LoRA (no unexpected keys in `worker.log`); and SDXL, Wan, Qwen, Flux and klein jobs again on the new diffusers pin and transformers.

    Live test on an A100 80 GB, high memory (2026-10-05, diffusers `c2798cc`, transformers 5.18.0): `flf2v` on both distilled variants, 768², 121 frames at 24 fps, seed 42, no upscale. The first frame was a library image and the last a 2× crop of it, so the clip is a push-in. Both kept in the library under the tag `ltx-flf2v-test`.
    - The cold copy ran far faster than estimated: about 3 min for LTX-2.5's 72 GB and 5 min for LTX-2.3 Distilled's 96 GB. A warm load took 5–30 s.
    - LTX-2.5: 8 steps in about 2.5 min, decode about 35 s, about 3.5 min a clip. LTX-2.3 Distilled: about 3 min of steps and 3.5 min a clip.
    - Both clips push in smoothly and end on the last frame, with stereo AAC at 48 kHz. LTX-2.5's sound is very quiet for this prompt (mean −54 dB); LTX-2.3's is louder (−31 dB).
    - Fixed on the way: `AutoencoderKLLTX2Audio` can't tile (diffusers raises `NotImplementedError`), so only the video VAE tiles. diffusers re-compresses conditioning images with H.264 through PyAV (`av`), which the image lacks and which diffusers checks for only when it is imported, so the bootstrap installs it. `encode_mp4`'s `-shortest` cut the last video frame, because LTX's audio is about 30 ms short; `encode_mp4` now pads the sound with silence, or cuts it, to the video's exact length (`fit_pcm`), with no `-shortest`. The two kept clips were made before that fix and have 120 frames.
    - On an A100 40 GB (2026-10-07), LTX-2.3 Distilled `i2v` with a LoRA ran out of memory at every size. diffusers' offload chain puts the VAE after the transformer, so the VAE that encoded the image stayed on the GPU through the denoise. The runner now moves it off after each encode, and after a failed job it puts every model back on the CPU (an out of memory had left the transformer on the GPU, failing the jobs after it). 512×512, 65 frames then ran in about 6 min.
    - Still to do: `t2v` and `i2v`; Upscale; LTX-2.3 (the full model); extend and stitch with sound; an LTX LoRA; and the other families on the new pin.

15. **Wan first and last frame.** ✅ Built and tested on an A100. `flf2v` on `i2v-a14b` (diffusers' `last_image`) and a new `wan21-flf2v-14b` variant (§4.3).

    Live test on an A100 80 GB, high memory (2026-10-06, diffusers 0.41.0.dev0 at the pinned commit, torch 2.11.0): 960², seed 42. The first frame was a 1024² library image and the last a 2× centre crop of it, so the clip is a push-in.
    - `wan21-flf2v-14b`, 81 frames, 30 steps: 44 minutes including the cold copy of the 90 GB folder and load, about 70 s a step. The clip pushes in smoothly. Its first and last frames match the inputs (mean difference about 5 of 255). The log notes `CLIPVisionModelWithProjection` where `CLIPVisionModel` is expected, which is harmless.
    - The same with Remade-AI's Squish LoRA (a Wan 2.1 I2V LoRA), 49 frames, 20 steps: 11.5 minutes. A hand enters and squeezes the figure, and the clip still ends on the last frame. No unexpected LoRA keys in `worker.log`.
    - `i2v-a14b` `flf2v`, 81 frames, 30 steps: 43 minutes including the cold copy of the 126 GB folder. It also ends on the last frame (mean difference about 4).
    - Regressions: `i2v` on `i2v-a14b` (49 frames, 20 steps, 12 minutes) and `t2v` on `ti2v-5b` (49 frames, 30 steps) both made sensible clips. Extending an FLF2V clip offers `wan21-i2v-14b` in `i2v` mode, with the first Wan 2.1 I2V folder in the index (480P here).

16. **FLUX 3.** ✅ Built and tested on an A100. The `flux3` family (§4.3): FLUX 3 Image Klein Base for `t2i` and `edit` with up to 10 images, with BFL's prompt upsampler in front of it. Its scene caption is kept with each result (`results.caption`, the saved config's `caption`) and shown in the viewer's wall label under the prompt. The Drive index takes `upsamplers/<family>/` folders and folders with a `modular_model_index.json`. The diffusers pin moved to `d42ebf9c` on `main`, which has FLUX 3 and its upsampler block.

    Live test on an A100 80 GB, high memory (2026-10-17, diffusers 0.41.0.dev0 at `d42ebf9c`, torch 2.11.0), on the real server. The ten FLUX 3 images are kept in the library under the tag `flux3-test`.
    - With the upsampler, everything stayed on the GPU (about 40 GB in use), so the offload chain hasn't run yet. The first job took 5.5 min, including the cold copy of both folders (39 GB) and the load. After that: 50 steps at 1024² in about 25 s (2 it/s). The upsampler adds roughly 10–20 s a job, and a two-image batch took 76 s. An edit with one image runs at 1.1 s a step (about 55 s), and with four images at 3.7 s a step (about 3 min).
    - `t2i` (a knitted lighthouse with the banner text "KNIT HARBOUR"): the caption is a full scene description of about 250 words, and the image follows it, with the text spelled right. `t2i` with *Upsample prompt* off gave a sane image and stored no caption.
    - An `edit` with no references ("the same scene at night under a knitted full moon") kept the composition and changed the lighting, sky and moon. An `edit` with 3 references at seed 42 kept the source almost unchanged, although the caption asked for the fox from the second image. With other seeds it works: the same four-image edit at seeds 7 and 8, and a two-image edit ("put the fox from the second image on the island next to the lighthouse") at 101 and 102, all put a knitted fox on the island, and the four-image ones take the snow and golden light from the other references. The way references reach the transformer (1 MP each, an `l` coordinate per reference, CFG with the references in both passes) matches BFL's reference code.
    - Step progress, the *Writing caption* phase and cancelling mid-denoise all work. The caption shows in the Results and Library viewers at phone width, and the kept items carry it in their config.
    - Regressions on the new pin: SDXL, FLUX.1 (fp8), Qwen-Image 2.1, klein, Wan 2.2 TI2V 5B (`t2v`) and LTX-2.5 (`t2v`) each ran one job.
    - Still to do: the offload path (a 40 GB A100 or an L4) and a FLUX 3 LoRA (none in Drive yet).

    Added later: *Uncensored* (§4.3), with Fibo 2.0's Qwen3-VL 4B Heretic as the text encoder. Live on an A100 40 GB, normal memory (2026-10-24, diffusers 0.42.0.dev0 at `de988b7e`, transformers 5.20.2, torch 2.11.0), on the real server; one image under `fibo2-heretic-test`.
    - FLUX 3's `text_encoder/` against the Heretic: the same 713 tensors, and the same 62 `o_proj` / `down_proj` weights differ by the same amounts as Fibo's, so the two models' encoders are the same stock Qwen3-VL-4B-Instruct.
    - The knitted lighthouse ("KNIT HARBOUR", seed 42, upsampled) with *Uncensored*: the same caption (the upsampler is unchanged) and almost the same picture, with the banner right and small changes in the knit. 311 s, with the cold copy of FLUX 3 and its upsampler and the offload chain on a 40 GB card.
    - The same job without it, next: the encoder swapped back without reloading the transformer, and the image was byte-identical to the regression image from 2026-10-21. 119 s. No missing or unexpected keys in `worker.log`.

17. **MOSS-Image.** ✅ Built and tested on an A100. The `moss` family (§4.3): MOSS-Image Base and SFT for `t2i`, a diffusers custom pipeline loaded with `trust_remote_code`. Shift and the negative branch's skipped block are under More settings. FLUX 3's scheduler-step progress moved to `runtime.scheduler_progress` so both families share it.

    Live test on an A100 40 GB, normal memory (2026-10-19, diffusers 0.41.0.dev0, transformers 5.18.0, torch 2.11.0), on the real server. The five MOSS images are kept in the library under the tag `moss-test`.
    - The 27 GB pipeline is under the offload share of a 40 GB card, so it stayed on the GPU (about 30 GB in use). The custom pipeline loaded from the Drive folder with no errors, only diffusers' `torch_dtype` deprecation warning. The first Base job took 2 min 36 s, including the cold copy and the load. The first SFT job took 1 min 37 s, including its own copy and the reload. After that, about 1 s a step at 1024² (two transformer passes a step): a two-image batch took 81 s, and a 1456² job 98 s.
    - `t2i` on Base and SFT at seed 42 (a bookshop window with a chalkboard reading "MOSS BOOKS — OPEN LATE"): both spelled the sign right, and the two models made clearly different pictures. SFT with *Skip layer* off (seeds 7 and 8) also spelled it right, with a slightly more photographic look. A short prompt on Base at 1456² with CFG 5 and shift 3 gave a clean red fox in snow.
    - Step progress and cancelling at step 10 work (the job stopped within a second).
    - Regression: a FLUX 3 `t2i` job (upsampling off) on the moved progress helper ran after MOSS and made a good knitted lighthouse. It's kept under `flux3-test`.
    - Still to do: the offload path (an L4).

    Added later: the `kromoss` variant (lodestones' fine-tune of Base) and LoRAs, applied as forward hooks, with Kromoss's 8-step turbo LoRA in `loras/moss/`.

    Live test of Kromoss and the turbo LoRA on an A100 40 GB, normal memory (2026-10-20, diffusers 0.41.0.dev0, transformers 5.20.1, torch 2.11.0), on the real server. The five images are kept under the tag `moss-test`, titled by job. All used the bookshop prompt at 1024², seed 42.
    - Kromoss at 50 steps and CFG 3 took 4 min 5 s, including the session start, the cold copy and the load. It spelled the sign right and was sharp.
    - Kromoss + turbo LoRA at 8 steps and CFG 1 took 18 s, including hooking in the 1.2 GB LoRA. The same job at weight 0.5 took 6 s with no reload, and gave a close variant. Both spelled the sign right with no loss of sharpness. No errors in the worker log, and about 32 GB in use on the GPU.
    - Kromoss again with no LoRA (39 s) gave a byte-identical PNG to the first job, so removing the hooks restores the model exactly.
    - Base + turbo LoRA (1 min 55 s, including Base's cold copy and the reload, which drops and re-hooks the LoRA) spelled the sign right, in a different composition.

18. **Fibo 2.0.** ✅ Built and tested on an A100. The `fibo2` family (§4.3): Bria's Fibo 2.0 Dev for `t2i` and `edit` with up to 5 images, with Bria's prompt-to-JSON VLMs in front of it (FIBO-vlm for `t2i`, FIBO-edit-vlm for an edit of one image), whose JSON is kept as the caption. The diffusers pin moved to `de988b7e` on `main`, which adds `BriaFibo2Pipeline`. The upsampler's missing-asset message no longer names FLUX 3's.

    Live test on an A100 40 GB, normal memory (2026-10-21, diffusers 0.42.0.dev0 at `de988b7e`, transformers 5.20.1, torch 2.11.0), on the real server. The eleven images are kept in the library under the tag `fibo2-test`, titled by job.
    - Fibo 2.0 and a VLM (about 35 GB) are over the offload share of a 40 GB card, so the hook chain ran: about 3.7 GB stays on the GPU between jobs. The first job took about 6 min from the session request, including the worker install, the cold copy of the model and FIBO-vlm (34 GB) and the load. After that, about 1 s a step at 1024² and 12 s to decode: `t2i` with *Upsample prompt* off took 80 s.
    - `t2i` (a bookshop window with a chalkboard reading "FIBO BOOKS — OPEN LATE", seed 42): FIBO-vlm wrote a full JSON caption, and the sign was spelled right in a sharp, detailed picture. With *Upsample prompt* off it was spelled right too, in a plainer scene. 1424×800 (a harbour at sunrise) took 137 s, about a minute of it the caption.
    - The VLMs write at about 7.5 tokens a second in the worker (18 in a script beside it; the offload hook accounts for only 15% of the difference), so a caption takes 1–2.5 min. A one-image edit at seed 7 took 215 s.
    - Fixed on the way: FIBO-edit-vlm names its rewording `reworded_prompt`, not `edit_instruction`, which the parser now maps. On the bookshop picture at seed 42 it looped over book titles until it ran out of 4096 tokens (9 min), and the job failed on the truncated JSON. Captions are now capped at 2048 tokens, and an answer that isn't JSON sends the prompt as written, with a warning in `worker.log`. The same edit then fell back after about 4.5 min.
    - Edits: "make it a snowy winter night, with snow piled on the window ledge and a wreath on the chalkboard" kept the window, books and sign. Through the caption (seed 7), it added falling snow and a wreath beside the readable sign. As written (the fallback), the wreath covered the sign. Neither looked like night. A two-image edit ("replace the chalkboard sign in `<image_1>` with a framed painting of the harbour scene from `<image_2>`") did exactly that and left the rest alone: 219 s, about 4 s a step with the second image's tokens.
    - Step progress, the *Writing caption* phase and cancelling work: cancelled at step 10, the job stopped in 1.1 s.
    - Regressions on the new pin, each including its cold copy: SDXL (Juggernaut, 96 s), klein `t2i` (235 s), MOSS SFT (201 s), a FLUX 3 one-image edit (181 s; one reference is now padded rather than cropped upstream, and the sign stayed intact) and a FLUX 3 upsampled `t2i` (255 s, banner spelled right). All looked right.
    - Still to do: an L4, a masked edit, and a faster VLM (the worker's token rate).

    *From image* (`i2i`, Bria's Inspire), live on an A100 40 GB (2026-10-22, same versions), on the real server; four more images under `fibo2-test`. Source: the harbour at sunrise (1424×800, seed 42).
    - No prompt: FIBO-vlm's `<inspire>` caption described the harbour, and Fibo drew a new picture of it (same red and blue boats, mist and hills, a different composition). 294 s, with the session's cold start.
    - With "the same scene in a snowstorm at night, the boats are bright red": FIBO-vlm ignored text after `<inspire>` (the caption, and so the picture, matched the unprompted one), and its `<refine>` didn't apply the instruction either (a probe beside the worker: as text it gave its input back unchanged, with the image it redescribed the source). FIBO-edit-vlm's `<edit>` caption carried the change (snowfall, night, a bright red boat in front), so a prompted Inspire now uses it: a snowy harbour, red boat in front, though more grey dusk than night, and only one boat red. 210 s.
    - The same prompt as an `edit` (FIBO-edit-vlm, with the cold copy of it) kept the source's layout and added the snow, 266 s.

    Added later: the *AusBoss* upsample style and *Uncensored* (§4.3). Live on an A100 40 GB, normal memory (2026-10-23, diffusers 0.42.0.dev0 at `de988b7e`, transformers 5.20.2, torch 2.11.0), on the real server; four images under `fibo2-heretic-test`, at seed 42.
    - Fibo's `text_encoder/` and Qwen3-VL 4B Heretic have the same 713 tensors and shapes; 62 differ, all `o_proj` and `down_proj` weights (about 2% each), and the rest are identical, so Fibo's encoder is the stock Qwen3-VL-4B-Instruct and the Heretic replaces it as is. No missing or unexpected keys in `worker.log`.
    - `t2i` bookshop, AusBoss style with *Uncensored* (both Heretics): Qwen3-VL 8B Heretic wrote a full caption in Bria's keys, adding a shopkeeper as "a young woman in her twenties" (AusBoss's rule), and the sign read "FIBO BOOKS / OPEN LATE" (without the dash). 6.5 min from the queue, with the session's cold copies of Fibo and both Heretics.
    - `edit` of the harbour, "make it snow …" (AusBoss + *Uncensored*): the caption described the source (red and blue boats, gulls, misty hills) with falling snow and a thin layer on the boats and quay, and kept the instruction as `edit_instruction`. The picture kept the layout and added snowfall and a light dusting. 4 min.
    - `i2i` of the harbour with *Uncensored* (FIBO-vlm writing, the Heretic encoding): a new composition of the same harbour. 258 s.
    - `t2i` bookshop, AusBoss style without *Uncensored* (Qwen-Image 2.1's encoder writing, Fibo's own encoder back without reloading the transformer): a caption close to the Heretic's, and the sign spelled right with its dash. 243 s, including the first copy of Qwen-Image 2.1 (36 GB).

19. **Krea 2.** ✅ Built and tested on an A100. The `krea2` family (§4.3): Krea 2 Raw and Turbo for `t2i` and `edit`. Krea ships no edit model, so an edit runs conradlocke's General or Identity Edit LoRA with comfyui-krea2edit's recipe ported to the stock pipeline: the images' clean VAE tokens put before the target's on their own RoPE frames, their attention boosted, and the prompt encoded together with the images. The descriptor adds the edit LoRA as the spec's `edit_lora`. Civitai's `Krea 2` and `Krea 2 Turbo` LoRAs import to `loras/krea2/`.

    Live test on an A100 40 GB, normal memory (2026-10-26, diffusers 0.42.0.dev0 at `de988b7e`, torch 2.11.0), on the real server. The fourteen images are kept in the library under the tag `krea2-test`, titled by job.
    - Each variant (36 GB) is over a 40 GB card's resident share, so it ran offloaded: under 1 GB stays on the GPU between jobs. The first Raw job finished 5 min after the session request: about 2 min for the cold copy, 15 s to load, then about 2 s a step at 1024² (28 steps, CFG 5.5) and 20 s to decode. Later Raw `t2i` jobs took 75–100 s.
    - Turbo, 8 steps: 40 s at 1024². At 2048² it fit on the 40 GB card, at about 15 s a step: 5.5 min with Turbo's cold copy (3 min). A full-size crop of its sky is smooth; the bands seen in a downscaled preview were the preview's.
    - Lenovo UltraReal (Civitai, `Krea 2`) loaded on both variants with no unexpected keys and changed the picture at the same seed (warmer, more film-like light).
    - Fixed on the way: conradlocke's edit LoRAs are in Krea's own module names with an alpha per module, and diffusers' converter refused them (`state_dict` should be empty). The runner now folds the alphas into `lora_A` (they equal the rank, 256), and all 512 matrices load with no unexpected keys.
    - General Edit on Raw (20 steps, CFG 3, seed 7, about 5 s a step with CFG and the image's tokens: 115–130 s): "make the fox's fur jet black" and "remove the tree trunk on the left" each changed only that, and the rest of the picture matched the source closely. On Turbo (8 steps, 50 s), both worked too, but the removal moved the fox.
    - A portrait source (832×1216) edited to a landscape output (1216×832, "make it snowing heavily"): no stretch and no seam, but the model reframed the shot, zooming in on the torso and cropping the head, instead of keeping the person at the source's scale.
    - Identity, the person alone ("she sits on a log beside a campfire…"): the raincoat and jeans carried over, but not the face or hair. Raw added glasses and brown hair; Turbo was closer in the face but had straight, tied-back hair. Identity with the fox scene as the source and the person as the second image ("the woman kneels in the snow next to the fox", 156 s): the face, red curls and coat matched, and the scene and fox were kept.
    - Still to do: an 80 GB A100 or an H100 (resident), and Identity's person-alone results at other settings.

## 11. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| `colab ssh` is a new CLI feature and could change or break | High: the transport depends on it | Keep the transport behind one interface (`colabkit.ssh`, shared with FrameHound). An `exec`-based fallback is proven to work (persistent kernel, streamed output, cancel via a flag file uploaded mid-job) |
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
| Krea 2's edits are a port of a community ComfyUI node, not Krea's or diffusers' code | Edits drift from what the LoRAs were trained with (geometry, positions, the grounded encode), and come out weaker or with seams | Ported line by line from the node pack, with the geometry unit-tested; the live test checks mismatched aspect ratios and both LoRAs; the trainer's code (`krea2edit-trainer`) is the reference if they disagree |
| Qwen-Image 2.1, LTX-2.5, FLUX 3 and Fibo 2.0 need diffusers `main` (not in 0.41.0), and LTX-2 needs transformers 5.10.1+ (Gemma 4) | The pinned commit and the newer transformers also run every other family, and could break them | Install each only when the image's lacks what's needed (`QwenImage21Pipeline`, `LTX2Pipeline` and `LTX2DurationHead`, FLUX 3's `upsample_prompt`; `Gemma4UnifiedForConditionalGeneration`); re-test every family on them; move to the first releases that have them |

## 12. Future work

- Video control for Wan 2.2 (VACE / Fun-Control): pose- or depth-driven video, reusing the preprocessors and control-unit UI.
- Additional families such as SD3.5, FLUX.2 [dev] and Hunyuan video, added through the plugin interface.
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
- **Prompt rewriting.** Qwen publishes two rewriting models, `Qwen/Qwen-Image-2.1-PE-T2I` and `-PE-I2I` (Qwen3.5-VL 9B fine-tunes). They expand a short prompt and suggest an aspect ratio. They're another ~9B model, so they'd compete with the image model for GPU memory. They would suit an *Expand prompt* action that shows the rewritten text for review before the job runs, rather than a hidden step. *Upsample prompt* (§4.3) does a lighter version with the loaded text encoder.
- **Faster attention and smaller weights.**
  - diffusers' `QwenImage21FlexAttnProcessor` is faster, but only with `torch.compile`: uncompiled, it runs out of memory at high resolution, and compiling costs time on the first job of each session. The ComfyUI write-up reports ~30% from SageAttention, but the model uses its own attention processors, so it doesn't plug straight in.
  - ComfyUI's int8 weights (6.9 GB transformer) don't load in diffusers. Quantizing when loading could make the text encoder fit an L4 without offload. The usual tool is torchao, which the bootstrap currently uninstalls because Colab's copy is too old for peft; that would need a newer torchao instead.
