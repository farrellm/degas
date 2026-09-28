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
- Training LoRAs.
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
4. Install packages only if something is missing. The Colab image already has torch, diffusers, transformers, peft, fastapi, uvicorn and ffmpeg, so this is usually a no-op. Extra packages such as `sam2` and DWPose dependencies are installed lazily, the first time they're used.
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
| POST | `/preprocess` | Run depth, pose, canny or SAM on a blob, returning the output as the response body |
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

**Interactive preprocessing.** SAM point prompts, and depth or pose extraction, use `POST /preprocess`. For SAM, the worker caches the image embedding by image hash, so each extra tap only runs the lightweight mask decoder. The expected round trip is 0.12 s of transport plus the inference time. When a generation job is running, a preprocessing request is served between denoising steps.

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
    modes: list[Mode]                # t2i, i2i, inpaint, outpaint, t2v, i2v
    lora_format: Literal["single", "paired_hi_lo"]
    supports_control: bool
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
  - `outpaint`: an inpaint on a padded canvas. The worker builds the mask automatically from the requested extension (pixels to add on each side).
- **Checkpoints:** any SDXL-architecture `.safetensors` file on Drive, including Pony and Illustrious derivatives. Loaded with `from_single_file`.
- **ControlNet:** one or more ControlNet units. Each unit has:
  - a ControlNet model,
  - a control image,
  - a conditioning scale,
  - a start and end fraction,
  - an optional mask (regional control).

  The ControlNet variants of the pipelines are used when units are present, and `MultiControlNetModel` when there is more than one unit.
- **Regional ControlNet:** implemented by multiplying each unit's down-block and mid-block residuals by that unit's downsampled mask. This requires a small custom pipeline subclass or a wrapper around the ControlNet forward pass.
- **Parameters:** prompt, negative prompt, width and height (with SDXL aspect-ratio presets), steps, CFG, sampler/scheduler, seed, clip skip, denoise strength (i2i and inpaint), mask blur and padding (inpaint), and an optional refiner.

**Wan 2.2 (`wan22`)**

- **Variants:**
  - `ti2v-5b`: text-to-video and image-to-video, minimum GPU L4.
  - `t2v-a14b`: text-to-video, minimum GPU A100.
  - `i2v-a14b`: image-to-video, minimum GPU A100.
- **Pipelines:** diffusers `WanPipeline` and `WanImageToVideoPipeline`.
- **LoRAs:** `paired_hi_lo` for the A14B variants. A LoRA entry names a high-noise file and a low-noise file, and each has its own weight. A LoRA entry for the 5B variant is a single file.
- **Parameters:** prompt, negative prompt, resolution preset, frame count, fps, steps, CFG (with separate values for the two experts on A14B), boundary ratio (A14B), seed, and a source image for i2v.
- **Output:** an MP4 encoded with H.264 in `yuv420p` pixel format, so it plays inline on iOS. A poster frame is extracted, and so is the last frame, which is used for video extension.

### 4.4 Preprocessors

Preprocessors run on the GPU, in the worker, as a family-independent registry:

| id | Model | Output |
|---|---|---|
| `depth` | Depth Anything V2 | Depth map image |
| `pose` | DWPose | Pose skeleton image (OpenPose format) |
| `canny` | OpenCV | Edge image (thresholds are parameters) |
| `sam` | SAM 2 | Mask from positive and negative point prompts (and an optional box) |

Preprocessor outputs are ordinary images. They go into the blob store and can be edited, saved, or reused.

## 5. Google Drive layout

```
MyDrive/degas/
  models/
    sdxl/          *.safetensors (+ optional *.yaml sidecar)
    wan22/         <variant>/…   (diffusers-format directories)
  loras/
    sdxl/          *.safetensors (+ *.yaml, preview *.jpg/png)
    wan22/         *.safetensors; A14B pairs named *_high_noise.safetensors / *_low_noise.safetensors or declared in sidecar
  controlnets/
    sdxl/          *.safetensors or diffusers dirs
  preprocessors/   depth-anything-v2/, dwpose/, sam2/
  vae/
    sdxl/
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
```

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
- **`assets`**: family, kind (`model|lora|controlnet|vae|preprocessor`), path, drive_file_id, size, mtime, md5, sidecar (JSON), preview_thumb, indexed_at.
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
  "model": { "path": "models/sdxl/juggernautXL_v10.safetensors", "size": 6938040682 },
  "loras": [
    { "path": "loras/sdxl/filmgrain.safetensors", "weight": 0.8 }
  ],
  "params": {
    "prompt": "…", "negative_prompt": "…",
    "width": 1024, "height": 1024, "steps": 30, "cfg": 5.5,
    "scheduler": "dpmpp_2m_karras", "seed": 1234, "strength": 0.75,
    "mask_blur": 8
  },
  "inputs": {
    "source": "sha256:…",
    "mask": "sha256:…",
    "transforms": {
      "sha256:<source>": { "original": "sha256:…",
        "ops": [ { "op": "rotate", "deg": 90 },
                 { "op": "crop", "x": 120, "y": 0, "w": 1536, "h": 1536 },
                 { "op": "resize", "w": 1024, "h": 1024, "filter": "lanczos" } ] }
    }
  },
  "control": [
    { "controlnet": "controlnets/sdxl/depth.safetensors",
      "image": "sha256:…", "preprocessor": { "id": "depth", "source": "sha256:…" },
      "scale": 0.7, "start": 0.0, "end": 0.8, "mask": "sha256:…" }
  ],
  "runtime": { "gpu": "L4", "diffusers": "0.x", "torch": "2.x", "duration_s": 14.2 }
}
```

For a Wan 2.2 A14B LoRA, the entry has the form `{ "high": {path, weight}, "low": {path, weight} }`. If an input image came from a URL, the config also records `inputs.origins: { "<sha256>": "<url>" }` for provenance. Replay always uses the stored blob, so the URL is never fetched again. Video extension records the parent video in `inputs.extends` (the parent's blob hash) and the extracted frame in `inputs.source`.

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

**Resize** is set in one of three ways: to the target resolution (default), to a preset, or to an explicit width × height with the aspect ratio locked. The editor shows the final pixel size, and warns when the image is being upscaled by more than 1.5×.

**Auto-fit.** If the user skips the editor and the source image's aspect ratio doesn't match the target size, the job spec gets a `fit` mode:

- `crop` (default): center-crop to the target aspect, then resize.
- `pad`: letterbox. For SDXL, the editor offers to switch the job to outpaint, with the pad area as the mask.
- `stretch`: resize without keeping the aspect ratio.

The server resolves `fit` into explicit transform operations when the job is submitted. The saved config therefore always records exact operations.

**Interaction with masks and control images.**

- A mask belongs to a specific source blob. Changing the source's transform makes its mask invalid. The UI asks before discarding the mask. It offers to transform the mask with the same operations instead, when the change is only a resize or rotate.
- A control image that has not been edited is automatically fitted to the output resolution using the same `fit` rules.

## 7. Server API

All endpoints are under `/api`. JSON unless noted.

| Method | Path | Purpose |
|---|---|---|
| GET | `/families` | Family descriptors, variants, modes |
| GET | `/families/{id}/schema?variant=&mode=` | Param JSON Schema for form |
| GET | `/assets?family=&kind=` | Cached Drive asset index |
| POST | `/assets/rescan` | Re-index Drive via the Drive API (no session needed) |
| GET | `/session` | Current session state, idle countdown |
| POST | `/session` | Start `{gpu, high_mem}` |
| DELETE | `/session` | Stop |
| GET | `/jobs` | Queue + recent jobs |
| POST | `/jobs` | Submit `{spec, batch_count, seed_mode}` |
| PATCH | `/jobs/{id}` | Reorder (`queue_position`) |
| DELETE | `/jobs/{id}` | Cancel |
| GET | `/results?cursor=` | Recent ephemeral results |
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
| POST | `/preprocess` | `{id: depth\|pose\|canny\|sam, image, params}` → `{blob}` |
| POST | `/push/subscribe` | Store Web Push subscription |
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
   - Inpaint: a mask slot that opens the mask editor on the source image. Outpaint: controls for how many pixels to add on each side, with a live preview of the padded canvas.
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
4. **Mask editor** (full screen):
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
  server/src/degas/        app.py, api/, colab/ (CLI wrapper, session mgr, dispatcher),
                           colab/tunnel.py (ssh), drive.py (OAuth, index), families/,
                           storage/, push.py, sweeper.py
  worker/src/degas_worker/ app.py (FastAPI), jobs.py, cache.py (rclone), families/, preprocess/
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
5. **Queue UX and push.** Batch generation and seed modes, cancel, reorder, the PWA manifest and service worker, and Web Push.
6. **i2i, inpaint and outpaint.** The mask editor with brush tools, and outpaint canvas extension.
7. **Control.** Preprocessors (depth, pose, canny), SDXL ControlNet units, SAM-assisted masking, and regional ControlNet.

## 11. Risks and open questions

| Risk | Impact | Mitigation |
|---|---|---|
| `colab ssh` is a new CLI feature and could change or break | High: the transport depends on it | Keep the transport behind one interface (`colab/tunnel.py`). An `exec`-based fallback is proven to work (persistent kernel, streamed output, cancel via a flag file uploaded mid-job) |
| A VM with an idle kernel may be reclaimed even while the worker is busy | Jobs killed partway through | Build the `exec` heartbeat in Phase 1 as a setting that is on by default (one trivial `exec` every 5 min, costing about 1.5 s each) |
| Drive OAuth app in "testing" mode issues refresh tokens that expire after 7 days | Weekly re-authorization | Publish the OAuth app for your own account only (drive.readonly is a restricted scope, but an unverified app used only by its owner is fine); the UI prompts to re-authorize when a refresh fails |
| SSH tunnel throughput (about 12 MB/s) | Slow transfer of large video outputs | Acceptable: a 10 MB MP4 takes about 1 s. Encode with a sensible CRF |
| Colab reclaims the VM or hits usage limits | Running job is lost | Detect the failure and move the session to `error`; queued jobs return to the queue; the notification explains what happened |
| Copying Wan A14B (two 14B experts) from Drive is slow, and it needs offloading | Cold start of about 7 min for the copy; slow generation; standard shapes have only 12 GB of RAM | Show copy progress; prefetch; keep the session warm; default to `--high-mem`; use 5B for drafts |
| Regional ControlNet needs a custom residual-masking path | Extra complexity | Isolate it in one pipeline wrapper; it is the last item in Phase 7 |
| Session bootstrap time | Measured: `colab new` 3.5 s (CPU), SSH 1.5 s, and packages are preinstalled, so bootstrap takes about 10–20 s before model copy | Good enough; the model copy dominates |
| Web Push on iOS | Needs a home-screen install and iOS 16.4+ | Document it; the in-app SSE path works regardless |
| diffusers API churn for newer Wan and ControlNet classes | Upgrades break the worker | Pin versions in `requirements-worker.txt`; record versions in each saved config |

## 12. Future work

- Video control for Wan 2.2 (VACE / Fun-Control): pose- or depth-driven video, reusing the preprocessors and control-unit UI.
- Additional families such as Flux, SD3.5 and Hunyuan/LTX video, added through the plugin interface.
- Upscaling / hires-fix passes, and video frame interpolation.
- Live latent previews during sampling using TAESD / TAEW.
- Regional LoRA.
- Multiple concurrent sessions, e.g. an L4 for images alongside an A100 for video.
