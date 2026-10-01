# IP-Adapter: research and plan

Status: Plans A, B and C built (Phases 11 to 13, 2026-09-30). FaceID was tested on an A100 on
2026-10-01 (§5.2); the rest is not yet tested on a live GPU. What
shipped is in [design.md](design.md) (§4.3, §5, §6.4, §10) and [ux.md](ux.md) (Phases 11 to 13).
Where the build differs from the plan below, §4.7, §5.1 and §6.1 say how.

## 1. What it is

IP-Adapter ("image prompt adapter", Ye et al. 2023) adds a second set of cross-attention layers to
a frozen UNet. A CLIP image encoder turns a reference picture into a few tokens, a small projection
maps them into the UNet's context space, and each attention block attends to the text tokens and
the image tokens separately, then adds the two results:

```
out = attn(q, text_k, text_v) + scale · attn(q, image_k, image_v)
```

So a picture works like a prompt: it says *what* and *in what manner* without fixing *where*. That
makes it the complement of ControlNet, which fixes layout and says nothing about content. The two
combine well, and with LoRAs, and in every SDXL mode (text-to-image, image-to-image, inpaint and
outpaint).

Why Degas wants it. SDXL is the only image family here that can't read a reference image. Qwen and
FLUX.2 [klein] read references natively (the *Images* row); SDXL can only copy a picture's layout
(ControlNet) or its pixels (i2i). IP-Adapter gives SDXL "like this one": a style from a painting, a
character's look without training a LoRA, a face.

## 2. The variants

### 2.1 SDXL (h94/IP-Adapter, `sdxl_models/`)

| File | Encoder | Tokens | What it carries | ~Size |
|---|---|---|---|---|
| `ip-adapter_sdxl` | ViT-bigG-14 | 4 (pooled) | Overall content and style, loosely | 0.7 GB |
| `ip-adapter_sdxl_vit-h` | ViT-H-14 | 4 (pooled) | Same, with the smaller encoder | 0.7 GB |
| `ip-adapter-plus_sdxl_vit-h` | ViT-H-14 | 16 (patch, resampler) | Much closer to the reference: objects, clothes, palette | 0.85 GB |
| `ip-adapter-plus-face_sdxl_vit-h` | ViT-H-14 | 16 (patch) | Trained on cropped faces: a likeness | 0.85 GB |

Encoders: `models/image_encoder` is OpenCLIP ViT-H-14 (632 M params, ~2.5 GB fp32), and
`sdxl_models/image_encoder` is ViT-bigG-14 (1.8 B, ~3.7 GB). All the useful SDXL adapters use
ViT-H, so **ViT-H is the only encoder Degas needs**; bigG is only for the oldest, weakest file.

### 2.2 FaceID (h94/IP-Adapter-FaceID)

FaceID replaces the CLIP embedding with an InsightFace ArcFace identity embedding (512-d, from
`buffalo_l`), so it transfers *who* rather than *what the photo looks like*. It is much less
sensitive to the reference's lighting, framing and style than plus-face.

| File | Needs | Notes |
|---|---|---|
| `ip-adapter-faceid_sdxl.bin` + `…_lora.safetensors` | ArcFace embed | ID only; the LoRA is part of the model |
| `ip-adapter-faceid-plusv2_sdxl.bin` + `…_lora.safetensors` | ArcFace embed **and** ViT-H embed of the aligned face | The one to use. `shortcut = True`, and a second weight for the CLIP structure path |
| `ip-adapter-faceid-portrait_sdxl.bin` (`_unnorm`) | ArcFace embeds of up to 5 faces, no LoRA | Portrait style; `unnorm` is very strong |

Best practice (community consensus, ComfyUI and A1111): adapter weight 0.7–0.85, the companion LoRA
at 0.55–0.65 (without it the likeness drops sharply), a clean front or three-quarter face in even
light, CFG 5–7, 28–35 steps. Diffusers documents DDIM or Euler for FaceID; DPM++ 2M Karras is fine
in practice. InsightFace's pretrained models are licensed for non-commercial research only, which
fits a personal app.

### 2.3 Community SDXL adapters

- **Composition** (`ip_plus_composition_sdxl`, ViT-H): layout and pose of the reference, not its
  look. Loads as a plus model.
- **NoobAI / Illustrious** (`kataragi/Noob_ipadapter`, `ip_adapter_Noobtest_800000.bin`, ViT-H):
  the h94 adapters were trained on SDXL base and drift on anime fine-tunes. This one is trained on
  NoobAI. Its author warns it is incompletely trained and breaks at high weight or late end steps.
  Worth a try because the user's checkpoints include Illustrious derivatives.
- **Kolors** adapters: not relevant (no Kolors family).

### 2.4 Layer targeting: InstantStyle and ComfyUI's weight types

The single most useful discovery since the paper: the adapter's effect isn't uniform across the
UNet. InstantStyle (Wang et al. 2024) found that in SDXL one attention block carries *style* and
one carries *layout*:

| ComfyUI index | diffusers block | Carries |
|---|---|---|
| 3 | `down.block_2`, attention 1 | Layout / composition |
| 6 | `up.block_0`, attention 1 | Style (colour, texture, stroke) |
| 0–2, 4, 5, 7–10 | the rest | Content, detail |

Diffusers takes this directly in `set_ip_adapter_scale`:

```python
style = {"up": {"block_0": [0.0, 1.0, 0.0]}}
layout = {"down": {"block_2": [0.0, 1.0]}}
both = {"down": {"block_2": [0.0, w_layout]}, "up": {"block_0": [0.0, w_style, 0.0]}}
```

Unlisted blocks are 0. ComfyUI_IPAdapter_plus (maintenance-only since April 2025, but still the
reference implementation) names the useful presets: *style transfer* (6), *composition* (3),
*style and composition* (3 and 6), *strong style transfer* (every block but 3), and *precise*
variants that leak a little (×0.1) into the other blocks. Style-only transfer lets the prompt
decide the subject, which is what "in the style of this picture" means. All-block transfer
follows the picture closely and reduces variety.

### 2.5 Other techniques that matter

- **Start and end steps.** Like ControlNet: ending at 0.6–0.8 keeps the reference's influence on
  the big shapes and leaves detail to the prompt. Diffusers has no argument for it; the scale can
  be changed from `callback_on_step_end` between steps.
- **Several images for one adapter.** Diffusers concatenates their tokens (ComfyUI's *concat*).
  ComfyUI also offers *average*, which needs our own embedding step. Concat is fine to start.
- **Several adapters at once.** `load_ip_adapter(weight_name=[a, b])` and a list of scales, e.g.
  plus at 0.7 for the scene and plus-face at 0.3 for the face. All must share one image encoder.
- **Masks (regional).** `IPAdapterMaskProcessor` plus `cross_attention_kwargs={"ip_adapter_masks":
  …}` confines each image to an area: two faces, two people. The same idea as regional ControlNet.
- **CLIP sees 224 px, centre-cropped.** Anything outside the centre square is lost, and fine detail
  is lost too. So: crop the reference to a square around what matters (the face, for plus-face),
  and don't expect text or small patterns to carry. ComfyUI's *Prep image for ClipVision* adds
  sharpening before the downscale.
- **Weights and CFG.** Start around 0.5–0.7 for all-block plus, 1.0 for style-only. At high weight
  the image drowns the prompt; lower the weight and raise the steps rather than raising CFG.
  Plus models are "very strong" and want lower weight than the base ones.
- **Negative image.** The unconditional branch gets zero embeddings by default. ComfyUI's noise
  negative helps a little; not worth doing first.
- **Offload order.** `enable_model_cpu_offload` must come *after* the encoder is attached, or the
  encoder is left on the CPU and errors.

### 2.6 Other families

| Family | Option | Verdict |
|---|---|---|
| FLUX.1 [dev] | XLabs `flux-ip-adapter` v2, CLIP-L encoder, in diffusers (`FluxIPAdapterMixin`) | Needs `true_cfg_scale` ~4 with a negative prompt, so each step costs twice. Widely judged weak. |
| | InstantX `FLUX.1-dev-IP-Adapter`, SigLIP-so400m | Better, but not in diffusers (custom pipeline code). |
| | **FLUX.1 Redux** (`FluxPriorReduxPipeline`, SigLIP + a small embedder) | Official, in diffusers, good. It makes *variations*: its output replaces the text embeds, with `prompt_embeds_scale` to mix the prompt back in. **The one to use for FLUX.1**, later. |
| Qwen-Image 2.1, FLUX.2 [klein] | Native multi-image references | Already built. No adapter needed. |
| Wan 2.2 | None. Subject-to-video would be Phantom or VACE | Out of scope (design.md §12). |
| SDXL, identity | InstantID (its own ControlNet plus an adapter, community pipeline); PuLID (not in diffusers) | Heavier and outside diffusers. FaceID Plus v2 covers the need. |

## 3. Recommendation

Build IP-Adapter for **SDXL only**, in two steps, then consider Redux for FLUX.1:

1. **Image prompts (A).** h94 adapters on ViT-H: plus, plus-face and composition, with
   purposes that set the block scales (*Everything*, *Style*, *Layout*, *Style and layout*),
   weight, steps, several images per unit, up to 2 units, and an optional area.
2. **Faces (B).** FaceID Plus v2 with its LoRA, embeddings from InsightFace run on
   onnxruntime (already used for DWPose), no `insightface` package.
3. **FLUX.1 Redux (C, optional).** Separate phase; different mechanism and UI.

## 4. Plan A: image prompts for SDXL

### 4.1 Drive and the index

```
MyDrive/degas/
  ip_adapters/
    sdxl/          *.safetensors / *.bin (+ *.yaml sidecar)
  image_encoders/
    sdxl/clip-vit-h-14/ config.json + model.safetensors (h94 models/image_encoder, the transformers folder)
```

- `drive.KINDS` gains `ip_adapters → ip_adapter` and `image_encoders → image_encoder`. An encoder
  folder with `config.json` is one asset, like a diffusers ControlNet.
- Sidecar fields: `encoder: clip-vit-h-14` (defaulting from the file name: `vit-h` in it, or any
  `plus`/`faceid-plus` file, means ViT-H; `ip-adapter_sdxl.*` means bigG), and `purpose:
  subject|face|composition` (defaulting from the name), which the UI uses the way ControlNets use
  `control:`.
- The `assets.kind` column accepts the two new kinds. Upload the fp16 ViT-H if one exists; the h94
  file is fp32 and is cast at load.

### 4.2 Spec

A new top-level list, beside `control`, recorded in the saved config (§6.4):

```json
"image_prompts": [
  { "adapter": { "path": "ip_adapters/sdxl/ip-adapter-plus_sdxl_vit-h.safetensors", "size": 847517512 },
    "images": ["sha256:…", "sha256:…"],
    "purpose": "style",
    "weight": 1.0, "start": 0.0, "end": 0.8,
    "mask": "sha256:…" }
],
"image_encoder": { "path": "image_encoders/sdxl/clip-vit-h-14", "size": null }
```

- `purpose`: `all | style | layout | style_layout`. It maps to block scales in the runner
  (§2.4). The table lives in `degas_worker/families/ip_adapter.py`, torch-free so it's testable.
- `validate_image_prompts` in `families/validation.py`, next to `validate_control`: at most 2 units, at
  most 4 images each, adapter under `ip_adapters/<family>/`, each adapter once, `start < end`,
  weight 0–2. All units must name the same encoder, which the descriptor resolves from the index
  and writes into `image_encoder` so it's prefetched and staged like the fp16 VAE.
- `Sdxl.validate` accepts the list in every mode. The descriptor grows `supports_image_prompts`,
  sent with `/families`, so the web shows the row only for SDXL.

### 4.3 Server

- `inputs.resolve`: images are **not** fitted to the output. If an image isn't square and hasn't
  been cropped, it's centre-cropped to a square and resized to 448 px, recorded in
  `inputs.transforms` like any fit. That's what CLIP would do anyway, but the saved config then
  shows exactly what the model saw. An area mask is fitted to the output size exactly as a control
  unit's area is (`_fit_mask`).
- Everywhere that walks `control` walks `image_prompts` too: `library.input_blobs` and the saved
  config (`library.py`), Remix's missing-asset check (`submission.py`), and the dispatcher's staging and
  prefetch (`dispatcher.py`). Worth a small helper, `spec_inputs(spec)`, so the next list doesn't
  need four edits again.

### 4.4 Worker runner

In `SdxlRunner`:

- `_load_image_prompts(adapters, encoder)`: if the requested adapter set differs from the loaded
  one, `unload_ip_adapter()` then `load_ip_adapter(dir, subfolder="", weight_name=[…],
  image_encoder_folder=None)`, and attach the encoder with `register_modules(image_encoder=
  CLIPVisionModelWithProjection.from_pretrained(…, torch_dtype=float16), feature_extractor=
  CLIPImageProcessor())`. Keep the encoder resident across jobs; drop it with the adapters. A job
  with no image prompts must unload them: the UNet refuses to run with IP layers and no image
  embeds.
- `from_pipe` pipelines share the UNet (so its IP layers) but copy components when made, so a
  derived pipeline made before the encoder was attached lacks it. Clear `self.derived` whenever
  the adapter set changes, as `_load_controlnets` already does for ControlNets.
- Under CPU offload (below 12 GB): `remove_all_hooks()` and `enable_model_cpu_offload()` again
  after attaching the encoder, so it gets its own hook.
- Per job: `set_ip_adapter_scale([scale_for(purpose, weight) …])`, `ip_adapter_image=[[images of
  unit 1], [images of unit 2]]`, and with areas `cross_attention_kwargs={"ip_adapter_masks":
  IPAdapterMaskProcessor().preprocess(…)}`. Units without an area get an all-white mask when any
  unit has one.
- Steps: in the existing `on_step` callback, recompute each unit's scale for the next step (0
  outside `start…end`) and call `set_ip_adapter_scale` when it changes. Only a handful of calls
  per job.
- *Around the mask* crops areas with `crop_areas`, as it does for ControlNet areas.
- Embeddings: diffusers encodes every call. Cache `prepare_ip_adapter_image_embeds` output by
  (encoder, image shas) for the life of the loaded set, so a batch of 8 encodes once.

### 4.5 Tests (no GPU)

- `validate_image_prompts`: limits, folder, duplicate adapter, mixed encoders, `start >= end`.
- Encoder resolution from sidecar and file name.
- `scale_for(purpose, weight)` and the step schedule, in the torch-free module.
- Server: a job with image prompts stages its images and encoder, a kept item keeps them, Remix
  flags a missing adapter, and non-square images get the square fit in `inputs.transforms`.
- Fake runner in `conftest.py` accepts and echoes `image_prompts`.

### 4.6 UX

The brief is set by ux.md: slate paper, chalk, rose only for what will change, hatching only for
"not drawn yet". Image prompts introduce no new colour or shape. What needs designing is the
language and where the row sits.

**Name.** *Image prompts*. It's what IP stands for, it says the picture works like the prompt,
and it keeps clear of Qwen's *Images* (which the prompt refers to by number) and of *ControlNet*
(which fixes layout). Each unit is *an image prompt*.

**What the chips say.** People don't think in blocks, and they shouldn't choose an adapter file to
say "copy the style". The unit editor leads with what to take from the picture, and the chip
picks the adapter, the way a trace chip picks a ControlNet:

| Chip | Adapter it picks | Blocks | Default weight |
|---|---|---|---|
| *Everything* | plus | all | 0.6 |
| *Style* | plus | style | 1.0 |
| *Layout* | composition, else plus | layout | 1.0 |
| *Style and layout* | plus | style, layout | 1.0 |
| *Face* | plus-face (FaceID in B) | all | 0.6 |

The Model row stays underneath for overriding the pick, and the sidecar's `purpose` warns about a
mismatch in the ControlNet way: *This is a face model; Style works best with a plus model.*

**Create.** The row sits under ControlNet, in every SDXL mode:

```
│ ControlNet                Add ▸ │
│   ▦ Depth · depth-sdxl   0.70   │
│ Image prompts             Add ▸ │
│   ▣▣ Style               1.00   │  two stacked thumbnails when a unit has several images
│   ▣  Face                0.60   │
```

**Unit editor** (a sheet, built from `ControlEditor`'s parts):

```
┌ Image prompt ──────────── Done ┐
│ ┌──┐┌──┐┌──┐  Add image        │  ordinary thumbnails, Crop / Remove on each
│ └──┘└──┘└──┘                   │
│ Take from it                   │
│ Everything  Style  Layout      │
│ Style and layout  Face         │
│ Model   plus_sdxl_vit-h      ▸ │
│ Weight  ───────●──────   1.00  │
│ Steps   ●━━━━━━━━━━━●──  1–24 of 30
│ Limit to an area               │
│ Remove this image prompt       │
└────────────────────────────────┘
```

- Under the chips, one line says what the choice does. *Style: its colour, texture and strokes;
  the prompt decides what's in the picture.* *Layout: where things are, not how they look.*
  *Face: crop close to the face.*
- **Crop.** The model sees a 224 px square from the middle of the picture, so *Crop* opens the
  crop editor at 1:1 with *Free* still available, and the readout says *The model sees the middle
  square.* instead of the upscale warning, which doesn't apply here. *Face* opens the crop tighter
  on the centre.
- **Steps** reuses `StepRange`, with the same note as ControlNet but for looks: *Ending early keeps
  its influence on the big shapes.*
- **Area** reuses the mask editor titled *Area*, painting over the output-sized canvas, since the
  area is where in the *output* the picture applies. Without a source image the canvas is the blank
  well at the output's shape.
- Thumbnails are plain images, never hatched or rose: they don't change.
- Discretion covers them like the other 34 px thumbnails.
- *Use as image prompt* joins *Use as source* in the viewer for images, opening Create's editor
  with the image already in a new unit.

Vocabulary additions for ux.md §3:

| Action | Button | Confirmation or state |
|---|---|---|
| Add an image prompt | Add (Image prompts row) | — |
| Choose what it carries | Everything / Style / Layout / Style and layout / Face | — |
| Drop a unit | Remove this image prompt | — |
| From the viewer | Use as image prompt | — |

**Checked against the defaults.** Two ideas were dropped as generic: a rose ring on image-prompt
thumbnails to show "influence" (rose means change, and a reference doesn't change), and a
per-block slider grid (the ComfyUI layout: accurate, unreadable on a phone). The chips carry the
block choice in words, and *Model* stays available for anyone who wants the file.

### 4.7 As built

- The encoder lives at `image_encoders/sdxl/clip-vit-h-14/` (bigG at `…/clip-vit-bigg-14/`),
  family-scoped like `vae/sdxl/`, so the index's family check applies. It's chosen from the
  adapter's file name (`ip-adapter_sdxl.*` or `bigg` in the name means bigG); a sidecar can't
  override it yet. Sidecars can say `purpose: subject|face|composition`.
- A file can be loaded for two units (style from one picture, layout from another).
- Pictures are fitted to squares no bigger than 1024 px, cropped or letterboxed (*Show it: The
  middle square / All of it, letterboxed*); stretch isn't offered.
- The area is painted over Create's source when it has the output's shape, else over a plain
  grey canvas of the output's size, uploaded as a blob.
- Remix restores units. A remixed area's picture underneath isn't recorded, so its editor
  opens over the source or a blank canvas with the area kept.
- *Use as image prompt* in the viewer isn't built.

### 4.8 Live test (an L4 and a T4)

- VRAM and time with ViT-H resident next to SDXL on a T4, with and without ControlNet.
- Style, Layout and Everything on a fixed seed against no image prompt.
- A LoRA plus an image prompt, and removing the image prompt again (no leftover IP layers).
- An area with two units (two faces).
- The step range: 0–0.6 against 0–1.
- The NoobAI adapter on an Illustrious checkpoint against h94 plus.

## 5. Plan B: faces (FaceID Plus v2)

- **Assets.** `ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin` (sidecar `purpose: faceid`,
  `lora: loras/sdxl/ip-adapter-faceid-plusv2_sdxl_lora.safetensors`), the LoRA itself, and
  `preprocessors/insightface/` holding `buffalo_l`'s `det_10g.onnx` and `w600k_r50.onnx`.
- **Preprocessor** `face` (`degas_worker/preprocess/face.py`): SCRFD detection and ArcFace
  recognition on onnxruntime (installed lazily, as for DWPose), the five-point alignment ported
  with `cv2.estimateAffinePartial2D` (so no `insightface` build on Colab). It returns the largest
  face's 512-d normed embedding and a 224 px aligned crop for the CLIP half. The server calls it
  through `/preprocess` when an image is added to a *Face* unit, so a picture with no face is
  refused in the editor (*No face found in this picture.*), not at the end of a job. The face box
  is drawn as an outline on the thumbnail (outline, as SAM's answer is: found, not changing).
- **Spec.** A FaceID unit records `faces: [{image, embed}]` where `embed` is a small `.npy` blob,
  so replay doesn't depend on the detector. Validation adds the sidecar's LoRA to `loras` at 0.6
  if it isn't there, visibly, so the LoRA row shows it and its weight can be changed.
- **Runner.** Load with `image_encoder_folder=None`, pass the ArcFace embeds as
  `ip_adapter_image_embeds` (zeros for the negative), set the projection layer's `clip_embeds`
  from the aligned crop and `shortcut = True`. A second slider, *Face structure* (the v2 shortcut
  weight, default 1.0), goes under Weight, advanced.
- **Mixing.** A FaceID unit and an ordinary unit need different embedding paths in one call.
  Diffusers can take embeds for all adapters at once; the runner builds CLIP embeds for the
  ordinary unit itself and passes the whole list as `ip_adapter_image_embeds`.
- Defaults from §2.2: weight 0.8, LoRA 0.6. The *Face* chip picks FaceID when it's in Drive, else
  plus-face.

### 5.1 As built

- **The LoRA is inside the `.bin`.** diffusers' `load_ip_adapter` reads the FaceID model's LoRA
  from its own weights (the `_lora.safetensors` file is the same LoRA for A1111 and ComfyUI) and
  loads it as a PEFT adapter named `faceid_<i>`, i being the model's place in the list it loaded.
  It then calls `set_adapters([faceid_i], [1.0])`, which switches the job's own LoRAs off. So the
  runner loads LoRAs, then image prompts, then weights both together (`_activate_loras`), and
  deletes the `faceid_*` adapters before `unload_ip_adapter`, which leaves them behind. There's no
  LoRA row entry: the unit's *Face LoRA* slider (`lora_weight`, 0.6) weights it.
- **Faces are found at job time.** The spec records the pictures, not embeddings: the runner
  runs InsightFace on each picture and uses its biggest face. The detector is an asset of the job
  (`face_detector: preprocessors/insightface`), prefetched like the encoder. A picture with no face
  fails the job with *Image prompt 1: no face found in picture 2*.
- **The editor still checks.** With a ready session, each picture in a FaceID unit is sent to
  `/preprocess` `face`, and its tile shows the aligned 224 px crop the model reads (*The biggest of
  3 faces* when there are several), or *No face found in this picture.* Without a session, it says
  the face is found when the job runs. No outline is drawn: the crop shows which face it took.
- **Alignment** is a least-squares similarity fit (the same as Umeyama's, which `norm_crop`
  uses) in plain Python (`degas_worker/faces.py`), then `cv2.warpAffine`; NMS is plain Python too,
  so both are tested without numpy.
- **Mixing** works as planned: `prepare_ip_adapter_image_embeds` encodes every unit (a FaceID
  unit's aligned crops, for its `clip_embeds`), then the FaceID slots are replaced by the ArcFace
  identities. `shortcut` is on for v2 files (by name); *Face structure* sets `shortcut_scale`.
- A FaceID unit's purpose must be *Everything*; the other chips don't apply to identity.

### 5.2 Checked against a settings guide, and tested (2026-10-01)

A survey of FaceID Plus v2 settings (*Precision Identity Conditioning in Stable Diffusion XL*, an
AI-written compilation of community posts) was checked against the build and against primary
sources: the h94/IP-Adapter-FaceID model card, tencent-ailab's `ip_adapter_faceid.py` and
`attention_processor_faceid.py`, diffusers' `embeddings.py`, `loaders/unet.py` and IP-Adapter
guide, InsightFace's `scrfd.py`, and ComfyUI_IPAdapter_plus (`IPAdapterPlus.py`, `utils.py`,
`CrossAttentionPatch.py`). The build already matched the sources; no default changed.

| The guide says | Primary sources | Here |
|---|---|---|
| CLIP ViT-H, not bigG | h94: `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | ViT-H |
| antelopev2 instead of buffalo_l | **Wrong.** h94 and diffusers use `buffalo_l`; ComfyUI_IPAdapter_plus names antelopev2 only for Kolors FaceID, which was trained on it. Another recognizer is another embedding space | `buffalo_l` |
| The LoRA at 0.55–0.65 | ComfyUI's loader defaults to 0.6; the reference code applies it at 1.0 | 0.6 |
| Weight 0.70–0.80 | Reference and ComfyUI default to 1.0 | 0.8 |
| `v2_weight` 1.0–1.4, never over 2 | h94 `s_scale=1.0`; ComfyUI 1.0 | 1.0, to 2 |
| CFG 4.0–5.5 | h94's examples use 7.5 | 5.5 (SDXL's default) |
| End at 0.8 with an "ease out" curve | **Misread.** ComfyUI's *ease out* ramps the weight across UNet blocks, not over time; `end_at` defaults to 1.0 | Steps 0–1 |
| DPM++ 2M SDE Karras | diffusers suggests DDIM or Euler | DPM++ 2M Karras; all are offered |
| Several pictures: a normalized mean of identities, CLIP tokens concatenated | In no source. Plus pairs each identity with its own CLIP crop; ComfyUI concatenates both by default | Both concatenated |
| Plus Face at 0.35–0.45 after FaceID | Community practice | Two units |
| InstantID, PuLID, FaceDetailer | Outside diffusers (§2.6) | Not built; inpaint *Around the mask* with a FaceID unit is the nearest |

Checking found one gap the guide doesn't mention: ComfyUI_IPAdapter_plus tries detection again
at 576, 512, 448, 384 and 320 px when 640 finds no face, because SCRFD misses a face that fills
the frame. `FaceAnalyzer.detect` now does the same.

**Test.** An A100, RealVisXL V5.0, 1024², 30 steps, CFG 5.5, DPM++ 2M Karras, one seed, a prompt
for a man in a garden. The reference is NASA's 1969 portrait of Neil Armstrong
([Commons](https://commons.wikimedia.org/wiki/File:Neil_Armstrong_pose.jpg), public domain),
with a second photo ([Commons](https://commons.wikimedia.org/wiki/File:Neil_Armstrong.jpg),
public domain) for the two-picture case. The 29 results are in the library, tagged
`faceid-test`. Likeness was judged by eye on one seed, so these are impressions, not
measurements:

- **Detection.** Two close crops of the face (440 px with a little margin, 300 px from brow
  to chin) were refused before the fallback and found after it. The full photos gave the same
  crop before and after.
- **Likeness.** FaceID at its defaults carries the colouring, eyes, nose and face shape; the
  hair and the age come from the checkpoint. It is a resemblance rather than a portrait. Plus
  Face at 0.6 (from the close crop) also carries the hair and reads at least as close.
  **FaceID 0.75 with Plus Face 0.4 was the closest of all**, which bears the guide out.
- **Weight** 0.6, 0.8, 1.0: little difference. **Face LoRA** 0 is darker and less like; 0.6
  and 1.0 are close (1.0 added stubble). **Face structure** 0 distorts the face; 1.0, 1.5
  and 2.0 are close, and 2.0 showed none of the waxiness the guide warns of.
- **CFG** 4, 5.5 and 7.5 all look sound; 7.5 didn't burn. **Ending at 0.8** barely differs from
  1.0. Euler, DDIM, DPM++ 2M Karras and DPM++ 2M SDE Karras all work.
- **Two pictures** mix the two readings: the second photo's smile and stubble came through.
- **A picture of two people** used the bigger face.
- **Checkpoints.** RealVisXL, Juggernaut XL and SDXL base all take it; base is the least like.
- **Inpaint** *Around the mask* over the face at strength 0.35 with the same unit runs and
  changes little, as a FaceDetailer pass would.

## 6. Plan C: FLUX.1 Redux (later)

Different enough to be its own phase: `FluxPriorReduxPipeline` (SigLIP-so400m and the Redux
embedder, `models/flux1/FLUX.1-Redux-dev/`) turns up to a few images into text embeds that
replace T5's, with `prompt_embeds_scale` and `pooled_prompt_embeds_scale` to weight each image and
the prompt. In the UI it would be an *Images* row on FLUX.1 (*Make variations of these*), not the
SDXL unit editor. XLabs' adapter isn't worth building: it doubles step cost for a weaker result.

### 6.1 As built

- **Appended, not summed.** diffusers' `FluxPriorReduxPipeline` makes, for each picture,
  `[T5(prompt), picture tokens]` scaled by its weight, and sums them; without its own text
  encoders the prompt is zeros. ComfyUI instead appends each picture's tokens after the prompt's,
  and the runner does that: the loaded `FluxPipeline.encode_prompt` gives the prompt's T5 and CLIP
  embeddings once, Redux's SigLIP and embedder (loaded from the folder's `image_encoder/`,
  `feature_extractor/` and `image_embedder/`) give each picture 729 tokens, and the pipeline gets
  `prompt_embeds` = prompt then pictures, with the prompt's pooled CLIP embedding.
- **How closely** (`downsample`, 1 to 5, default 3): each picture's 27 × 27 grid is averaged down
  to 27, 13, 9, 6 or 5 a side before it's appended, as ComfyUI's Redux Advanced node does; at full
  size Redux drowns the prompt. The chips are *Closely* (1), *Somewhat* (2), *Loosely* (3) and
  *Just the gist* (5). The weight (default 1) multiplies the tokens (ComfyUI's "multiply"
  strength; its attention-bias strength would need an attention processor).
- **The same row, fewer controls.** FLUX.1 uses SDXL's *Image prompts* row and sheet. Each
  family's descriptor sends `image_prompt_options` (purposes, areas, steps, faces, detail), and
  the sheet shows what they allow: for FLUX.1, only pictures, *How closely*, the model and the
  weight. The server refuses areas, step ranges and other purposes on FLUX.1.
- The Redux folder is `ip_adapters/flux1/FLUX.1-Redux-dev/` (gated on Hugging Face, FLUX.1 [dev]
  Non-Commercial License), indexed as one asset. SigLIP squashes pictures to 384 px squares, so
  the server's square fit applies here too.

## 7. Risks

| Risk | Mitigation |
|---|---|
| diffusers' IP-Adapter loader and PEFT LoRAs on the same attention layers: load order, unload leaving processors behind | Live test adds, swaps and removes each with the other loaded; unload IP-Adapters before touching LoRAs if it misbehaves |
| `from_pipe` and IP-Adapter components drift | Clear derived pipelines when the adapter set changes |
| ViT-H plus SDXL plus three ControlNets on a T4 | Measure; offload already kicks in below 12 GB |
| h94 adapters drift on Pony/Illustrious checkpoints | NoobAI adapter; the purpose chips make lower block sets easy to reach |
| InsightFace model licence | Non-commercial; Degas is personal |
| ComfyUI_IPAdapter_plus is in maintenance mode | Used only as a reference for weights and presets |

## Sources

- [diffusers: IP-Adapter guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/ip_adapter)
- [diffusers: FLUX pipelines (IP-Adapter, Redux)](https://huggingface.co/docs/diffusers/main/en/api/pipelines/flux)
- [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter)
- [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) (`IPAdapterPlus.py`: weight types and SDXL layer indices)
- [kataragi/Noob_ipadapter](https://huggingface.co/kataragi/Noob_ipadapter)
- [InstantX/FLUX.1-dev-IP-Adapter](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter)
- [IP-Adapter FaceID Plus v2 settings](https://www.sozee.ai/resources/ip-adapter-faceid-tutorial-2026/), [Stable Diffusion Art: IP-Adapters](https://stable-diffusion-art.com/ip-adapter/)
