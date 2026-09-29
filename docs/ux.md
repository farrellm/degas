# Degas — UX and visual design

Status: v1 · 2026-09-27. Companion to [design.md](design.md) §8. This covers the visual system built in the Phase 1 redesign and the UX plan for Phases 2–7.

## 1. Point of view

Degas is a one-person studio on a phone. The loop is: write a prompt, send it to a rented GPU, watch images arrive, keep the good ones. Two facts shape the interface:

- **The images are the content.** The interface should recede so images read true. It uses a mid-tone toned-paper ground rather than white or black, the way you judge a picture against a grey wall.
- **The GPU is a meter that's running.** The session is always visible, and the time left before it stops for idleness is never more than a glance away.

The visual language comes from Degas's working materials. He drew in pastel on toned paper, usually blue-grey or grey-green, and many of his pictures stop partway, with the hatching still visible. So the app is pastel on toned paper, and **hatching means "not drawn yet"** everywhere: an image being denoised, a queued image, the area outpainting will fill.

## 2. Tokens

Defined in `web/src/index.css`. Dark is the default look on the phone; light follows the system setting.

| Token | Dark (slate paper) | Light (grey-green paper) | Use |
|---|---|---|---|
| `--paper` | `#333b46` | `#dadfd8` | Page ground |
| `--paper-raised` | `#3e4753` | `#e8ebe5` | Prompt block, selected controls |
| `--well` | `#232930` | `#c4cbc3` | Behind every image and in the viewer |
| `--ink` | `#ece8e1` | `#1f262d` | Text (chalk / charcoal) |
| `--ink-muted` | `#a6aeb8` | `#545e68` | Labels, metadata |
| `--accent` | `#efa3b5` | `#a83f60` | Rose chalk: the primary action, focus, the hatching, and later the mask overlay |
| `--ok` / `--warn` / `--danger` | viridian / ochre / vermilion chalks | darker equivalents | Session state, errors |

**Rules for the accent.** Rose means "your action" or "what will change". It's used for the one primary button per screen, focus rings, sliders, the in-progress hatching and, from Phase 6, the painted mask and outpaint margins, both drawn as hatching. It's never used for decoration.

**Type.** Two families, self-hosted through `@fontsource-variable` so the installed PWA works offline:

- *Newsreader* (serif, optical sizes) is for the words you write: the prompt field, prompt captions in the feed, and the viewer's wall label. Prompts are shown in italic, as the title of the work.
- *Schibsted Grotesk* is for the interface: controls, settings, numbers. Numbers that update in place (countdowns, slider values, progress) use tabular figures. Nothing else does, because this face's tabular punctuation is wide.

Scale: 13 / 15 / 16 / 20 / 26 px (`--step--1` … `--step-2`). Labels are sentence case, never all caps.

**Shape.** Radius follows the role: 3 px for image tiles (they're sheets of paper), 6 px for the prompt block, 10–12 px for buttons, full pills for the session chip and counts. There are no drop shadows; separation comes from hairline rules (`--rule`) and paper tone.

**Motion.** There are three moments: sheets rise, the viewer fades in, and hatching fills in as denoising progresses. An indeterminate phase (copying or loading a model) slowly pulses its hatching. All of it is off under `prefers-reduced-motion`.

## 3. Structure

```
┌───────────────────────────────┐
│ Degas               ● L4 12:40│  header: wordmark + session chip (opens GPU sheet)
├───────────────────────────────┤
│                               │
│   screen                      │
│                               │
├───────────────────────────────┤
│ [– 2 +] [ Generate 2 images ] │  generate bar (Create only)
├───────────────────────────────┤
│    Create      Results (2)    │  tabs; Library joins in Phase 3
└───────────────────────────────┘
```

- **Session chip** (replaces the Session tab). It shows *No GPU*, *L4 starting*, *L4 12:40* (idle countdown) or *L4* while a job runs, with a coloured state dot. Tapping it opens the **GPU session sheet**: start (GPU choice, high memory), status, VRAM, stop, keep running, restart worker, and Drive status and rescan.
- **Results** merges the old Queue and Results tabs into one contact-sheet feed. Each job is a group with its prompt as an italic caption, the model and size underneath, and its images in a grid (three columns for portrait and square, two for landscape). Running and queued jobs come first, with sketch tiles for images that aren't finished. The corner of each group shows *Cancel* while it's pending, then the time the images have left.
- **Create** stays put after Generate. A confirmation line in the generate bar says *Queued 2 images* and links to Results. When no GPU is running, the same line says jobs will wait and offers *Start a session*.
- **Viewer** is full screen on the image well, with swipe, arrow keys and ‹ › to move through the feed. Under the image is a **wall label**: the prompt as the title, then the model, the size and seed, and the sampling settings on separate lines. Actions: *Save to Photos* and *Reuse settings*, which loads the spec and its seed into Create.

### Vocabulary

One name per action, used in buttons, confirmations and empty states:

| Action | Button | Confirmation or state |
|---|---|---|
| Submit a job | Generate / Generate 3 images | Queued 3 images. |
| Stop a job | Cancel | Cancelled |
| Load a result's spec into Create | Reuse settings (becomes **Remix** in Phase 3) | — |
| Keep a result (Phase 3) | Keep | Kept |
| Export to the phone | Save to Photos | — |
| Delete every finished result now | Clear results | Delete every finished image and clip? Kept ones stay in the library. |
| GPU | Start L4 session / Stop session / Keep running | Starting L4 / Ready on NVIDIA L4 / Generating on … |
| Worker | Restart worker | — |
| Drive | Rescan Drive | Models indexed 3 h ago. |
| Open the mask editor | Paint the area to redraw / Edit mask | — |
| Close the mask editor | Done | — |
| Use a SAM selection | Add / Subtract / Replace | Shown as an outline. |
| Re-crop a masked image | — | The mask moved with the crop. / The crop left out the whole mask. |
| Letterboxed source | Outpaint the bars | — |
| Add a ControlNet unit | Add (ControlNet row) | — |
| Make a control image | As is / Depth / Pose / Edges | Tracing on the GPU… |
| Limit a unit to part of the image | Limit to an area / Edit area | — |
| Drop a unit | Remove this ControlNet | — |
| Add a reference image (Qwen or FLUX.2 [klein] edit) | Add image (Images row) | — |
| Reorder or drop a reference | Earlier / Remove | — |
| Discretion mode (Phase 9) | Cover images (header switch) | Covered tiles read *Show image 1*; prompts *Show prompt* |

"Keep" and "Save to Photos" are deliberately different words. Keeping is about retention inside Degas; Save to Photos exports to the phone.

## 4. Phase plans

Each phase lists the screens it adds or changes, then any new components. Wireframes are phone-width.

### Phase 2: Assets and LoRA

**Model picker sheet.** The Model row opens a sheet instead of a native select once sidecars exist. Each row shows the sidecar label (falling back to the file name), the size, and whether the file is already on the GPU. The cold copy costs about 90 s, so that last fact matters when choosing.

```
┌ Model ─────────────────── Done ┐
│ [search models             ]   │
│ ▣ Studio XL v10                │
│   6.9 GB, on the GPU           │
│ ▢ SDXL base 1.0                │
│   6.9 GB, about 90 s to copy   │
│ ─────────────────────────────  │
│ Indexed 3 h ago.  Rescan Drive │
└────────────────────────────────┘
```

**LoRAs in Create.** A *LoRAs* row sits under Model in the settings list. Each added LoRA is a row with its name, a weight slider and a remove button. Trigger words from the sidecar appear as small chips under the row, and tapping one inserts it at the cursor in the prompt. *Add LoRA* opens the same picker sheet filtered to LoRAs, with preview thumbnails from the index.

```
│ Model            Studio XL v10 ▸│
│ LoRAs                     Add ▸ │
│   Film Grain v3   ──●──  0.80 ✕ │
│   filmgrain                     │  ← trigger-word chip
```

**Copy progress.** The sketch tile already shows *Copying model 45%*. The session sheet gains an *On the GPU* section listing cached files and disk used out of the 150 GB budget.

### Phase 3: Save, library and remix

Built. Notes on what shipped: *Kept* in the viewer is a toggle, so tapping it again removes the image from the library (the result stays in the feed until it expires). The kept mark is a rose dog-ear with a paper edge; the *Kept* button repeats it beside its label. The library's viewer adds a *Tags* field (comma-separated, saved on blur) so tag search has something to match. The library grid keeps each image's proportions in three columns under day headings (*Today*, *Yesterday*, *Sep 24*). Deleting from the library asks for confirmation inline.

- **Keep.** In the viewer, *Keep* becomes the primary action and *Save to Photos* moves to second place. Kept tiles carry a small rose corner mark in the feed. A group whose images are all kept shows *Kept* instead of the time left, and the time left turns ochre under 2 hours.
- **Library tab** (the third tab). It has a search field for prompt text and tags, and an *Images / Prompts* switch. The images grid is by date, and opens the same viewer and wall label with *Remix*, *Delete* and (from Phase 4) *Use as source*.
- **Remix** replaces *Reuse settings*. It restores LoRAs and inputs as well as parameters. Any asset that's no longer in the Drive index is flagged on its row: *Not found in Drive. Pick another model.*
- **Saved prompts.** A *Prompts* button at the top right of the prompt block opens a sheet listing saved prompts. Tapping one replaces the prompt fields, and *Save this prompt* sits at the top of the sheet. The Library's Prompts view lists the same items with rename and delete.
- The empty-state copy changes to *Images you don't keep are deleted 24 hours after the GPU session ends.*

### Phase 4: Wan 2.2 video, image picker, crop editor

Built. Notes on what shipped: the *Image / Video* switch keeps a separate draft per family, and carries the prompt over to a family that has none. The variant follows the model (each variant's models live in their own Drive folder); its name appears under the Model row when the model's label doesn't already say it. A *Fit* row (*Crop to fit*, *Letterbox*, *Stretch*) appears under the source only when its shape differs from the output size. Applying a crop of a different shape (*Free* or a preset) changes the form's size to the crop's, snapped to the model's step. The crop frame's corners are rose chalk L-marks; in *Free* they are the drag handles, and the frame settles back to the middle of the stage when released. The viewer plays clips with the native controls, since Wan clips have no sound to unmute. *Use as source* on an image switches Create to the first image-to-video mode. A14B LoRAs are pairs (`…_high_noise` / `…_low_noise`) with a weight slider per expert. *High memory* in the session sheet starts ticked when the Create model is an A14B.

**Choosing what to make.** A segmented control at the top of Create picks *Image* or *Video* (the family's media type). Below the prompt, a row of mode chips lists only the modes the variant supports: *From text* or *From image*, later *Inpaint* and *Outpaint*. The variant appears in the Model row. If the variant's minimum GPU is above the running session's, a warning line appears under the row: *Needs an A100; this L4 session may run it slowly.*

**Source slot.** It's a settings row with a thumbnail. Tapping it opens the image picker, and a filled slot has *Crop* and *Remove*.

```
┌ Choose image ──────────── Done ┐
│ Recent  Library  Photos  Link  │
│ ┌──┐┌──┐┌──┐┌──┐                │
│ └──┘└──┘└──┘└──┘                │
│ ──── after picking ──────────   │
│ [ preview, 3024 × 2016 ]        │
│ [ Crop ]         [ Use image ]  │
└────────────────────────────────┘
```

The Link tab has a URL field and a *Paste* button. For a video, the picker offers *First frame*, *Last frame*, or a scrubber.

**Crop editor** (full screen, on the image well):

```
│ Cancel      Crop       Apply │
│ ┌───────────────────────────┐ │
│ │ ░░░┌───────────────┐░░░░░ │ │  image pans and zooms under a fixed frame
│ │ ░░░│               │░░░░░ │ │
│ │ ░░░└───────────────┘░░░░░ │ │
│ └───────────────────────────┘ │
│ Match size  1:1  3:2  16:9  Free│
│ ⟲ rotate   ⇋ flip       Reset │
│ 832 × 1216 from 2016 × 3024   │
└───────────────────────────────┘
```

When the crop scales the image up more than 1.5×, the readout turns ochre and says so.

**Video results.** Tiles show the poster frame with the duration in the corner. The viewer plays the clip inline, looped and muted, and tapping it unmutes. The wall label adds a line for frames, fps and duration. *Extend* opens Create in i2v mode with the last frame as the source. A stitched chain appears as its own group captioned *Extended, 3 clips*.

### Phase 5: Queue UX and push

Built. Notes on what shipped: the drag handle is three strokes of hatching, since a queued job isn't drawn yet. A lifted group is raised paper that extends past its edges (a spread, not a shadow) and follows the finger; a rose line marks where it will land. The handle also moves a group with the arrow keys. With a mouse the group lifts at once. *Move to top* sits under *Cancel* in the group's corner. The *Cancelled / Undo* toast is an inverted pill above the tab bar, and only appears for queued jobs, since a running job can't be restored. A batch's Seed row offers *Random seeds* and *Count up*, with a *from* field under *Count up*. An empty field means counting up from a random seed. The notification question is a raised-paper strip under the header. The sheet's switch reads *When images finish*, and when Degas isn't installed it explains how to add it to the Home Screen instead. The icon keeps the upper half of the *D* in solid chalk and leaves the lower half in hatching (`web/public/icon.svg`, PNGs from `web/scripts/icons.sh`).

- **Reorder and cancel.** Queued groups in Results get a drag handle (long-press to lift) and a *Move to top* action. Cancelling a queued job shows an *Undo* link in a toast for 5 s.
- **Seeds for batches.** When the batch size is above 1, the Seed row becomes a two-way choice: *Random seeds* or *Count up from 1234*. A fixed seed with a batch is disallowed, as the design doc says.
- **PWA.** The icon is an italic *D* in chalk on slate paper, with rose hatching across its lower half. `theme-color` is already set per scheme.
- **Notifications.** Degas asks once, after the first job finishes while installed to the home screen: *Get a notification when images finish?* with *Allow* and *Not now*. After that it's a toggle in the GPU session sheet. The idle warning 2 minutes before shutdown uses the same wording as the sheet: *L4 stops in 2:00 if no job runs.*

### Phase 6: Image-to-image, inpaint and outpaint

Built. Notes on what shipped, including where it changed from the plan below:

- **The mask is hatching, not a wash.** The plan said rose at 50%. A translucent red wash is every inpaint tool's default and muddies the picture; the masked area is literally what isn't drawn yet, so it's drawn in the same rose hatching as a sketch tile, and you see the image between the strokes. The brush paints with a hatch pattern anchored to the canvas, so strokes join without seams. The hatching is sized for the fitted zoom and scales with pinch-zoom like paper.
- **Mask editor.** *Brush*, *Erase* and *Select* in a three-way switch; a *Size* slider with a ring under the pointer; undo and redo, *Invert*, *Clear*, and *Blur*, which previews the form's mask blur. An *Image* slider fades the photo, not the mask. It paints at up to 2048 px a side; the server scales the mask to the image. One finger paints, two pinch and pan; a stroke started just before a pinch is dropped.
- **Select (SAM 3, pulled forward from Phase 7).** Tap to include (a rose dot), long-press or *Exclude* to leave out (a hollow ring), or type what to select and *Find*. The answer is an outline, not hatching, until *Add*, *Subtract* or *Replace*, so the mask's meaning stays "will be redrawn". *Smaller* and *Bigger* move between SAM's candidates; *Grow* (default 8 px, up to 128) gives the selection a margin to blend into, and the outline shows it grown; it sits above the candidates so it can be set before selecting. On a short screen the tools scroll under the image. Without a ready session, or without SAM 3 in Drive, the Select panel says which.
- **Mask row.** Under Source in Inpaint: the source thumbnail with the mask laid over in rose, *Edit mask*, and *Clear*. Generate stays disabled until there's a mask. Cropping the source again carries the mask with it.
- **Outpaint** is placement on the canvas rather than a stepper per side: the Size row is the canvas, and the source sits on it with hatched margins. Drag it, scale it with *Image size*, or push it *Left*, *Right*, *Top*, *Bottom* or back to the *Centre* (buttons that wouldn't move it are dimmed). The readout says *+288 px left, +288 px right*. The default fits the source inside the canvas, centred; if that fills it, the source starts at 75%. Choosing *Letterbox* in the Fit row offers *Outpaint the bars*.
- The mode chips scroll sideways instead of wrapping when four don't fit. Choosing an inpainting checkpoint limits them to *Inpaint* and *Outpaint*, and its variant name shows under the Model row.
- *Use as source* sends an image to the family Create has open: an image goes to *From image* (or stays in *Inpaint* or *Outpaint*), and only a video draft sends it on to image-to-video.

The plan as written before building:

**Mask editor** (full screen). The mask is painted in rose at 50% over the image, so "affected area" uses the same colour as every other "what will change" cue.

```
│ Cancel       Mask        Done │
│ ┌───────────────────────────┐ │
│ │   image + rose mask       │ │  two fingers pinch and pan; one finger paints
│ └───────────────────────────┘ │
│  Brush  Erase     ●───── size │
│  ↶ ↷   Invert  Clear  ◐ image │
└───────────────────────────────┘
```

**Outpaint.** A preview shows the source inside its padded canvas, and the new margins are drawn in the same hatching as a sketch tile, since that's where new pixels will go. Each side has a stepper, the edges can be dragged, and *Fill to target size* pads to the current Size preset.

A *Denoise strength* slider row appears for i2i and inpaint, from the schema.

### Phase 7: Control

Built. Notes on what shipped, including where it changed from the plan below:

- **A control image is a study, not a picture.** In the ControlNet rows, each unit's image is drawn in chalk on the paper: greyscale and screen-blended on the slate, inverted and multiplied as charcoal on the grey-green. An edge map reads as a line drawing and a depth map as tone. Thin lines are level-stretched so they survive the 34 px thumbnail. With an area, the trace fades to a quarter outside it, since the unit only guides there. No new colour is involved.
- **The editor shows what the model reads**, in its true colours on the well (a pose's colours tell the model which limb is which). *Over the picture* lays the trace on the photo it came from, dimmed, to check they line up.
- **Tracing is a tap.** The chips are *As is*, *Depth*, *Pose* and *Edges*; tapping one traces the picture straight away (there's no *Run*), with the preview in pulsing hatching while the GPU works. *Edges* adds a *Detail* slider (0 to 100%, one value for Canny's two thresholds) that traces again when it rests. Without a ready session the trace chips are disabled and a note says why. A trace picks a ControlNet that reads it when the unit has none, and a model that reads another kind gets a warning: *This looks like an edges model; the image is a depth map.*
- **The picture comes first.** An empty unit asks for the picture whose layout the image should follow, with *Use the source* (when Create has one) and *Choose image*. Cropping crops the picture, not the trace, and traces it again; an area follows the crop.
- **Weight, not strength**, so it doesn't collide with *Denoise strength* in the same form, and matches the LoRA rows.
- **Steps** is two thumbs on one track counted in the form's steps, with the guided span in rose and a readout, *1–24 of 30*, rather than fractions. The note under it says what the range is for: *Early steps set the layout; ending early leaves the details free.*
- **Area** opens the mask editor titled *Area*, painting over the photo the trace came from (same size) and selecting in it with SAM. The blur preview is hidden, since areas aren't blurred.
- A ControlNet model can guide one unit; up to three units. A unit closed with neither an image nor a model is dropped. Generate stays disabled until every unit has both.

The plan as written before building:

- **ControlNet section** in Create, under LoRAs. Each unit is a row showing a thumbnail of its control image, the model name and the scale. Tapping a unit opens the **control editor sheet**: an image slot (using the picker), preprocessor chips (*Depth*, *Pose*, *Edges*) with a *Run* button and a preview, the model, strength, a two-thumb start–end range, and *Limit to an area*, which opens the mask editor for regional control.
- SAM in the mask editor shipped in Phase 6.

### Phase 8: Qwen-Image 2.1

The plan:

- **A second image family, no new screen.** Under Image, the Model picker lists SDXL checkpoints and Qwen models together, each row naming its family; choosing one from the other family switches Create to it, carrying the prompt. Qwen's mode chips are *From text*, *Edit* and *Inpaint*. The form comes from the schema, so CFG, Steps, Schedule and *Remove VAE grid* need no new widgets. CFG's note says what the number does: *Above 1 uses the negative prompt and takes twice as long.*
- **Images row.** In *Edit* and *Inpaint*, a row under Source lists the other images the prompt can refer to, as thumbnails numbered from 2. The source is image 1, and the Source row says so (*Image 1*). *Add image* opens the image picker. Each thumbnail has *Crop*, *Earlier* (move up one) and *Remove*. There are up to 9, and the row says *Refer to them by number: "the jacket from image 2".* A reference has no Fit row, because the model takes each at its own shape. Its *Crop* (on the row, or in the picker) opens the crop editor without *Match size*: the crop starts as the whole image, is *Free* by default, and keeps its own pixel size. The readout turns ochre when the model will scale the crop up more than 1.5× (to the output's pixel count): *The model scales it up 2.4×, so fine detail will be soft.*
- **Inpaint** reads as it does for SDXL (Source, Mask), plus the Images row. There's no *Denoise strength*, because the model redraws the whole picture and the mask decides what is kept.
- **Size.** The presets come in two tiers, 1K and 2K. 2048² is selected by default. The GPU warning under Model already covers a T4 session.
- **Nothing is rose that isn't changing:** reference thumbnails are ordinary image thumbnails, not hatched.

### Phase 9: Discretion

Built. For the subway or the couch: a switch in the header, left of the session chip, covers every image and prompt until you touch it. It's an eye, closed and on raised paper while it's on (`aria-pressed`, labelled *Cover images*), and it's remembered on the device. An inline script in `index.html` sets `<html data-discreet>` before the first paint, so a relaunch never flashes an image.

- **Glassine, not hatching.** A covered image lies under glassine, the paper laid over a pastel to protect it: blurred past recognition, desaturated and washed with paper. Hatching still means only "not drawn yet". A covered prompt keeps the shape of its words but not the words.
- **Where.** Tiles in Results, Library and the image picker's Recent and Library grids, the viewer's image and its wall-label prompts, the feed's captions and model names (uncovered together), the viewer's model and LoRA line (uncovered with its image), and a failed job's error, whose blur stays vermilion so a failure still shows. Create's 34 px source, reference, mask and control thumbnails stay covered, as there's no tap to spare on them. The full-screen editors (crop, mask, placement) and the picked preview are clear, because you open them on purpose.
- **Uncovering.** Tap a covered tile to uncover it, and tap again to open it. Hold it to see it only while your finger is down (a move first means scrolling). In the viewer, each image starts covered unless its tile was uncovered, and a tap uncovers it. An uncovered image shows its prompt too; a covered image's prompt can still be uncovered on its own. Swipes don't count as taps. Tap a prompt to uncover it.
- **Covering again.** Closing the viewer, changing tabs, and leaving the app cover everything.
- **App switcher.** When the app loses focus, a plain paper shield with the wordmark covers the screen. It's set on the DOM directly, not through React, so it paints as soon as possible. It drops when the app has focus again. A share sheet or photo picker also raises it, but only hiding the app covers what you'd uncovered. iOS may take its snapshot before the shield paints, so covered is also the resting state: at worst, the snapshot shows what you uncovered just before leaving.
- **Notifications.** While the mode is on, the service worker drops the body (the prompt, or an error) and keeps the title, *2 images finished*.

### Phase 10: Flux

Built. Two new families, found under the Model button like Qwen. They need no new widgets, because each form is built from its family's schema.

- **FLUX.1 [dev]** has only *From text*. Its settings are *Guidance* (3.5), *Steps* (28) and the size. There's no *Negative prompt* and no *CFG*, because the model is guidance-distilled, and the Guidance field says so.
- **FLUX.2 [klein]** has only *Edit*. Its settings are *Steps* (4, and the field says the model is distilled to 4) and the size. The *Images* row takes up to 3 images after the source, since klein reads 4 in all. Its *Add image* is disabled at that limit, as Qwen's is at 9: the limit comes from the variant (`max_refs`). A family whose variant declares no references has no Images row, even in *Edit*.
- **Crop readout.** Klein scales references down to 1 megapixel and never up, so cropping a reference small doesn't bring the ochre *The model scales it up* warning that Qwen shows.

## 5. Component inventory

| Component | Status | Notes |
|---|---|---|
| `Sheet` | Phase 1 | Bottom sheet; Escape and scrim close it; focus returns to the opener |
| `SessionChip`, `SessionSheet` | Phase 1 | |
| `SchemaForm` | Phase 1 | Prompt block + settings rows; `leadingRows` slot for non-schema rows |
| Sketch tile | Phase 1 | `.tile.sketch` with `--p` in 0..1; `.waiting` and `.indeterminate` variants |
| Viewer + wall label | Phase 1 | Shared by Results and Library |
| `AssetPicker` | Phase 2 | Models and LoRAs (ControlNets in Phase 7); search from 7 rows |
| `LoraList` | Phase 2 | Single weight 0–2; paired high/low weights in Phase 4 |
| `Viewer`, `SaveToPhotos` | Phase 3 | Shared by Results and Library; actions and extra wall-label rows are slots |
| `PromptSheet` | Phase 3 | Saved prompts from the prompt block |
| `LibraryScreen` | Phase 3 | Images / Prompts, search, day-grouped grid |
| `ImagePicker` | Phase 4 | Recent, Library, Photos, Link; frame choice for videos |
| `CropEditor` | Phase 4 | Full screen; geometry in `crop.ts`; `free` for references (Phase 8) |
| `MaskEditor`, `MaskThumb` | Phase 6 | Full screen, canvas; helpers in `mask.ts`; SAM 3 Select |
| `PlaceEditor` | Phase 6 | Outpaint placement; geometry in `place.ts` |
| `ControlList`, `ControlThumb` | Phase 7 | Unit rows; the chalk-study thumbnail |
| `ControlEditor` | Phase 7 | Sheet; hosts the image picker, crop, area and model pickers in its place; `StepRange`; helpers in `control.ts` |
| `RefList` | Phase 8 | Ordered reference images for Qwen edits; *Crop*, *Earlier* and *Remove* |
| `Tile`, `CoveredText`, `DiscretionToggle` | Phase 9 | Result/library/picker tile with covering; a covered prompt; the header switch. State, `useCover` and the shield in `discretion.ts` |

## 6. Quality floor

- Tap targets are at least 44 px. Inputs are 16 px so iOS doesn't zoom on focus.
- Every control has a visible label or an `aria-label`. Sketch tiles are `role="img"` with their progress as the label.
- Focus is visible (a 2 px rose ring) and sheets move focus in and back out.
- `prefers-reduced-motion` turns off all animation, including the pulsing hatching.
- Safe-area insets are respected at the top, in the tab bar, and in sheets.
