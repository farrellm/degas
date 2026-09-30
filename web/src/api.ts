// Typed client for the Degas server API (/api, design §7).

export interface Variant {
  id: string
  label: string
  min_gpu: string
  modes: string[]
  /** Drive folder of this variant's models, when the family has several variants. */
  model_dir: string | null
  lora_format: 'single' | 'paired_hi_lo'
  /** How many images an edit reads after its source (0: none). */
  max_refs: number
  /** References are scaled down to this and never up; null: to the output's pixel count. */
  ref_max_pixels?: number | null
  size_constraints: {
    multiple_of: number
    min_pixels: number
    max_pixels: number
    presets: [number, number][]
  }
}

export interface Family {
  id: string
  label: string
  media: 'image' | 'video'
  lora_format: 'single' | 'paired_hi_lo'
  /** Whether jobs take ControlNet units (SDXL). */
  supports_control: boolean
  /** Whether jobs take image prompts (IP-Adapter for SDXL, Redux for FLUX.1). */
  supports_image_prompts: boolean
  /** What those image prompts can do. */
  image_prompt_options?: ImagePromptOptions | null
  variants: Variant[]
}

export interface ImagePromptOptions {
  /** Which UNet blocks a unit can act in. */
  purposes: Purpose[]
  /** Limited to an area of the output. */
  areas: boolean
  /** Limited to a range of steps. */
  steps: boolean
  /** FaceID models. */
  faces: boolean
  /** Redux: how closely to follow the picture (`downsample`). */
  detail: boolean
}

export interface ParamProp {
  type: 'string' | 'integer' | 'number' | 'boolean'
  title?: string
  /** A line under a checkbox's title, or under a select. */
  description?: string
  default?: string | number | boolean
  minimum?: number
  maximum?: number
  multipleOf?: number
  minLength?: number
  enum?: string[]
  'x-enum-labels'?: string[]
  'x-widget'?: 'prompt' | 'slider' | 'number' | 'select' | 'seed' | 'aspect'
  'x-advanced'?: boolean
  /** Slider step when it isn't `multipleOf` (Wan frame counts go 17, 21, 25…). */
  'x-step'?: number
}

export interface ParamSchema {
  type: 'object'
  required?: string[]
  properties: Record<string, ParamProp>
}

/** Optional YAML next to an asset in Drive (design §5). */
export interface Sidecar {
  label?: string
  trigger_words?: string[]
  default_weight?: number
  notes?: string
  /** Wan 2.2: the variants a LoRA is for. */
  variants?: string[]
  /** Wan 2.2 A14B: the high- and low-noise files of a pair, in the same folder. */
  pair?: { high: string; low: string }
  /** ControlNets: the kind of control image the model reads. */
  control?: TraceId
  /** Image prompt models (IP-Adapters): what they were trained to carry. */
  purpose?: AdapterKind
  /** Where it came from, e.g. its Civitai page. */
  source?: string
}

export interface Asset {
  path: string
  family: string | null
  kind: string
  size: number | null
  sidecar: Sidecar | null
  /** Blob sha of the preview image, served from /api/thumbs. */
  preview_thumb: string | null
  indexed_at: string
}

export interface CachedFile {
  path: string
  size: number
  last_used: number
}

export type SessionState = 'starting' | 'ready' | 'busy' | 'stopping' | 'stopped' | 'error'

export interface SessionSnapshot {
  session: {
    id: string
    gpu: string
    high_mem: boolean
    state: SessionState
    started_at: string
    ended_at: string | null
    last_activity_at: string
    error: string | null
  } | null
  step: string | null
  worker: {
    gpu: string | null
    vram_free: number | null
    vram_total: number | null
    disk_free: number
    /** The VM's model cache; absent from workers older than Phase 2. */
    cache?: { used: number; budget: number; files: CachedFile[] }
  } | null
  idle_deadline: string | null
  idle_timeout_min: number
  drive: {
    configured: boolean
    authorized: boolean
    error: string | null
    push_error: string | null
  }
  gpus: string[]
}

export interface DriveStatus {
  configured: boolean
  authorized: boolean
  error: string | null
  indexed_at: string | null
}

export type Params = Record<string, string | number | boolean | null>

export interface LoraRef {
  path: string
  weight: number
  size?: number | null
}

/** A Wan 2.2 A14B LoRA: one file per expert (either may be missing). */
export interface LoraPair {
  high?: LoraRef
  low?: LoraRef
}

export type LoraEntry = LoraRef | LoraPair

export const isPair = (l: LoraEntry): l is LoraPair => !('path' in l)

export type Fit = 'crop' | 'pad' | 'stretch'

export interface Inputs {
  /** `sha256:…` of the source image (i2i, edit, inpaint, outpaint, i2v). */
  source?: string
  /** Edit (and Qwen's inpaint): `sha256:…` of the images after the source, in reading order. */
  refs?: string[]
  /** `sha256:…` of the inpaint mask, painted over the source (white is redrawn). */
  mask?: string
  /** Outpaint: where the source sits on the canvas, in canvas pixels. */
  place?: { x: number; y: number; w: number; h: number }
  fit?: Fit
  /** `sha256:…` of the clip a video extension continues. */
  extends?: string
  transforms?: Record<string, { original: string; ops: Op[] }>
}

/** The preprocessors that trace a control image from a picture. */
export type TraceId = 'depth' | 'pose' | 'canny'

/** A ControlNet unit in a job spec (design §6.4). */
export interface ControlSpec {
  controlnet: { path: string; size?: number | null }
  /** `sha256:…` of the control image the ControlNet reads. */
  image: string
  fit?: Fit
  scale: number
  /** The fraction of the steps the unit guides, from start to end. */
  start: number
  end: number
  /** `sha256:…` of the area it's limited to, painted over the control image. */
  mask?: string
  /** How the control image was traced, and from which picture. */
  preprocessor?: { id: TraceId; source: string; params: Params }
}

/** What an image prompt model was trained to carry: a picture's subject, a face (from CLIP,
 * or an InsightFace identity for FaceID), or its composition. */
export type AdapterKind = 'subject' | 'face' | 'faceid' | 'composition'

/** Which of the UNet's blocks an image prompt acts in. */
export type Purpose = 'all' | 'style' | 'layout' | 'style_layout'

/** An image prompt (IP-Adapter) in a job spec (docs/ip-adapter.md). */
export interface ImagePromptSpec {
  adapter: { path: string; size?: number | null }
  /** `sha256:…` of its pictures. */
  images: string[]
  fit?: Fit
  purpose: Purpose
  weight: number
  /** The fraction of the steps it acts on, from start to end. */
  start: number
  end: number
  /** `sha256:…` of the area of the output it's limited to. */
  mask?: string
  /** FaceID: how much of CLIP's reading of the face Plus v2 adds. */
  structure?: number
  /** FaceID: the weight of the LoRA the model carries. */
  lora_weight?: number
  /** Redux: its 27 × 27 grid of tokens is shrunk by this factor (1 to 5). */
  downsample?: number
}

export interface Spec {
  family: string
  variant: string
  mode: string
  model: { path: string; size?: number | null }
  loras?: LoraEntry[]
  params: Params
  inputs?: Inputs
  control?: ControlSpec[]
  image_prompts?: ImagePromptSpec[]
}

/** A transform operation (design §6.5). Rotation is clockwise. */
export type Op =
  | { op: 'rotate'; deg: 90 | 180 | 270 }
  | { op: 'flip_h' }
  | { op: 'flip_v' }
  | { op: 'crop'; x: number; y: number; w: number; h: number }
  | { op: 'resize'; w: number; h: number; filter?: string }
  | { op: 'pad'; w: number; h: number }
  | { op: 'paste'; x: number; y: number; w: number; h: number }

/** A stored image or video, as returned by uploads, imports, frames and transforms. */
export interface BlobInfo {
  sha256: string
  media_type: string
  width: number | null
  height: number | null
  duration?: number | null
}

export interface Progress {
  job: string
  item: number
  phase: string
  step: number
  steps: number
  /** The Drive asset being copied, during the `copy` phase. */
  asset?: string | null
}

export type JobStatus = 'queued' | 'running' | 'done' | 'cancelled' | 'error'

export type SeedMode = 'increment' | 'random'

export interface Job {
  id: string
  status: JobStatus
  /** Order in the queue: lower runs sooner. */
  queue_position: number
  spec: Spec
  seeds: number[]
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: string | null
  progress: Progress | null
}

/** The replayable config a kept image carries (design §6.4): a spec plus its runtime. */
export interface SavedConfig extends Spec {
  degas_version: number
  /** A stitched video: the configs of its clips, oldest first. */
  segments?: SavedConfig[]
  runtime?: { gpu: string | null; duration_s?: number; diffusers?: string; torch?: string }
}

export interface LibraryItem {
  id: string
  kind: 'image' | 'video'
  blob_sha: string
  media_type: string
  width: number | null
  height: number | null
  duration: number | null
  config: SavedConfig
  title: string | null
  tags: string[]
  created_at: string
  source_result_id: string | null
}

export interface SavedPrompt {
  id: string
  name: string
  prompt: string
  negative_prompt: string
  family: string | null
  tags: string[]
  created_at: string
}

export interface Result {
  id: string
  job_id: string
  item_index: number
  blob_sha: string
  media_type: string
  seed: number | null
  width: number | null
  height: number | null
  duration: number | null
  /** A stitched video extension: the configs of the clips it chains. */
  segments: SavedConfig[] | null
  created_at: string
  expires_at: string | null
  /** The library item keeping this result, if it was kept. */
  library_id: string | null
  spec: Spec | null
}

/** What importing a Civitai link would do. */
export interface CivitaiPlan {
  model_name: string
  version_name: string
  base_model: string
  family: string
  label: string
  trigger_words: string[]
  weight: number
  files: { civitai_name: string; path: string; size: number; half: 'high' | 'low' | null }[]
  warnings: string[]
}

/** The latest Civitai import; `import` events follow it. */
export interface CivitaiImport {
  id: string
  label: string
  family: string
  paths: string[]
  state: 'copying' | 'finishing' | 'done' | 'failed'
  done: number
  total: number
  error: string | null
  warnings: string[]
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const raw = body instanceof Blob
  const res = await fetch(`/api${path}`, {
    method,
    headers:
      body === undefined
        ? undefined
        : { 'Content-Type': raw ? body.type || 'application/octet-stream' : 'application/json' },
    body: body === undefined ? undefined : raw ? body : JSON.stringify(body),
  })
  if (!res.ok) {
    let message = `HTTP ${String(res.status)}`
    try {
      const data = (await res.json()) as { detail?: unknown }
      if (typeof data.detail === 'string') message = data.detail
    } catch {
      // not JSON
    }
    throw new ApiError(res.status, message)
  }
  return (await res.json()) as T
}

export const api = {
  families: () => request<Family[]>('GET', '/families'),
  schema: (family: string, variant: string, mode: string) =>
    request<ParamSchema>(
      'GET',
      `/families/${family}/schema?variant=${encodeURIComponent(variant)}&mode=${encodeURIComponent(mode)}`,
    ),
  assets: () => request<Asset[]>('GET', '/assets'),
  rescan: () => request<{ count: number; indexed_at: string }>('POST', '/assets/rescan'),
  drive: () => request<DriveStatus>('GET', '/drive'),
  civitaiPlan: (url: string) => request<CivitaiPlan>('POST', '/civitai/plan', { url }),
  civitaiImport: (url: string) => request<CivitaiImport>('POST', '/civitai/import', { url }),
  civitaiImportState: () => request<CivitaiImport | null>('GET', '/civitai/import'),
  session: () => request<SessionSnapshot>('GET', '/session'),
  startSession: (gpu: string, highMem: boolean) =>
    request<SessionSnapshot>('POST', '/session', { gpu, high_mem: highMem }),
  stopSession: () => request<SessionSnapshot>('DELETE', '/session'),
  touchSession: () => request<SessionSnapshot>('POST', '/session/touch'),
  resetWorker: () => request<SessionSnapshot>('POST', '/session/reset-worker'),
  jobs: () => request<Job[]>('GET', '/jobs'),
  submitJob: (spec: Spec, batchCount: number, seedMode: SeedMode) =>
    request<Job>('POST', '/jobs', { spec, batch_count: batchCount, seed_mode: seedMode }),
  cancelJob: (id: string) => request<{ cancelled: boolean }>('DELETE', `/jobs/${id}`),
  /** Move a queued job to `position` in the queue (0 runs next). */
  moveJob: (id: string, position: number) => request<Job>('PATCH', `/jobs/${id}`, { position }),
  /** Undo cancelling a queued job. */
  restoreJob: (id: string) => request<Job>('POST', `/jobs/${id}/restore`),
  pushKey: () => request<{ public_key: string }>('GET', '/push'),
  pushSubscribe: (sub: PushSubscriptionJSON) =>
    request<{ subscribed: boolean }>('POST', '/push/subscribe', sub),
  pushUnsubscribe: (endpoint: string) =>
    request<{ unsubscribed: boolean }>('POST', '/push/unsubscribe', { endpoint }),
  results: (cursor?: string) =>
    request<{ results: Result[]; cursor: string | null }>(
      'GET',
      `/results${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''}`,
    ),
  /** Delete finished jobs and their results now; kept images stay in the library. */
  clearResults: () => request<{ results: number; jobs: number }>('DELETE', '/results'),
  /** Delete one finished job's results: its stitched chain with `chain`, else the rest. */
  deleteJobResults: (jobId: string, chain: boolean) =>
    request<{ results: number; jobs: number }>(
      'DELETE',
      `/jobs/${jobId}/results${chain ? '?chain=true' : ''}`,
    ),
  keep: (resultId: string) => request<LibraryItem>('POST', `/results/${resultId}/save`),
  library: (q: string, cursor?: string) =>
    request<{ items: LibraryItem[]; cursor: string | null }>(
      'GET',
      `/library${query({ q, cursor })}`,
    ),
  editLibraryItem: (id: string, edit: { title?: string; tags?: string[] }) =>
    request<LibraryItem>('PATCH', `/library/${id}`, edit),
  deleteLibraryItem: (id: string) => request<{ deleted: boolean }>('DELETE', `/library/${id}`),
  prompts: (q = '') => request<SavedPrompt[]>('GET', `/prompts${query({ q })}`),
  savePrompt: (p: { prompt: string; negative_prompt: string; family: string | null }) =>
    request<SavedPrompt>('POST', '/prompts', p),
  editPrompt: (id: string, edit: { name?: string; tags?: string[] }) =>
    request<SavedPrompt>('PATCH', `/prompts/${id}`, edit),
  deletePrompt: (id: string) => request<{ deleted: boolean }>('DELETE', `/prompts/${id}`),
  upload: (file: Blob) => request<BlobInfo>('POST', '/blobs', file),
  fromUrl: (url: string) => request<BlobInfo>('POST', '/blobs/from-url', { url }),
  frame: (sha: string, at: 'first' | 'last' | number) =>
    request<BlobInfo>('POST', `/blobs/${sha}/frame`, { at }),
  transform: (sha: string, ops: Op[]) =>
    request<BlobInfo>('POST', `/blobs/${sha}/transform`, { ops }),
  getTransform: (sha: string) =>
    request<{ original: string; ops: Op[] }>('GET', `/blobs/${sha}/transform`),
  /** Store a mask painted over image `source` (a PNG; alpha or white is redrawn). */
  uploadMask: (source: string, png: Blob) =>
    request<BlobInfo>('POST', `/blobs/${source}/mask`, png),
  /** Carry a mask from the image it was painted on to a new crop of that image. */
  remapMask: (mask: string, source: string, to: string) =>
    request<BlobInfo & { empty: boolean }>('POST', `/blobs/${mask}/remap`, { source, to }),
  /** SAM 3: candidate masks for taps (`include` false leaves out) and/or a description. */
  select: (image: string, params: { points: SelectPoint[]; text?: string }) =>
    request<Selection>('POST', '/preprocess', { id: 'sam', image, params }),
  /** Trace a control image (a depth map, a pose, edges) from a picture, at its size. */
  trace: (id: TraceId, image: string, params: Params) =>
    request<{ image: BlobInfo }>('POST', '/preprocess', { id, image, params }),
  /** The aligned crop of a picture's main face, which a FaceID image prompt reads. */
  findFace: (image: string) =>
    request<{ image: BlobInfo; faces: number }>('POST', '/preprocess', {
      id: 'face',
      image,
      params: {},
    }),
  extendResult: (id: string) => request<Extension>('POST', `/results/${id}/extend`),
  extendLibraryItem: (id: string) => request<Extension>('POST', `/library/${id}/extend`),
}

/** A tap for SAM, in the image's pixels. */
export interface SelectPoint {
  x: number
  y: number
  include: boolean
}

/** SAM's answer: masks from smallest to largest, and the one it rates best. */
export interface Selection {
  candidates: (BlobInfo & { score: number | null })[]
  chosen: number | null
}

/** A spec continuing a clip from its last frame, and that frame. */
export interface Extension {
  spec: Spec
  source: BlobInfo
}

/** "?q=cat&cursor=…", leaving out empty values. */
function query(params: Record<string, string | undefined>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v) q.set(k, v)
  const s = q.toString()
  return s ? `?${s}` : ''
}

export function isActive(snapshot: SessionSnapshot | undefined): boolean {
  const state = snapshot?.session?.state
  return state === 'starting' || state === 'ready' || state === 'busy'
}

export const blobUrl = (sha: string) => `/api/blobs/${sha}`
export const thumbUrl = (sha: string) => `/api/thumbs/${sha}`
export const isVideo = (mediaType: string) => mediaType.startsWith('video/')
export const unref = (value: string) => value.replace(/^sha256:/, '')
