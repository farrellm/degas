// Typed client for the Degas server API (/api, design §7).

export interface Variant {
  id: string
  label: string
  min_gpu: string
  modes: string[]
  /** Drive folder of this variant's models, when the family has several variants. */
  model_dir: string | null
  lora_format: 'single' | 'paired_hi_lo'
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
  variants: Variant[]
}

export interface ParamProp {
  type: 'string' | 'integer' | 'number'
  title?: string
  default?: string | number
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

export type Params = Record<string, string | number | null>

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
  /** `sha256:…` of the source image (i2v). */
  source?: string
  fit?: Fit
  /** `sha256:…` of the clip a video extension continues. */
  extends?: string
  transforms?: Record<string, { original: string; ops: Op[] }>
}

export interface Spec {
  family: string
  variant: string
  mode: string
  model: { path: string; size?: number | null }
  loras?: LoraEntry[]
  params: Params
  inputs?: Inputs
}

/** A transform operation (design §6.5). Rotation is clockwise. */
export type Op =
  | { op: 'rotate'; deg: 90 | 180 | 270 }
  | { op: 'flip_h' }
  | { op: 'flip_v' }
  | { op: 'crop'; x: number; y: number; w: number; h: number }
  | { op: 'resize'; w: number; h: number; filter?: string }
  | { op: 'pad'; w: number; h: number }

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

export interface Job {
  id: string
  status: JobStatus
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
  session: () => request<SessionSnapshot>('GET', '/session'),
  startSession: (gpu: string, highMem: boolean) =>
    request<SessionSnapshot>('POST', '/session', { gpu, high_mem: highMem }),
  stopSession: () => request<SessionSnapshot>('DELETE', '/session'),
  touchSession: () => request<SessionSnapshot>('POST', '/session/touch'),
  resetWorker: () => request<SessionSnapshot>('POST', '/session/reset-worker'),
  jobs: () => request<Job[]>('GET', '/jobs'),
  submitJob: (spec: Spec, batchCount: number, seedMode: 'increment' | 'random') =>
    request<Job>('POST', '/jobs', { spec, batch_count: batchCount, seed_mode: seedMode }),
  cancelJob: (id: string) => request<{ cancelled: boolean }>('DELETE', `/jobs/${id}`),
  results: (cursor?: string) =>
    request<{ results: Result[]; cursor: string | null }>(
      'GET',
      `/results${cursor ? `?cursor=${encodeURIComponent(cursor)}` : ''}`,
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
  extendResult: (id: string) => request<Extension>('POST', `/results/${id}/extend`),
  extendLibraryItem: (id: string) => request<Extension>('POST', `/library/${id}/extend`),
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
