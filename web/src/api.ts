// Typed client for the Degas server API (/api, design §7).

export interface Variant {
  id: string
  label: string
  min_gpu: string
  modes: string[]
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

export interface Spec {
  family: string
  variant: string
  mode: string
  model: { path: string; size?: number | null }
  loras?: LoraRef[]
  params: Params
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

export interface Result {
  id: string
  job_id: string
  item_index: number
  blob_sha: string
  media_type: string
  seed: number | null
  width: number | null
  height: number | null
  created_at: string
  expires_at: string | null
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
  const res = await fetch(`/api${path}`, {
    method,
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
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
}

export function isActive(snapshot: SessionSnapshot | undefined): boolean {
  const state = snapshot?.session?.state
  return state === 'starting' || state === 'ready' || state === 'busy'
}

export const blobUrl = (sha: string) => `/api/blobs/${sha}`
export const thumbUrl = (sha: string) => `/api/thumbs/${sha}`
