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

export interface Asset {
  path: string
  family: string | null
  kind: string
  size: number | null
  indexed_at: string
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

export interface Spec {
  family: string
  variant: string
  mode: string
  model: { path: string; size?: number | null }
  params: Params
}

export interface Progress {
  job: string
  item: number
  phase: string
  step: number
  steps: number
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
  assets: (family: string, kind: string) =>
    request<Asset[]>('GET', `/assets?family=${family}&kind=${kind}`),
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
