// Typed client for the Degas server API (/api, design §7).

import type {
  Asset,
  BlobInfo,
  CivitaiImport,
  CivitaiPlan,
  DriveStatus,
  Extension,
  Family,
  Job,
  LibraryItem,
  Op,
  Params,
  ParamSchema,
  Result,
  SavedPrompt,
  SeedMode,
  Selection,
  SelectPoint,
  SessionSnapshot,
  Spec,
  TraceId,
} from './types'

export class ApiError extends Error {
  override readonly name = 'ApiError'
  readonly status: number

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
  /** Move LoRA files (both halves of a pair together) to Drive's trash. */
  deleteLoras: (paths: string[]) =>
    request<{ deleted: string[] }>(
      'DELETE',
      `/assets?${paths.map((p) => `path=${encodeURIComponent(p)}`).join('&')}`,
    ),
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

/** "?q=cat&cursor=…", leaving out empty values. */
function query(params: Record<string, string | undefined>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) if (v) q.set(k, v)
  const s = q.toString()
  return s ? `?${s}` : ''
}
