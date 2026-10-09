// The server's event stream (/api/events), checked as each event arrives: what one carries goes
// straight into the query cache. Objects are loose, keeping fields beyond the ones checked.

import * as v from 'valibot'

import type {
  CachedFile,
  CivitaiImport,
  Job,
  JobStatus,
  Params,
  Progress,
  SessionSnapshot,
  SessionState,
  Spec,
} from './types'

const SESSION_STATES: SessionState[] = ['starting', 'ready', 'busy', 'stopping', 'stopped', 'error']
const JOB_STATUSES: JobStatus[] = ['queued', 'running', 'done', 'cancelled', 'error']
const IMPORT_STATES: CivitaiImport['state'][] = ['copying', 'finishing', 'done', 'failed']

const params: v.GenericSchema<unknown, Params> = v.record(
  v.string(),
  v.union([v.string(), v.number(), v.boolean(), v.null()]),
)

const cachedFile: v.GenericSchema<unknown, CachedFile> = v.looseObject({
  path: v.string(),
  size: v.number(),
  last_used: v.number(),
})

const sessionEntries = {
  session: v.nullable(
    v.looseObject({
      id: v.string(),
      gpu: v.string(),
      high_mem: v.boolean(),
      state: v.picklist(SESSION_STATES),
      started_at: v.string(),
      ended_at: v.nullable(v.string()),
      last_activity_at: v.string(),
      error: v.nullable(v.string()),
    }),
  ),
  step: v.nullable(v.string()),
  worker: v.nullable(
    v.looseObject({
      gpu: v.nullable(v.string()),
      vram_free: v.nullable(v.number()),
      vram_total: v.nullable(v.number()),
      disk_free: v.number(),
      cache: v.optional(
        v.looseObject({ used: v.number(), budget: v.number(), files: v.array(cachedFile) }),
      ),
    }),
  ),
  idle_deadline: v.nullable(v.string()),
  idle_timeout_min: v.number(),
  drive: v.looseObject({
    configured: v.boolean(),
    authorized: v.boolean(),
    error: v.nullable(v.string()),
    push_error: v.nullable(v.string()),
  }),
  gpus: v.array(v.string()),
}

const progressEntries = {
  job: v.string(),
  item: v.number(),
  phase: v.string(),
  step: v.number(),
  steps: v.number(),
  asset: v.nullish(v.string()),
}
const progress: v.GenericSchema<unknown, Progress> = v.looseObject(progressEntries)

// The spec is the server's to check (it validated it at submit): only its outline is checked
// here, what every job's row and tile reads.
const spec = v.pipe(
  v.looseObject({
    family: v.string(),
    variant: v.string(),
    mode: v.string(),
    model: v.looseObject({ path: v.string() }),
    params,
  }),
  v.transform((s) => s as Spec),
)

const job: v.GenericSchema<unknown, Job> = v.looseObject({
  id: v.string(),
  status: v.picklist(JOB_STATUSES),
  queue_position: v.number(),
  spec,
  seeds: v.array(v.number()),
  created_at: v.string(),
  started_at: v.nullable(v.string()),
  finished_at: v.nullable(v.string()),
  error: v.nullable(v.string()),
  progress: v.nullable(progress),
})

const civitaiImport: v.GenericSchema<unknown, CivitaiImport> = v.looseObject({
  id: v.string(),
  label: v.string(),
  family: v.string(),
  paths: v.array(v.string()),
  state: v.picklist(IMPORT_STATES),
  done: v.number(),
  total: v.number(),
  error: v.nullable(v.string()),
  warnings: v.array(v.string()),
})

const signal = <T extends string>(type: T) => v.looseObject({ type: v.literal(type) })

export type ServerEvent =
  | { type: 'hello' }
  | ({ type: 'session' } & SessionSnapshot)
  | { type: 'job'; job: Job }
  | ({ type: 'progress' } & Progress)
  | { type: 'result' }
  | { type: 'assets' }
  | { type: 'library' }
  | { type: 'prompts' }
  | { type: 'swept' }
  | { type: 'import'; import: CivitaiImport }

// Typed as the union above, so each branch is checked against the wire types at compile time.
const serverEvent: v.GenericSchema<unknown, ServerEvent> = v.variant('type', [
  signal('hello'),
  v.looseObject({ type: v.literal('session'), ...sessionEntries }),
  v.looseObject({ type: v.literal('job'), job }),
  v.looseObject({ type: v.literal('progress'), ...progressEntries }),
  signal('result'),
  signal('assets'),
  signal('library'),
  signal('prompts'),
  signal('swept'),
  v.looseObject({ type: v.literal('import'), import: civitaiImport }),
])

/** An event from the stream, or null when it isn't one this app knows how to read. */
export function parseEvent(data: string): ServerEvent | null {
  let json: unknown
  try {
    json = JSON.parse(data)
  } catch {
    return null
  }
  const result = v.safeParse(serverEvent, json)
  return result.success ? result.output : null
}
