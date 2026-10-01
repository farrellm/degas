import type { Asset, Job, Result, Spec } from '@/api/types'
import { assetLabel } from '@/lib/assets'
import { isVideo } from '@/lib/image'

// The Results feed's groups: one per job (and one per stitched chain), in queue order.

/** One job's worth of results: the contact-sheet row under a prompt caption. */
export interface Group {
  id: string
  jobId: string
  spec: Spec | null
  job: Job | undefined
  results: Result[]
  at: string
  /** A group of stitched video extensions, apart from the clips that made them. */
  chain: boolean
}

export const PENDING = new Set(['queued', 'running'])

/** "Studio XL v10 + 2 LoRAs" for a group caption. */
export function modelLine(spec: Spec, assets: Asset[] | undefined): string {
  const n = spec.loras?.length ?? 0
  const model = assetLabel(spec.model.path, assets)
  return n === 0 ? model : `${model} + ${String(n)} ${n === 1 ? 'LoRA' : 'LoRAs'}`
}

export function buildGroups(jobs: Job[], results: Result[]): Group[] {
  const byId = new Map<string, Group>()
  for (const job of jobs) {
    if (job.status === 'done' || job.status === 'cancelled') continue
    byId.set(job.id, {
      id: job.id,
      jobId: job.id,
      spec: job.spec,
      job,
      results: [],
      at: job.created_at,
      chain: false,
    })
  }
  for (const r of results) {
    const chain = !!r.segments
    const id = chain ? `${r.job_id}:chain` : r.job_id
    let g = byId.get(id)
    if (!g) {
      const job = jobs.find((j) => j.id === r.job_id)
      g = {
        id,
        jobId: r.job_id,
        spec: r.spec ?? job?.spec ?? null,
        job,
        results: [],
        at: r.created_at,
        chain,
      }
      byId.set(id, g)
    }
    g.results.push(r)
    if (r.created_at > g.at) g.at = r.created_at
  }
  const rank = (g: Group) =>
    g.chain ? 2 : g.job?.status === 'running' ? 0 : g.job?.status === 'queued' ? 1 : 2
  const queueOrder = (a: Group, b: Group) =>
    rank(a) === 1 && rank(b) === 1 ? (a.job?.queue_position ?? 0) - (b.job?.queue_position ?? 0) : 0
  return [...byId.values()]
    .map((g) => ({ ...g, results: g.results.toSorted((a, b) => a.item_index - b.item_index) }))
    .toSorted(
      (a, b) => rank(a) - rank(b) || queueOrder(a, b) || (a.at < b.at ? 1 : a.at > b.at ? -1 : 0),
    )
}

/** The jobs cache with `id` moved to `position` among the queued jobs. */
export function reorder(jobs: Job[], id: string, position: number): Job[] {
  const queued = jobs
    .filter((j) => j.status === 'queued')
    .toSorted((a, b) => a.queue_position - b.queue_position)
  const slots = queued.map((j) => j.queue_position)
  const ids = queued.map((j) => j.id).filter((x) => x !== id)
  ids.splice(position, 0, id)
  const at = new Map(ids.map((x, i) => [x, slots[i] ?? 0]))
  return jobs.map((j) => (at.has(j.id) ? { ...j, queue_position: at.get(j.id) ?? 0 } : j))
}

/** "Delete these 4 images? Kept ones stay in the library." */
export function deleteQuestion(results: Result[]): string {
  const n = results.length
  if (n === 0) return 'Delete this failed job?'
  const noun = results.every((r) => isVideo(r.media_type)) ? 'clip' : 'image'
  const what = n === 1 ? `this ${noun}` : `these ${String(n)} ${noun}s`
  const kept = results.filter((r) => r.library_id).length
  const note =
    kept === 0 ? '' : kept === n ? ' They stay in the library.' : ' Kept ones stay in the library.'
  return `Delete ${what}?${note}`
}
