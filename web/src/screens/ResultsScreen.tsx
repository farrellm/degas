import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState, type CSSProperties } from 'react'
import { api, thumbUrl, type Asset, type Job, type Result, type Spec } from '../api'
import { assetLabel, useAssets } from '../assets'
import { SaveToPhotos, Viewer } from '../components/Viewer'
import { draftFromSpec } from '../draft'
import { copyText, itemFraction, phaseText, size } from '../format'
import { hoursLeft, shortTime, timeLeft, useNow } from '../time'

/** One job's worth of results: the contact-sheet row under a prompt caption. */
interface Group {
  id: string
  spec: Spec | null
  job: Job | undefined
  results: Result[]
  at: string
}

const PENDING = new Set(['queued', 'running'])

/** "Studio XL v10 + 2 LoRAs" for a group caption. */
function modelLine(spec: Spec, assets: Asset[] | undefined): string {
  const n = spec.loras?.length ?? 0
  const model = assetLabel(spec.model.path, assets)
  return n === 0 ? model : `${model} + ${String(n)} ${n === 1 ? 'LoRA' : 'LoRAs'}`
}

// Unkept images closer than this to deletion get a warning colour.
const SOON_HOURS = 2

function buildGroups(jobs: Job[], results: Result[]): Group[] {
  const byId = new Map<string, Group>()
  for (const job of jobs) {
    if (job.status === 'done' || job.status === 'cancelled') continue
    byId.set(job.id, { id: job.id, spec: job.spec, job, results: [], at: job.created_at })
  }
  for (const r of results) {
    let g = byId.get(r.job_id)
    if (!g) {
      const job = jobs.find((j) => j.id === r.job_id)
      g = { id: r.job_id, spec: r.spec ?? job?.spec ?? null, job, results: [], at: r.created_at }
      byId.set(r.job_id, g)
    }
    g.results.push(r)
    if (r.created_at > g.at) g.at = r.created_at
  }
  const rank = (g: Group) => (g.job?.status === 'running' ? 0 : g.job?.status === 'queued' ? 1 : 2)
  return [...byId.values()]
    .map((g) => ({ ...g, results: g.results.toSorted((a, b) => a.item_index - b.item_index) }))
    .toSorted((a, b) => rank(a) - rank(b) || (a.at < b.at ? 1 : a.at > b.at ? -1 : 0))
}

export function ResultsScreen({
  onRemix,
  onCreate,
}: {
  onRemix: () => void
  onCreate: () => void
}) {
  const qc = useQueryClient()
  const now = useNow(30_000)
  const assets = useAssets()
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const results = useQuery({ queryKey: ['results'], queryFn: () => api.results() })
  const cancel = useMutation({
    mutationFn: api.cancelJob,
    onSettled: () => qc.invalidateQueries({ queryKey: ['jobs'] }),
  })
  const [open, setOpen] = useState<string | null>(null)
  const keep = useMutation({
    mutationFn: async (r: Result) => {
      await (r.library_id ? api.deleteLibraryItem(r.library_id) : api.keep(r.id))
    },
    onSettled: () =>
      Promise.all([
        qc.invalidateQueries({ queryKey: ['results'] }),
        qc.invalidateQueries({ queryKey: ['library'] }),
      ]),
  })

  if (jobs.isPending || results.isPending) return <p className="loading">Loading…</p>
  if (jobs.error ?? results.error) {
    return <p role="alert">{(jobs.error ?? results.error)?.message}</p>
  }

  const groups = buildGroups(jobs.data, results.data.results)
  const flat = groups.flatMap((g) => g.results)
  const openIndex = flat.findIndex((r) => r.id === open)

  if (groups.length === 0) {
    return (
      <div className="feed-empty">
        <div className="tile sketch" aria-hidden />
        <p className="lede">Nothing here yet.</p>
        <p>
          Images appear here as they finish. Images you don’t keep are deleted 24 hours after the
          GPU session ends.
        </p>
        <button type="button" className="btn" onClick={onCreate}>
          Write a prompt
        </button>
      </div>
    )
  }

  return (
    <>
      <div className="feed">
        {groups.map((g) => (
          <GroupView
            key={g.id}
            group={g}
            now={now}
            assets={assets.data}
            onOpen={setOpen}
            onCancel={(id) => {
              cancel.mutate(id)
            }}
          />
        ))}
      </div>
      {openIndex >= 0 && (
        <Viewer
          items={flat}
          assets={assets.data}
          index={openIndex}
          onIndex={(i) => {
            setOpen(flat[i]?.id ?? null)
          }}
          onClose={() => {
            setOpen(null)
          }}
          actions={(r) => (
            <>
              <button
                type="button"
                className={r.library_id ? 'btn quiet kept wide' : 'btn wide'}
                aria-pressed={!!r.library_id}
                disabled={keep.isPending}
                onClick={() => {
                  keep.mutate(r)
                }}
              >
                {r.library_id ? 'Kept' : 'Keep'}
              </button>
              <SaveToPhotos item={r} />
              <button
                type="button"
                className="btn quiet"
                disabled={!r.spec}
                onClick={() => {
                  if (r.spec) draftFromSpec(r.spec, r.seed)
                  onRemix()
                }}
              >
                Remix
              </button>
              {keep.error && (
                <p className="viewer-note" role="alert">
                  {keep.error.message}
                </p>
              )}
            </>
          )}
        />
      )}
    </>
  )
}

function GroupView({
  group,
  now,
  assets,
  onOpen,
  onCancel,
}: {
  group: Group
  now: number
  assets: Asset[] | undefined
  onOpen: (id: string) => void
  onCancel: (jobId: string) => void
}) {
  const { spec, job, results } = group
  const params = spec?.params ?? {}
  const prompt = String(params.prompt ?? '').trim()
  const w = Number(params.width ?? results[0]?.width ?? 1)
  const h = Number(params.height ?? results[0]?.height ?? 1)
  const style = {
    '--ratio': `${String(w)} / ${String(h)}`,
    '--cols': w > h ? 2 : 3,
  } as CSSProperties
  const pending = job && PENDING.has(job.status)
  const expiresAt = results[0]?.expires_at ?? null
  const allKept = results.length > 0 && results.every((r) => r.library_id)
  const expiry = allKept ? 'Kept' : timeLeft(expiresAt, now)
  const soon = !allKept && expiresAt !== null && hoursLeft(expiresAt, now) < SOON_HOURS
  const total = job?.seeds.length ?? results.length
  const done = new Set(results.map((r) => r.item_index))

  return (
    <section aria-label={prompt || 'Untitled'}>
      <header className="group-caption">
        <p className={prompt ? 'title' : 'title untitled'}>{prompt || 'No prompt'}</p>
        <p className="meta">{spec ? `${modelLine(spec, assets)}, ${size(w, h)}` : size(w, h)}</p>
        <div className="aside">
          {pending ? (
            <button
              type="button"
              className="btn quiet small"
              onClick={() => {
                onCancel(group.id)
              }}
            >
              Cancel
            </button>
          ) : (
            <span className={soon ? 'soon' : undefined}>{expiry ?? shortTime(group.at, now)}</span>
          )}
        </div>
      </header>
      {job?.status === 'error' && job.error && <p className="group-error">{job.error}</p>}
      <div className="contact" style={style}>
        {Array.from({ length: Math.max(total, results.length) }, (_, i) => {
          const r = results.find((x) => x.item_index === i)
          if (r) {
            return (
              <button
                key={r.id}
                type="button"
                className={r.library_id ? 'tile kept' : 'tile'}
                onClick={() => {
                  onOpen(r.id)
                }}
                aria-label={`Open image ${String(i + 1)}, seed ${String(r.seed)}${r.library_id ? ', kept' : ''}`}
              >
                <img src={thumbUrl(r.blob_sha)} alt="" loading="lazy" />
              </button>
            )
          }
          if (!pending) return null
          return <SketchTile key={`s${String(i)}`} job={job} item={i} done={done} />
        })}
      </div>
    </section>
  )
}

/** A tile that isn't finished yet, drawn in pastel hatching as it denoises. */
function SketchTile({ job, item, done }: { job: Job; item: number; done: Set<number> }) {
  const current = job.status === 'running' && (job.progress?.item ?? 0) === item && !done.has(item)
  const fraction = current ? itemFraction(job) : null
  const cls = !current ? 'waiting' : fraction === null ? 'indeterminate' : ''
  const label = current
    ? phaseText(job)
    : job.status !== 'queued'
      ? 'Waiting'
      : job.progress?.phase === 'copy'
        ? `Queued, copying ${copyText(job)}` // prefetched while another job runs
        : 'Queued'
  return (
    <div
      className={`tile sketch ${cls}`}
      style={{ '--p': fraction ?? 0 } as CSSProperties}
      role="img"
      aria-label={`Image ${String(item + 1)}: ${label}`}
    >
      {(current || item === 0) && <span className="sketch-label">{label}</span>}
    </div>
  )
}
