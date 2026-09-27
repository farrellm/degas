import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { api, blobUrl, thumbUrl, type Asset, type Job, type Result, type Spec } from '../api'
import { assetLabel, useAssets } from '../assets'
import { draftFromSpec } from '../draft'
import { copyText, itemFraction, phaseText, size } from '../format'
import { shortTime, timeLeft, useNow } from '../time'

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

/** "Studio XL v10, with Film Grain v3 at 0.8" for the wall label. */
function modelWithLoras(spec: Spec, assets: Asset[] | undefined): string {
  const model = assetLabel(spec.model.path, assets)
  const loras = (spec.loras ?? []).map(
    (l) => `${assetLabel(l.path, assets)} at ${String(Number(l.weight.toFixed(2)))}`,
  )
  return loras.length ? `${model}, with ${loras.join(' and ')}` : model
}

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
  onReuse,
  onCreate,
}: {
  onReuse: () => void
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
          Images appear here as they finish. Unsaved images are deleted 24 hours after the GPU
          session ends.
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
          results={flat}
          assets={assets.data}
          index={openIndex}
          onIndex={(i) => {
            setOpen(flat[i]?.id ?? null)
          }}
          onClose={() => {
            setOpen(null)
          }}
          onReuse={(r) => {
            if (r.spec) draftFromSpec(r.spec, r.seed)
            onReuse()
          }}
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
  const expiry = timeLeft(results[0]?.expires_at ?? null, now)
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
            <span>{expiry ?? shortTime(group.at, now)}</span>
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
                className="tile"
                onClick={() => {
                  onOpen(r.id)
                }}
                aria-label={`Open image ${String(i + 1)}, seed ${String(r.seed)}`}
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

function Viewer({
  results,
  assets,
  index,
  onIndex,
  onClose,
  onReuse,
}: {
  results: Result[]
  assets: Asset[] | undefined
  index: number
  onIndex: (i: number) => void
  onClose: () => void
  onReuse: (r: Result) => void
}) {
  const r = results[index]
  const ref = useRef<HTMLDivElement>(null)
  const swipe = useRef<number | null>(null)
  const [shareError, setShareError] = useState<string | null>(null)

  useEffect(() => {
    ref.current?.focus()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowRight' && index < results.length - 1) onIndex(index + 1)
      if (e.key === 'ArrowLeft' && index > 0) onIndex(index - 1)
    }
    document.addEventListener('keydown', onKey)
    const overflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = overflow
    }
  }, [index, results.length, onIndex, onClose])

  const spec = r?.spec
  const schema = useQuery({
    queryKey: ['schema', spec?.family, spec?.variant, spec?.mode],
    queryFn: () => api.schema(spec?.family ?? '', spec?.variant ?? '', spec?.mode ?? ''),
    enabled: !!spec,
    staleTime: Infinity,
  })

  if (!r) return null
  const params = spec?.params ?? {}
  const sampler = schema.data?.properties.scheduler
  const samplerIndex = sampler?.enum?.indexOf(String(params.scheduler)) ?? -1
  const samplerLabel = sampler?.['x-enum-labels']?.[samplerIndex] ?? String(params.scheduler)

  const share = async () => {
    setShareError(null)
    const blob = await (await fetch(blobUrl(r.blob_sha))).blob()
    const file = new File([blob], `degas-${String(r.seed)}.png`, { type: r.media_type })
    if ('canShare' in navigator && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file] })
    } else {
      const a = document.createElement('a')
      a.href = blobUrl(r.blob_sha)
      a.download = file.name
      a.click()
    }
  }

  return (
    <div
      ref={ref}
      className="viewer"
      role="dialog"
      aria-modal="true"
      aria-label="Image"
      tabIndex={-1}
    >
      <div className="viewer-bar">
        <button type="button" className="btn quiet small" onClick={onClose}>
          Close
        </button>
        <span className="position">
          {index + 1} of {results.length}
        </span>
        <span className="sheet-actions">
          <button
            type="button"
            className="btn quiet small"
            aria-label="Previous image"
            disabled={index === 0}
            onClick={() => {
              onIndex(index - 1)
            }}
          >
            ‹
          </button>
          <button
            type="button"
            className="btn quiet small"
            aria-label="Next image"
            disabled={index === results.length - 1}
            onClick={() => {
              onIndex(index + 1)
            }}
          >
            ›
          </button>
        </span>
      </div>
      <div
        className="viewer-image"
        onPointerDown={(e) => {
          swipe.current = e.clientX
        }}
        onPointerUp={(e) => {
          if (swipe.current === null) return
          const dx = e.clientX - swipe.current
          swipe.current = null
          if (dx < -50 && index < results.length - 1) onIndex(index + 1)
          if (dx > 50 && index > 0) onIndex(index - 1)
        }}
      >
        <img src={blobUrl(r.blob_sha)} alt={String(params.prompt ?? '')} draggable={false} />
      </div>
      <div className="wall-label">
        <div>
          <p className="title">{String(params.prompt ?? '') || 'No prompt'}</p>
          {params.negative_prompt ? (
            <p className="avoid">Negative: {String(params.negative_prompt)}</p>
          ) : null}
        </div>
        <p className="lines">
          <span>{r.spec ? modelWithLoras(r.spec, assets) : 'Unknown model'}</span>
          <span>
            {size(r.width, r.height)}, seed {r.seed}
          </span>
          <span>
            {String(params.steps)} steps, CFG {String(params.cfg)}, {samplerLabel}
          </span>
        </p>
        <div className="viewer-actions">
          <button
            type="button"
            className="btn"
            onClick={() => {
              void share().catch((e: unknown) => {
                if (!(e instanceof DOMException && e.name === 'AbortError')) {
                  setShareError('Saving failed. Try again, or long-press the image.')
                }
              })
            }}
          >
            Save to Photos
          </button>
          <button
            type="button"
            className="btn quiet"
            disabled={!r.spec}
            onClick={() => {
              onReuse(r)
            }}
          >
            Reuse settings
          </button>
        </div>
        {shareError && <p role="alert">{shareError}</p>}
      </div>
    </div>
  )
}
