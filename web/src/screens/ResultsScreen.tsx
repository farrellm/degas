import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent,
  type PointerEvent,
} from 'react'
import { api, isVideo, type Asset, type Job, type Result, type Spec } from '../api'
import { assetLabel, useAssets } from '../assets'
import { CoveredText } from '../components/CoveredText'
import { Tile } from '../components/Tile'
import { SaveToPhotos, Viewer } from '../components/Viewer'
import { draftFromSpec, draftWithSource, useSourceTarget } from '../draft'
import { copyText, duration, itemFraction, phaseText, size } from '../format'
import { hoursLeft, shortTime, timeLeft, useNow } from '../time'

/** One job's worth of results: the contact-sheet row under a prompt caption. */
interface Group {
  id: string
  jobId: string
  spec: Spec | null
  job: Job | undefined
  results: Result[]
  at: string
  /** A group of stitched video extensions, apart from the clips that made them. */
  chain: boolean
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
function reorder(jobs: Job[], id: string, position: number): Job[] {
  const queued = jobs
    .filter((j) => j.status === 'queued')
    .toSorted((a, b) => a.queue_position - b.queue_position)
  const slots = queued.map((j) => j.queue_position)
  const ids = queued.map((j) => j.id).filter((x) => x !== id)
  ids.splice(position, 0, id)
  const at = new Map(ids.map((x, i) => [x, slots[i] ?? 0]))
  return jobs.map((j) => (at.has(j.id) ? { ...j, queue_position: at.get(j.id) ?? 0 } : j))
}

/** A queued group being dragged: where it started, and where it would land. */
interface Drag {
  id: string
  from: number
  to: number
  startY: number
  dy: number
  /** Vertical middles of the other queued groups, in queue order, at lift time. */
  mids: number[]
}

const UNDO_MS = 5000

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
  const [undo, setUndo] = useState<string | null>(null)
  const cancel = useMutation({
    mutationFn: (job: Job) => api.cancelJob(job.id),
    onSuccess: (_, job) => {
      // A job that hasn't started can come back; a running one is already stopping.
      setUndo(job.status === 'queued' ? job.id : null)
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ['jobs'] }),
  })
  const restore = useMutation({
    mutationFn: api.restoreJob,
    onSuccess: () => {
      setUndo(null)
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ['jobs'] }),
  })
  const move = useMutation({
    mutationFn: ({ id, position }: { id: string; position: number }) => api.moveJob(id, position),
    onMutate: ({ id, position }) => {
      qc.setQueryData<Job[]>(['jobs'], (js) => js && reorder(js, id, position))
    },
    onSettled: () => qc.invalidateQueries({ queryKey: ['jobs'] }),
  })
  const [drag, setDrag] = useState<Drag | null>(null)
  const [announce, setAnnounce] = useState('')
  const sections = useRef(new Map<string, HTMLElement>())
  const refocus = useRef<string | null>(null)

  useEffect(() => {
    if (undo === null) return
    const id = setTimeout(() => {
      setUndo(null)
    }, UNDO_MS)
    return () => {
      clearTimeout(id)
    }
  }, [undo])

  // Keep focus on a handle moved with the arrow keys as its group changes place.
  useEffect(() => {
    const id = refocus.current
    if (!id) return
    refocus.current = null
    sections.current.get(id)?.querySelector<HTMLElement>('.drag-handle')?.focus()
  })
  const [open, setOpen] = useState<string | null>(null)
  const sourceTarget = useSourceTarget()
  const extend = useMutation({
    mutationFn: (r: Result) => api.extendResult(r.id),
    onSuccess: (ext) => {
      draftFromSpec(ext.spec, null, ext.source)
      onRemix()
    },
  })
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

  const clear = useMutation({
    mutationFn: api.clearResults,
    onSuccess: () => {
      setOpen(null)
    },
    onSettled: () =>
      Promise.all([
        qc.invalidateQueries({ queryKey: ['results'] }),
        qc.invalidateQueries({ queryKey: ['jobs'] }),
      ]),
  })

  const remove = useMutation({
    mutationFn: (g: Group) => api.deleteJobResults(g.jobId, g.chain),
    onSuccess: (_, g) => {
      if (g.results.some((r) => r.id === open)) setOpen(null)
    },
    onSettled: () =>
      Promise.all([
        qc.invalidateQueries({ queryKey: ['results'] }),
        qc.invalidateQueries({ queryKey: ['jobs'] }),
      ]),
  })

  if (jobs.isPending || results.isPending) return <p className="loading">Loading…</p>
  if (jobs.error ?? results.error) {
    return <p role="alert">{(jobs.error ?? results.error)?.message}</p>
  }

  const groups = buildGroups(jobs.data, results.data.results)
  const flat = groups.flatMap((g) => g.results)
  const openIndex = flat.findIndex((r) => r.id === open)
  const queue = groups.filter((g) => !g.chain && g.job?.status === 'queued').map((g) => g.id)
  const finished = groups.some((g) => !['queued', 'running'].includes(g.job?.status ?? ''))

  const moveTo = (id: string, position: number) => {
    const n = queue.length
    setAnnounce(`Moved to ${String(position + 1)} of ${String(n)} in the queue.`)
    move.mutate({ id, position })
  }

  const lift = (id: string, y: number) => {
    const others = queue.filter((q) => q !== id)
    const mids = others.map((q) => {
      const r = sections.current.get(q)?.getBoundingClientRect()
      return r ? r.top + r.height / 2 : 0
    })
    const from = queue.indexOf(id)
    setDrag({ id, from, to: from, startY: y, dy: 0, mids })
  }

  const dragTo = (y: number) => {
    setDrag((d) => d && { ...d, dy: y - d.startY, to: d.mids.filter((m) => m < y).length })
  }

  const drop = () => {
    if (drag && drag.to !== drag.from) moveTo(drag.id, drag.to)
    setDrag(null)
  }

  // Where the drop mark goes: before the group now at `to`, or after the last one.
  const others = drag ? queue.filter((q) => q !== drag.id) : []
  const markBefore = drag && drag.to !== drag.from ? others[drag.to] : undefined
  const markAfter =
    drag && drag.to !== drag.from && drag.to === others.length ? others.at(-1) : undefined

  const undoToast = undo && (
    <p className="undo-toast" role="status">
      <span>Cancelled</span>
      <button
        type="button"
        className="link"
        disabled={restore.isPending}
        onClick={() => {
          restore.mutate(undo)
        }}
      >
        Undo
      </button>
    </p>
  )

  if (groups.length === 0) {
    return (
      <div className="feed-empty">
        <div className="tile sketch" aria-hidden />
        <p className="lede">Nothing here yet.</p>
        <p>
          Images and clips appear here as they finish. Anything you don’t keep is deleted 24 hours
          after the GPU session ends.
        </p>
        <button type="button" className="btn" onClick={onCreate}>
          Write a prompt
        </button>
        {undoToast}
      </div>
    )
  }

  return (
    <>
      <div className="feed">
        {groups.map((g) => {
          const index = queue.indexOf(g.id)
          const queued: Queued | undefined =
            index < 0
              ? undefined
              : {
                  index,
                  length: queue.length,
                  lifted: drag?.id === g.id ? drag.dy : null,
                  mark: markBefore === g.id ? 'before' : markAfter === g.id ? 'after' : null,
                  onTop: () => {
                    moveTo(g.id, 0)
                  },
                  onStep: (delta) => {
                    const to = index + delta
                    if (to < 0 || to >= queue.length) return
                    refocus.current = g.id
                    moveTo(g.id, to)
                  },
                  onLift: (y) => {
                    lift(g.id, y)
                  },
                  onDrag: dragTo,
                  onDrop: drop,
                }
          return (
            <GroupView
              key={g.id}
              group={g}
              now={now}
              assets={assets.data}
              queued={queued}
              sectionRef={(el) => {
                if (el) sections.current.set(g.id, el)
                else sections.current.delete(g.id)
              }}
              onOpen={setOpen}
              onCancel={(job) => {
                cancel.mutate(job)
              }}
              deleting={remove.isPending && remove.variables.id === g.id}
              onDelete={() => {
                remove.mutate(g)
              }}
            />
          )
        })}
      </div>
      {finished && (
        <ClearResults
          pending={clear.isPending}
          error={clear.error?.message}
          onClear={() => {
            clear.mutate()
          }}
        />
      )}
      <p className="visually-hidden" id="drag-hint">
        Hold and drag to change the order, or use the up and down arrow keys.
      </p>
      <p className="visually-hidden" aria-live="polite">
        {announce}
      </p>
      {(move.error ?? cancel.error ?? restore.error ?? remove.error) && (
        <p className="feed-error" role="alert">
          {(move.error ?? cancel.error ?? restore.error ?? remove.error)?.message}
        </p>
      )}
      {undoToast}
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
              {isVideo(r.media_type) ? (
                <button
                  type="button"
                  className="btn quiet"
                  disabled={extend.isPending || r.spec?.family !== 'wan22'}
                  onClick={() => {
                    extend.mutate(r)
                  }}
                >
                  {extend.isPending ? 'Extending…' : 'Extend'}
                </button>
              ) : (
                <button
                  type="button"
                  className="btn quiet"
                  disabled={!sourceTarget || r.width === null || r.height === null}
                  onClick={() => {
                    if (!sourceTarget || r.width === null || r.height === null) return
                    draftWithSource(sourceTarget.family, sourceTarget.mode, {
                      sha: r.blob_sha,
                      width: r.width,
                      height: r.height,
                    })
                    onRemix()
                  }}
                >
                  Use as source
                </button>
              )}
              {(keep.error ?? extend.error) && (
                <p className="viewer-note" role="alert">
                  {(keep.error ?? extend.error)?.message}
                </p>
              )}
            </>
          )}
        />
      )}
    </>
  )
}

/** A queued group's place in the queue, and the ways to change it. */
interface Queued {
  index: number
  length: number
  /** How far the group has been dragged, while it's lifted. */
  lifted: number | null
  /** Show where a dragged group would land: before or after this one. */
  mark: 'before' | 'after' | null
  onTop: () => void
  onStep: (delta: -1 | 1) => void
  onLift: (clientY: number) => void
  onDrag: (clientY: number) => void
  onDrop: () => void
}

function GroupView({
  group,
  now,
  assets,
  queued,
  sectionRef,
  onOpen,
  onCancel,
  deleting,
  onDelete,
}: {
  group: Group
  now: number
  assets: Asset[] | undefined
  queued?: Queued
  sectionRef: (el: HTMLElement | null) => void
  onOpen: (id: string) => void
  onCancel: (job: Job) => void
  deleting: boolean
  onDelete: () => void
}) {
  const { spec, job, results, chain } = group
  const [confirming, setConfirming] = useState(false)
  const params = spec?.params ?? {}
  const prompt = String(params.prompt ?? '').trim()
  const w = Number(params.width ?? results[0]?.width ?? 1)
  const h = Number(params.height ?? results[0]?.height ?? 1)
  const style = {
    '--ratio': `${String(w)} / ${String(h)}`,
    '--cols': w > h ? 2 : 3,
  } as CSSProperties
  const pending = !chain && job && PENDING.has(job.status)
  const expiresAt = results[0]?.expires_at ?? null
  const allKept = results.length > 0 && results.every((r) => r.library_id)
  const expiry = allKept ? 'Kept' : timeLeft(expiresAt, now)
  const soon = !allKept && expiresAt !== null && hoursLeft(expiresAt, now) < SOON_HOURS
  const total = chain ? 0 : (job?.seeds.length ?? results.length)
  const done = new Set(results.map((r) => r.item_index))
  const video = results[0] ? isVideo(results[0].media_type) : false
  const clips = results[0]?.segments?.length ?? 0
  const length = results[0]?.duration
  const shape = [size(w, h), video && length ? duration(length) : null].filter(Boolean).join(', ')
  // The model uncovers with the prompt: one tap on either shows the group's words.
  const meta = chain ? (
    `Extended, ${String(clips)} clips${length ? `, ${duration(length)}` : ''}`
  ) : spec ? (
    <>
      <CoveredText id={`prompt:${group.id}`} label="Show model">
        {`${modelLine(spec, assets)},`}
      </CoveredText>{' '}
      {shape}
    </>
  ) : (
    shape
  )
  const tile = (r: Result, i: number) => {
    const what = `${isVideo(r.media_type) ? 'clip' : 'image'} ${String(i + 1)}`
    return (
      <Tile
        key={r.id}
        id={r.id}
        blobSha={r.blob_sha}
        mediaType={r.media_type}
        duration={r.duration}
        kept={!!r.library_id}
        label={`Open ${what}, seed ${String(r.seed)}${r.library_id ? ', kept' : ''}`}
        coveredLabel={`Show ${what}`}
        onOpen={() => {
          onOpen(r.id)
        }}
      />
    )
  }

  const lifted = queued?.lifted ?? null
  const classes = [
    'group',
    lifted !== null && 'lifted',
    queued?.mark === 'before' && 'drop-before',
    queued?.mark === 'after' && 'drop-after',
    (confirming || deleting) && 'doomed',
  ]

  return (
    <section
      ref={sectionRef}
      aria-label={prompt || 'Untitled'}
      className={classes.filter(Boolean).join(' ')}
      style={lifted !== null ? { transform: `translateY(${String(lifted)}px)` } : undefined}
    >
      <header className={queued ? 'group-caption queued' : 'group-caption'}>
        {queued && <DragHandle queued={queued} />}
        <p className={prompt ? 'title' : 'title untitled'}>
          {prompt ? <CoveredText id={`prompt:${group.id}`}>{prompt}</CoveredText> : 'No prompt'}
        </p>
        <p className="meta">{meta}</p>
        <div className="aside">
          {pending ? (
            <>
              <button
                type="button"
                className="btn quiet small"
                onClick={() => {
                  onCancel(job)
                }}
              >
                Cancel
              </button>
              {queued && queued.index > 0 && (
                <button type="button" className="btn quiet small" onClick={queued.onTop}>
                  Move to top
                </button>
              )}
            </>
          ) : (
            <>
              <span className={soon ? 'soon' : undefined}>
                {expiry ?? shortTime(group.at, now)}
              </span>
              {!confirming && (
                <button
                  type="button"
                  className="btn quiet small"
                  disabled={deleting}
                  onClick={() => {
                    setConfirming(true)
                  }}
                >
                  {deleting ? 'Deleting…' : 'Delete'}
                </button>
              )}
            </>
          )}
        </div>
      </header>
      {confirming && (
        <div className="confirm group-confirm" role="group" aria-label="Confirm delete">
          <p>{deleteQuestion(results)}</p>
          <button
            type="button"
            className="btn danger"
            onClick={() => {
              onDelete()
              setConfirming(false)
            }}
          >
            Delete
          </button>
          <button
            type="button"
            className="btn quiet"
            onClick={() => {
              setConfirming(false)
            }}
          >
            Cancel
          </button>
        </div>
      )}
      {job?.status === 'error' && job.error && (
        <p className="group-error">
          <CoveredText id={`error:${group.id}`} label="Show error">
            {job.error}
          </CoveredText>
        </p>
      )}
      <div className="contact" style={style}>
        {chain && results.map(tile)}
        {Array.from({ length: Math.max(total, chain ? 0 : results.length) }, (_, i) => {
          const r = results.find((x) => x.item_index === i)
          if (r) return tile(r, i)
          if (!pending) return null
          return <SketchTile key={`s${String(i)}`} job={job} item={i} done={done} />
        })}
      </div>
    </section>
  )
}

/** "Delete these 4 images? Kept ones stay in the library." */
function deleteQuestion(results: Result[]): string {
  const n = results.length
  if (n === 0) return 'Delete this failed job?'
  const noun = results.every((r) => isVideo(r.media_type)) ? 'clip' : 'image'
  const what = n === 1 ? `this ${noun}` : `these ${String(n)} ${noun}s`
  const kept = results.filter((r) => r.library_id).length
  const note =
    kept === 0 ? '' : kept === n ? ' They stay in the library.' : ' Kept ones stay in the library.'
  return `Delete ${what}?${note}`
}

// Hold this long on the handle to lift a group; moving first means scrolling.
const LONG_PRESS_MS = 250
const SLOP_PX = 10

/** Long-press and drag to move a queued group, or focus it and use the arrow keys. */
function DragHandle({ queued }: { queued: Queued }) {
  const press = useRef<{ timer: number; y: number; lifted: boolean } | null>(null)

  const end = () => {
    if (!press.current) return
    clearTimeout(press.current.timer)
    if (press.current.lifted) queued.onDrop()
    press.current = null
  }

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return
    const el = e.currentTarget
    const { pointerId, clientY } = e
    const liftNow = () => {
      if (!press.current) return
      press.current.lifted = true
      el.setPointerCapture(pointerId)
      queued.onLift(press.current.y)
    }
    press.current = { timer: 0, y: clientY, lifted: false }
    if (e.pointerType === 'mouse') liftNow()
    else press.current.timer = window.setTimeout(liftNow, LONG_PRESS_MS)
  }

  const onPointerMove = (e: PointerEvent<HTMLButtonElement>) => {
    const p = press.current
    if (!p) return
    if (p.lifted) queued.onDrag(e.clientY)
    else if (Math.abs(e.clientY - p.y) > SLOP_PX) end()
  }

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
    e.preventDefault()
    queued.onStep(e.key === 'ArrowUp' ? -1 : 1)
  }

  return (
    <button
      type="button"
      className="drag-handle"
      aria-label={`Queue position ${String(queued.index + 1)} of ${String(queued.length)}`}
      aria-describedby="drag-hint"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onKeyDown={onKeyDown}
      onContextMenu={(e) => {
        e.preventDefault() // a long press shouldn't open the callout
      }}
    >
      <svg viewBox="0 0 20 20" aria-hidden>
        <path d="M4 11 9 3M8 15l7-11M12 17l5-8" />
      </svg>
    </button>
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

/** Delete everything finished at once, after asking inline. */
function ClearResults({
  pending,
  error,
  onClear,
}: {
  pending: boolean
  error: string | undefined
  onClear: () => void
}) {
  const [confirming, setConfirming] = useState(false)
  return (
    <div className="feed-clear">
      {confirming ? (
        <div className="confirm" role="group" aria-label="Confirm clear">
          <p>Delete every finished image and clip? Kept ones stay in the library.</p>
          <button
            type="button"
            className="btn danger"
            disabled={pending}
            onClick={() => {
              onClear()
              setConfirming(false)
            }}
          >
            Clear results
          </button>
          <button
            type="button"
            className="btn quiet"
            onClick={() => {
              setConfirming(false)
            }}
          >
            Cancel
          </button>
        </div>
      ) : (
        <button
          type="button"
          className="btn quiet"
          disabled={pending}
          onClick={() => {
            setConfirming(true)
          }}
        >
          {pending ? 'Clearing…' : 'Clear results'}
        </button>
      )}
      {error && <p role="alert">{error}</p>}
    </div>
  )
}
