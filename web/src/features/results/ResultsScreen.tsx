import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { api } from '@/api/client'
import { queries, queryKeys } from '@/api/queries'
import type { Job, Result } from '@/api/types'
import { Viewer } from '@/components/Viewer/Viewer'
import { draftFromSpec } from '@/features/create/draft'
import { useAssets } from '@/hooks/useAssets'
import { useAutoDismiss } from '@/hooks/useAutoDismiss'
import { useNow } from '@/hooks/useNow'

import { ClearResults } from './ClearResults'
import { buildGroups, type Group, PENDING, reorder } from './groups'
import { ResultActions } from './ResultActions'
import { ResultGroup } from './ResultGroup'
import { useQueueDrag } from './useQueueDrag'

// How long "Cancelled · Undo" stays up.
const UNDO_MS = 5000

export interface ResultsScreenProps {
  /** The Create draft was replaced from a result: go there. */
  onRemix: () => void
  onCreate: () => void
}

/** Results: the queue, then what's finished, newest first, a group per job. */
export function ResultsScreen({ onRemix, onCreate }: ResultsScreenProps) {
  const qc = useQueryClient()
  const now = useNow(30_000)
  const assets = useAssets()
  const jobs = useQuery(queries.jobs())
  const results = useQuery(queries.results())
  // The job just cancelled, while it can still be brought back.
  const [undo, setUndo] = useState<string | null>(null)
  const [open, setOpen] = useState<string | null>(null)

  const refreshJobs = () => qc.invalidateQueries({ queryKey: queryKeys.jobs })
  const refreshResultsAnd = (other: 'jobs' | 'library') => () =>
    Promise.all([
      qc.invalidateQueries({ queryKey: queryKeys.results }),
      qc.invalidateQueries({ queryKey: queryKeys[other] }),
    ])

  const cancel = useMutation({
    mutationFn: (job: Job) => api.cancelJob(job.id),
    onSuccess: (_, job) => {
      // A job that hasn't started can come back; a running one is already stopping.
      setUndo(job.status === 'queued' ? job.id : null)
    },
    onSettled: refreshJobs,
  })
  const restore = useMutation({
    mutationFn: api.restoreJob,
    onSuccess: () => {
      setUndo(null)
    },
    onSettled: refreshJobs,
  })
  const move = useMutation({
    mutationFn: ({ id, position }: { id: string; position: number }) => api.moveJob(id, position),
    onMutate: ({ id, position }) => {
      qc.setQueryData(queries.jobs().queryKey, (js) => js && reorder(js, id, position))
    },
    onSettled: refreshJobs,
  })
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
    onSettled: refreshResultsAnd('library'),
  })
  const clear = useMutation({
    mutationFn: api.clearResults,
    onSuccess: () => {
      setOpen(null)
    },
    onSettled: refreshResultsAnd('jobs'),
  })
  const remove = useMutation({
    mutationFn: (g: Group) => api.deleteJobResults(g.jobId, g.chain),
    onSuccess: (_, g) => {
      if (g.results.some((r) => r.id === open)) setOpen(null)
    },
    onSettled: refreshResultsAnd('jobs'),
  })

  const queueDrag = useQueueDrag((id, position) => {
    move.mutate({ id, position })
  })
  useAutoDismiss(undo, UNDO_MS, () => {
    setUndo(null)
  })

  if (jobs.isPending || results.isPending) return <p className="loading">Loading…</p>
  if (jobs.error ?? results.error) {
    return <p role="alert">{(jobs.error ?? results.error)?.message}</p>
  }

  const groups = buildGroups(jobs.data, results.data.results)
  const flat = groups.flatMap((g) => g.results)
  const openIndex = flat.findIndex((r) => r.id === open)
  const queue = groups.filter((g) => !g.chain && g.job?.status === 'queued').map((g) => g.id)
  const finished = groups.some((g) => !PENDING.has(g.job?.status ?? ''))
  const error = move.error ?? cancel.error ?? restore.error ?? remove.error

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
        {groups.map((g) => (
          <ResultGroup
            key={g.id}
            group={g}
            now={now}
            assets={assets.data}
            queued={queueDrag.place(queue, g.id)}
            sectionRef={queueDrag.sectionRef(g.id)}
            onOpen={setOpen}
            onCancel={(job) => {
              cancel.mutate(job)
            }}
            onRemix={(job) => {
              draftFromSpec(job.spec, null)
              onRemix()
            }}
            deleting={remove.isPending && remove.variables.id === g.id}
            onDelete={() => {
              remove.mutate(g)
            }}
          />
        ))}
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
        {queueDrag.announce}
      </p>
      {error && (
        <p className="feed-error" role="alert">
          {error.message}
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
            <ResultActions
              result={r}
              keeping={keep.isPending}
              extending={extend.isPending}
              error={(keep.error ?? extend.error)?.message}
              onKeep={() => {
                keep.mutate(r)
              }}
              onExtend={() => {
                extend.mutate(r)
              }}
              onRemix={onRemix}
            />
          )}
        />
      )}
    </>
  )
}
