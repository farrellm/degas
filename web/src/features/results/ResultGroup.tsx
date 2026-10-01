import { type CSSProperties, useState } from 'react'

import type { Asset, Job, Result } from '@/api/types'
import { CoveredText } from '@/components/CoveredText'
import { Tile } from '@/components/Tile'
import { formatDuration, formatSize } from '@/lib/format'
import { isVideo } from '@/lib/image'
import { hoursLeft, shortTime, timeLeft } from '@/lib/time'

import { DragHandle, type Queued } from './DragHandle'
import { deleteQuestion, type Group, modelLine, PENDING } from './groups'
import { SketchTile } from './SketchTile'

// Unkept images closer than this to deletion get a warning colour.
const SOON_HOURS = 2

export function ResultGroup({
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
  const shape = [formatSize(w, h), video && length ? formatDuration(length) : null]
    .filter(Boolean)
    .join(', ')
  // The model uncovers with the prompt: one tap on either shows the group's words.
  const meta = chain ? (
    `Extended, ${String(clips)} clips${length ? `, ${formatDuration(length)}` : ''}`
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
