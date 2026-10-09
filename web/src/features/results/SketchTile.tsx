import type { CSSProperties } from 'react'

import type { Job } from '@/api/types'
import { copyText, itemFraction, phaseText } from '@/lib/format'

/**
 * A tile that isn't finished yet, drawn in pastel hatching as it denoises. With `onOpen` it's
 * a button in the feed; without, the picture in the viewer.
 */
export function SketchTile({
  job,
  item,
  done,
  onOpen,
}: {
  job: Job
  item: number
  done: Set<number>
  onOpen?: () => void
}) {
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
  const className = `tile sketch ${cls}`
  const style = { '--p': fraction ?? 0 } as CSSProperties
  const what = `${job.spec.params.num_frames == null ? 'image' : 'clip'} ${item + 1}`
  const caption = (current || item === 0 || !onOpen) && (
    <span className="sketch-label">{label}</span>
  )
  if (!onOpen) {
    return (
      <div className={className} style={style} role="img" aria-label={label}>
        {caption}
      </div>
    )
  }
  return (
    <button
      type="button"
      className={className}
      style={style}
      aria-label={`Open ${what}: ${label}`}
      onClick={onOpen}
    >
      {caption}
    </button>
  )
}
