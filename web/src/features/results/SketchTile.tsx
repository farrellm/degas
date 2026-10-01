import type { CSSProperties } from 'react'

import type { Job } from '@/api/types'
import { copyText, itemFraction, phaseText } from '@/lib/format'

/** A tile that isn't finished yet, drawn in pastel hatching as it denoises. */
export function SketchTile({ job, item, done }: { job: Job; item: number; done: Set<number> }) {
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
