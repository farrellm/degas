const MAX_BATCH = 8

export interface GenerateBarProps {
  video: boolean
  batchCount: number
  onBatchCount: (change: (n: number) => number) => void
  /** The job is being queued. */
  pending: boolean
  /** The form isn't ready to submit. */
  blocked: boolean
  error: string | null
  /** How many were just queued, while the confirmation shows. */
  queued: number | null
  /** No session is running, so jobs will wait. */
  noGpu: boolean
  onShowResults: () => void
  onOpenSession: () => void
}

/** The Generate bar, sticky above the tab bar: the batch size, the button, and what just happened. */
export function GenerateBar({
  video,
  batchCount,
  onBatchCount,
  pending,
  blocked,
  error,
  queued,
  noGpu,
  onShowResults,
  onOpenSession,
}: GenerateBarProps) {
  const noun = (n: number) => (video ? (n === 1 ? 'clip' : 'clips') : n === 1 ? 'image' : 'images')

  return (
    <div className="generate-bar">
      <div className="generate-bar-inner">
        {error ? (
          <p className="bar-note error" role="alert">
            {error}
          </p>
        ) : queued !== null ? (
          <p className="toast" role="status">
            <span>
              Queued {queued} {noun(queued)}.
            </span>
            <button type="button" className="link" onClick={onShowResults}>
              See results
            </button>
          </p>
        ) : noGpu ? (
          <p className="bar-note">
            No GPU is running, so jobs will wait.{' '}
            <button type="button" className="link" onClick={onOpenSession}>
              Start a session
            </button>
          </p>
        ) : null}
        <div className="generate-row">
          <div className="stepper" role="group" aria-label="Batch size">
            <button
              type="button"
              aria-label={`Fewer ${noun(2)}`}
              disabled={batchCount <= 1}
              onClick={() => {
                onBatchCount((n) => Math.max(1, n - 1))
              }}
            >
              −
            </button>
            <output aria-label={`${video ? 'Clips' : 'Images'} per job`}>{batchCount}</output>
            <button
              type="button"
              aria-label={`More ${noun(2)}`}
              disabled={batchCount >= MAX_BATCH}
              onClick={() => {
                onBatchCount((n) => Math.min(MAX_BATCH, n + 1))
              }}
            >
              +
            </button>
          </div>
          <button type="submit" className="btn" disabled={pending || blocked}>
            {pending
              ? 'Queuing…'
              : batchCount === 1
                ? 'Generate'
                : `Generate ${String(batchCount)} ${noun(batchCount)}`}
          </button>
        </div>
      </div>
    </div>
  )
}
