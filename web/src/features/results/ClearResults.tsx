import { useState } from 'react'

/** Delete everything finished at once, after asking inline. */
export function ClearResults({
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
