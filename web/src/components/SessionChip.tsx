import { useQuery } from '@tanstack/react-query'
import { api, isActive } from '../api'
import { countdown, useNow } from '../time'

const STATE_TEXT: Record<string, string> = {
  starting: 'starting',
  stopping: 'stopping',
  error: 'error',
}

/** Header chip: which GPU is running and how long until it stops for idleness. */
export function SessionChip({ onOpen }: { onOpen: () => void }) {
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const now = useNow(1000)
  const snap = session.data
  const s = snap?.session

  if (!snap || !s || (!isActive(snap) && s.state !== 'error' && s.state !== 'stopping')) {
    return (
      <button type="button" className="session-chip off" onClick={onOpen}>
        <span className="dot" aria-hidden />
        No GPU
      </button>
    )
  }

  const idle = countdown(snap.idle_deadline, now)
  const note = STATE_TEXT[s.state] ?? (s.state === 'ready' && idle ? idle : null)
  return (
    <button
      type="button"
      className="session-chip"
      onClick={onOpen}
      aria-label={`${s.gpu} session ${s.state}${idle ? `, stops in ${idle} if idle` : ''}`}
    >
      <span className={`dot ${s.state}`} aria-hidden />
      {s.gpu}
      {note && <span className="countdown">{note}</span>}
    </button>
  )
}
