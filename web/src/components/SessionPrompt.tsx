import { useQuery } from '@tanstack/react-query'
import { api, isActive } from '../api'

/** Nudge shown on Create and Queue when jobs would wait for a GPU. */
export function SessionPrompt() {
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  if (!session.data || isActive(session.data)) return null
  return (
    <p className="notice">
      No GPU session is running. Jobs wait in the queue until you start one from the{' '}
      <strong>Session</strong> tab.
    </p>
  )
}
