import type { ReactNode } from 'react'
import { reveal, useCovered } from '../discretion'

/** A prompt under glassine in discretion mode; a tap uncovers it. */
export function CoveredText({
  id,
  label = 'Show prompt',
  children,
}: {
  id: string
  label?: string
  children: ReactNode
}) {
  const covered = useCovered(id)
  if (!covered) return children
  return (
    <button
      type="button"
      className="covered-text"
      aria-label={label}
      onClick={() => {
        reveal(id)
      }}
    >
      <span aria-hidden>{children}</span>
    </button>
  )
}
