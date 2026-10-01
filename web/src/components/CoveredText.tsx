import type { ReactNode } from 'react'

import { useCovered } from '@/hooks/useDiscretion'
import { reveal } from '@/lib/discretion'

/**
 * A prompt under glassine in discretion mode; a tap uncovers it. `shown` uncovers
 * it with something else, such as its image in the viewer.
 */
export function CoveredText({
  id,
  label = 'Show prompt',
  shown = false,
  children,
}: {
  id: string
  label?: string
  shown?: boolean
  children: ReactNode
}) {
  const covered = useCovered(id)
  if (!covered || shown) return children
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
