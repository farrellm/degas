import { useEffect, useRef, useState } from 'react'

import type { Queued } from './DragHandle'

/** A queued group being dragged: where it started, and where it would land. */
interface Drag {
  id: string
  from: number
  to: number
  startY: number
  dy: number
  /** Vertical middles of the other queued groups, in queue order, at lift time. */
  mids: number[]
}

/**
 * Reordering the queue by dragging a group's handle or with its arrow keys. `onMove` moves
 * a job to a position in the queue (0 runs next).
 */
export function useQueueDrag(onMove: (id: string, position: number) => void) {
  const [drag, setDrag] = useState<Drag | null>(null)
  const [announce, setAnnounce] = useState('')
  const sections = useRef(new Map<string, HTMLElement>())
  const refocus = useRef<string | null>(null)

  // Keep focus on a handle moved with the arrow keys as its group changes place.
  useEffect(() => {
    const id = refocus.current
    if (!id) return
    refocus.current = null
    sections.current.get(id)?.querySelector<HTMLElement>('.drag-handle')?.focus()
  })

  /** `id`'s place in `queue` (the queued groups' ids, in order), or undefined when it isn't queued. */
  const place = (queue: string[], id: string): Queued | undefined => {
    const index = queue.indexOf(id)
    if (index < 0) return undefined

    const moveTo = (moved: string, position: number) => {
      setAnnounce(`Moved to ${position + 1} of ${queue.length} in the queue.`)
      onMove(moved, position)
    }
    // Where the drop mark goes: before the group now at `to`, or after the last one.
    const others = drag ? queue.filter((q) => q !== drag.id) : []
    const markBefore = drag && drag.to !== drag.from ? others[drag.to] : undefined
    const markAfter =
      drag && drag.to !== drag.from && drag.to === others.length ? others.at(-1) : undefined

    return {
      index,
      length: queue.length,
      lifted: drag?.id === id ? drag.dy : null,
      mark: markBefore === id ? 'before' : markAfter === id ? 'after' : null,
      onTop: () => moveTo(id, 0),
      onStep: (delta) => {
        const to = index + delta
        if (to < 0 || to >= queue.length) return
        refocus.current = id
        moveTo(id, to)
      },
      onLift: (y) => {
        const mids = queue
          .filter((q) => q !== id)
          .map((q) => {
            const r = sections.current.get(q)?.getBoundingClientRect()
            return r ? r.top + r.height / 2 : 0
          })
        setDrag({ id, from: index, to: index, startY: y, dy: 0, mids })
      },
      onDrag: (y) =>
        setDrag((d) => d && { ...d, dy: y - d.startY, to: d.mids.filter((m) => m < y).length }),
      onDrop: () => {
        if (drag && drag.to !== drag.from) moveTo(drag.id, drag.to)
        setDrag(null)
      },
    }
  }

  return {
    /** What the last move did, for screen readers. */
    announce,
    place,
    /** A ref callback for the group `id`'s section, so drags can be measured against it. */
    sectionRef: (id: string) => (el: HTMLElement | null) => {
      if (el) sections.current.set(id, el)
      else sections.current.delete(id)
    },
  }
}
