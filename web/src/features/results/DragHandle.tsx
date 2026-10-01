import { type KeyboardEvent, type PointerEvent, useRef } from 'react'

/** A queued group's place in the queue, and the ways to change it. */
export interface Queued {
  index: number
  length: number
  /** How far the group has been dragged, while it's lifted. */
  lifted: number | null
  /** Show where a dragged group would land: before or after this one. */
  mark: 'before' | 'after' | null
  onTop: () => void
  onStep: (delta: -1 | 1) => void
  onLift: (clientY: number) => void
  onDrag: (clientY: number) => void
  onDrop: () => void
}

// Hold this long on the handle to lift a group; moving first means scrolling.
const LONG_PRESS_MS = 250
const SLOP_PX = 10

/** Long-press and drag to move a queued group, or focus it and use the arrow keys. */
export function DragHandle({ queued }: { queued: Queued }) {
  const press = useRef<{ timer: number; y: number; lifted: boolean } | null>(null)

  const end = () => {
    if (!press.current) return
    clearTimeout(press.current.timer)
    if (press.current.lifted) queued.onDrop()
    press.current = null
  }

  const onPointerDown = (e: PointerEvent<HTMLButtonElement>) => {
    if (e.button !== 0) return
    const el = e.currentTarget
    const { pointerId, clientY } = e
    const liftNow = () => {
      if (!press.current) return
      press.current.lifted = true
      el.setPointerCapture(pointerId)
      queued.onLift(press.current.y)
    }
    press.current = { timer: 0, y: clientY, lifted: false }
    if (e.pointerType === 'mouse') liftNow()
    else press.current.timer = window.setTimeout(liftNow, LONG_PRESS_MS)
  }

  const onPointerMove = (e: PointerEvent<HTMLButtonElement>) => {
    const p = press.current
    if (!p) return
    if (p.lifted) queued.onDrag(e.clientY)
    else if (Math.abs(e.clientY - p.y) > SLOP_PX) end()
  }

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
    e.preventDefault()
    queued.onStep(e.key === 'ArrowUp' ? -1 : 1)
  }

  return (
    <button
      type="button"
      className="drag-handle"
      aria-label={`Queue position ${String(queued.index + 1)} of ${String(queued.length)}`}
      aria-describedby="drag-hint"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={end}
      onPointerCancel={end}
      onKeyDown={onKeyDown}
      onContextMenu={(e) => {
        e.preventDefault() // a long press shouldn't open the callout
      }}
    >
      <svg viewBox="0 0 20 20" aria-hidden>
        <path d="M4 11 9 3M8 15l7-11M12 17l5-8" />
      </svg>
    </button>
  )
}
