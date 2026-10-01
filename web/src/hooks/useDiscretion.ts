import { type PointerEvent, useRef, useState, useSyncExternalStore } from 'react'

import { isCovered, isDiscreet, reveal, subscribe } from '@/lib/discretion'

export function useDiscretion(): boolean {
  return useSyncExternalStore(subscribe, isDiscreet)
}

/** Whether `id` is covered right now (never when the mode is off). */
export function useCovered(id: string): boolean {
  return useSyncExternalStore(subscribe, () => isCovered(id))
}

// Hold this long to peek at a covered image; moving first means scrolling.
const PEEK_MS = 250
const SLOP_PX = 10

/**
 * A covered tile: a tap uncovers it, a second tap opens it, and holding it shows
 * it only while the finger is down. Spread `press` on the tile and route its
 * click through `tap`.
 */
export function useCover(id: string) {
  const covered = useCovered(id)
  const [peek, setPeek] = useState(false)
  const hold = useRef<{ timer: number; x: number; y: number } | null>(null)
  // A peek ends in a click, which mustn't uncover or open the tile.
  const peeked = useRef(false)

  const end = () => {
    if (hold.current) clearTimeout(hold.current.timer)
    hold.current = null
    setPeek(false)
  }

  const press = covered
    ? {
        onPointerDown: (e: PointerEvent) => {
          peeked.current = false
          if (e.button !== 0) return
          const timer = window.setTimeout(() => {
            peeked.current = true
            setPeek(true)
          }, PEEK_MS)
          hold.current = { timer, x: e.clientX, y: e.clientY }
        },
        onPointerMove: (e: PointerEvent) => {
          const h = hold.current
          if (h && Math.hypot(e.clientX - h.x, e.clientY - h.y) > SLOP_PX) end()
        },
        onPointerUp: end,
        onPointerCancel: end,
        onPointerLeave: end,
        onContextMenu: (e: { preventDefault: () => void }) => {
          e.preventDefault() // a long press shouldn't open the image callout
        },
      }
    : {}

  const tap = (open: () => void) => {
    if (peeked.current) {
      peeked.current = false
      return
    }
    if (covered) reveal(id)
    else open()
  }

  return { covered: covered && !peek, peek, press, tap }
}
