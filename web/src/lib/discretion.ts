import { useRef, useState, useSyncExternalStore, type PointerEvent } from 'react'

/*
 * Discretion mode: images and prompts are covered, like pastels under a glassine
 * sheet, until tapped (ux.md Phase 9). `<html data-discreet>` drives the CSS;
 * index.html sets it before the first paint so a relaunch never flashes an image.
 */

const KEY = 'degas.discretion'

function stored(): boolean {
  try {
    return localStorage.getItem(KEY) === '1'
  } catch {
    return false
  }
}

let on = stored()
let revealed = new Set<string>()
const listeners = new Set<() => void>()

function emit() {
  for (const l of listeners) l()
}

function subscribe(l: () => void) {
  listeners.add(l)
  return () => {
    listeners.delete(l)
  }
}

/** Tell the service worker, so a notification leaves the prompt out while covered. */
function tellWorker() {
  if (!('serviceWorker' in navigator)) return
  void navigator.serviceWorker.ready.then((reg) => {
    reg.active?.postMessage({ type: 'discretion', on })
  })
}

export function useDiscretion(): boolean {
  return useSyncExternalStore(subscribe, () => on)
}

export function setDiscretion(value: boolean) {
  on = value
  revealed = new Set()
  try {
    localStorage.setItem(KEY, value ? '1' : '0')
  } catch {
    // private browsing: the mode lasts until the app closes
  }
  if (value) document.documentElement.dataset.discreet = ''
  else delete document.documentElement.dataset.discreet
  tellWorker()
  emit()
}

/** Uncover one image or prompt until the next `coverAll`. */
export function reveal(id: string) {
  if (revealed.has(id)) return
  revealed = new Set(revealed).add(id)
  emit()
}

/** Cover everything again: on a tab change, closing the viewer, and leaving the app. */
export function coverAll() {
  if (revealed.size === 0) return
  revealed = new Set()
  emit()
}

/** Whether `id` is covered right now (never when the mode is off). */
export function useCovered(id: string): boolean {
  return useSyncExternalStore(subscribe, () => on && !revealed.has(id))
}

/*
 * The app switcher's snapshot: a shield of plain paper goes up the moment the app
 * loses focus, set on the DOM directly so it paints without waiting for React.
 * iOS may still take its snapshot first, so covered is also the resting state.
 */
export function installShield() {
  const root = document.documentElement
  const up = () => {
    if (on) root.dataset.shield = ''
  }
  const down = () => {
    delete root.dataset.shield
  }
  const hidden = () => {
    up()
    coverAll()
  }
  window.addEventListener('blur', up)
  window.addEventListener('pagehide', hidden)
  window.addEventListener('focus', down)
  window.addEventListener('pageshow', down)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') hidden()
    else down()
  })
  tellWorker()
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
