import { readStored, writeStored } from './storage'

/*
 * Discretion mode: images and prompts are covered, like pastels under a glassine
 * sheet, until tapped (ux.md Phase 9). `<html data-discreet>` drives the CSS;
 * index.html sets it before the first paint so a relaunch never flashes an image.
 */

const KEY = 'degas.discretion'

let on = readStored(KEY) === '1'
let revealed = new Set<string>()
const listeners = new Set<() => void>()

function emit() {
  for (const l of listeners) l()
}

/** For `useSyncExternalStore`: see hooks/useDiscretion.ts. */
export function subscribe(l: () => void) {
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

export const isDiscreet = () => on

export function setDiscretion(value: boolean) {
  on = value
  revealed = new Set()
  // Where it can't be stored (private browsing) the mode lasts until the app closes.
  writeStored(KEY, value ? '1' : '0')
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
export const isCovered = (id: string) => on && !revealed.has(id)

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
