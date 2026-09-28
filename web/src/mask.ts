// Pure helpers for the mask editor (design §8.2): working size, the view, the undo
// history, and turning a stored mask (white = redraw) into canvas alpha and back.
import type { Rect, Size, View } from './crop'

/** Masks are painted at most this big on a side; iOS limits canvas memory. */
export const MAX_WORKING = 2048
const MAX_ZOOM = 8
const STAGE_MARGIN = 12

/** The size the mask is painted at: the image's, scaled down to fit `max`. */
export function workingSize(image: Size, max = MAX_WORKING): Size {
  const k = Math.min(1, max / Math.max(image.w, image.h))
  return { w: Math.max(1, Math.round(image.w * k)), h: Math.max(1, Math.round(image.h * k)) }
}

/** The whole image centred on the stage. */
export function fitView(stage: Size, image: Size): View {
  const s = Math.min((stage.w - 2 * STAGE_MARGIN) / image.w, (stage.h - 2 * STAGE_MARGIN) / image.h)
  return { s, tx: (stage.w - image.w * s) / 2, ty: (stage.h - image.h * s) / 2 }
}

/**
 * Keep the zoom between fitting the stage and 8× that, and keep the image on the stage:
 * smaller than the stage it stays centred, larger it can't leave a gap at an edge.
 */
export function clampView(view: View, stage: Size, image: Size): View {
  const fit = fitView(stage, image)
  const s = Math.min(Math.max(view.s, fit.s), fit.s * MAX_ZOOM)
  const axis = (t: number, room: number, length: number) => {
    const size = length * s
    if (size <= room) return (room - size) / 2
    return Math.min(0, Math.max(room - size, t))
  }
  return { s, tx: axis(view.tx, stage.w, image.w), ty: axis(view.ty, stage.h, image.h) }
}

/** A stage point in image (working) pixels. */
export function toImage(view: View, x: number, y: number): { x: number; y: number } {
  return { x: (x - view.tx) / view.s, y: (y - view.ty) / view.s }
}

/** Undo history: `done` actions are applied, `undone` ones can be redone. */
export interface History<T> {
  done: T[]
  undone: T[]
}

export const emptyHistory = <T>(): History<T> => ({ done: [], undone: [] })

export function push<T>(h: History<T>, action: T): History<T> {
  return { done: [...h.done, action], undone: [] }
}

export function undo<T>(h: History<T>): History<T> {
  const last = h.done.at(-1)
  return last === undefined ? h : { done: h.done.slice(0, -1), undone: [last, ...h.undone] }
}

export function redo<T>(h: History<T>): History<T> {
  const [next, ...rest] = h.undone
  return next === undefined ? h : { done: [...h.done, next], undone: rest }
}

/** Replace the last action (a stroke growing as the finger moves). */
export function amend<T>(h: History<T>, action: T): History<T> {
  return { done: [...h.done.slice(0, -1), action], undone: h.undone }
}

/** Stored masks are grey (white = redraw); the canvas keeps the mask in its alpha. */
export function lumaToAlpha(data: Uint8ClampedArray): void {
  for (let i = 0; i < data.length; i += 4) {
    const v = data[i] ?? 0
    data[i] = 255
    data[i + 1] = 255
    data[i + 2] = 255
    data[i + 3] = v
  }
}

/** Whether any pixel of an RGBA buffer has alpha. */
export function hasAlpha(data: Uint8ClampedArray): boolean {
  for (let i = 3; i < data.length; i += 4) if ((data[i] ?? 0) > 0) return true
  return false
}

/** Offsets that, drawn over each other, grow a shape by `r` px in every direction. */
export function growOffsets(r: number): { x: number; y: number }[] {
  if (r <= 0) return []
  const out: { x: number; y: number }[] = []
  for (const radius of r > 4 ? [r / 2, r] : [r]) {
    const n = Math.max(8, Math.ceil(radius * 1.5))
    for (let i = 0; i < n; i++) {
      const a = (i / n) * 2 * Math.PI
      out.push({ x: Math.cos(a) * radius, y: Math.sin(a) * radius })
    }
  }
  return out
}

/** The stage rect of the image at a view. */
export function imageRect(view: View, image: Size): Rect {
  return { x: view.tx, y: view.ty, w: image.w * view.s, h: image.h * view.s }
}
