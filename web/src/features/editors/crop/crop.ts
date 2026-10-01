// Geometry for the crop editor (design §6.5). The image pans and zooms under a fixed
// frame; everything here is in stage pixels (CSS px) or image pixels, never both at once.
import type { Op } from './api'

export type Rotation = 0 | 90 | 180 | 270

export interface Size {
  w: number
  h: number
}

export interface Rect {
  x: number
  y: number
  w: number
  h: number
}

/** How the (rotated) image sits on the stage: `s` stage px per image px, top-left at `tx, ty`. */
export interface View {
  s: number
  tx: number
  ty: number
}

export interface Constraints {
  multiple_of: number
  min_pixels: number
  max_pixels: number
}

/** Aspect choices: 'match' locks to the form's size, 'free' is any rectangle. */
export type Aspect = 'match' | 'free' | `${number}:${number}`

export const ASPECTS: { id: Aspect; label: string }[] = [
  { id: 'match', label: 'Match size' },
  { id: '1:1', label: '1:1' },
  { id: '3:2', label: '3:2' },
  { id: '2:3', label: '2:3' },
  { id: '16:9', label: '16:9' },
  { id: '9:16', label: '9:16' },
  { id: 'free', label: 'Free' },
]

/** Aspect choices for a free crop, which has no form size to match. */
export const FREE_ASPECTS = ASPECTS.filter((a) => a.id !== 'match')

// Beyond this, the crop is upscaled enough that the readout warns.
export const UPSCALE_WARN = 1.5
const FRAME_MARGIN = 20
const MAX_ZOOM = 8
const MIN_FRAME = 40

export function rotated(natural: Size, rot: Rotation): Size {
  return rot === 90 || rot === 270 ? { w: natural.h, h: natural.w } : natural
}

/** Width ÷ height of an aspect choice, or null for free. */
export function ratio(aspect: Aspect, target: Size): number | null {
  if (aspect === 'free') return null
  if (aspect === 'match') return target.w / target.h
  const [w, h] = aspect.split(':').map(Number)
  return (w ?? 1) / (h ?? 1)
}

/** The largest frame of the given aspect centred on the stage. */
export function frameFor(stage: Size, aspect: number): Rect {
  const room = { w: stage.w - 2 * FRAME_MARGIN, h: stage.h - 2 * FRAME_MARGIN }
  const w = Math.max(1, Math.min(room.w, room.h * aspect))
  const h = Math.max(1, w / aspect)
  return { x: (stage.w - w) / 2, y: (stage.h - h) / 2, w, h }
}

/** The smallest scale at which the image covers the frame. */
export function coverScale(frame: Rect, image: Size): number {
  return Math.max(frame.w / image.w, frame.h / image.h)
}

/** Keep the frame covered by the image, and the zoom within bounds. */
export function clampView(view: View, frame: Rect, image: Size): View {
  const min = coverScale(frame, image)
  const s = Math.min(Math.max(view.s, min), min * MAX_ZOOM)
  const tx = Math.min(frame.x, Math.max(frame.x + frame.w - image.w * s, view.tx))
  const ty = Math.min(frame.y, Math.max(frame.y + frame.h - image.h * s, view.ty))
  return { s, tx, ty }
}

/** The image centred under the frame, just covering it. */
export function coverView(frame: Rect, image: Size): View {
  const s = coverScale(frame, image)
  return {
    s,
    tx: frame.x + (frame.w - image.w * s) / 2,
    ty: frame.y + (frame.h - image.h * s) / 2,
  }
}

/** Zoom by `factor` about the stage point `(cx, cy)`. */
export function zoomAt(view: View, factor: number, cx: number, cy: number): View {
  const s = view.s * factor
  return { s, tx: cx - (cx - view.tx) * factor, ty: cy - (cy - view.ty) * factor }
}

/** The part of the (rotated) image under the frame, in whole image pixels. */
export function cropOf(view: View, frame: Rect, image: Size): Rect {
  const x = Math.max(0, Math.round((frame.x - view.tx) / view.s))
  const y = Math.max(0, Math.round((frame.y - view.ty) / view.s))
  const w = Math.min(image.w - x, Math.max(1, Math.round(frame.w / view.s)))
  const h = Math.min(image.h - y, Math.max(1, Math.round(frame.h / view.s)))
  return { x, y, w, h }
}

/** The view that puts `crop` exactly under the frame (to reopen an earlier crop). */
export function viewFor(crop: Rect, frame: Rect): View {
  const s = frame.w / crop.w
  return { s, tx: frame.x - crop.x * s, ty: frame.y - crop.y * s }
}

/** The frame's rect on the stage for a crop at the given view. */
export function frameOf(crop: Rect, view: View): Rect {
  return {
    x: view.tx + crop.x * view.s,
    y: view.ty + crop.y * view.s,
    w: crop.w * view.s,
    h: crop.h * view.s,
  }
}

/** Move one corner of a free frame, keeping it on the image and on the stage. */
export function dragCorner(
  frame: Rect,
  corner: 'nw' | 'ne' | 'sw' | 'se',
  dx: number,
  dy: number,
  bounds: Rect,
): Rect {
  let left = frame.x
  let top = frame.y
  let right = frame.x + frame.w
  let bottom = frame.y + frame.h
  if (corner.includes('w')) left = Math.min(right - MIN_FRAME, Math.max(bounds.x, left + dx))
  else right = Math.max(left + MIN_FRAME, Math.min(bounds.x + bounds.w, right + dx))
  if (corner.includes('n')) top = Math.min(bottom - MIN_FRAME, Math.max(bounds.y, top + dy))
  else bottom = Math.max(top + MIN_FRAME, Math.min(bounds.y + bounds.h, bottom + dy))
  return { x: left, y: top, w: right - left, h: bottom - top }
}

/** The intersection of two rects. */
export function intersect(a: Rect, b: Rect): Rect {
  const x = Math.max(a.x, b.x)
  const y = Math.max(a.y, b.y)
  return {
    x,
    y,
    w: Math.max(0, Math.min(a.x + a.w, b.x + b.w) - x),
    h: Math.max(0, Math.min(a.y + a.h, b.y + b.h) - y),
  }
}

/**
 * The pixel size the crop is resized to: the form's size when matching it, otherwise
 * the crop's shape at about the form's pixel count, snapped to the model's `multiple_of`.
 */
export function outputSize(crop: Rect, aspect: Aspect, target: Size, c: Constraints): Size {
  if (aspect === 'match') return target
  const m = c.multiple_of
  const r = crop.w / crop.h
  const pixels = Math.min(c.max_pixels, Math.max(c.min_pixels, target.w * target.h))
  const snap = (v: number) => Math.max(m, Math.round(v / m) * m)
  let w = snap(Math.sqrt(pixels * r))
  let h = snap(w / r)
  // Rounding can step just past the limits: shrink the longer side a notch at a time.
  while (w * h > c.max_pixels && Math.max(w, h) > m) {
    if (w >= h) w -= m
    else h -= m
  }
  return { w, h }
}

/** How much the crop is scaled up to reach the output size (1 = not at all). */
export function upscale(crop: Rect, out: Size): number {
  return Math.max(out.w / crop.w, out.h / crop.h)
}

/** How much the model scales a reference up, sizing it to the output's pixel count. */
export function modelUpscale(crop: Size, target: Size): number {
  return Math.sqrt((target.w * target.h) / (crop.w * crop.h))
}

/** Operations for the server, leaving out any that do nothing. */
export function buildOps(rot: Rotation, flip: boolean, crop: Rect, image: Size, out: Size): Op[] {
  const ops: Op[] = []
  if (rot) ops.push({ op: 'rotate', deg: rot })
  if (flip) ops.push({ op: 'flip_h' })
  if (crop.x || crop.y || crop.w !== image.w || crop.h !== image.h) {
    ops.push({ op: 'crop', ...crop })
  }
  if (out.w !== crop.w || out.h !== crop.h) ops.push({ op: 'resize', w: out.w, h: out.h })
  return ops
}

/** Read back what `buildOps` wrote (to reopen the editor on a derived image). */
export function parseOps(ops: Op[]): {
  rot: Rotation
  flip: boolean
  crop: Rect | null
  resized: boolean
} {
  let rot: Rotation = 0
  let flip = false
  let crop: Rect | null = null
  let resized = false
  for (const op of ops) {
    if (op.op === 'rotate') rot = op.deg
    else if (op.op === 'flip_h') flip = true
    else if (op.op === 'crop') crop = { x: op.x, y: op.y, w: op.w, h: op.h }
    else if (op.op === 'resize') resized = true
  }
  return { rot, flip, crop, resized }
}

/** The crop flipped left-to-right within the image. */
export function mirror(crop: Rect, image: Size): Rect {
  return { ...crop, x: image.w - crop.x - crop.w }
}

/** The part of the image under the frame, unrounded (the editor's working state). */
export function exactCrop(view: View, frame: Rect): Rect {
  return {
    x: (frame.x - view.tx) / view.s,
    y: (frame.y - view.ty) / view.s,
    w: frame.w / view.s,
    h: frame.h / view.s,
  }
}

/** The largest rect of aspect `r` centred in the image. */
export function centered(image: Size, r: number): Rect {
  const w = Math.min(image.w, image.h * r)
  const h = w / r
  return { x: (image.w - w) / 2, y: (image.h - h) / 2, w, h }
}

/** A crop of aspect `r` with about the same area and centre as `crop`, kept on the image. */
export function reshape(crop: Rect, r: number, image: Size): Rect {
  const area = crop.w * crop.h
  let w = Math.sqrt(area * r)
  let h = w / r
  const fit = Math.min(1, image.w / w, image.h / h)
  w *= fit
  h *= fit
  const cx = crop.x + crop.w / 2
  const cy = crop.y + crop.h / 2
  const x = Math.min(image.w - w, Math.max(0, cx - w / 2))
  const y = Math.min(image.h - h, Math.max(0, cy - h / 2))
  return { x, y, w, h }
}
