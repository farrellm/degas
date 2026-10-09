// Where an outpaint puts its source on the canvas (the form's Size), in canvas pixels.
import type { Rect, Size } from '@/lib/geometry'

export type Place = Rect

/** Positions snap to this step, which SDXL's latents also use. */
const STEP = 8
export const MIN_SIDE = 64
/** How much of the canvas the source fills by default, when fitting it would fill it all. */
const ROOM = 0.75

const snap = (v: number) => Math.round(v / STEP) * STEP

/** The largest size of the source's shape that fits inside the canvas. */
export function fitInside(source: Size, canvas: Size): Size {
  const k = Math.min(canvas.w / source.w, canvas.h / source.h)
  return { w: source.w * k, h: source.h * k }
}

/** The source at `scale` (1 = fit inside) of the canvas, snapped, at least MIN_SIDE a side. */
export function sized(source: Size, canvas: Size, scale: number): Size {
  const fit = fitInside(source, canvas)
  const min = MIN_SIDE / Math.min(fit.w, fit.h)
  const k = Math.min(1, Math.max(min, scale))
  return {
    w: Math.min(canvas.w, Math.max(MIN_SIDE, snap(fit.w * k))),
    h: Math.min(canvas.h, Math.max(MIN_SIDE, snap(fit.h * k))),
  }
}

/** Keep a placement inside the canvas, on the snapping grid. */
export function clampPlace(p: Place, canvas: Size): Place {
  const w = Math.min(canvas.w, p.w)
  const h = Math.min(canvas.h, p.h)
  return {
    x: Math.min(canvas.w - w, Math.max(0, snap(p.x))),
    y: Math.min(canvas.h - h, Math.max(0, snap(p.y))),
    w,
    h,
  }
}

/** The source fitted inside the canvas and centred, leaving room around it to draw. */
export function defaultPlace(source: Size, canvas: Size): Place {
  const fit = sized(source, canvas, 1)
  const fills = fit.w >= canvas.w - STEP && fit.h >= canvas.h - STEP
  const size = fills ? sized(source, canvas, ROOM) : fit
  return centre(size, canvas)
}

export function centre(size: Size, canvas: Size): Place {
  return clampPlace(
    { x: (canvas.w - size.w) / 2, y: (canvas.h - size.h) / 2, w: size.w, h: size.h },
    canvas,
  )
}

/** Rescale about the placement's centre. */
export function rescale(p: Place, source: Size, canvas: Size, scale: number): Place {
  const size = sized(source, canvas, scale)
  return clampPlace({ x: p.x + (p.w - size.w) / 2, y: p.y + (p.h - size.h) / 2, ...size }, canvas)
}

/** The current scale, as a fraction of fitting inside. */
export function scaleOf(p: Place, source: Size, canvas: Size): number {
  return p.w / fitInside(source, canvas).w
}

/** Whether a placement is usable for this canvas and source shape. */
export function validPlace(p: Place | null | undefined, source: Size, canvas: Size): boolean {
  if (!p) return false
  const inside = p.x >= 0 && p.y >= 0 && p.x + p.w <= canvas.w && p.y + p.h <= canvas.h
  const shape = Math.abs(p.w / p.h / (source.w / source.h) - 1) < 0.05
  const room = p.w < canvas.w || p.h < canvas.h
  return inside && shape && room && p.w >= MIN_SIDE && p.h >= MIN_SIDE
}

export type Edge = 'left' | 'right' | 'top' | 'bottom'

/** Push the source against an edge (or back to the middle of that axis). */
export function align(p: Place, canvas: Size, to: Edge | 'centre'): Place {
  switch (to) {
    case 'left':
      return { ...p, x: 0 }
    case 'right':
      return { ...p, x: canvas.w - p.w }
    case 'top':
      return { ...p, y: 0 }
    case 'bottom':
      return { ...p, y: canvas.h - p.h }
    case 'centre':
      return centre(p, canvas)
  }
}

/** The new pixels on each side, for the readout: "+256 px left, +256 px right". */
export function margins(p: Place, canvas: Size): string {
  const sides: [string, number][] = [
    ['left', p.x],
    ['right', canvas.w - p.x - p.w],
    ['top', p.y],
    ['bottom', canvas.h - p.y - p.h],
  ]
  const parts = sides.filter(([, v]) => v > 0).map(([side, v]) => `+${v} px ${side}`)
  return parts.length ? parts.join(', ') : 'No new pixels'
}
