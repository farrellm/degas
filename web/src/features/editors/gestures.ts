// Pointer maths the crop and mask editors share: both pan and pinch an image on a stage.

import type { View } from '@/lib/geometry'

import { zoomAt } from './crop/crop'

export interface Point {
  x: number
  y: number
}

/** Where a pointer is on the stage, in stage px. */
export function stagePoint(stage: Element | null, e: { clientX: number; clientY: number }): Point {
  const r = stage?.getBoundingClientRect()
  return { x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0) }
}

/**
 * One finger of a pinch moved from `prev` to `next` while the `other` stayed: zoom about
 * their midpoint by the change in spread, and follow the midpoint.
 */
export function pinch(view: View, prev: Point, next: Point, other: Point): View {
  const before = Math.hypot(prev.x - other.x, prev.y - other.y)
  const after = Math.hypot(next.x - other.x, next.y - other.y)
  const mid = { x: (next.x + other.x) / 2, y: (next.y + other.y) / 2 }
  const z = before > 0 ? zoomAt(view, after / before, mid.x, mid.y) : view
  return { ...z, tx: z.tx + (next.x - prev.x) / 2, ty: z.ty + (next.y - prev.y) / 2 }
}

/**
 * What a key does to the view: the arrows pan by `step` stage px, and + (or =) and − zoom
 * by `factor` about `centre`. Null for any other key.
 */
export function keyMove(
  key: string,
  step: number,
  factor: number,
  centre: Point,
): ((view: View) => View) | null {
  switch (key) {
    case 'ArrowLeft':
      return (v) => ({ ...v, tx: v.tx + step })
    case 'ArrowRight':
      return (v) => ({ ...v, tx: v.tx - step })
    case 'ArrowUp':
      return (v) => ({ ...v, ty: v.ty + step })
    case 'ArrowDown':
      return (v) => ({ ...v, ty: v.ty - step })
    case '+':
    case '=':
      return (v) => zoomAt(v, factor, centre.x, centre.y)
    case '-':
      return (v) => zoomAt(v, 1 / factor, centre.x, centre.y)
    default:
      return null
  }
}
