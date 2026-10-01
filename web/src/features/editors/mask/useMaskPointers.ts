import { type PointerEvent, type RefObject, useRef, useState } from 'react'

import { pinch, type Point, stagePoint } from '@/features/editors/gestures'
import type { View } from '@/lib/geometry'

import type { Stroke } from './canvas'
import { toImage } from './mask'

const TAP_SLOP = 8
const LONG_PRESS_MS = 500

/**
 * Fingers on the mask editor's stage: one paints a stroke (or, with `selecting`, taps a
 * point for SAM, a long press leaving it out), two pinch and pan the view.
 */
export function useMaskPointers({
  stageRef,
  ready,
  selecting,
  erase,
  brush,
  view,
  moveView,
  paint,
  onAbandon,
  onStroke,
  onTap,
}: {
  stageRef: RefObject<HTMLElement | null>
  /** The mask has loaded and can be painted. */
  ready: boolean
  selecting: boolean
  erase: boolean
  /** The brush's width, in working px. */
  brush: number
  view: View
  moveView: (change: (view: View) => View) => void
  /** Draw a stroke in progress from its point `from` on. */
  paint: (stroke: Stroke, from: number) => void
  /** A stroke turned out to be the start of a pinch: draw the mask without it. */
  onAbandon: () => void
  onStroke: (stroke: Stroke) => void
  /** A tap at `at` (working px) while selecting. */
  onTap: (at: Point, long: boolean) => void
}) {
  // Where the brush is, in stage px, to draw its outline.
  const [ring, setRing] = useState<Point | null>(null)
  const stroke = useRef<Stroke | null>(null)
  const pointers = useRef(new Map<number, Point>())
  const tap = useRef<(Point & { long: boolean; timer: number }) | null>(null)

  const cancelTap = () => {
    if (tap.current) window.clearTimeout(tap.current.timer)
    tap.current = null
  }

  const onPointerDown = (e: PointerEvent<HTMLDivElement>) => {
    if (!ready || (e.pointerType === 'mouse' && e.button !== 0)) return
    e.currentTarget.setPointerCapture(e.pointerId)
    const p = stagePoint(stageRef.current, e)
    pointers.current.set(e.pointerId, p)
    if (pointers.current.size > 1) {
      // A second finger: this is a pinch, not paint.
      cancelTap()
      if (stroke.current) {
        stroke.current = null
        onAbandon()
      }
      return
    }
    if (selecting) {
      const timer = window.setTimeout(() => {
        if (tap.current) tap.current.long = true
      }, LONG_PRESS_MS)
      tap.current = { ...p, long: false, timer }
      return
    }
    const at = toImage(view, p.x, p.y)
    stroke.current = { kind: 'stroke', erase, size: brush, points: [[at.x, at.y]] }
    paint(stroke.current, 0)
  }

  const onPointerMove = (e: PointerEvent<HTMLDivElement>) => {
    const p = stagePoint(stageRef.current, e)
    if (!selecting) setRing(p)
    const prev = pointers.current.get(e.pointerId)
    if (!prev) return
    const other = [...pointers.current].find(([id]) => id !== e.pointerId)?.[1]
    pointers.current.set(e.pointerId, p)
    if (other) {
      moveView((v) => pinch(v, prev, p, other))
      return
    }
    if (tap.current && Math.hypot(p.x - tap.current.x, p.y - tap.current.y) > TAP_SLOP) {
      cancelTap()
    }
    const s = stroke.current
    if (s) {
      const at = toImage(view, p.x, p.y)
      s.points.push([at.x, at.y])
      paint(s, s.points.length - 1)
    }
  }

  const onPointerUp = (e: PointerEvent<HTMLDivElement>) => {
    pointers.current.delete(e.pointerId)
    const s = stroke.current
    if (s) {
      stroke.current = null
      onStroke(s)
    }
    const t = tap.current
    if (t && selecting) onTap(toImage(view, t.x, t.y), t.long)
    cancelTap()
  }

  return {
    ring: selecting ? null : ring,
    onPointerDown,
    onPointerMove,
    onPointerUp,
    onPointerCancel: onPointerUp,
    onPointerLeave: () => {
      setRing(null)
    },
  }
}
