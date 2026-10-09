import { type RefObject, useCallback, useEffect, useEffectEvent, useRef, useState } from 'react'

import type { Size } from '@/lib/geometry'

import {
  type Action,
  apply,
  canvas,
  context,
  drawStroke,
  hatchPattern,
  loadLayer,
  type Stroke,
} from './canvas'
import { hasAlpha, type History } from './mask'

/**
 * The mask being painted. It lives off screen as an alpha layer at the working size `work`:
 * the stored mask it started from, then every action done in `history`. What's on screen
 * (`displayRef`) is that layer filled with hatching, `hatchGap` working px between strokes.
 */
export function useMaskCanvas({
  mask,
  displayRef,
  work,
  history,
  hatchGap,
}: {
  /** The stored mask to start from, if there is one. */
  mask: string | null
  /** The canvas on screen. */
  displayRef: RefObject<HTMLCanvasElement | null>
  work: Size
  history: History<Action>
  hatchGap: number
}) {
  const [ready, setReady] = useState(!mask)
  const maskCanvas = useRef<HTMLCanvasElement | null>(null)
  const base = useRef<HTMLCanvasElement | null>(null)
  const pattern = useRef<CanvasPattern | string>('#efa3b5')

  // The mask itself lives off screen; the display is the mask filled with hatching.
  const recomposite = useCallback(() => {
    const m = maskCanvas.current
    const ctx = context(displayRef.current)
    if (!m || !ctx) return
    ctx.save()
    ctx.clearRect(0, 0, work.w, work.h)
    ctx.drawImage(m, 0, 0)
    ctx.globalCompositeOperation = 'source-in'
    ctx.fillStyle = pattern.current
    ctx.fillRect(0, 0, work.w, work.h)
    ctx.restore()
  }, [displayRef, work.w, work.h])

  /** Draw the mask again from its base and `done`: after an undo, or to drop a stroke in progress. */
  const render = useCallback(
    (done: Action[]) => {
      maskCanvas.current ??= canvas(work)
      const ctx = context(maskCanvas.current)
      if (!ctx) return
      ctx.clearRect(0, 0, work.w, work.h)
      if (base.current) ctx.drawImage(base.current, 0, 0)
      for (const a of done) apply(ctx, a, work)
      recomposite()
    },
    [work, recomposite],
  )

  // The hatching is sized for the fitted zoom, so it reads like the feed's sketch tiles.
  useEffect(() => {
    const ctx = context(displayRef.current)
    if (ctx) pattern.current = hatchPattern(ctx, hatchGap)
    recomposite()
  }, [displayRef, hatchGap, recomposite])

  // Load once, at open: the mask and size the editor opened with.
  const loadBase = useEffectEvent(() => (mask ? loadLayer(mask, work) : null))
  useEffect(() => {
    const loading = loadBase()
    if (!loading) return
    let live = true
    loading
      .then((layer) => {
        if (!live) return
        base.current = layer
        setReady(true)
      })
      .catch(() => {
        if (live) setReady(true) // start from an empty mask
      })
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    if (ready) render(history.done)
  }, [ready, history, render])

  /** Draw a stroke in progress, from its point `from` on, without waiting for a render. */
  const paint = (s: Stroke, from: number) => {
    const m = context(maskCanvas.current)
    const d = context(displayRef.current)
    if (m) drawStroke(m, s, '#fff', from)
    if (d) drawStroke(d, s, pattern.current, from)
  }

  /** The mask as a PNG, or null when nothing is painted. */
  const toPng = async (): Promise<Blob | null> => {
    const m = maskCanvas.current
    const ctx = context(m)
    if (!m || !ctx) throw new Error('This browser can’t paint masks.')
    if (!hasAlpha(ctx.getImageData(0, 0, work.w, work.h).data)) return null
    const png = await new Promise<Blob | null>((resolve) => m.toBlob(resolve, 'image/png'))
    if (!png) throw new Error('Couldn’t read the painted mask.')
    return png
  }

  return { ready, render, paint, toPng }
}
