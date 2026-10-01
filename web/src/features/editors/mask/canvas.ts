import { blobUrl } from '@/api/urls'
import type { Size } from '@/lib/geometry'

import { binarizeAlpha, growSteps, lumaToAlpha, ringOffsets } from './mask'

// Canvas drawing for the mask editor: the mask is an alpha layer at the working size, which
// strokes and selections are composited onto.

export type Combine = 'add' | 'subtract' | 'replace'

export interface Stroke {
  kind: 'stroke'
  erase: boolean
  size: number
  points: [number, number][]
}

export type Action =
  | Stroke
  | { kind: 'invert' }
  | { kind: 'clear' }
  | { kind: 'selection'; op: Combine; layer: HTMLCanvasElement }

export function canvas(size: Size): HTMLCanvasElement {
  const c = document.createElement('canvas')
  c.width = size.w
  c.height = size.h
  return c
}

export const context = (c: HTMLCanvasElement | null) => c?.getContext('2d') ?? null

/** Rose hatching as a canvas pattern; strokes drawn with it join without seams. */
export function hatchPattern(ctx: CanvasRenderingContext2D, gap: number): CanvasPattern | string {
  const color =
    getComputedStyle(document.documentElement).getPropertyValue('--hatch').trim() || '#efa3b5'
  const t = Math.max(3, Math.round(gap))
  const tile = canvas({ w: t, h: t })
  const tc = context(tile)
  if (!tc) return color
  tc.strokeStyle = color
  tc.lineWidth = Math.max(1, t / 4.5)
  tc.lineCap = 'square'
  tc.beginPath()
  for (const o of [-t, 0, t]) {
    tc.moveTo(o, t)
    tc.lineTo(o + t, 0)
  }
  tc.stroke()
  return ctx.createPattern(tile, 'repeat') ?? color
}

export function drawStroke(
  ctx: CanvasRenderingContext2D,
  stroke: Stroke,
  style: CanvasPattern | string,
  from = 0,
) {
  const pts = stroke.points.slice(Math.max(0, from - 1))
  const first = pts[0]
  if (!first) return
  ctx.save()
  ctx.globalCompositeOperation = stroke.erase ? 'destination-out' : 'source-over'
  ctx.strokeStyle = style
  ctx.fillStyle = style
  ctx.lineWidth = stroke.size
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.beginPath()
  if (pts.length === 1) {
    ctx.arc(first[0], first[1], stroke.size / 2, 0, 2 * Math.PI)
    ctx.fill()
  } else {
    ctx.moveTo(first[0], first[1])
    for (const [x, y] of pts.slice(1)) ctx.lineTo(x, y)
    ctx.stroke()
  }
  ctx.restore()
}

/** Apply one action to the mask (white where it will be redrawn, in the alpha channel). */
export function apply(ctx: CanvasRenderingContext2D, action: Action, size: Size) {
  switch (action.kind) {
    case 'stroke':
      drawStroke(ctx, action, '#fff')
      break
    case 'clear':
      ctx.clearRect(0, 0, size.w, size.h)
      break
    case 'invert':
      ctx.save()
      ctx.globalCompositeOperation = 'xor'
      ctx.fillStyle = '#fff'
      ctx.fillRect(0, 0, size.w, size.h)
      ctx.restore()
      break
    case 'selection':
      ctx.save()
      if (action.op === 'replace') ctx.clearRect(0, 0, size.w, size.h)
      ctx.globalCompositeOperation = action.op === 'subtract' ? 'destination-out' : 'source-over'
      ctx.drawImage(action.layer, 0, 0)
      ctx.restore()
  }
}

/** A stored grey mask as an alpha layer at the working size. */
export async function loadLayer(sha: string, size: Size): Promise<HTMLCanvasElement> {
  const img = new Image()
  img.src = blobUrl(sha)
  await img.decode()
  const layer = canvas(size)
  const ctx = context(layer)
  if (ctx) {
    ctx.drawImage(img, 0, 0, size.w, size.h)
    const data = ctx.getImageData(0, 0, size.w, size.h)
    lumaToAlpha(data.data)
    ctx.putImageData(data, 0, 0)
  }
  return layer
}

/** A layer grown by `r` px, to give a selection a margin to blend into. */
export function grow(layer: HTMLCanvasElement, r: number): HTMLCanvasElement {
  const steps = growSteps(r)
  if (!steps.length) return layer
  const size = { w: layer.width, h: layer.height }
  let from = canvas(size)
  const first = context(from)
  if (!first) return layer
  first.drawImage(layer, 0, 0)
  const data = first.getImageData(0, 0, size.w, size.h)
  binarizeAlpha(data.data)
  first.putImageData(data, 0, 0)
  for (const step of steps) {
    const out = canvas(size)
    const ctx = context(out)
    if (!ctx) return from
    ctx.imageSmoothingEnabled = false
    ctx.drawImage(from, 0, 0)
    for (const o of ringOffsets(step)) ctx.drawImage(from, o.x, o.y)
    from = out
  }
  return from
}

/** The edge of a layer, `width` px thick, for showing a selection before it's used. */
export function outline(layer: HTMLCanvasElement, width: number, style: string): HTMLCanvasElement {
  const size = { w: layer.width, h: layer.height }
  const inner = canvas(size)
  const ic = context(inner)
  const out = canvas(size)
  const oc = context(out)
  if (!ic || !oc) return out
  ic.drawImage(layer, 0, 0)
  ic.globalCompositeOperation = 'destination-in'
  for (const [dx, dy] of [
    [width, 0],
    [-width, 0],
    [0, width],
    [0, -width],
  ] as const) {
    ic.drawImage(layer, dx, dy)
  }
  oc.drawImage(layer, 0, 0)
  oc.globalCompositeOperation = 'destination-out'
  oc.drawImage(inner, 0, 0)
  oc.globalCompositeOperation = 'source-in'
  oc.fillStyle = style
  oc.fillRect(0, 0, size.w, size.h)
  return out
}
