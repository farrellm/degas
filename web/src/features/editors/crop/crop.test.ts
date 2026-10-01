import { describe, expect, it } from 'vitest'
import {
  buildOps,
  centered,
  clampView,
  cropOf,
  dragCorner,
  exactCrop,
  FREE_ASPECTS,
  frameFor,
  modelUpscale,
  outputSize,
  parseOps,
  reshape,
  rotated,
  upscale,
  viewFor,
  zoomAt,
} from './crop'

const SDXL = { multiple_of: 8, min_pixels: 512 * 512, max_pixels: 1536 * 1536 }
const WAN_5B = { multiple_of: 32, min_pixels: 480 * 832, max_pixels: 1280 * 736 }

describe('crop geometry', () => {
  it('fits the frame to the stage and keeps it covered by the image', () => {
    const frame = frameFor({ w: 360, h: 480 }, 16 / 9)
    expect(frame.w).toBeCloseTo(320)
    expect(frame.x).toBeCloseTo(20)
    expect(frame.y + frame.h / 2).toBeCloseTo(240)

    const image = { w: 3000, h: 2000 }
    // Zoomed out too far and dragged away: pulled back to cover the frame.
    const view = clampView({ s: 0.01, tx: 500, ty: 500 }, frame, image)
    expect(image.w * view.s).toBeGreaterThanOrEqual(frame.w - 1e-9)
    expect(image.h * view.s).toBeGreaterThanOrEqual(frame.h - 1e-9)
    expect(view.tx).toBeLessThanOrEqual(frame.x)
    expect(view.ty).toBeLessThanOrEqual(frame.y)
  })

  it('round-trips a crop through the view', () => {
    const frame = frameFor({ w: 360, h: 480 }, 1)
    const crop = { x: 400, y: 120, w: 900, h: 900 }
    const view = viewFor(crop, frame)
    expect(cropOf(view, frame, { w: 2000, h: 1500 })).toEqual(crop)
    expect(exactCrop(view, frame).w).toBeCloseTo(900)
    const zoomed = zoomAt(view, 2, frame.x + frame.w / 2, frame.y + frame.h / 2)
    const tighter = exactCrop(zoomed, frame)
    expect(tighter.w).toBeCloseTo(450) // zooming about the middle keeps it centred
    expect(tighter.x + tighter.w / 2).toBeCloseTo(850)
  })

  it('sizes the output to the form, or to the crop at about the same pixel count', () => {
    const crop = { x: 0, y: 0, w: 2000, h: 1000 }
    expect(outputSize(crop, 'match', { w: 832, h: 1216 }, SDXL)).toEqual({ w: 832, h: 1216 })
    const free = outputSize(crop, 'free', { w: 1024, h: 1024 }, SDXL)
    expect(free.w % 8).toBe(0)
    expect(free.w / free.h).toBeCloseTo(2, 1)
    const wan = outputSize({ x: 0, y: 0, w: 1000, h: 1000 }, '1:1', { w: 1280, h: 704 }, WAN_5B)
    expect(wan.w % 32).toBe(0)
    expect(wan.w * wan.h).toBeLessThanOrEqual(WAN_5B.max_pixels)
    expect(upscale({ x: 0, y: 0, w: 400, h: 300 }, { w: 1280, h: 704 })).toBeCloseTo(3.2)
  })

  it('writes only the operations that change something, and reads them back', () => {
    const image = { w: 1000, h: 800 }
    expect(buildOps(0, false, { x: 0, y: 0, ...image }, image, { w: 1000, h: 800 })).toEqual([])
    const ops = buildOps(90, true, { x: 10, y: 0, w: 600, h: 800 }, rotated(image, 90), {
      w: 480,
      h: 640,
    })
    expect(ops).toEqual([
      { op: 'rotate', deg: 90 },
      { op: 'flip_h' },
      { op: 'crop', x: 10, y: 0, w: 600, h: 800 },
      { op: 'resize', w: 480, h: 640 },
    ])
    expect(parseOps(ops)).toEqual({
      rot: 90,
      flip: true,
      crop: { x: 10, y: 0, w: 600, h: 800 },
      resized: true,
    })
    expect(parseOps(ops.slice(0, 3)).resized).toBe(false)
  })

  it('keeps a free crop at its own size, and says how far the model scales it up', () => {
    expect(FREE_ASPECTS.map((a) => a.id)).not.toContain('match')
    const image = { w: 1000, h: 800 }
    const crop = { x: 100, y: 50, w: 300, h: 400 }
    expect(buildOps(0, false, crop, image, { w: 300, h: 400 })).toEqual([{ op: 'crop', ...crop }])
    expect(modelUpscale({ w: 512, h: 512 }, { w: 2048, h: 2048 })).toBeCloseTo(4)
    expect(modelUpscale({ w: 4096, h: 2048 }, { w: 2048, h: 2048 })).toBeLessThan(1)
  })

  it('reshapes a crop about its centre, staying on the image', () => {
    const image = { w: 1000, h: 1000 }
    expect(centered(image, 2)).toEqual({ x: 0, y: 250, w: 1000, h: 500 })
    const r = reshape({ x: 800, y: 0, w: 200, h: 200 }, 16 / 9, image)
    expect(r.w / r.h).toBeCloseTo(16 / 9)
    expect(r.x + r.w).toBeLessThanOrEqual(1000)
  })

  it('drags a free corner without leaving the image or collapsing', () => {
    const frame = { x: 50, y: 50, w: 200, h: 200 }
    const bounds = { x: 0, y: 0, w: 300, h: 300 }
    expect(dragCorner(frame, 'se', 500, 500, bounds)).toEqual({ x: 50, y: 50, w: 250, h: 250 })
    expect(dragCorner(frame, 'nw', 500, 0, bounds).w).toBe(40)
  })
})
