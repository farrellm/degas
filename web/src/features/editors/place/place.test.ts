import { describe, expect, it } from 'vitest'

import { align, clampPlace, defaultPlace, margins, rescale, scaleOf, validPlace } from './place'

const canvas = { w: 1344, h: 768 }
const square = { w: 1024, h: 1024 }

describe('outpaint placement', () => {
  it('fits the source inside and centres it', () => {
    expect(defaultPlace(square, canvas)).toEqual({ x: 288, y: 0, w: 768, h: 768 })
    expect(margins(defaultPlace(square, canvas), canvas)).toBe('+288 px left, +288 px right')
  })

  it('leaves room to draw when the source already has the canvas shape', () => {
    const p = defaultPlace({ w: 1344, h: 768 }, canvas)
    expect(p).toEqual({ x: 168, y: 96, w: 1008, h: 576 })
    expect(validPlace(p, { w: 1344, h: 768 }, canvas)).toBe(true)
    expect(validPlace({ x: 0, y: 0, ...canvas }, canvas, canvas)).toBe(false)
  })

  it('keeps placements inside the canvas on an 8 px grid', () => {
    expect(clampPlace({ x: -30, y: 5, w: 768, h: 768 }, canvas)).toEqual({
      x: 0,
      y: 0,
      w: 768,
      h: 768,
    })
    expect(clampPlace({ x: 1000, y: 0, w: 768, h: 768 }, canvas).x).toBe(576)
  })

  it('rescales about the centre and aligns to edges', () => {
    const p = defaultPlace(square, canvas)
    const half = rescale(p, square, canvas, 0.5)
    expect(half).toEqual({ x: 480, y: 192, w: 384, h: 384 })
    expect(scaleOf(half, square, canvas)).toBeCloseTo(0.5)
    expect(align(half, canvas, 'right').x).toBe(960)
    expect(align(half, canvas, 'top').y).toBe(0)
    expect(rescale(p, square, canvas, 0.01).w).toBe(64)
  })
})
