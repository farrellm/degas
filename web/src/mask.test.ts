import { describe, expect, it } from 'vitest'
import {
  amend,
  clampView,
  emptyHistory,
  fitView,
  growOffsets,
  hasAlpha,
  lumaToAlpha,
  push,
  redo,
  undo,
  workingSize,
} from './mask'

describe('mask editor helpers', () => {
  it('paints big images at a smaller working size', () => {
    expect(workingSize({ w: 4032, h: 3024 })).toEqual({ w: 2048, h: 1536 })
    expect(workingSize({ w: 1024, h: 1024 })).toEqual({ w: 1024, h: 1024 })
  })

  it('fits, clamps and centres the view', () => {
    const stage = { w: 400, h: 600 }
    const image = { w: 1000, h: 1000 }
    const fit = fitView(stage, image)
    expect(fit.s).toBeCloseTo(0.376)
    expect(fit.tx).toBeCloseTo(12)
    // Zoomed out past fitting: back to fitting, centred.
    expect(clampView({ s: 0.1, tx: 0, ty: 0 }, stage, image)).toEqual(fit)
    // Zoomed in: no gap at the left edge.
    const zoomed = clampView({ s: 1, tx: 50, ty: -2000 }, stage, image)
    expect(zoomed).toEqual({ s: 1, tx: 0, ty: -400 })
  })

  it('undoes and redoes', () => {
    let h = push(push(emptyHistory<string>(), 'a'), 'b')
    h = amend(h, 'b2')
    h = undo(h)
    expect(h).toEqual({ done: ['a'], undone: ['b2'] })
    expect(redo(h)).toEqual({ done: ['a', 'b2'], undone: [] })
    expect(push(h, 'c').undone).toEqual([])
    expect(undo(emptyHistory())).toEqual(emptyHistory())
  })

  it('reads grey masks into alpha', () => {
    const data = new Uint8ClampedArray([0, 0, 0, 255, 200, 200, 200, 255])
    expect(hasAlpha(data)).toBe(true)
    lumaToAlpha(data)
    expect([...data]).toEqual([255, 255, 255, 0, 255, 255, 255, 200])
    expect(hasAlpha(new Uint8ClampedArray([9, 9, 9, 0]))).toBe(false)
  })

  it('grows in every direction', () => {
    const offsets = growOffsets(8)
    expect(Math.max(...offsets.map((o) => o.x))).toBeCloseTo(8)
    expect(Math.min(...offsets.map((o) => o.y))).toBeCloseTo(-8)
    expect(growOffsets(0)).toEqual([])
  })
})
