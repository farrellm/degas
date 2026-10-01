import { describe, expect, it } from 'vitest'
import type { Asset } from './api'
import {
  controlKind,
  edgeDetail,
  edgeParams,
  mismatch,
  newUnit,
  stepSpan,
  stepsLabel,
  underlay,
  unitFromSpec,
  unitSpec,
  unitSummary,
} from './control'

const asset = (path: string, control?: 'depth' | 'pose' | 'canny'): Asset => ({
  path,
  family: 'sdxl',
  kind: 'controlnet',
  size: 1,
  sidecar: control ? { control } : null,
  preview_thumb: null,
  indexed_at: '',
})

describe('steps', () => {
  it('counts the steps diffusers runs a unit on', () => {
    expect(stepSpan(0, 1, 30)).toEqual({ first: 1, last: 30 })
    expect(stepSpan(0, 0.8, 30)).toEqual({ first: 1, last: 24 })
    expect(stepSpan(0.1, 0.5, 30)).toEqual({ first: 4, last: 15 })
    expect(stepSpan(0.5, 0.51, 30)).toBeNull()
  })

  it('reads as a range of steps', () => {
    expect(stepsLabel(0, 0.8, 30)).toBe('Steps 1–24 of 30')
    expect(stepsLabel(0, 1 / 30, 30)).toBe('Step 1 of 30')
    expect(stepsLabel(0.5, 0.51, 30)).toBe('No steps. Widen the range.')
  })
})

describe('models and traces', () => {
  it('reads the kind from the sidecar, else the file name', () => {
    expect(controlKind(asset('controlnets/sdxl/xinsir.safetensors', 'pose'))).toBe('pose')
    expect(controlKind(asset('controlnets/sdxl/diffusers-depth-xl'))).toBe('depth')
    expect(controlKind(asset('controlnets/sdxl/OpenPoseXL2.safetensors'))).toBe('pose')
    expect(controlKind(asset('controlnets/sdxl/canny-sdxl.safetensors'))).toBe('canny')
    expect(controlKind(asset('controlnets/sdxl/tile.safetensors'))).toBeNull()
  })

  it('warns when the model reads another kind of image', () => {
    const canny = asset('controlnets/sdxl/canny.safetensors')
    expect(mismatch(canny, canny.path, 'depth')).toBe(
      'This looks like an edges model; the image is a depth map.',
    )
    expect(mismatch(canny, canny.path, 'canny')).toBeNull()
    expect(mismatch(canny, canny.path, null)).toBeNull()
  })

  it('maps edge detail to Canny thresholds and back', () => {
    expect(edgeParams(0.25)).toEqual({ low: 100, high: 200 })
    expect(edgeParams(1)).toEqual({ low: 25, high: 50 })
    expect(edgeDetail(edgeParams(0.6))).toBe(0.6)
  })
})

describe('units', () => {
  const photo = { sha: 'p', width: 1600, height: 1200 }
  const traced = {
    ...newUnit(),
    model: 'controlnets/sdxl/depth.safetensors',
    image: { sha: 't', width: 1600, height: 1200 },
    trace: { id: 'depth' as const, params: {}, from: photo },
    area: 'm',
    end: 0.8,
  }

  it('summarizes a unit for its row', () => {
    expect(unitSummary(traced, 30)).toBe('Depth, weight 0.70, steps 1–24, in an area')
    expect(unitSummary(newUnit(), 30)).toBe('No image yet, weight 0.70')
  })

  it('paints the area over the photo a trace came from', () => {
    expect(underlay(traced)).toBe('p')
    expect(underlay({ ...traced, trace: { ...traced.trace, from: { ...photo, width: 0 } } })).toBe(
      't',
    )
  })

  it('round-trips through a spec', () => {
    const spec = unitSpec(traced)
    expect(spec).toEqual({
      controlnet: { path: 'controlnets/sdxl/depth.safetensors' },
      image: 'sha256:t',
      fit: 'crop',
      scale: 0.7,
      start: 0,
      end: 0.8,
      mask: 'sha256:m',
      preprocessor: { id: 'depth', source: 'sha256:p', params: {} },
    })
    const back = unitFromSpec(spec, { w: 1024, h: 768 })
    expect(back.image).toEqual({ sha: 't', width: 1024, height: 768 })
    expect(back.trace?.from).toEqual({ sha: 'p', width: 0, height: 0 })
    expect(back.area).toBe('m')
    expect(back.end).toBe(0.8)
  })
})
