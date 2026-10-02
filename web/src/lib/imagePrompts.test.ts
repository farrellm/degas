import { describe, expect, it } from 'vitest'

import type { ImagePromptSpec } from '@/api/types'

import { specSummary } from './imagePrompts'

const unit = (over: Partial<ImagePromptSpec>): ImagePromptSpec => ({
  adapter: { path: 'ip_adapters/sdxl/ip-adapter-plus_sdxl_vit-h.safetensors' },
  images: ['sha256:a'],
  purpose: 'all',
  weight: 0.6,
  start: 0,
  end: 1,
  ...over,
})

describe('a finished image prompt', () => {
  it('says what it took, how much, when and where', () => {
    expect(specSummary(unit({}), 30)).toBe('Everything, weight 0.60')
    expect(specSummary(unit({ purpose: 'style', weight: 1, end: 0.8, mask: 'sha256:m' }), 30)).toBe(
      'Style, weight 1.00, steps 1–24, in an area',
    )
    // Its pictures are shown beside it, so they aren't counted.
    expect(specSummary(unit({ images: ['sha256:a', 'sha256:b'] }), 30)).toBe(
      'Everything, weight 0.60',
    )
  })

  it('reads a face model as Face', () => {
    const adapter = { path: 'ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin' }
    expect(specSummary(unit({ adapter, weight: 0.8 }), 30)).toBe('Face, weight 0.80')
  })

  it('reads Redux by how closely it followed the picture', () => {
    const adapter = { path: 'redux/flux1-redux-dev.safetensors' }
    expect(specSummary(unit({ adapter, weight: 1, downsample: 3 }), 28)).toBe(
      'Loosely, weight 1.00',
    )
  })
})
