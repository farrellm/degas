import { describe, expect, it } from 'vitest'
import type { AdapterKind, Asset } from './api'
import {
  adapterKind,
  anyOblong,
  mismatch,
  modelFor,
  newPrompt,
  promptFromSpec,
  promptSpec,
  promptSummary,
} from './imagePrompt'

const asset = (name: string, purpose?: AdapterKind): Asset => ({
  path: `ip_adapters/sdxl/${name}`,
  family: 'sdxl',
  kind: 'ip_adapter',
  size: 1,
  sidecar: purpose ? { purpose } : null,
  preview_thumb: null,
  indexed_at: '2026-09-30T00:00:00Z',
})

describe('image prompts', () => {
  it('tells a model’s kind from its sidecar, else its name', () => {
    expect(adapterKind(asset('ip-adapter-plus-face_sdxl_vit-h.safetensors'))).toBe('face')
    expect(adapterKind(asset('ip_plus_composition_sdxl.safetensors'))).toBe('composition')
    expect(adapterKind(asset('noob.bin'))).toBe('subject')
    expect(adapterKind(asset('noob.bin', 'face'))).toBe('face')
  })

  it('picks a plus model of the kind a choice wants', () => {
    const base = asset('ip-adapter_sdxl_vit-h.safetensors')
    const plus = asset('ip-adapter-plus_sdxl_vit-h.safetensors')
    const face = asset('ip-adapter-plus-face_sdxl_vit-h.safetensors')
    expect(modelFor([base, plus, face], 'style')).toBe(plus)
    expect(modelFor([base, face], 'all')).toBe(base)
    expect(modelFor([base, plus, face], 'face')).toBe(face)
    // Layout prefers a composition model, and falls back to a subject one.
    expect(modelFor([plus], 'layout')).toBe(plus)
    expect(modelFor([plus, asset('composition.bin')], 'layout')?.path).toMatch(/composition/)
    expect(modelFor([base], 'face')).toBeUndefined()
  })

  it('warns when the model doesn’t suit the choice', () => {
    const face = asset('plus-face.bin')
    expect(mismatch(face, face.path, 'style')).toBe(
      'This is a face model; it reads faces, not style.',
    )
    expect(mismatch(undefined, 'ip_adapters/sdxl/plus.bin', 'face')).toMatch(/isn’t a face model/)
    expect(mismatch(undefined, 'ip_adapters/sdxl/composition.bin', 'style')).toMatch(/layout/)
    expect(mismatch(undefined, 'ip_adapters/sdxl/plus.bin', 'style_layout')).toBeNull()
  })

  it('summarizes a unit for its row', () => {
    const unit = {
      ...newPrompt(),
      take: 'style' as const,
      weight: 1,
      pictures: [
        { sha: 'a', width: 10, height: 10 },
        { sha: 'b', width: 20, height: 10 },
      ],
      end: 0.5,
    }
    expect(promptSummary(unit, 30)).toBe('Style, weight 1.00, 2 pictures, steps 1–15')
    expect(anyOblong(unit)).toBe(true)
  })

  it('round-trips through a spec, remembering a face choice', () => {
    const unit = {
      ...newPrompt(),
      model: 'ip_adapters/sdxl/ip-adapter-plus-face_sdxl_vit-h.safetensors',
      take: 'face' as const,
      pictures: [{ sha: 'a', width: 512, height: 512 }],
      area: { sha: 'm', over: { sha: 'c', width: 1024, height: 768 } },
    }
    const spec = promptSpec(unit)
    expect(spec).toMatchObject({ purpose: 'all', images: ['sha256:a'], mask: 'sha256:m' })
    const back = promptFromSpec(spec, { w: 1024, h: 768 })
    expect(back).toMatchObject({
      model: unit.model,
      take: 'face',
      pictures: [{ sha: 'a' }],
      area: { sha: 'm', over: { width: 1024, height: 768 } },
    })
  })
})
