import { beforeEach, describe, expect, it } from 'vitest'
import { loadDraft, switchFamily } from './draft'

describe('loadDraft', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  it('fills in fields that units saved before they existed lack', () => {
    // A FaceID image prompt from before Phase 12 (no structure or LoRA weight) and a
    // ControlNet unit missing its fit.
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: {
          sdxl: {
            model: 'm',
            loras: [],
            params: {},
            prompts: [
              {
                key: 'p',
                model: 'ip_adapters/sdxl/ip-adapter-faceid-plusv2_sdxl.bin',
                take: 'face',
                pictures: [],
                area: null,
                weight: 0.8,
                start: 0,
                end: 1,
                fit: 'crop',
              },
            ],
            control: [{ key: 'c', model: 'x', image: null, trace: null, area: null, scale: 0.5 }],
          },
        },
      }),
    )
    const fd = loadDraft().families.sdxl
    expect(fd?.prompts?.[0]).toMatchObject({
      key: 'p',
      weight: 0.8,
      structure: 1,
      loraWeight: 0.6,
      downsample: 3,
    })
    expect(fd?.control?.[0]).toMatchObject({ key: 'c', scale: 0.5, start: 0, end: 1, fit: 'crop' })
  })
})

describe('switchFamily', () => {
  beforeEach(() => {
    localStorage.clear()
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: { flux: { model: 'm', params: { prompt: 'old', negative_prompt: 'blur' } } },
      }),
    )
  })

  it('fills only an empty prompt when switching Image ⇄ Video', () => {
    switchFamily('flux', { prompt: 'a cat' })
    expect(loadDraft().families.flux?.params?.prompt).toBe('old')
  })

  it('keeps the prompt over the other family’s when changing model', () => {
    switchFamily('flux', { prompt: 'a cat', negative: '', keep: true })
    expect(loadDraft().families.flux?.params).toMatchObject({
      prompt: 'a cat',
      negative_prompt: '',
    })
  })

  it('keeps the chosen images over the other family’s when changing model', () => {
    const img = (sha: string) => ({ sha, width: 64, height: 64 })
    localStorage.setItem(
      'degas.create.draft',
      JSON.stringify({
        family: 'sdxl',
        families: {
          flux: {
            model: 'm',
            params: {},
            source: img('old'),
            mask: { sha: 'k', source: 'old' },
            refs: [img('r0')],
          },
        },
      }),
    )
    switchFamily('flux', { prompt: '', source: img('s'), refs: [img('r1')], keep: true })
    const fd = loadDraft().families.flux
    expect(fd?.source?.sha).toBe('s')
    expect(fd?.mask).toBeNull()
    expect(fd?.refs?.map((r) => r.sha)).toEqual(['r1'])

    // Without images chosen, the other family's are cleared too.
    switchFamily('flux', { prompt: '', source: null, refs: [], keep: true })
    expect(loadDraft().families.flux).toMatchObject({ source: null, refs: [] })
  })
})
