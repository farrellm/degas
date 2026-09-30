import { beforeEach, describe, expect, it } from 'vitest'
import { loadDraft } from './draft'

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
