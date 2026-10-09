import { act, renderHook } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'

import { loadDraft, switchFamily } from './draft'
import { useDraft } from './useDraft'

const KEY = 'degas.create.draft'

describe('loadDraft', () => {
  beforeEach(() => localStorage.clear())

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

describe('loadDraft, from a draft it can’t use as it is', () => {
  beforeEach(() => localStorage.clear())

  it('drops only the fields of the wrong type', () => {
    localStorage.setItem(
      KEY,
      JSON.stringify({
        family: 'flux',
        families: {
          flux: { model: 'm', loras: 'none', fit: 'sideways', params: { prompt: 'a cat' } },
          sdxl: 'not a draft',
        },
        batchCount: 0,
        seedMode: 'sometimes',
      }),
    )
    const draft = loadDraft()
    expect(draft).toMatchObject({ family: 'flux', batchCount: 1, seedMode: 'random' })
    expect(draft.families.flux).toEqual({ model: 'm', params: { prompt: 'a cat' } })
    expect(draft.families.sdxl).toEqual({})
  })

  it('starts over from what isn’t a draft at all', () => {
    for (const stored of ['{not json', '[1, 2]', '"draft"', 'null']) {
      localStorage.setItem(KEY, stored)
      expect(loadDraft()).toEqual({
        family: 'sdxl',
        families: {},
        batchCount: 1,
        seedMode: 'random',
        recent: ['sdxl'],
      })
    }
  })

  it('reads the same draft back until it changes', () => {
    localStorage.setItem(KEY, JSON.stringify({ family: 'flux' }))
    const draft = loadDraft()
    expect(loadDraft()).toBe(draft)
    switchFamily('sdxl', { prompt: '' })
    expect(loadDraft()).not.toBe(draft)
  })
})

describe('useDraft', () => {
  beforeEach(() => localStorage.clear())

  it('follows the draft as it is saved, here or in another tab', () => {
    const { result } = renderHook(() => useDraft())
    expect(result.current.family).toBe('sdxl')

    act(() => switchFamily('flux', { prompt: 'a cat' }))
    expect(result.current.family).toBe('flux')

    act(() => {
      localStorage.setItem(KEY, JSON.stringify({ family: 'wan' }))
      window.dispatchEvent(new StorageEvent('storage', { key: KEY }))
    })
    expect(result.current.family).toBe('wan')
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
