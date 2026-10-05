import { describe, expect, it } from 'vitest'

import type { Asset, Family, Variant } from '@/api/types'

import { modelForMode, resolveSelection, variantOf } from './selection'

const variant = (id: string, modes: string[], extra: Partial<Variant> = {}): Variant => ({
  id,
  label: id,
  min_gpu: 'T4',
  modes,
  model_dir: null,
  lora_format: 'single',
  max_refs: 0,
  size_constraints: { multiple_of: 8, min_pixels: 0, max_pixels: 0, presets: [] },
  ...extra,
})

const family = (id: string, variants: Variant[], extra: Partial<Family> = {}): Family => ({
  id,
  label: id,
  media: 'image',
  lora_format: 'single',
  supports_control: false,
  supports_image_prompts: false,
  extendable: false,
  variants,
  ...extra,
})

const asset = (path: string, fam: string, kind = 'model'): Asset => ({
  path,
  family: fam,
  kind,
  size: null,
  sidecar: null,
  preview_thumb: null,
  indexed_at: '',
})

const SDXL = family('sdxl', [variant('base', ['t2i', 'i2i', 'inpaint'])], {
  supports_control: true,
})
const QWEN = family('qwen', [
  variant('image', ['t2i'], { model_dir: 'models/qwen/image' }),
  variant('edit', ['edit', 'inpaint'], { model_dir: 'models/qwen/edit', max_refs: 2 }),
])
const WAN = family('wan22', [variant('ti2v', ['t2v', 'i2v'])], { media: 'video' })
const FAMILIES = [SDXL, QWEN, WAN]
const ASSETS = [
  asset('models/sdxl/a.safetensors', 'sdxl'),
  asset('models/sdxl/b.safetensors', 'sdxl'),
  asset('models/qwen/image/q', 'qwen'),
  asset('models/qwen/edit/e', 'qwen'),
  asset('loras/sdxl/film.safetensors', 'sdxl', 'lora'),
  asset('controlnets/sdxl/depth.safetensors', 'sdxl', 'controlnet'),
]

const resolve = (familyId: string, chosenModel = '', chosenMode?: string) =>
  resolveSelection({ families: FAMILIES, assets: ASSETS, familyId, chosenModel, chosenMode })

describe('resolveSelection', () => {
  it('starts on the first model and its first mode', () => {
    const s = resolve('sdxl')
    expect(s.model).toBe('models/sdxl/a.safetensors')
    expect(s.mode).toBe('t2i')
    expect(s.variant?.id).toBe('base')
    expect(s.needsSource).toBe(false)
    expect(s.withControl).toBe(true)
  })

  it('offers the modes of every family making the same media, in chip order', () => {
    expect(resolve('sdxl').allModes).toEqual(['t2i', 'i2i', 'edit', 'inpaint'])
    expect(resolve('wan22').allModes).toEqual(['t2v', 'i2v'])
  })

  it('offers every model of the same media that can do the mode', () => {
    expect(resolve('sdxl', '', 'inpaint').pickable?.map((a) => a.path)).toEqual([
      'models/sdxl/a.safetensors',
      'models/sdxl/b.safetensors',
      'models/qwen/edit/e',
    ])
  })

  it('takes the first model that can do the mode when the chosen one cannot', () => {
    const s = resolve('qwen', 'models/qwen/image/q', 'edit')
    expect(s.model).toBe('models/qwen/edit/e')
    expect(s.variant?.id).toBe('edit')
    expect(s.needsSource).toBe(true)
    expect(s.takesRefs).toBe(true)
    expect(s.maxRefs).toBe(2)
  })

  it('keeps a model that left Drive, flagged', () => {
    const s = resolve('sdxl', 'models/sdxl/gone.safetensors')
    expect(s.model).toBe('models/sdxl/gone.safetensors')
    expect(s.modelMissing).toBe(true)
  })

  it("lists only the family's own LoRAs and ControlNets", () => {
    expect(resolve('sdxl').loras?.map((a) => a.path)).toEqual(['loras/sdxl/film.safetensors'])
    expect(resolve('qwen').controlnets).toEqual([])
  })

  it('waits for the families and the index', () => {
    const s = resolveSelection({
      families: undefined,
      assets: undefined,
      familyId: 'sdxl',
      chosenModel: '',
      chosenMode: undefined,
    })
    expect(s).toMatchObject({ family: undefined, mode: undefined, model: '', modelMissing: false })
  })
})

describe('variantOf', () => {
  it("is the model's variant, else the first that does the mode, else the first", () => {
    expect(variantOf(QWEN, 'models/qwen/edit/e', 't2i')?.id).toBe('edit')
    expect(variantOf(QWEN, undefined, 'inpaint')?.id).toBe('edit')
    expect(variantOf(QWEN, undefined, undefined)?.id).toBe('image')
  })
})

describe('modelForMode', () => {
  it("finds a family's first model for a mode", () => {
    expect(modelForMode(ASSETS, QWEN, 'edit')?.path).toBe('models/qwen/edit/e')
    expect(modelForMode(ASSETS, WAN, 't2v')).toBeUndefined()
  })
})
