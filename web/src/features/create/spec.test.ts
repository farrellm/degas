import { describe, expect, it } from 'vitest'

import { buildJob, type FormState } from './spec'

const FORM: FormState = {
  family: 'sdxl',
  variant: 'base',
  mode: 't2i',
  model: 'models/sdxl/a.safetensors',
  loras: [],
  params: { prompt: 'a lighthouse', seed: 7 },
  inputs: null,
  control: [],
  prompts: [],
  promptOptions: { purposes: ['all'], areas: true, steps: true, faces: true, detail: false },
  batchCount: 1,
  seedMode: 'random',
}

const SOURCE = { sha: 'aaa', width: 1024, height: 1024 }
const INPUTS: NonNullable<FormState['inputs']> = {
  source: SOURCE,
  fit: 'crop',
  extends: null,
  mask: null,
  place: null,
  refs: [],
}

describe('buildJob', () => {
  it('uses the seed as set for one image', () => {
    const job = buildJob(FORM)
    expect(job.seedMode).toBe('increment')
    expect(job.spec).toEqual({
      family: 'sdxl',
      variant: 'base',
      mode: 't2i',
      model: { path: 'models/sdxl/a.safetensors' },
      loras: [],
      params: { prompt: 'a lighthouse', seed: 7 },
    })
  })

  it('ignores the seed for a batch with random seeds, and counts up from it otherwise', () => {
    const random = buildJob({ ...FORM, batchCount: 4 })
    expect(random.seedMode).toBe('random')
    expect(random.spec.params.seed).toBe(-1)

    const counting = buildJob({ ...FORM, batchCount: 4, seedMode: 'increment' })
    expect(counting.seedMode).toBe('increment')
    expect(counting.spec.params.seed).toBe(7)
  })

  it('names the source, and the mask only when inpainting', () => {
    const mask = { sha: 'mmm', source: 'aaa' }
    const inputs = { ...INPUTS, mask }
    expect(buildJob({ ...FORM, mode: 'i2i', inputs }).spec.inputs).toEqual({
      source: 'sha256:aaa',
      fit: 'crop',
    })
    expect(buildJob({ ...FORM, mode: 'inpaint', inputs }).spec.inputs).toEqual({
      source: 'sha256:aaa',
      fit: 'crop',
      mask: 'sha256:mmm',
    })
  })

  it('places an outpaint, continues a clip, and lists the images after the source', () => {
    const place = { x: 0, y: 64, w: 512, h: 512 }
    expect(
      buildJob({ ...FORM, mode: 'outpaint', inputs: { ...INPUTS, place } }).spec.inputs,
    ).toEqual({ source: 'sha256:aaa', fit: 'crop', place })
    const refs = [{ sha: 'bbb', width: 0, height: 0 }]
    const inputs = { ...INPUTS, extends: 'sha256:clip', refs }
    expect(buildJob({ ...FORM, mode: 'edit', inputs }).spec.inputs).toEqual({
      source: 'sha256:aaa',
      fit: 'crop',
      extends: 'sha256:clip',
      refs: ['sha256:bbb'],
    })
  })
})
