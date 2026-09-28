import { describe, expect, it } from 'vitest'
import type { ParamSchema } from './api'
import { enumLabel, initialParams } from './schema'

const SCHEMA: ParamSchema = {
  type: 'object',
  properties: {
    steps: { type: 'integer', default: 30 },
    scheduler: {
      type: 'string',
      default: 'dpmpp_2m',
      enum: ['dpmpp_2m', 'euler'],
      'x-enum-labels': ['DPM++ 2M', 'Euler'],
    },
  },
}

describe('initialParams', () => {
  it('keeps saved values and drops choices no longer offered', () => {
    expect(initialParams(SCHEMA, { steps: 20, scheduler: 'euler', gone: 1 })).toEqual({
      steps: 20,
      scheduler: 'euler',
    })
    expect(initialParams(SCHEMA, { scheduler: 'dpmpp_2m_karras' })).toEqual({
      steps: 30,
      scheduler: 'dpmpp_2m',
    })
  })
})

describe('enumLabel', () => {
  it('labels a value, falling back to the value itself', () => {
    const prop = SCHEMA.properties.scheduler
    expect(enumLabel(prop, 'euler')).toBe('Euler')
    expect(enumLabel(prop, 'ddim')).toBe('ddim')
    expect(enumLabel(undefined, 'ddim')).toBe('ddim')
    expect(enumLabel(prop, null)).toBeNull()
  })
})
