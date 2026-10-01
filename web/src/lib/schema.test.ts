import { describe, expect, it } from 'vitest'

import type { ParamProp, ParamSchema } from '@/api/types'

import { enumLabel, initialParams, resetLabel } from './schema'

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

describe('resetLabel', () => {
  const steps: ParamProp = { type: 'integer', default: 30 }
  const scheduler: ParamProp = {
    type: 'string',
    default: 'dpmpp_2m',
    enum: ['dpmpp_2m', 'euler'],
    'x-enum-labels': ['DPM++ 2M', 'Euler'],
  }
  it('names the default only once the value has left it', () => {
    expect(resetLabel(steps, 30)).toBeNull()
    expect(resetLabel(steps, 32)).toBe('30')
    expect(resetLabel({ type: 'number', default: 3.5 }, 0.1 + 3.4)).toBeNull()
  })

  it("uses an enum default's label", () => {
    expect(resetLabel(scheduler, 'dpmpp_2m')).toBeNull()
    expect(resetLabel(scheduler, 'euler')).toBe('DPM++ 2M')
  })

  it('has nothing to reset to without a default', () => {
    expect(resetLabel({ type: 'integer' }, 4)).toBeNull()
  })
})
