import { describe, expect, it } from 'vitest'

import { parseEvent } from './events'

const JOB = {
  id: 'j1',
  session_id: null,
  status: 'running',
  queue_position: 1,
  spec: { family: 'sdxl', variant: 'base', mode: 't2i', model: { path: 'm' }, params: {} },
  seeds: [7],
  created_at: '2026-10-08T12:00:00Z',
  started_at: null,
  finished_at: null,
  error: null,
  runtime: null,
  progress: null,
}

const event = (data: unknown) => parseEvent(JSON.stringify(data))

describe('parseEvent', () => {
  it('reads an event, keeping fields beyond the ones it checks', () => {
    const progress = { type: 'progress', t: 'progress', job: 'j1', item: 0, phase: 'denoise' }
    expect(event({ ...progress, step: 3, steps: 30 })).toEqual({ ...progress, step: 3, steps: 30 })
    expect(event({ type: 'job', job: JOB })).toEqual({ type: 'job', job: JOB })
    expect(event({ type: 'swept', results: 2, jobs: 1 })).toMatchObject({ type: 'swept' })
  })

  it('is null for what this app can’t read', () => {
    expect(parseEvent('{not json')).toBeNull()
    expect(event({ type: 'weather' })).toBeNull()
    expect(event({ type: 'job', job: { ...JOB, id: undefined } })).toBeNull()
    expect(event({ type: 'job', job: { ...JOB, status: 'paused' } })).toBeNull()
    expect(event({ type: 'progress', job: 'j1', item: 0, phase: 'denoise', step: '3' })).toBeNull()
  })
})
