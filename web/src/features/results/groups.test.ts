import { describe, expect, it } from 'vitest'

import type { Job, Result, Spec } from '@/api/types'

import { buildGroups, deleteQuestion, modelLine, reorder } from './groups'

const SPEC: Spec = {
  family: 'sdxl',
  variant: 'base',
  mode: 't2i',
  model: { path: 'models/sdxl/juggernaut_v10.safetensors' },
  params: { prompt: 'a lighthouse' },
}

const job = (id: string, status: Job['status'], extra: Partial<Job> = {}): Job => ({
  id,
  status,
  queue_position: 0,
  spec: SPEC,
  seeds: [1],
  created_at: '2026-09-27T12:00:00Z',
  started_at: null,
  finished_at: null,
  error: null,
  progress: null,
  ...extra,
})

const result = (id: string, jobId: string, extra: Partial<Result> = {}): Result => ({
  id,
  job_id: jobId,
  item_index: 0,
  blob_sha: id,
  media_type: 'image/png',
  seed: 1,
  width: 1024,
  height: 1024,
  duration: null,
  segments: null,
  created_at: '2026-09-27T12:00:00Z',
  expires_at: null,
  library_id: null,
  spec: SPEC,
  ...extra,
})

describe('buildGroups', () => {
  it('puts the running job first, then the queue in order, then finished work newest first', () => {
    const jobs = [
      job('old', 'done'),
      job('q2', 'queued', { queue_position: 2 }),
      job('new', 'done'),
      job('q1', 'queued', { queue_position: 1 }),
      job('run', 'running'),
    ]
    const results = [
      result('r-old', 'old', { created_at: '2026-09-27T12:01:00Z' }),
      result('r-new', 'new', { created_at: '2026-09-27T12:05:00Z' }),
    ]
    expect(buildGroups(jobs, results).map((g) => g.id)).toEqual(['run', 'q1', 'q2', 'new', 'old'])
  })

  it('orders a group by item, and keeps a stitched chain apart from its clips', () => {
    const jobs = [job('j', 'done')]
    const results = [
      result('b', 'j', { item_index: 1 }),
      result('a', 'j', { item_index: 0 }),
      result('chain', 'j', { segments: [] }),
    ]
    const groups = buildGroups(jobs, results)
    expect(groups.map((g) => [g.id, g.chain, g.results.map((r) => r.id)])).toEqual(
      expect.arrayContaining([
        ['j', false, ['a', 'b']],
        ['j:chain', true, ['chain']],
      ]),
    )
  })

  it('leaves out a cancelled job with nothing to show', () => {
    expect(buildGroups([job('c', 'cancelled')], [])).toEqual([])
  })
})

describe('reorder', () => {
  it('moves a queued job, reusing the queue positions already handed out', () => {
    const jobs = [
      job('a', 'queued', { queue_position: 10 }),
      job('b', 'queued', { queue_position: 20 }),
      job('c', 'queued', { queue_position: 30 }),
      job('run', 'running', { queue_position: 5 }),
    ]
    const moved = reorder(jobs, 'c', 0)
    expect(moved.map((j) => [j.id, j.queue_position])).toEqual([
      ['a', 20],
      ['b', 30],
      ['c', 10],
      ['run', 5],
    ])
  })
})

describe('captions', () => {
  it('names the model and counts its LoRAs', () => {
    expect(modelLine(SPEC, undefined)).toBe('juggernaut v10')
    const loras = [
      { path: 'a', weight: 1 },
      { path: 'b', weight: 1 },
    ]
    expect(modelLine({ ...SPEC, loras }, undefined)).toBe('juggernaut v10 + 2 LoRAs')
  })

  it('asks before deleting, saying what stays', () => {
    expect(deleteQuestion([])).toBe('Delete this failed job?')
    expect(deleteQuestion([result('a', 'j')])).toBe('Delete this image?')
    expect(deleteQuestion([result('a', 'j'), result('b', 'j', { library_id: 'k' })])).toBe(
      'Delete these 2 images? Kept ones stay in the library.',
    )
    expect(deleteQuestion([result('a', 'j', { media_type: 'video/mp4', library_id: 'k' })])).toBe(
      'Delete this clip? They stay in the library.',
    )
  })
})
