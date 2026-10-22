import { describe, expect, it } from 'vitest'

import { balanceNow, duration, hoursLeft, sessionSpend } from './units'

const T0 = Date.parse('2026-10-22T12:00:00Z')
const session = {
  id: 's1',
  gpu: 'L4',
  high_mem: false,
  state: 'ready' as const,
  started_at: '2026-10-22T11:58:00Z',
  ended_at: null,
  last_activity_at: '2026-10-22T12:00:00Z',
  error: null,
  allocated_at: '2026-10-22T12:00:00Z',
  unit_rate: 1.6,
}

describe('sessionSpend', () => {
  it('runs from allocation at the VM rate', () => {
    expect(sessionSpend(session, T0 + 90 * 60_000)).toBeCloseTo(2.4)
  })

  it('stops at the end', () => {
    const ended = { ...session, ended_at: '2026-10-22T13:00:00Z' }
    expect(sessionSpend(ended, T0 + 10 * 3_600_000)).toBeCloseTo(1.6)
  })

  it('is unknown without a rate or before allocation', () => {
    expect(sessionSpend({ ...session, unit_rate: null }, T0)).toBeNull()
    expect(sessionSpend({ ...session, allocated_at: null }, T0)).toBeNull()
    expect(sessionSpend(null, T0)).toBeNull()
  })
})

describe('balanceNow', () => {
  const units = { balance: 100, rate: 2, assignments: 1, checked_at: '2026-10-22T12:00:00Z' }

  it('runs down at the account rate since the last check', () => {
    expect(balanceNow(units, T0 + 30 * 60_000)).toBeCloseTo(99)
  })

  it('never goes below zero', () => {
    expect(balanceNow(units, T0 + 100 * 3_600_000)).toBe(0)
  })
})

describe('duration', () => {
  it('reads in hours, then days', () => {
    expect(hoursLeft(100, 0)).toBeNull()
    expect(duration(0.5)).toBe('under an hour')
    expect(duration(1.2)).toBe('1 hour')
    expect(hoursLeft(196, 2)).toBe(98)
    expect(duration(121)).toBe('5 days')
    expect(duration(47.9)).toBe('47 hours')
  })
})
