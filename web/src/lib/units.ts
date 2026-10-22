// Colab compute units: what a session has used and how long the balance lasts.

import type { SessionSnapshot, Units } from '@/api/types'

const HOUR = 3_600_000

type Session = NonNullable<SessionSnapshot['session']>

/** Units a session has used: its VM's rate since allocation, or null if unknown. */
export function sessionSpend(session: Session | null, now: number): number | null {
  if (!session?.allocated_at || session.unit_rate == null) return null
  const end = session.ended_at ? new Date(session.ended_at).getTime() : now
  const hours = Math.max(0, end - new Date(session.allocated_at).getTime()) / HOUR
  return session.unit_rate * hours
}

/** The balance now: the last one Colab gave, run down at the account's rate since. */
export function balanceNow(units: Units, now: number): number {
  const hours = Math.max(0, now - new Date(units.checked_at).getTime()) / HOUR
  return Math.max(0, units.balance - units.rate * hours)
}

/** Hours `balance` lasts at `rate`, or null when nothing is running. */
export function hoursLeft(balance: number, rate: number): number | null {
  return rate > 0 ? balance / rate : null
}

/** "under an hour", "121 hours", "6 days". */
export function duration(hours: number): string {
  if (hours < 1) return 'under an hour'
  if (hours < 48) return `${Math.floor(hours)} ${Math.floor(hours) === 1 ? 'hour' : 'hours'}`
  return `${Math.floor(hours / 24)} days`
}

export function formatUnits(units: number): string {
  return units.toFixed(2)
}
