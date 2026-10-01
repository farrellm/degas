import { useEffect, useState } from 'react'

/** Current time, re-rendered every `intervalMs`. */
export function useNow(intervalMs: number) {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const id = setInterval(() => {
      setNow(Date.now())
    }, intervalMs)
    return () => {
      clearInterval(id)
    }
  }, [intervalMs])
  return now
}

/** `m:ss` until `deadline`, or null without one. */
export function countdown(deadline: string | null, now: number): string | null {
  if (!deadline) return null
  const s = Math.max(0, Math.round((new Date(deadline).getTime() - now) / 1000))
  return `${String(Math.floor(s / 60))}:${String(s % 60).padStart(2, '0')}`
}

/** Hours (fractional) until `iso`. */
export function hoursLeft(iso: string, now: number): number {
  return (new Date(iso).getTime() - now) / 3_600_000
}

/** Hours or minutes left before `expiresAt`, e.g. "17 h left". */
export function timeLeft(expiresAt: string | null, now: number): string | null {
  if (!expiresAt) return null
  const hours = hoursLeft(expiresAt, now)
  if (hours <= 0) return 'Deleting soon'
  return hours >= 1
    ? `${String(Math.floor(hours))} h left`
    : `${String(Math.ceil(hours * 60))} min left`
}

/** Clock time for today, otherwise a short date. */
export function shortTime(iso: string, now: number): string {
  const d = new Date(iso)
  const sameDay = new Date(now).toDateString() === d.toDateString()
  return sameDay
    ? d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
    : d.toLocaleDateString([], { month: 'short', day: 'numeric' })
}

/** "3 h ago" style age. */
export function ago(iso: string, now: number): string {
  const min = Math.round((now - new Date(iso).getTime()) / 60_000)
  if (min < 1) return 'just now'
  if (min < 60) return `${String(min)} min ago`
  const h = Math.round(min / 60)
  if (h < 48) return `${String(h)} h ago`
  return `${String(Math.round(h / 24))} days ago`
}
