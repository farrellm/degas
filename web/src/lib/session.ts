import type { SessionSnapshot } from '@/api/types'

/** Whether a session is starting or running (not stopping, stopped or failed). */
export function isActive(snapshot: SessionSnapshot | undefined): boolean {
  const state = snapshot?.session?.state
  return state === 'starting' || state === 'ready' || state === 'busy'
}

/** Whether the session's GPU can take work now: preprocessors (SAM, traces, faces) need it. */
export function isGpuReady(snapshot: SessionSnapshot | undefined): boolean {
  const state = snapshot?.session?.state
  return state === 'ready' || state === 'busy'
}
