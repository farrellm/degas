import type { Job } from '@/api/types'

/** "models/sdxl/juggernautXL_v10.safetensors" → "juggernautXL v10". */
export function modelName(path: string): string {
  const file = path.split('/').pop() ?? path
  return file.replace(/\.(safetensors|ckpt|pt|bin)$/, '').replaceAll('_', ' ')
}

export function formatSize(w: unknown, h: unknown): string {
  return `${String(w)} × ${String(h)}`
}

/** "0:05": a clip's length, as on a video tile. */
export function formatClock(seconds: number): string {
  const s = Math.max(1, Math.round(seconds))
  return `${String(Math.floor(s / 60))}:${String(s % 60).padStart(2, '0')}`
}

/** "5.0 s" or "12 s": a clip's length in running text. */
export function formatDuration(seconds: number): string {
  return seconds < 10 ? `${seconds.toFixed(1)} s` : `${String(Math.round(seconds))} s`
}

const PHASES: Record<string, string> = {
  load: 'Loading model',
  denoise: 'Denoising',
  decode: 'Decoding',
  encode: 'Encoding',
}

/** What a running job is doing to the given item, for its sketch tile. */
export function phaseText(job: Job): string {
  const p = job.progress
  if (!p) return 'Starting'
  if (p.phase === 'copy') return `Copying ${copyText(job)}`
  const phase = PHASES[p.phase] ?? p.phase
  return p.steps > 0 && p.phase === 'denoise'
    ? `${phase} ${String(p.step)}/${String(p.steps)}`
    : phase
}

/** "LoRA 45%": what a job's `copy` progress is copying to the GPU, and how far along. */
export function copyText(job: Job): string {
  const p = job.progress
  const what = p?.asset?.startsWith('loras/') ? 'LoRA' : 'model'
  return p && p.steps > 0 ? `${what} ${String(Math.round((100 * p.step) / p.steps))}%` : what
}

/** How far the current item is drawn (0..1), or null while it can't be measured. */
export function itemFraction(job: Job): number | null {
  const p = job.progress
  if (!p || p.steps === 0) return null
  if (p.phase === 'denoise') return p.step / p.steps
  if (p.phase === 'decode' || p.phase === 'encode') return 1
  return null
}

const GB = 1000 ** 3
export const MB = 1000 ** 2

/** "6.9 GB", "144 MB". */
export function formatBytes(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(n >= 100 * GB ? 0 : 1)} GB`
  return `${String(Math.max(1, Math.round(n / MB)))} MB`
}
