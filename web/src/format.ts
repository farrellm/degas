import type { Job } from './api'

/** "models/sdxl/juggernautXL_v10.safetensors" → "juggernautXL v10". */
export function modelName(path: string): string {
  const file = path.split('/').pop() ?? path
  return file.replace(/\.(safetensors|ckpt|pt|bin)$/, '').replaceAll('_', ' ')
}

export function size(w: unknown, h: unknown): string {
  return `${String(w)} × ${String(h)}`
}

const PHASES: Record<string, string> = {
  copy: 'Copying model',
  load: 'Loading model',
  denoise: 'Denoising',
  decode: 'Decoding',
  encode: 'Encoding',
}

/** What a running job is doing to the given item, for its sketch tile. */
export function phaseText(job: Job): string {
  const p = job.progress
  if (!p) return 'Starting'
  const phase = PHASES[p.phase] ?? p.phase
  if (p.phase === 'copy' && p.steps > 0) {
    return `${phase} ${String(Math.round((100 * p.step) / p.steps))}%`
  }
  return p.steps > 0 && p.phase === 'denoise'
    ? `${phase} ${String(p.step)}/${String(p.steps)}`
    : phase
}

/** How far the current item is drawn (0..1), or null while it can't be measured. */
export function itemFraction(job: Job): number | null {
  const p = job.progress
  if (!p || p.steps === 0) return null
  if (p.phase === 'denoise') return p.step / p.steps
  if (p.phase === 'decode' || p.phase === 'encode') return 1
  return null
}

export const GiB = 1024 ** 3

export const GPU_VRAM: Record<string, string> = {
  T4: '16 GB',
  L4: '24 GB',
  A100: '40–80 GB',
  H100: '80 GB',
}
