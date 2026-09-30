import type { Job } from './api'

/** "models/sdxl/studioXL_v10.safetensors" → "studioXL v10". */
export function modelName(path: string): string {
  const file = path.split('/').pop() ?? path
  return file.replace(/\.(safetensors|ckpt|pt|bin)$/, '').replaceAll('_', ' ')
}

export function size(w: unknown, h: unknown): string {
  return `${String(w)} × ${String(h)}`
}

/** "0:05": a clip's length, as on a video tile. */
export function clock(seconds: number): string {
  const s = Math.max(1, Math.round(seconds))
  return `${String(Math.floor(s / 60))}:${String(s % 60).padStart(2, '0')}`
}

/** "5.0 s" or "12 s": a clip's length in running text. */
export function duration(seconds: number): string {
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

export const GiB = 1024 ** 3

/** Colab GPUs, ascending capability (matches the server's `GPUS`). */
export const GPUS = ['T4', 'L4', 'A100', 'H100']

/** Whether `gpu` is below `min`, i.e. a model needing `min` may run slowly on it or not at all. */
export function belowGpu(gpu: string, min: string): boolean {
  return GPUS.indexOf(gpu) < GPUS.indexOf(min)
}

/** "needs an L4": a model's minimum GPU, or null when any GPU runs it. */
export function needsGpu(min: string): string | null {
  return GPUS.indexOf(min) > 0 ? `needs an ${min}` : null
}

export const GPU_VRAM: Record<string, string> = {
  T4: '16 GB',
  L4: '24 GB',
  A100: '40–80 GB',
  H100: '80 GB',
}
