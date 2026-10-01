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
