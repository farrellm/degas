import { readStored, writeStored } from '@/lib/storage'

const GPU_KEY = 'degas.session.gpu'

/** The GPU the last session was started on, to offer again. */
export function lastGpu(): string {
  return readStored(GPU_KEY) ?? 'L4'
}

export function rememberGpu(gpu: string) {
  writeStored(GPU_KEY, gpu)
}
