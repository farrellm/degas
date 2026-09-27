import { useQuery } from '@tanstack/react-query'
import { api, isActive, type Asset } from './api'
import { modelName } from './format'

/** The whole Drive index, fetched once and filtered on the phone. */
export function useAssets() {
  return useQuery({ queryKey: ['assets'], queryFn: api.assets })
}

/** Sidecar label, else a readable file name. */
export function assetLabel(path: string, assets: Asset[] | undefined): string {
  return assets?.find((a) => a.path === path)?.sidecar?.label ?? modelName(path)
}

/** Paths copied to the running session's GPU, or null when there's no session to ask. */
export function useCachedPaths(): Set<string> | null {
  const session = useQuery({ queryKey: ['session'], queryFn: api.session })
  const cache = session.data?.worker?.cache
  if (!isActive(session.data) || !cache) return null
  return new Set(cache.files.map((f) => f.path))
}

const GB = 1000 ** 3
const MB = 1000 ** 2

/** "6.9 GB", "144 MB". */
export function bytes(n: number): string {
  if (n >= GB) return `${(n / GB).toFixed(n >= 100 * GB ? 0 : 1)} GB`
  return `${String(Math.max(1, Math.round(n / MB)))} MB`
}

// Measured Drive → VM copy speed (Phase 1 live test: ~77 MB/s).
const COPY_BYTES_PER_S = 70 * MB

/** How long a cold copy to the GPU takes, e.g. "about 100 s to copy". */
export function copyEstimate(size: number): string {
  const s = size / COPY_BYTES_PER_S
  if (s < 10) return 'a few seconds to copy'
  if (s < 120) return `about ${String(Math.round(s / 10) * 10)} s to copy`
  return `about ${String(Math.round(s / 60))} min to copy`
}
