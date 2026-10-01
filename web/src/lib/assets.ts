import type { Asset, Family, Variant } from '@/api/types'

import { MB, modelName } from './format'

/** Sidecar label, else a readable file name. */
export function assetLabel(path: string, assets: Asset[] | undefined): string {
  return assets?.find((a) => a.path === path)?.sidecar?.label ?? modelName(path)
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

/** The variant a model belongs to: the one whose Drive folder holds it. */
export function variantFor(path: string, family: Family): Variant | undefined {
  const inDir = family.variants.find(
    (v) => v.model_dir && (path === v.model_dir || path.startsWith(`${v.model_dir}/`)),
  )
  if (inDir) return inDir
  return family.variants.find((v) => !v.model_dir)
}
