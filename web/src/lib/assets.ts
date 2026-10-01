import { useQuery } from '@tanstack/react-query'
import { api, isActive, isPair, type Asset, type Family, type LoraEntry, type Variant } from './api'
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

/** The variant a model belongs to: the one whose Drive folder holds it. */
export function variantFor(path: string, family: Family): Variant | undefined {
  const inDir = family.variants.find(
    (v) => v.model_dir && (path === v.model_dir || path.startsWith(`${v.model_dir}/`)),
  )
  if (inDir) return inDir
  return family.variants.find((v) => !v.model_dir)
}

// foo_high_noise.safetensors / foo_low_noise.safetensors (design §5)
const HALF = /^(.*?)[_-](high|low)[_-]?noise\.[^.]+$/i
const PAIR_PREFIX = 'pair:'

/** "Motion v2" for "loras/wan22/motion_v2_high_noise.safetensors". */
export function pairName(path: string): string {
  const file = path.split('/').pop() ?? path
  return modelName(file.replace(HALF, '$1'))
}

interface Pair {
  high?: Asset
  low?: Asset
}

/**
 * The LoRAs to offer for a variant, as picker rows. A14B pairs become one row each
 * (a synthetic asset whose path starts with `pair:`); `entryFor` turns a row into a spec entry.
 */
export function loraChoices(index: Asset[] | undefined, variant: Variant | undefined) {
  const paired = variant?.lora_format === 'paired_hi_lo'
  const pairs = new Map<string, Pair>()
  const inPair = new Set<string>()
  const byPath = new Map((index ?? []).map((a) => [a.path, a]))
  for (const a of index ?? []) {
    const dir = a.path.slice(0, a.path.lastIndexOf('/') + 1)
    const declared = a.sidecar?.pair
    if (declared) {
      const high = byPath.get(dir + declared.high)
      const low = byPath.get(dir + declared.low)
      if (high || low) {
        pairs.set(dir + (high ? pairName(high.path) : pairName(declared.low)), { high, low })
        for (const h of [high, low]) if (h) inPair.add(h.path)
      }
    }
  }
  for (const a of index ?? []) {
    const m = HALF.exec(a.path.split('/').pop() ?? '')
    if (!m || inPair.has(a.path)) continue
    const key = a.path.slice(0, a.path.lastIndexOf('/') + 1) + (m[1] ?? '')
    const pair = pairs.get(key) ?? {}
    pair[(m[2] ?? '').toLowerCase() === 'high' ? 'high' : 'low'] = a
    pairs.set(key, pair)
    inPair.add(a.path)
  }

  const fits = (a: Asset, isPairRow: boolean) =>
    a.sidecar?.variants
      ? !!variant && a.sidecar.variants.includes(variant.id)
      : isPairRow === paired
  const rows: Asset[] = []
  const pairOf = new Map<string, Pair>()
  for (const [key, pair] of pairs) {
    const first = pair.high ?? pair.low
    if (!first || !fits(first, true)) continue
    const path = PAIR_PREFIX + key
    pairOf.set(path, pair)
    rows.push({
      ...first,
      path,
      size: (pair.high?.size ?? 0) + (pair.low?.size ?? 0) || null,
      sidecar: { label: pairName(first.path), ...pair.high?.sidecar, ...pair.low?.sidecar },
      preview_thumb: pair.high?.preview_thumb ?? pair.low?.preview_thumb ?? null,
    })
  }
  for (const a of index ?? []) {
    if (!inPair.has(a.path) && fits(a, false)) rows.push(a)
  }
  rows.sort((a, b) => assetLabel(a.path, rows).localeCompare(assetLabel(b.path, rows)))

  const entryFor = (row: Asset): LoraEntry => {
    const weight = row.sidecar?.default_weight ?? 1
    const pair = pairOf.get(row.path)
    if (!pair) return { path: row.path, weight }
    return {
      ...(pair.high && { high: { path: pair.high.path, weight } }),
      ...(pair.low && { low: { path: pair.low.path, weight } }),
    }
  }
  return { rows, entryFor }
}

/** The files behind a LoRA in the form (both halves of a pair). */
export function loraPaths(entry: LoraEntry): string[] {
  return isPair(entry)
    ? [entry.high?.path, entry.low?.path].filter((p): p is string => !!p)
    : [entry.path]
}

/** The index entries behind a LoRA in the form (both halves of a pair). */
export function loraAssets(entry: LoraEntry, index: Asset[] | undefined): (Asset | undefined)[] {
  return loraPaths(entry).map((p) => index?.find((a) => a.path === p))
}

/** A LoRA's display name: the sidecar label, else its file (or its pair's shared) name. */
export function loraLabel(entry: LoraEntry, index: Asset[] | undefined): string {
  const found = loraAssets(entry, index).find((a) => a?.sidecar?.label)
  if (found?.sidecar?.label) return found.sidecar.label
  if (!isPair(entry)) return modelName(entry.path)
  return pairName(entry.high?.path ?? entry.low?.path ?? '')
}

/** A stable identity for a LoRA in the form. */
export function loraKey(entry: LoraEntry): string {
  return isPair(entry) ? `${entry.high?.path ?? ''}|${entry.low?.path ?? ''}` : entry.path
}
