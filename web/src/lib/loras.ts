import type { Asset, LoraEntry, LoraPair, Variant } from '@/api/types'

import { assetLabel } from './assets'
import { modelName } from './format'

export const isPair = (l: LoraEntry): l is LoraPair => !('path' in l)

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
    if (!pair) {
      // A single file offered to an A14B variant (by its sidecar) goes into both experts.
      const ref = { path: row.path, weight }
      return paired ? { high: ref, low: { ...ref } } : ref
    }
    return {
      ...(pair.high && { high: { path: pair.high.path, weight } }),
      ...(pair.low && { low: { path: pair.low.path, weight } }),
    }
  }
  return { rows, entryFor }
}

/** The files behind a LoRA in the form (both halves of a pair, once if they share a file). */
export function loraPaths(entry: LoraEntry): string[] {
  if (!isPair(entry)) return [entry.path]
  const paths = [entry.high?.path, entry.low?.path].filter((p): p is string => !!p)
  return [...new Set(paths)]
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

/** Whether two entries are the same LoRA files, whatever their weights. */
export function sameLora(a: LoraEntry, b: LoraEntry): boolean {
  return loraKey(a) === loraKey(b)
}
